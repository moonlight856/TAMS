import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import glob
import shutil
import torch
import h5py
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
from random import randint, seed  # randint/seed kept for 50salads path


def get_edge_info(num_frame, args):
    skip = args.skip_factor

    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source = []
    node_target = []
    edge_attr = []
    for i in range(num_frame):
        for j in range(num_frame):
            # Frame difference between the i-th and j-th nodes
            frame_diff = i - j

            # The edge ij connects the i-th node and j-th node
            # Positive edge_attr indicates that the edge ij is backward (negative: forward)
            if abs(frame_diff) <= args.tauf:
                node_source.append(i)
                node_target.append(j)
                edge_attr.append(np.sign(frame_diff))

            # Make additional connections between non-adjacent nodes
            # This can help reduce over-segmentation of predictions in some cases
            elif skip:
                if (frame_diff % skip == 0) and (abs(frame_diff) <= skip * args.tauf):
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

    return node_source, node_target, edge_attr


def generate_sum_temporal_graph(video_data, args, path_graphs):
    """
    Generate temporal graphs of a single video from video summarization data set
    """
    video_id = video_data["global_id"]
    out_folder = video_data["purpose"]
    features = video_data["features"]
    gtscore_1d = np.asarray(video_data["gtscore"], dtype=np.float32)
    num_samples = features.shape[0]
    if gtscore_1d.shape[0] != num_samples:
        picks_np = np.asarray(video_data["picks"], dtype=np.int64)
        if gtscore_1d.shape[0] > num_samples and picks_np.shape[0] == num_samples:
            gtscore_1d = gtscore_1d[picks_np]
        else:
            gtscore_1d = gtscore_1d[:num_samples]
    gtscore = gtscore_1d[:, None]
    labels = torch.tensor(video_data['gtsummary'])
    usr_smy = torch.tensor(video_data['user_summary'])
    picks = torch.tensor(video_data['picks'])

    cp_raw = video_data.get('change_points')
    if cp_raw is not None and isinstance(cp_raw, np.ndarray) and cp_raw.ndim == 2 and cp_raw.shape[1] == 2:
        change_points = torch.tensor(cp_raw.astype(np.int64), dtype=torch.long)
    else:
        change_points = torch.zeros((0, 2), dtype=torch.long)
    n_frames_full = torch.tensor([video_data.get('n_frames', num_samples)], dtype=torch.long)

    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source, node_target, edge_attr = get_edge_info(num_samples, args)

    graphs = Data(x=torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32),
                  g=video_id,
                  edge_index=torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                  edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                  y=torch.tensor(gtscore, dtype=torch.float32),
                  labels=labels, user_summary=usr_smy, picks=picks,
                  change_points=change_points, n_frames_full=n_frames_full)

    torch.save(graphs, os.path.join(path_graphs, f'{out_folder}/{video_id}.pt'))


def generate_temporal_graph(data_file, args, path_graphs, actions, train_ids, all_ids):
    """
    Generate temporal graphs of a single video
    """

    video_id = os.path.splitext(os.path.basename(data_file))[0]
    feature = np.transpose(np.load(data_file))
    num_frame = feature.shape[0]

    # Get a list of ground-truth action labels
    with open(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth/{video_id}.txt')) as f:
        label = [actions[line.strip()] for line in f]

    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source, node_target, edge_attr = get_edge_info(num_frame, args)

    # x: features
    # g: global_id
    # edge_index: information on how the graph nodes are connected
    # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
    # y: labels
    graphs = Data(x=torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
                  g=all_ids.index(video_id),
                  edge_index=torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                  edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                  y=torch.tensor(np.array(label, dtype=np.int64)[::args.sample_rate], dtype=torch.long))

    if video_id in train_ids:
        torch.save(graphs, os.path.join(path_graphs, 'train', f'{video_id}.pt'))
    else:
        torch.save(graphs, os.path.join(path_graphs, 'val', f'{video_id}.pt'))


def _resolve_sum_tv_h5_path(root_data, dataset, h5_name_or_path):
    """Resolve H5 path for SumMe/TVSum; supports basename or absolute/relative file path."""
    if h5_name_or_path.endswith('.h5') and os.path.isfile(h5_name_or_path):
        return h5_name_or_path
    candidate = os.path.join(root_data, f'annotations/{dataset}/{h5_name_or_path}.h5')
    if os.path.isfile(candidate):
        return candidate
    # Also allow passing basename with .h5
    if h5_name_or_path.endswith('.h5'):
        candidate2 = os.path.join(root_data, f'annotations/{dataset}/{h5_name_or_path}')
        if os.path.isfile(candidate2):
            return candidate2
    raise FileNotFoundError(
        f'Cannot find h5 file from --features={h5_name_or_path}. Tried:\n'
        f'  {candidate}\n'
        f'  {os.path.join(root_data, f"annotations/{dataset}/{h5_name_or_path}") if h5_name_or_path.endswith(".h5") else "(n/a)"}\n'
        f'  {h5_name_or_path}'
    )


def _build_feature_matrix_from_modalities(video_key, h5_vis, h5_text=None, h5_audio=None, modality_mode='v'):
    """Return [T, D] merged features according to modality_mode in {'v','vt','va','vta'}."""
    v = np.array(h5_vis.get(video_key + '/features'), dtype=np.float32)
    if modality_mode == 'v':
        return v

    if modality_mode == 'va':
        if h5_audio is None:
            raise ValueError('modality_mode=va requires audio features, but audio h5 is None')
        if video_key not in h5_audio:
            raise KeyError(f'{video_key} not found in audio h5')
        a = np.array(h5_audio.get(video_key), dtype=np.float32)
        if v.shape[0] != a.shape[0]:
            raise ValueError(f'T mismatch for {video_key}: visual T={v.shape[0]}, audio T={a.shape[0]}')
        return np.concatenate([v, a], axis=1).astype(np.float32)

    if h5_text is None:
        raise ValueError('modality_mode requires text features, but text h5 is None')
    if video_key not in h5_text:
        raise KeyError(f'{video_key} not found in text h5')
    t = np.array(h5_text.get(video_key), dtype=np.float32)

    if v.shape[0] != t.shape[0]:
        raise ValueError(f'T mismatch for {video_key}: visual T={v.shape[0]}, text T={t.shape[0]}')

    if modality_mode == 'vt':
        return np.concatenate([v, t], axis=1).astype(np.float32)

    if h5_audio is None:
        raise ValueError('modality_mode=vta requires audio features, but audio h5 is None')
    if video_key not in h5_audio:
        raise KeyError(f'{video_key} not found in audio h5')
    a = np.array(h5_audio.get(video_key), dtype=np.float32)

    if v.shape[0] != a.shape[0]:
        raise ValueError(f'T mismatch for {video_key}: visual T={v.shape[0]}, audio T={a.shape[0]}')

    return np.concatenate([v, t, a], axis=1).astype(np.float32)


def _load_sum_tv_dataset(args):
    """
    Load SumMe/TVSum data and return a list-like dataset indexed by (video_id-1).

    Backward compatible behavior (default):
      --modality_mode v, and --features points to a single H5 with /video_x/features.

    New behavior for separated modality H5:
      --modality_mode vt/vta with --text_h5 / --audio_h5.
    """
    path_vis = _resolve_sum_tv_h5_path(args.root_data, args.dataset, args.features)

    # Open optional text/audio h5 if needed
    h5_text = None
    h5_audio = None
    if args.modality_mode in ('vt', 'vta'):
        if not args.text_h5:
            raise ValueError('--modality_mode vt/vta requires --text_h5')
        text_path = _resolve_sum_tv_h5_path(args.root_data, args.dataset, args.text_h5)
        h5_text = h5py.File(text_path, 'r')
    if args.modality_mode in ('va', 'vta'):
        if not args.audio_h5:
            raise ValueError(f'--modality_mode {args.modality_mode} requires --audio_h5')
        audio_path = _resolve_sum_tv_h5_path(args.root_data, args.dataset, args.audio_h5)
        h5_audio = h5py.File(audio_path, 'r')

    try:
        with h5py.File(path_vis, 'r') as hdf:
            all_videos = list(hdf.keys())

            all_ids = []
            dataset = [None] * len(all_videos)
            for video in all_videos:
                idx = int(video.split('_')[1])
                all_ids.append(idx - 1)

                data = {}
                data['global_id'] = idx
                data['purpose'] = 'train'
                data['features'] = _build_feature_matrix_from_modalities(
                    video,
                    hdf,
                    h5_text=h5_text,
                    h5_audio=h5_audio,
                    modality_mode=args.modality_mode,
                )
                data['gtscore'] = np.array(hdf.get(video + '/gtscore'))
                data['gtsummary'] = np.array(hdf.get(video + '/gtsummary'))
                data['user_summary'] = np.array(hdf.get(video + '/user_summary'))
                data['picks'] = np.array(hdf.get(video + '/picks'))

                cp_key = video + '/change_points'
                nf_key = video + '/n_frames'
                data['change_points'] = np.array(hdf.get(cp_key)) if cp_key in hdf else None
                data['n_frames'] = int(hdf.get(nf_key)[()]) if nf_key in hdf else int(data['gtscore'].shape[0])

                dataset[idx - 1] = data
    finally:
        if h5_text is not None:
            h5_text.close()
        if h5_audio is not None:
            h5_audio.close()

    return dataset, all_ids


def _graph_base_name(args):
    """Graph folder base name; keep old behavior if --graph_tag is empty."""
    base = f'{args.dataset}_{args.tauf}_{args.skip_factor}'
    if args.graph_tag:
        return f'{base}_{args.graph_tag}'
    return base


if __name__ == '__main__':
    """
    Generate temporal graphs from the extracted features
    """

    parser = argparse.ArgumentParser()
    # Default paths for the training process
    parser.add_argument('--root_data', type=str, help='Root directory to the data', default='./data')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default='50salads')
    parser.add_argument('--features', type=str, help='Name/path of visual or merged features', required=True)

    # Hyperparameters for the graph generation
    parser.add_argument('--tauf', type=int, help='Maximum frame difference between neighboring nodes', required=True)
    parser.add_argument('--skip_factor', type=int, help='Make additional connections between non-adjacent nodes', default=10)
    parser.add_argument('--sample_rate', type=int, help='Downsampling rate for the input', default=2)

    # New (SumMe/TVSum only): separated modality H5 support + ablation-friendly graph naming
    parser.add_argument('--modality_mode', type=str, default='v', choices=['v', 'vt', 'va', 'vta'],
                        help='For SumMe/TVSum: build graph x from V-only, V+T, V+A, or V+T+A')
    parser.add_argument('--text_h5', type=str, default=None,
                        help='SumMe/TVSum text feature h5 basename/path (required for vt/vta)')
    parser.add_argument('--audio_h5', type=str, default=None,
                        help='SumMe/TVSum audio feature h5 basename/path (required for vta)')
    parser.add_argument('--graph_tag', type=str, default='',
                        help='Optional suffix for graph folder, e.g. vonly / vt / vta (avoids overwriting)')

    args = parser.parse_args()

    print('This process might take a few minutes')

    actions = {}
    all_ids = []
    if args.dataset == '50salads':

        # Build a mapping from action classes to action ids
        with open(os.path.join(args.root_data, f'annotations/{args.dataset}/mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                actions[cls] = int(aid)

        # Get a list of all video ids
        all_ids = sorted([os.path.splitext(v)[0] for v in
                          os.listdir(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth'))])

        # Iterate over different splits
        list_splits = sorted(os.listdir(os.path.join(args.root_data, f'features/{args.features}')))
        for split in list_splits:
            # Get a list of training video ids
            with open(os.path.join(args.root_data, f'annotations/{args.dataset}/splits/train.{split}.bundle')) as f:
                train_ids = [os.path.splitext(line.strip())[0] for line in f]

            path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.tauf}_{args.skip_factor}/{split}')
            os.makedirs(os.path.join(path_graphs, 'train'), exist_ok=True)
            os.makedirs(os.path.join(path_graphs, 'val'), exist_ok=True)

            list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{split}/*.npy')))

            num_workers = min(20, (os.cpu_count() or 4))
            with Pool(processes=num_workers) as pool:
                pool.map(partial(generate_temporal_graph, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids), list_data_files)

            print(f'Graph generation for {split} is finished')

    elif args.dataset == 'SumMe' or args.dataset == 'TVSum':

        dataset, all_ids = _load_sum_tv_dataset(args)

        import random as _rng
        _rng.seed(42)
        shuffled_ids = all_ids.copy()
        _rng.shuffle(shuffled_ids)

        n = len(shuffled_ids)
        fold_size = n // 5
        folds = []
        for fi in range(5):
            start = fi * fold_size
            end = start + fold_size if fi < 4 else n
            folds.append(set(shuffled_ids[start:end]))

        graph_base = _graph_base_name(args)
        for split_i in range(1, 6):
            val_set = folds[split_i - 1]
            for i in all_ids:
                dataset[i]['purpose'] = 'val' if i in val_set else 'train'

            path_graphs = os.path.join(args.root_data, f'graphs/{graph_base}/split{split_i}')
            train_dir = os.path.join(path_graphs, 'train')
            val_dir = os.path.join(path_graphs, 'val')
            for d in (train_dir, val_dir):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

            for video_data in dataset:
                generate_sum_temporal_graph(video_data, args, path_graphs)

            print(f'Graph generation for split{split_i} is finished')

        # Small summary for sanity check
        sample_i = all_ids[0]
        sample_dim = dataset[sample_i]['features'].shape[1]
        print(f'Done. modality_mode={args.modality_mode}, feature_dim={sample_dim}, graph_base={graph_base}')
