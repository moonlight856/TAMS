import os
import glob
import torch
import pickle  #nosec
import numpy as np


def get_formatting_data_dict(cfg):
    """
    Get a dictionary that is used to format the results following the formatting rules of the evaluation tool
    """

    root_data = cfg['root_data']
    dataset = cfg['dataset']
    data_dict = {}

    if 'AVA' in cfg['eval_type']:
        # Get a list of the feature files
        features = '_'.join(cfg['graph_name'].split('_')[:-3])
        list_data_files = sorted(glob.glob(os.path.join(root_data, f'features/{features}/val/*.pkl')))

        for data_file in list_data_files:
            video_id = os.path.splitext(os.path.basename(data_file))[0]

            with open(data_file, 'rb') as f:
                data = pickle.load(f) #nosec

            # Get a list of frame_timestamps
            list_fts = sorted([float(frame_timestamp) for frame_timestamp in data.keys()])

            # Iterate over all the frame_timestamps and retrieve the required data for evaluation
            for fts in list_fts:
                frame_timestamp = f'{fts:g}'
                for entity in data[frame_timestamp]:
                    data_dict[entity['global_id']] = {'video_id': video_id,
                                                      'frame_timestamp': frame_timestamp,
                                                      'person_box': entity['person_box'],
                                                      'person_id': entity['person_id']}
    elif 'AS' in cfg['eval_type']:
        # Build a mapping from action ids to action classes
        data_dict['actions'] = {}
        with open(os.path.join(root_data, 'annotations', dataset, 'mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                data_dict['actions'][int(aid)] = cls

        # Get a list of all video ids
        data_dict['all_ids'] = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(root_data, f'annotations/{dataset}/groundTruth'))])

    return data_dict


def get_formatted_preds(cfg, logits, g, data_dict):
    """
    Get a list of formatted predictions from the model output, which is used to compute the evaluation score
    """

    eval_type = cfg['eval_type']
    preds = []
    if 'AVA' in eval_type:
        # Compute scores from the logits
        scores_all = torch.sigmoid(logits.detach().cpu()).numpy()

        # Iterate over all the nodes and get the formatted predictions for evaluation
        for scores, global_id in zip(scores_all, g):
            data = data_dict[global_id]
            video_id = data['video_id']
            frame_timestamp = float(data['frame_timestamp'])
            x1, y1, x2, y2 = [float(c) for c in data['person_box'].split(',')]

            if eval_type == 'AVA_ASD':
                # Line formatted following Challenge #2: http://activity-net.org/challenges/2019/tasks/guest_ava.html
                person_id = data['person_id']
                score = scores.item()
                pred = [video_id, frame_timestamp, x1, y1, x2, y2, 'SPEAKING_AUDIBLE', person_id, score]
                preds.append(pred)

            elif eval_type == 'AVA_AL':
                # Line formatted following Challenge #1: http://activity-net.org/challenges/2019/tasks/guest_ava.html
                for action_id, score in enumerate(scores, 1):
                    pred = [video_id, frame_timestamp, x1, y1, x2, y2, action_id, score]
                    preds.append(pred)
    elif 'AS' in eval_type:
        tmp = logits
        if cfg['use_ref']:
            tmp = logits[-1]

        tmp = torch.softmax(tmp.detach().cpu(), dim=1).max(dim=1)[1].tolist()

        # Upsample the predictions to fairly compare with the ground-truth labels
        preds = []
        for pred in tmp:
            preds.extend([data_dict['actions'][pred]] * cfg['sample_rate'])

        # Pair the final predictions with the video_id
        (g,) = g
        video_id = data_dict['all_ids'][g]
        preds = [(video_id, preds)]

    elif 'VS' in eval_type:
        tmp = logits.squeeze().cpu()
        score_mode = cfg.get("eval_score_mode", "sigmoid")
        eval_temp = float(cfg.get("eval_temperature", 1.0))
        if score_mode == "minmax":
            lo, hi = tmp.min(), tmp.max()
            tmp = ((tmp - lo) / (hi - lo + 1e-8)).numpy()
        elif score_mode == "zscore":
            tmp = ((tmp - tmp.mean()) / (tmp.std() + 1e-8) * eval_temp).numpy()
            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-8)
        elif score_mode == "percentile":
            arr = tmp.numpy()
            from scipy.stats import rankdata as _rankdata
            tmp = (_rankdata(arr) - 1) / max(len(arr) - 1, 1)
        else:
            if cfg.get("eval_center_logits", False):
                tmp = tmp - tmp.mean()
            tmp = torch.sigmoid(tmp * eval_temp).numpy()
        power = float(cfg.get("eval_power", 1.0))
        if power != 1.0:
            tmp = np.power(np.clip(tmp, 1e-8, None), power)
        (gid,) = g
        vid = f"video_{gid}"
        if cfg.get("score_level") == "shot":
            import h5py

            path_h5 = os.path.join(
                cfg["root_data"],
                f'annotations/{cfg["dataset"]}/eccv16_dataset_{cfg["dataset"].lower()}_google_pool5.h5',
            )
            if os.path.isfile(path_h5):
                with h5py.File(path_h5, "r") as hdf:
                    if vid in hdf:
                        from gravit.utils.score_adapter import (
                            change_points_valid,
                            shot_scores_to_sampled_frame_scores,
                            uniform_change_points,
                        )

                        n_frames = int(hdf[vid + "/n_frames"][()])
                        picks = np.array(hdf[vid + "/picks"][()])
                        cp = np.array(hdf[vid + "/change_points"][()])
                        n_samples = int(hdf[vid + "/n_steps"][()])
                        if not change_points_valid(cp):
                            cp = uniform_change_points(n_frames, max(1, int(tmp.size)))
                        tmp = shot_scores_to_sampled_frame_scores(tmp, picks, cp, n_frames, n_samples)
        tmp = tmp.tolist()
        preds.append([vid, tmp])

    return preds
