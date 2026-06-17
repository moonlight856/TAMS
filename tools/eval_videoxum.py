import setup_paths  # noqa: F401

import os
import glob
import torch
import argparse
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr
from torch.utils.data import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
from gravit.utils.cfg_defaults import merge_defaults
from model import build_model
from videoxum_dataset import VideoXumDataset, videoxum_collate_fn
from tqdm import tqdm


def evaluate(cfg):
    """
    Run the evaluation process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    split = cfg.get("split")
    if split is not None:
        path_graphs = os.path.join(path_graphs, f"split{split}")
        path_result = os.path.join(path_result, f"split{split}")

    # Prepare the logger
    logger = get_logger(path_result, file_name='eval')
    logger.info(cfg['exp_name'])
    logger.info(path_result)
    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device_str = cfg.get("device", "cuda:0")
    if "cuda" in device_str and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    merge_defaults(cfg)
    cpu = torch.device("cpu")
    val_ds = VideoXumDataset.from_cfg("val", cfg, device=cpu)
    x0, _, _, _, _ = val_ds[0]
    cfg["num_modality"] = val_ds.num_modalities
    cfg["videoxum_modality_dim"] = int(x0.shape[1]) // val_ds.num_modalities

    model = build_model(cfg).to(device)
    print(model)
    val_loader = DataLoader(val_ds, collate_fn=videoxum_collate_fn)
    num_val_graphs = len(val_loader)

    # Load the trained model
    logger.info('Loading the trained model')
    state_dict = torch.load(
        os.path.join(path_result, 'ckpt_best.pt'),
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Run the evaluation process
    logger.info('Evaluation process started')

    f1_max = []
    f1_mean = []
    rho = []
    tau = []

    from modules_step0 import build_modality_scale_videoxum

    nm = int(cfg.get("num_modality", 1))
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader, 1)):
            x, y, e, e_attr, vid = data
            x = x.to(device)
            y = y
            e = e.to(device)
            e_attr = e_attr.to(device)
            c = None
            ms = build_modality_scale_videoxum(
                cfg, str(vid), x.shape[0], nm, device, x.dtype
            )
            logits = model(x, e, e_attr, c, modality_scale=ms)

            # Change the format of the model output
            preds = torch.sigmoid(logits.squeeze().cpu()).numpy()

            # logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')

            gt_score = torch.mean(y, dim=0).detach().cpu().numpy()

            rho_coeff, _ = spearmanr(preds, gt_score)
            tau_coeff, _ = kendalltau(rankdata(-preds), rankdata(-gt_score))

            pred = np.percentile(preds, 37.6)
            pred = (preds > pred).astype(int)

            y = y.detach().cpu().numpy()

            tp = pred[None, :] * y
            precision = np.sum(tp, 1) / max(np.sum(pred), 1)
            recall = np.sum(tp, 1) / np.maximum(np.sum(y, 1), 1)

            f1 = (2 * precision * recall * 100 / (precision + recall + 1e-10))

            f1_mean.append(np.mean(f1))
            f1_max.append(np.max(f1))
            rho.append(rho_coeff)
            tau.append(tau_coeff)


    # Compute the evaluation score
    logger.info(f'Computing the evaluation score')
    # print(preds_all)

    f1_max = np.array(f1_max)
    f1_mean = np.array(f1_mean)
    tau = np.array(tau)
    rho = np.array(rho)

    print(f"f1_max: {np.mean(f1_max)}, f1_mean: {np.mean(f1_mean)}, tau: {np.mean(tau)}, rho: {np.mean(rho)}")


if __name__ == "__main__":
    """
    Evaluate the trained model from the experiment "exp_name"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset')
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment', required=True)
    parser.add_argument('--eval_type',     type=str,   help='Type of the evaluation', required=True)
    parser.add_argument('--split',         type=int,   help='Split to evaluate')
    parser.add_argument('--all_splits',    action='store_true',   help='Evaluate all splits')

    args = parser.parse_args()

    path_exp = os.path.join(args.root_result, args.exp_name)

    results = []
    if args.all_splits:
        results = sorted(glob.glob(os.path.join(path_exp, "split*", "cfg.yaml")))
        if not results:
            raise ValueError(
                f'No split*/cfg.yaml under "{path_exp}". '
                f'Train with --split N (saves to splitN/) or check --root_result.'
            )
    else:
        if args.split is not None:
            path_result = os.path.join(path_exp, f"split{args.split}")
        else:
            path_result = path_exp
        if not os.path.isdir(path_result):
            raise ValueError(
                f'Experiment output not found: {path_result}. '
                f'Train first, or pass --split N if you trained with --split N.'
            )
        cfg_path = os.path.join(path_result, "cfg.yaml")
        if not os.path.isfile(cfg_path):
            raise ValueError(f"Missing {cfg_path}")
        results.append(cfg_path)

    for result in results:
        args.cfg = result
        cfg = get_cfg(args)
        evaluate(cfg)

