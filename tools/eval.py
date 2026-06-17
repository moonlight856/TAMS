import setup_paths  # noqa: F401

import os
import glob

import h5py
import torch
import argparse
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
from model import build_model, build_dataloaders
from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds
from gravit.utils.vs import avg_splits
from gravit.utils import protocol
from gravit.utils.score_adapter import change_points_valid, synthetic_shot_segment_count, uniform_change_points


def _forward_eval(model, cfg, data, device):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    c = data.c.to(device) if cfg.get("use_spf", False) else None
    if cfg.get("model_name") == "TAMS":
        from model import TAMS

        if isinstance(model, TAMS):
            q = getattr(data, "query_emb", None)
            qe = q.to(device) if q is not None else None
            from modules_step0 import resolve_node_confidence

            nc = resolve_node_confidence(cfg, data, x.shape[0], device, x.dtype)
            ms = None
            if cfg.get("apply_modality_scale") and cfg.get("step0_multimodal_json") and cfg.get("use_step0"):
                from modules_step0 import build_modality_scale_videoxum, _graph_id_str
                vid = _graph_id_str(data)
                if vid is not None:
                    ms = build_modality_scale_videoxum(
                        cfg, vid, x.shape[0], cfg.get("num_modality", 1), device, x.dtype,
                    )
            return model(
                x, edge_index, edge_attr, c,
                query_emb=qe, node_conf=nc, modality_scale=ms,
            )
    return model(x, edge_index, edge_attr, c)


def _graph_id_to_list(g) -> list:
    """Graph `g` may be int, Tensor, or list (PyG / legacy .pt)."""
    if isinstance(g, torch.Tensor):
        return g.view(-1).tolist()
    if isinstance(g, (list, tuple)):
        return list(g)
    return [int(g)]


def get_eval_score(cfg, preds):
    eval_type = cfg["eval_type"]
    str_score = ""
    if eval_type == "VS_max" or eval_type == "VS_avg":
        path_dataset = os.path.join(
            cfg["root_data"],
            f'annotations/{cfg["dataset"]}/eccv16_dataset_{cfg["dataset"].lower()}_google_pool5.h5',
        )
        with h5py.File(path_dataset, "r") as hdf:
            all_f1_scores = []
            all_taus = []
            all_rhos = []
            proto = cfg.get("protocol_mode", "strict_main")

            for video, scores in preds:
                n_samples = int(hdf.get(video + "/n_steps")[()])
                n_frames = int(hdf.get(video + "/n_frames")[()])
                gt_segments = np.array(hdf.get(video + "/change_points"))
                if not change_points_valid(gt_segments):
                    n_seg = synthetic_shot_segment_count(n_frames, n_samples)
                    gt_segments = uniform_change_points(n_frames, n_seg)
                gt_samples = np.array(hdf.get(video + "/picks"))
                gt_scores = np.array(hdf.get(video + "/gtscore"))
                user_summaries = np.array(hdf.get(video + "/user_summary"))

                frame_scores = protocol.frame_scores_from_sampled_scores(np.array(scores, dtype=np.float32), gt_samples, n_frames)

                s_scores, s_lengths = protocol.segment_scores_and_lengths(frame_scores, gt_segments)
                final_len = int(n_frames * 0.15)
                segments = protocol.select_segments(proto, final_len, s_scores, s_lengths)

                pred_summary = np.zeros(n_frames, dtype=np.int8)
                for seg in segments:
                    pred_summary[gt_segments[seg][0] : gt_segments[seg][1]] = 1

                user_summary = np.zeros(n_frames, dtype=np.int8)
                n_user_sums = user_summaries.shape[0]
                f1_scores = np.empty(n_user_sums)

                for u_sum_idx in range(n_user_sums):
                    user_summary[:n_frames] = user_summaries[u_sum_idx]
                    tp = pred_summary & user_summary
                    p_den = max(int(pred_summary.sum()), 1)
                    r_den = max(int(user_summary.sum()), 1)
                    precision = float(tp.sum()) / p_den
                    recall = float(tp.sum()) / r_den
                    if (precision + recall) == 0:
                        f1_scores[u_sum_idx] = 0
                    else:
                        f1_scores[u_sum_idx] = 2 * precision * recall * 100 / (precision + recall)

                pred_imp_score = np.array(scores)
                ref_imp_scores = gt_scores
                rho_coeff, _ = spearmanr(pred_imp_score, ref_imp_scores)
                tau_coeff, _ = kendalltau(rankdata(-pred_imp_score), rankdata(-ref_imp_scores))

                all_taus.append(tau_coeff)
                all_rhos.append(rho_coeff)

                if eval_type == "VS_max":
                    f1 = max(f1_scores)
                else:
                    f1 = np.mean(f1_scores)
                all_f1_scores.append(f1)

        f1_score = sum(all_f1_scores) / len(all_f1_scores)
        tau = sum(all_taus) / len(all_taus)
        rho = sum(all_rhos) / len(all_rhos)
        str_score = f"F1-Score = {f1_score}, Tau = {tau}, Rho = {rho}"
    return str_score


def evaluate(cfg):
    path_graphs = os.path.join(cfg["root_data"], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg["root_result"], f'{cfg["exp_name"]}')
    split = cfg.get("split")
    if split is not None:
        path_graphs = os.path.join(path_graphs, f"split{split}")
        path_result = os.path.join(path_result, f"split{split}")

    logger = get_logger(path_result, file_name="eval")
    logger.info(cfg["exp_name"])
    logger.info(path_result)
    logger.info("Preparing a model and data loaders")

    device_str = cfg.get("device", "cuda:0")
    if not torch.cuda.is_available() and "cuda" in device_str:
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    model = build_model(cfg).to(device)
    _, val_loader = build_dataloaders(cfg, path_graphs)
    num_val_graphs = len(val_loader)

    ckpt_name = cfg.get("eval_checkpoint", "ckpt_best.pt")
    logger.info(f"Loading the trained model: {ckpt_name}")
    state_dict = torch.load(
        os.path.join(path_result, ckpt_name),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = set(state_dict.keys()) - model_keys
    if skipped:
        logger.info(f"Skipping checkpoint keys not in current model: {sorted(skipped)}")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    logger.info("Retrieving the formatting dictionary")
    data_dict = get_formatting_data_dict(cfg)

    logger.info("Evaluation process started")
    preds_all = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 1):
            g = _graph_id_to_list(data.g)
            logits = _forward_eval(model, cfg, data, device)
            preds = get_formatted_preds(cfg, logits, g, data_dict)
            preds_all.extend(preds)
            logger.info(f"[{i:04d}|{num_val_graphs:04d}] processed")

    logger.info("Computing the evaluation score")
    eval_score = get_eval_score(cfg, preds_all)
    logger.info(f'{cfg["eval_type"]} evaluation finished: {eval_score}\n')
    return eval_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data", type=str, help="Root directory to the data", default="./data")
    parser.add_argument("--root_result", type=str, help="Root directory to output", default="./results")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--exp_name", type=str, help="Name of the experiment", required=True)
    parser.add_argument("--eval_type", type=str, help="Type of the evaluation", required=True)
    parser.add_argument("--split", type=int, help="Split to evaluate")
    parser.add_argument("--all_splits", action="store_true", help="Evaluate all splits")
    parser.add_argument(
        "--protocol_mode",
        type=str,
        default=None,
        choices=["strict_main", "extended"],
        help="Override cfg: strict_main (knapsack) vs extended (greedy Track-B).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Eval-time logit temperature (>1 sharpens sigmoid, helps F1 without changing Tau/Rho).",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep temperatures [1,2,3,5,8] and report best F1 (Tau/Rho stay identical).",
    )
    parser.add_argument(
        "--center", action="store_true",
        help="Enable logit centering (subtract mean before sigmoid).",
    )
    parser.add_argument(
        "--score-mode", type=str, default=None,
        choices=["sigmoid", "minmax", "zscore", "percentile"],
        help="Score transform: sigmoid (default), minmax, zscore, percentile.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint filename to load (default: ckpt_best.pt). E.g. ckpt_ep040.pt",
    )
    parser.add_argument(
        "--power", type=float, default=None,
        help="Power transform: score^p. Monotonic → τ/ρ invariant, but changes segment averages → F1 changes.",
    )
    parser.add_argument(
        "--sweep-power", action="store_true",
        help="Sweep power values [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0] combined with temperature sweep.",
    )
    parser.add_argument(
        "--sweep-ckpts", action="store_true",
        help="Sweep all ckpt_ep*.pt checkpoints and report results for each.",
    )

    args = parser.parse_args()

    path_result = os.path.join(args.root_result, args.exp_name)
    if not os.path.isdir(path_result):
        raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

    results = []
    if args.all_splits:
        results = glob.glob(os.path.join(path_result, "*", "cfg.yaml"))
    else:
        if args.split is not None:
            path_result = os.path.join(path_result, f"split{args.split}")
            if not os.path.isdir(path_result):
                raise ValueError(f'Please run the training experiment "{args.exp_name}" first')
        results.append(os.path.join(path_result, "cfg.yaml"))

    temperatures = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0] if args.sweep else [args.temperature or 1.0]
    if args.temperature is not None and not args.sweep:
        temperatures = [args.temperature]

    score_mode = getattr(args, "score_mode", None) or "sigmoid"
    powers = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0] if getattr(args, "sweep_power", False) else [getattr(args, "power", None) or 1.0]

    checkpoints = ["ckpt_best.pt"]
    if getattr(args, "sweep_ckpts", False) and args.all_splits:
        first_split_dir = os.path.dirname(results[0])
        ep_ckpts = sorted(glob.glob(os.path.join(first_split_dir, "ckpt_ep*.pt")))
        checkpoints = ["ckpt_best.pt"] + [os.path.basename(p) for p in ep_ckpts]
    elif getattr(args, "checkpoint", None):
        checkpoints = [args.checkpoint]

    for ckpt_name in checkpoints:
        for pw in powers:
            for temp in temperatures:
                all_eval_results = []
                for result in results:
                    args.cfg = result
                    cfg = get_cfg(args)
                    cfg["eval_temperature"] = temp
                    cfg["eval_center_logits"] = getattr(args, "center", False)
                    cfg["eval_score_mode"] = score_mode
                    cfg["eval_power"] = pw
                    cfg["eval_checkpoint"] = ckpt_name
                    ckpt_path = os.path.join(os.path.dirname(result), ckpt_name)
                    if not os.path.isfile(ckpt_path):
                        continue
                    all_eval_results.append(evaluate(cfg))

                if "VS" in args.eval_type and args.all_splits and all_eval_results:
                    if args.sweep or len(checkpoints) > 1 or len(powers) > 1:
                        print(f"\n===== ckpt={ckpt_name}  T={temp:.2f}  p={pw:.1f}  mode={score_mode} =====")
                    avg_splits.print_results(all_eval_results)
