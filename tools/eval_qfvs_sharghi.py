"""
QFVS evaluation under Sharghi CVPR'17 *shot-level* protocol (dense tag IoU + matching).

- **Baseline** (--source random): reproducible random shot scores + top-frac selection
  (same protocol as FCSNA-style ``top 2%`` papers; does not import external GitHub code).
- **TAMS** (--source tams): load val PyG graphs + ``ckpt_best.pt``, map node scores
  to shots (max over nodes whose pick falls in the shot), then same Sharghi P/R/F1.

Does **not** replace ``eval_qfvs.py`` (node-level set metrics); use both for different tables.

Examples::

  # Baseline: random scores, top 2% shots, 4-fold macro
  python tools/eval_qfvs_sharghi.py --source random --all_splits --seed 0

  # Trained TAMS, split 4 only
  python tools/eval_qfvs_sharghi.py --source tams --cfg configs/QFVS/TAMS_Net_v2_tuned.yaml --split 4

  # Per-split top_frac from YAML (qfvs_sharghi_top_frac_by_split); CLI --top_frac overrides all splits
  python tools/eval_qfvs_sharghi.py --source tams --cfg configs/QFVS/TAMS_Net_v4_push.yaml --all_splits
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import importlib.util

import setup_paths  # noqa: F401

from eval_qfvs import _forward
from gravit.utils.cfg_defaults import DEFAULT_KEYS, merge_defaults
from gravit.utils.qfvs_sharghi_protocol import (
    gt_shot_set_union_users,
    load_shot_tag_sets,
    sharghi_prf1_from_sets,
    top_frac_shot_mask,
)
from model import build_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_build_qfvs_module():
    path = _repo_root() / "data" / "build_qfvs_ut_graphs.py"
    spec = importlib.util.spec_from_file_location("build_qfvs_ut_graphs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bqg = _load_build_qfvs_module()
PARTICIPANTS = _bqg.PARTICIPANTS
scan_groups = _bqg.scan_groups


def _val_participant(split: int) -> str:
    if split < 1 or split > 4:
        raise ValueError("split must be 1..4")
    return PARTICIPANTS[split - 1]


def _lookup_split_top_frac(cfg: dict | None, split: int) -> float | None:
    """Return cfg qfvs_sharghi_top_frac_by_split[split] if present, else None."""
    if cfg is None:
        return None
    m = cfg.get("qfvs_sharghi_top_frac_by_split")
    if not isinstance(m, dict):
        return None
    for key in (split, str(split)):
        if key in m and m[key] is not None:
            return float(m[key])
    return None


def _resolve_top_frac_for_split(cfg: dict | None, split: int, cli_top_frac: float | None) -> float:
    if cli_top_frac is not None:
        return float(cli_top_frac)
    v = _lookup_split_top_frac(cfg, split)
    if v is not None:
        return v
    c = dict(cfg or {})
    merge_defaults(c)
    return float(c.get("qfvs_sharghi_top_frac", DEFAULT_KEYS["qfvs_sharghi_top_frac"]))


def eval_sharghi_tams_model(
    model,
    *,
    origin_root: Path,
    groups: dict,
    split: int,
    top_frac: float | None,
    shot_stride: int,
    root_data: Path,
    graph_name: str,
    cfg: dict,
    device: torch.device,
) -> tuple[float, float, float]:
    """Sharghi P/R/F1 macro on the held-out participant for split (model already loaded)."""
    cfg = {**cfg, "split": split, "dataset": "QFVS"}
    merge_defaults(cfg)
    path_graphs = os.path.join(str(root_data), "graphs", graph_name, f"split{split}", "val")
    model.eval()
    val_p = _val_participant(split)
    stride = max(int(shot_stride), 1)
    tag_cache: dict[str, list[frozenset[str]]] = {}
    ps, rs, f1s = [], [], []
    tf = _resolve_top_frac_for_split(cfg, split, top_frac)

    keys = sorted((k, v) for k, v in groups.items() if k[0] == val_p)
    for (participant, query_key), users in keys:
        if participant not in tag_cache:
            tag_cache[participant] = load_shot_tag_sets(origin_root, participant)
        tag_sets = tag_cache[participant]
        n_shots = len(tag_sets)
        gt_shots = gt_shot_set_union_users(users, shot_stride=stride)
        gt_shots = {s for s in gt_shots if 0 <= s < n_shots}

        safe = f"{participant}__{query_key}".replace(os.sep, "_")
        pt_path = os.path.join(path_graphs, f"{safe}.pt")
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(pt_path)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        scores = np.full(n_shots, -np.inf, dtype=np.float64)
        with torch.no_grad():
            logits = _forward(model, cfg, data, device)
            prob = torch.sigmoid(logits.squeeze(-1)).detach().float().cpu().numpy()
        picks = data.picks.numpy()
        if prob.ndim != 1:
            prob = prob.reshape(-1)
        for i in range(int(picks.shape[0])):
            fr = int(picks[i])
            sh = fr // stride
            if 0 <= sh < n_shots:
                scores[sh] = max(float(scores[sh]), float(prob[i]))
        scores = np.where(np.isfinite(scores), scores, 0.0)
        pred_shots = top_frac_shot_mask(scores, tf)
        p, r, f1 = sharghi_prf1_from_sets(gt_shots, pred_shots, tag_sets)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))


def _eval_split(
    *,
    origin_root: Path,
    groups: dict,
    split: int,
    source: str,
    top_frac: float,
    shot_stride: int,
    seed: int,
    root_data: Path,
    graph_name: str,
    cfg: dict | None,
    device: torch.device,
) -> tuple[float, float, float]:
    val_p = _val_participant(split)
    stride = max(int(shot_stride), 1)
    rng = np.random.default_rng(seed + split * 10007)

    tag_cache: dict[str, list[frozenset[str]]] = {}
    ps, rs, f1s = [], [], []

    if source == "tams":
        if cfg is None:
            raise ValueError("tams requires cfg")
        cfg = {**cfg, "split": split, "dataset": "QFVS"}
        merge_defaults(cfg)
        path_result = os.path.join(cfg["root_result"], cfg["exp_name"], f"split{split}")
        ckpt = os.path.join(path_result, "ckpt_best.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        return eval_sharghi_tams_model(
            model,
            origin_root=origin_root,
            groups=groups,
            split=split,
            top_frac=top_frac,
            shot_stride=shot_stride,
            root_data=root_data,
            graph_name=graph_name,
            cfg=cfg,
            device=device,
        )

    keys = sorted((k, v) for k, v in groups.items() if k[0] == val_p)
    for (participant, query_key), users in keys:
        if participant not in tag_cache:
            tag_cache[participant] = load_shot_tag_sets(origin_root, participant)
        tag_sets = tag_cache[participant]
        n_shots = len(tag_sets)
        gt_shots = gt_shot_set_union_users(users, shot_stride=stride)
        gt_shots = {s for s in gt_shots if 0 <= s < n_shots}

        scores = rng.random(n_shots).astype(np.float64)
        pred_shots = top_frac_shot_mask(scores, top_frac)
        p, r, f1 = sharghi_prf1_from_sets(gt_shots, pred_shots, tag_sets)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser(description="QFVS Sharghi shot-level IoU+matching metrics")
    ap.add_argument(
        "--source",
        type=str,
        choices=("random", "tams"),
        required=True,
        help="random=in-repo stochastic baseline; tams=TAMS ckpt on val graphs",
    )
    ap.add_argument(
        "--origin_root",
        type=str,
        default=None,
        help="QFVS origin_data (default: <repo>/data/annotations/QFVS/origin_data)",
    )
    ap.add_argument("--root_data", type=str, default="./data")
    ap.add_argument("--graph_name", type=str, default="QFVS_ut")
    ap.add_argument("--split", type=int, default=None, help="1..4 (omit with --all_splits)")
    ap.add_argument("--all_splits", action="store_true")
    ap.add_argument(
        "--top_frac",
        type=float,
        default=None,
        help=(
            "Sharghi predicted-summary fraction of shots; overrides cfg for every split. "
            "If omitted: use qfvs_sharghi_top_frac_by_split[split] when present, else qfvs_sharghi_top_frac."
        ),
    )
    ap.add_argument(
        "--shot_stride",
        type=int,
        default=1,
        help=(
            "User-summary index (1-based) and graph pick (0-based) → dense shot id via floor(index/stride). "
            "Default 1 matches public QFVS origin_data (one step per dense tag line). "
            "Use round(fps*5) e.g. 150 only if summaries are raw video frames at that fps."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=str, default=None, help="YAML for tams (e.g. configs/QFVS/TAMS_vq_ut_p2.yaml)")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (default: cfg or cuda if available)")
    args = ap.parse_args()

    repo = _repo_root()
    origin = Path(args.origin_root) if args.origin_root else repo / "data" / "annotations" / "QFVS" / "origin_data"
    if not origin.is_dir():
        print(f"ERROR: origin_data not found: {origin}", file=sys.stderr)
        sys.exit(2)

    groups = scan_groups(origin)
    if not groups:
        print("ERROR: no QFVS query groups found.", file=sys.stderr)
        sys.exit(2)

    cfg = None
    device = None
    if args.source == "tams":
        if not args.cfg:
            ap.error("--cfg is required for --source tams")
        with open(args.cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        merge_defaults(cfg)
        cfg["root_data"] = str(Path(cfg.get("root_data", "./data")))
        if not Path(cfg["root_data"]).is_absolute():
            cfg["root_data"] = str((repo / cfg["root_data"]).resolve())
        rr = cfg.get("root_result", "./results")
        if not Path(rr).is_absolute():
            cfg["root_result"] = str((repo / rr).resolve())
        dev = args.device or cfg.get("device", "cuda:0")
        if "cuda" in dev and not torch.cuda.is_available():
            dev = "cpu"
        device = torch.device(dev)

    root_data = Path(args.root_data)
    if not root_data.is_absolute():
        root_data = (repo / root_data).resolve()

    splits = list(range(1, 5)) if args.all_splits else [int(args.split)]
    if not args.all_splits and args.split is None:
        ap.error("Provide --split 1..4 or --all_splits")

    def _resolve_top_frac_baseline() -> float:
        """Single global frac (random baseline or cfg fallback when no per-split map)."""
        if args.top_frac is not None:
            return float(args.top_frac)
        if cfg is not None:
            merge_defaults(cfg)
            return float(cfg.get("qfvs_sharghi_top_frac", DEFAULT_KEYS["qfvs_sharghi_top_frac"]))
        if args.cfg:
            with open(args.cfg, encoding="utf-8") as f:
                y = yaml.safe_load(f)
            merge_defaults(y)
            return float(y.get("qfvs_sharghi_top_frac", DEFAULT_KEYS["qfvs_sharghi_top_frac"]))
        return float(DEFAULT_KEYS["qfvs_sharghi_top_frac"])

    global_top_frac = _resolve_top_frac_baseline()

    rows = []
    for s in splits:
        tf = (
            float(args.top_frac)
            if args.top_frac is not None
            else (_resolve_top_frac_for_split(cfg, s, None) if args.source == "tams" else global_top_frac)
        )
        if args.source == "random":
            tf = global_top_frac
        p, r, f1 = _eval_split(
            origin_root=origin,
            groups=groups,
            split=s,
            source=args.source,
            top_frac=tf,
            shot_stride=args.shot_stride,
            seed=args.seed,
            root_data=root_data,
            graph_name=args.graph_name,
            cfg=cfg,
            device=device or torch.device("cpu"),
        )
        rows.append((s, p, r, f1))
        print(
            f"split{s} Sharghi (IoU+match) P={p*100:.2f} R={r*100:.2f} F1={f1*100:.2f} "
            f"(source={args.source}, top_frac={tf}, shot_stride={args.shot_stride})"
        )

    if len(rows) > 1:
        mp = float(np.mean([x[1] for x in rows]))
        mr = float(np.mean([x[2] for x in rows]))
        mf = float(np.mean([x[3] for x in rows]))
        print(f"QFVS 4-split macro (Sharghi): P={mp*100:.2f} R={mr*100:.2f} F1={mf*100:.2f}")


if __name__ == "__main__":
    main()
