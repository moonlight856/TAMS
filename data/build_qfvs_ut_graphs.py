#!/usr/bin/env python3
"""
Build PyG graphs for QFVS (Sharghi et al., CVPR 2017 / arXiv:1707.04960) from
official ``origin_data`` annotations under ``data/annotations/QFVS/``.

Experiment layout (4 splits on QFVS, leave-one-participant-out):
  - Leave-one-participant-out: split k holds participant P0{k} in *val*, others in *train*.

Each graph is one (participant, query) pair with 3 user summary files (*_user{1,2,3}.txt).
Frame indices in those files are treated as **1-based** (as in the released summaries).

Graph schema (matches ``tools/train_qfvs.py`` / SumMe frame graphs):
  - ``user_summary``: [3, n_frames] binary at **full video** resolution
  - ``picks``: [n_nodes] — frame index per graph node (subset / stride over 0..n_frames-1)
  - ``x``: [n_nodes, feat_dim] — demo noise, per-query .npy, or participant H5 (shared across queries)
  - ``y``: [n_nodes, 1] mean of user summaries at picked frames (soft target)
  - ``query_emb`` / ``query_emb_neg``: L2-normalized query vector (``--query_dim``); use ``--glove_txt`` for GloVe

Feature sources (choose exactly one):
  - ``--demo``: random [T, feat_dim]
  - ``{features_root}/{participant}/{query_key}.npy``  shape [T, feat_dim]
  - ``--features_h5_dir``: ``V{n}_{backbone}.h5`` per participant (P01->V1, ...), dataset ``feature`` [T, D]

Usage:
  python data/build_qfvs_ut_graphs.py --demo --stride 4 --query_dim 1024

  python data/build_qfvs_ut_graphs.py --features_root data/features/QFVS_ut --stride 2 --query_dim 1024

  python data/build_qfvs_ut_graphs.py --features_h5_dir data/annotations/QFVS/features \\
    --h5_backbone C3D --glove_txt path/to/glove.840B.300d.txt --stride 4 --query_dim 300
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from torch_geometric.data import Data

from gravit.utils.qfvs_glove import QFVSQueryGlove
from gravit.utils.qfvs_h5_io import aligned_features, load_feature_matrix

PARTICIPANTS = ("P01", "P02", "P03", "P04")
USER_RE = re.compile(r"^(.+)_user([123])\.txt$", re.IGNORECASE)


def _parse_user_file(path: Path) -> list[int]:
    frames: list[int] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(int(line))
            except ValueError:
                continue
    return frames


def scan_groups(origin_root: Path) -> dict[tuple[str, str], dict[int, list[int]]]:
    """
    Returns mapping (participant, query_key) -> {user_id (1..3): list of 1-based frame indices}.
    """
    base = origin_root / "Query-Focused_Summaries" / "User_Summaries"
    if not base.is_dir():
        raise FileNotFoundError(f"Missing QFVS user summaries: {base}")

    raw: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for pdir in sorted(base.iterdir()):
        if not pdir.is_dir() or pdir.name not in PARTICIPANTS:
            continue
        participant = pdir.name
        for fp in sorted(pdir.glob("*.txt")):
            if fp.name.lower() == "readme.txt":
                continue
            m = USER_RE.match(fp.name)
            if not m:
                continue
            query_key, uid_s = m.group(1), m.group(2)
            uid = int(uid_s)
            frames = _parse_user_file(fp)
            raw[(participant, query_key)][uid] = frames

    groups: dict[tuple[str, str], dict[int, list[int]]] = {}
    for key, users in raw.items():
        if set(users.keys()) != {1, 2, 3}:
            continue
        groups[key] = {u: users[u] for u in (1, 2, 3)}
    return groups


def _edges_for_picks(picks: np.ndarray, tauf: int, skip_factor: int) -> tuple[list[int], list[int], list[float]]:
    n = int(picks.shape[0])
    src, tgt, attr = [], [], []
    for i in range(n):
        for j in range(n):
            d = int(picks[i]) - int(picks[j])
            ad = abs(d)
            if ad <= tauf:
                src.append(i)
                tgt.append(j)
                attr.append(float(np.sign(d)))
            elif skip_factor and ad % skip_factor == 0 and ad <= skip_factor * tauf:
                src.append(i)
                tgt.append(j)
                attr.append(float(np.sign(d)))
    return src, tgt, attr


def _query_vec(text: str, dim: int) -> np.ndarray:
    seed = (hash(text) % (2**31)) + dim * 17
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v) + 1e-8)
    return (v / n).astype(np.float32)


def _load_features(
    features_root: Path | None,
    participant: str,
    query_key: str,
    n_frames: int,
    feat_dim: int,
    demo: bool,
) -> np.ndarray:
    if demo:
        rng = np.random.default_rng((hash((participant, query_key)) % (2**31)) & 0xFFFFFFFF)
        return rng.standard_normal((n_frames, feat_dim)).astype(np.float32)
    assert features_root is not None
    cand = features_root / participant / f"{query_key}.npy"
    if not cand.is_file():
        raise FileNotFoundError(f"Expected features file: {cand}")
    x = np.load(cand, mmap_mode="r")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != feat_dim:
        raise ValueError(f"{cand}: expected [T,{feat_dim}], got {x.shape}")
    if x.shape[0] < n_frames:
        raise ValueError(f"{cand}: T={x.shape[0]} < n_frames={n_frames}")
    return np.array(x[:n_frames], dtype=np.float32)


def build_one_graph(
    participant: str,
    query_key: str,
    users: dict[int, list[int]],
    *,
    feat_dim: int,
    stride: int,
    tauf: int,
    skip_factor: int,
    query_dim: int,
    features_root: Path | None,
    demo: bool,
    h5_dir: Path | None = None,
    h5_backbone: str = "C3D",
    glove: QFVSQueryGlove | None = None,
) -> Data:
    all_idx: list[int] = []
    for u in (1, 2, 3):
        all_idx.extend(users[u])
    if not all_idx:
        raise ValueError(f"Empty summaries for {participant}/{query_key}")
    max_f1 = max(all_idx)
    n_frames_ann = max(max_f1, 1)

    if h5_dir is not None:
        full, _inferred_d = load_feature_matrix(h5_dir, participant, h5_backbone)
        feats_ctx, n_use = aligned_features(full, n_frames_ann)
    else:
        feats_ctx = _load_features(features_root, participant, query_key, n_frames_ann, feat_dim, demo)
        n_use = n_frames_ann

    user_summary = np.zeros((3, n_use), dtype=np.float32)
    for u in (1, 2, 3):
        for f1 in users[u]:
            if f1 < 1:
                continue
            j = f1 - 1
            if 0 <= j < n_use:
                user_summary[u - 1, j] = 1.0

    picks = np.arange(0, n_use, stride, dtype=np.int64)
    if picks.size == 0:
        picks = np.array([0], dtype=np.int64)
    x = feats_ctx[picks].astype(np.float32)

    y = user_summary[:, picks].mean(axis=0, keepdims=True).T.astype(np.float32)

    ns, nt, ea = _edges_for_picks(picks, tauf=tauf, skip_factor=skip_factor)
    if glove is not None:
        q = glove.embed(query_key)
        if int(q.shape[0]) != int(query_dim):
            raise ValueError(f"query_emb dim {q.shape[0]} != query_dim {query_dim}")
    else:
        q = _query_vec(f"{participant}::{query_key}", query_dim)
    gid = f"{participant}__{query_key}"

    return Data(
        x=torch.from_numpy(x),
        g=gid,
        edge_index=torch.tensor([ns, nt], dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.float32),
        y=torch.from_numpy(y),
        labels=torch.zeros(picks.shape[0], dtype=torch.float64),
        user_summary=torch.from_numpy(user_summary.astype(np.float32)),
        picks=torch.from_numpy(picks.copy()),
        query_emb=torch.from_numpy(q),
    )


def assign_neg_queries(graphs: list[Data]) -> None:
    """In-place: query_emb_neg[i] = query_emb[i+1] (cyclic)."""
    if not graphs:
        return
    qs = [g.query_emb.clone() for g in graphs]
    n = len(graphs)
    for i, g in enumerate(graphs):
        g.query_emb_neg = qs[(i + 1) % n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QFVS UT graphs from origin_data annotations.")
    ap.add_argument(
        "--origin_root",
        type=str,
        default=None,
        help="Path to .../origin_data (default: <repo>/data/annotations/QFVS/origin_data)",
    )
    ap.add_argument("--out_root", type=str, default="data/graphs/QFVS_ut", help="Output graphs root")
    ap.add_argument("--feat_dim", type=int, default=1024)
    ap.add_argument("--query_dim", type=int, default=1024, help="Must match yaml query_in_dim")
    ap.add_argument("--stride", type=int, default=4, help="Subsample every stride frames as nodes")
    ap.add_argument("--tauf", type=int, default=5)
    ap.add_argument("--skip_factor", type=int, default=0)
    ap.add_argument("--features_root", type=str, default=None, help="Optional npy features tree")
    ap.add_argument(
        "--features_h5_dir",
        type=str,
        default=None,
        help="Directory with V1_C3D.h5 … V4_*.h5 (participant P01..P04 maps to V1..V4)",
    )
    ap.add_argument("--h5_backbone", type=str, default="C3D", help="H5 stem, e.g. C3D or resnet_avg")
    ap.add_argument(
        "--glove_txt",
        type=str,
        default=None,
        help="Path to glove.*.txt; builds cached subset under annotations processed/ (first run may scan full file)",
    )
    ap.add_argument("--glove_cache", type=str, default=None, help="Optional path for glove_qfvs_subset.npz")
    ap.add_argument("--demo", action="store_true", help="Random visual features (no .npy required)")
    args = ap.parse_args()

    here = _REPO
    origin = Path(args.origin_root) if args.origin_root else (here / "data" / "annotations" / "QFVS" / "origin_data")
    if not origin.is_dir():
        raise FileNotFoundError(origin)

    groups = scan_groups(origin)
    if not groups:
        raise RuntimeError("No complete (3-user) query groups found under annotations.")

    n_modes = int(args.demo) + int(bool(args.features_root)) + int(bool(args.features_h5_dir))
    if n_modes != 1:
        ap.error("Choose exactly one of: --demo, --features_root, --features_h5_dir")

    features_root = Path(args.features_root) if args.features_root else None
    h5_dir = Path(args.features_h5_dir) if args.features_h5_dir else None
    if h5_dir is not None and not h5_dir.is_dir():
        raise FileNotFoundError(h5_dir)

    glove: QFVSQueryGlove | None = None
    if args.glove_txt:
        gpath = Path(args.glove_txt)
        cache = Path(args.glove_cache) if args.glove_cache else None
        glove = QFVSQueryGlove(gpath, groups, cache_path=cache)
        if glove.dim != args.query_dim:
            ap.error(f"With --glove_txt, set --query_dim to {glove.dim} (got {args.query_dim})")

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = here / out_root

    by_participant: dict[str, list[Data]] = defaultdict(list)
    for (participant, query_key), users in sorted(groups.items()):
        try:
            g = build_one_graph(
                participant,
                query_key,
                users,
                feat_dim=args.feat_dim,
                stride=args.stride,
                tauf=args.tauf,
                skip_factor=args.skip_factor,
                query_dim=args.query_dim,
                features_root=features_root,
                demo=args.demo,
                h5_dir=h5_dir,
                h5_backbone=str(args.h5_backbone),
                glove=glove,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP {participant}/{query_key}: {e}")
            continue
        by_participant[participant].append(g)

    for plist in by_participant.values():
        assign_neg_queries(plist)

    for si, test_p in enumerate(PARTICIPANTS, start=1):
        split_dir = out_root / f"split{si}"
        tr = split_dir / "train"
        va = split_dir / "val"
        tr.mkdir(parents=True, exist_ok=True)
        va.mkdir(parents=True, exist_ok=True)
        for p, graphs in by_participant.items():
            sub = va if p == test_p else tr
            for g in graphs:
                safe = str(g.g).replace(os.sep, "_")
                torch.save(g, sub / f"{safe}.pt")

        print(f"split{si}: val={test_p} — wrote {len(by_participant[test_p])} val, "
              f"train={sum(len(by_participant[q]) for q in PARTICIPANTS if q != test_p)} train graphs")

    print(f"Done. Total unique (participant,query) groups: {len(groups)}")


if __name__ == "__main__":
    main()
