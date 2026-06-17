#!/usr/bin/env python3
"""
Build QFVS graph .pt files for tools/train_qfvs.py.

Requires you to provide per-query video features (e.g. npy per clip) and query
embeddings. This script documents the expected torch_geometric.Data schema:

  x, edge_index, edge_attr, y, user_summary, picks, g, query_emb
  optional: query_emb_neg (mismatched query for contrastive loss)

Example stub (random data) for pipeline testing:
  python data/build_qfvs_query_graphs.py --out data/graphs/QFVS_stub/split1 --nodes 20 --feat 1024
"""

import argparse
import os
import numpy as np
import torch
from torch_geometric.data import Data


def _get_edge_info(num_frame: int, tauf: int, skip_factor: int = 0):
    node_source, node_target, edge_attr = [], [], []
    for i in range(num_frame):
        for j in range(num_frame):
            frame_diff = i - j
            if abs(frame_diff) <= tauf:
                node_source.append(i)
                node_target.append(j)
                edge_attr.append(np.sign(frame_diff))
            elif skip_factor and (frame_diff % skip_factor == 0) and (abs(frame_diff) <= skip_factor * tauf):
                node_source.append(i)
                node_target.append(j)
                edge_attr.append(np.sign(frame_diff))
    return node_source, node_target, edge_attr


def make_random_graph(nodes: int, feat_dim: int, video_id: str, seed: int) -> Data:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((nodes, feat_dim)).astype(np.float32)
    tauf = min(5, max(nodes - 1, 1))
    ns, nt, ea = _get_edge_info(nodes, tauf, 0)
    y = rng.random((nodes, 1)).astype(np.float32)
    usr = (rng.random((3, nodes)) > 0.7).astype(np.float32)
    picks = np.arange(nodes, dtype=np.int64)
    qemb = rng.standard_normal(feat_dim).astype(np.float32)
    qneg = rng.standard_normal(feat_dim).astype(np.float32)
    return Data(
        x=torch.from_numpy(x),
        g=video_id,
        edge_index=torch.tensor([ns, nt], dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.float32),
        y=torch.from_numpy(y),
        labels=torch.zeros(1),
        user_summary=torch.from_numpy(usr),
        picks=torch.from_numpy(picks),
        query_emb=torch.from_numpy(qemb),
        query_emb_neg=torch.from_numpy(qneg),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="e.g. data/graphs/QFVS_stub/split1")
    p.add_argument("--nodes", type=int, default=20)
    p.add_argument("--feat", type=int, default=1024)
    args = p.parse_args()
    for sub in ("train", "val"):
        d = os.path.join(args.out, sub)
        os.makedirs(d, exist_ok=True)
    for i, split in enumerate(["train", "train", "val"]):
        path = os.path.join(args.out, split, f"stub_{i}.pt")
        g = make_random_graph(args.nodes, args.feat, f"qfvs_stub_{i}", seed=i)
        torch.save(g, path)
    print(f"Wrote stub graphs under {args.out}/train and {args.out}/val")


if __name__ == "__main__":
    main()
