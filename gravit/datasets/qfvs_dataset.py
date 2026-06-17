import os
import glob
import torch
from torch_geometric.data import Dataset


class QFVSDataset(Dataset):
    """
    Loads .pt graphs for QFVS. Each file should be torch_geometric Data with at least:
    x, edge_index, edge_attr, y, g, picks, user_summary (and query_emb when using QueryBridge).
    """

    def __init__(self, path_graphs: str):
        super().__init__()
        self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, "*.pt")))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        return torch.load(self.all_graphs[idx], weights_only=False)
