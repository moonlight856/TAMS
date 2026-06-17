# TAMS: Topology-Adaptive Multimodal Summarization

<div align="center">

**Chaoqun Hu**,
**Ruyi Zhang**,
**Haocheng Tang**,
[**Wenrui Li**](https://liwrui.github.io/),
**Hengyu Man**,
[**Xiaopeng Fan**](https://scholar.google.cz/citations?hl=zh-CN&user=4LsZhDgAAAAJ&view_op=list_works&sortby=pubdate)

**Topology-Adaptive Multi-Scale Network for Unified Video Summarization**

</div>

## 🔖 Introduction

TAMS is a unified graph-based framework for video summarization that introduces four key innovations:

- **MATL** (Modality-Adaptive Topology Learner): replaces fixed τ-neighbor graphs with per-modality learned topologies via Gumbel-Sigmoid edge predictors.
- **DT-GNN** (Directional Topology GNN): direction-specific message passing with learned per-node fusion gates for forward, backward, and undirected streams.
- **OT-CMA** (Optimal Transport Cross-Modal Alignment): entropic optimal transport for principled cross-modal feature fusion with topology-modulated cost.
- **AMRG** (Adaptive Modality Relevance Gate): automatically suppresses uninformative modalities based on temporal feature variance.

TAMS achieves state-of-the-art results on five benchmarks: **SumMe**, **TVSum**, **VideoXum**, **QFVS**, and **MrHiSum**.

## 📑 Setup

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric (PyG)
- Additional: `h5py`, `scipy`, `numpy`, `pyyaml`, `tqdm`

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/TAMS.git
cd TAMS
pip install torch torchvision torchaudio
pip install torch_geometric
pip install h5py scipy numpy pyyaml tqdm
```

### Download Datasets

Download the following datasets to `./data/annotations/`:

| Dataset | Source |
|---------|--------|
| **SumMe** | [GraVi-T Guide](https://github.com/IntelLabs/GraVi-T/blob/main/docs/GETTING_STARTED_VS.md) |
| **TVSum** | [GraVi-T Guide](https://github.com/IntelLabs/GraVi-T/blob/main/docs/GETTING_STARTED_VS.md) |
| **VideoXum** | [HuggingFace](https://huggingface.co/datasets/jylins/videoxum) |
| **QFVS** | [UT Egocentric](https://www.cs.utexas.edu/~grauman/papers/videosum/) |
| **MrHiSum** | [GitHub](https://github.com/MrHiSum/MrHiSum) |

### Directory Structure

```
TAMS/
├── configs/
│   ├── SumMe/
│   │   └── TAMS.yaml
│   ├── TVSum/
│   │   └── TAMS.yaml
│   ├── VideoXum/
│   │   └── TAMS.yaml
│   ├── QFVS/
│   │   └── TAMS.yaml
│   └── MrHiSum/
│       └── TAMS.yaml
├── data/
│   ├── annotations/
│   │   ├── SumMe/
│   │   │   └── eccv16_dataset_summe_google_pool5.h5
│   │   ├── TVSum/
│   │   │   └── eccv16_dataset_tvsum_google_pool5.h5
│   │   ├── VideoXum/
│   │   │   ├── train_videoxum.json
│   │   │   ├── val_videoxum.json
│   │   │   └── test_videoxum.json
│   │   ├── QFVS/
│   │   │   └── origin_data/
│   │   └── MrHiSum/
│   │       ├── mrhisum_feat_visual_inceptionv3.h5
│   │       └── mrhisum_split.json
│   ├── graphs/
│   ├── generate_temporal_graphs.py
│   ├── build_qfvs_ut_graphs.py
│   └── build_qfvs_query_graphs.py
├── gravit/
├── tools/
├── results/
└── README.md
```

## 🛠️ Preprocessing

Generate temporal graphs offline before training on **SumMe** and **TVSum**:

```bash
python data/generate_temporal_graphs.py --dataset SumMe --features eccv16_dataset_summe_google_pool5 --tauf 10 --skip_factor 0
```

```bash
python data/generate_temporal_graphs.py --dataset TVSum --features eccv16_dataset_tvsum_google_pool5 --tauf 5 --skip_factor 0
```

For multimodal graphs (V+T+A), add `--modalities vta`:

```bash
python data/generate_temporal_graphs.py --dataset TVSum --features eccv16_dataset_tvsum_google_pool5 --tauf 30 --skip_factor 0 --modalities vta
```

Build **QFVS** graphs:

```bash
python data/build_qfvs_ut_graphs.py --cfg configs/QFVS/TAMS.yaml --all_splits
```

**VideoXum** and **MrHiSum** generate graphs online during training.

## 🚀 Training

Training on **SumMe** (5-fold cross-validation):

```bash
python tools/train.py --cfg configs/SumMe/TAMS.yaml --all_splits
```

Training on **TVSum**:

```bash
python tools/train.py --cfg configs/TVSum/TAMS.yaml --all_splits
```

Training on **VideoXum**:

```bash
python tools/train_videoxum.py --cfg configs/VideoXum/TAMS.yaml --all_splits
```

Training on **QFVS**:

```bash
python tools/train_qfvs.py --cfg configs/QFVS/TAMS.yaml --all_splits
```

Training on **MrHiSum**:

```bash
python tools/train_mrhisum.py --cfg configs/MrHiSum/TAMS.yaml
```

## 👀 Evaluation

Evaluation on **SumMe**:

```bash
python tools/eval.py --exp_name TAMS_SumMe --eval_type VS_max --all_splits
```

Evaluation on **TVSum**:

```bash
python tools/eval.py --exp_name TAMS_TVSum --eval_type VS_avg --all_splits
```

Evaluation on **VideoXum**:

```bash
python tools/eval_videoxum.py --exp_name TAMS_VideoXum --eval_type VS_avg --all_splits
```

Evaluation on **QFVS** (Sharghi protocol):

```bash
python tools/eval_qfvs_sharghi.py --source tams --cfg configs/QFVS/TAMS.yaml --all_splits
```

Evaluation on **MrHiSum**:

```bash
python tools/eval_mrhisum.py --exp_name TAMS_MrHiSum
```

## 📊 Results

| Dataset  | F1    | Kendall's τ | Spearman's ρ |
| -------- | ----- | ----------- | ------------ |
| SumMe    | 55.13 | 0.170       | 0.230        |
| TVSum    | 59.36 | 0.342       | 0.483        |
| VideoXum | 32.47 | 0.232       | 0.305        |

| Dataset | Avg F1 (Sharghi) |
|---------|------------------|
| QFVS    | 56.03            |

| Dataset | Kendall's τ | Spearman's ρ | mAP@50 | mAP@15 |
| ------- | ----------- | ------------ | ------ | ------ |
| MrHiSum | 0.269       | 0.364        | 71.04  | 42.31  |

## 📦 Model Zoo

Pre-trained checkpoints will be released soon.

## 🙏 Acknowledgements

This project builds upon the following works:

- [GraVi-T](https://github.com/IntelLabs/GraVi-T) — Graph-based Video Transformer
- [TripleSumm](https://github.com/smkim37/TripleSumm) — MrHiSum evaluation protocol

## 📄 Citation

If you find TAMS useful in your research, please cite:

```bibtex
@article{tams,
  title={TAMS: Topology-Adaptive Multi-Scale Network for Unified Video Summarization},
  author={},
  journal={},
  year={}
}
```
