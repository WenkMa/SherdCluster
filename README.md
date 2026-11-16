# SherdCluster Reproducibility

English | [简体中文](README.zh-CN.md)

This repository provides code and data to reproduce the SherdCluster method for archaeological pottery sherd clustering.

## Quick Start

**1. Download assets** (model weights and features) from [GitHub Release](https://github.com/WenkMa/SherdCluster/releases/tag/reproducibility-assets-v1):

```bash
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.npz' --dir data/features
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.pth' --pattern '*.pt' --dir weights
```

**2. Set up environment:**

```bash
conda create -n sherdcluster python=3.10 -y
conda activate sherdcluster
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

**3. Reproduce results using supplied features:**

```bash
python run.py reproduce --device cuda:0
```

Results are saved to [outputs/evaluation/metrics.json](outputs/evaluation/metrics.json).

## Reproducing from Images

To reproduce from scratch using your own images:

**1. Prepare images** in `data/images/`:
```
JD00001_exterior.png
JD00001_interior.png
```

**2. Run full pipeline** (training → extraction → clustering → evaluation):

```bash
python run_fresh.py \
  --image-dir data/images \
  --collection-ids data/collection_ids.txt \
  --output outputs/fresh_run \
  --device cuda:0 \
  --workers 4
```

Monitor progress: `tail -f outputs/fresh_run/training.log`

## Expected Results

Using the supplied features:

| Features | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: |
| Exterior | 30.31% | 51.55% | 61.24% |
| Interior | 21.24% | 36.49% | 46.80% |
| Both (concatenated) | 52.78% | 75.67% | 81.86% |

Clustering: 8,128 clusters from 17,609 specimens; 61.35% known-pair retention.
