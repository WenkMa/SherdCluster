# SherdCluster Reproducibility

English | [简体中文](README.zh-CN.md)

This repository provides code and data to reproduce the SherdCluster method for archaeological pottery sherd clustering.

The original images are publicly available in the [Hugging Face image dataset](https://huggingface.co/datasets/Mwk19990801/SherdCluster_images): **35,189 individual PNG files, approximately 106.8 GiB**. Images retain their original filenames and contents and can be [browsed and downloaded individually](https://huggingface.co/datasets/Mwk19990801/SherdCluster_images/tree/main/images). Model weights and precomputed features are distributed through the GitHub Release below.

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

Results are saved to `outputs/evaluation/metrics.json`. This workflow uses the supplied features and does not require downloading the images.

## Reproducing from Images

This pipeline starts a new fine-tuning run from the supplied ImageNet initialization, then extracts features, clusters the sherds, and evaluates the results.

**1. Prepare images.** If `data/images/` already contains the original images, skip the download and directory preparation below. Otherwise, download the public dataset from the repository root:

```bash
python -m pip install huggingface_hub
hf download Mwk19990801/SherdCluster_images --repo-type dataset --local-dir data/images/.hf_download
```

The dataset stores images in 18 ID-prefix directories, such as `images/JD00/JD00001_exterior.png`. The training scripts read a flat image directory. Prepare that layout with the following command, which reuses existing destination files and creates hard links where possible. If hard links are unavailable, it copies the files and requires additional disk space.

```bash
python - <<'PY'
from pathlib import Path
import os
import shutil

source = Path("data/images/.hf_download/images")
destination = Path("data/images")
images = sorted(source.rglob("*.png"))
if not images:
    raise SystemExit("No downloaded PNG images found.")
if len({image.name for image in images}) != len(images):
    raise SystemExit("Duplicate image filenames found.")
destination.mkdir(parents=True, exist_ok=True)
added = 0
for image in images:
    target = destination / image.name
    if target.exists() or target.is_symlink():
        continue
    try:
        os.link(image.resolve(), target)
    except OSError:
        shutil.copy2(image, target)
    added += 1
print(f"Found {len(images)} PNG images; added {added} files to {destination}.")
PY
```

The resulting files are directly inside `data/images/`:
```
JD00001_exterior.png
JD00001_interior.png
```

The collection covers 17,609 specimens, including 17,580 with complete exterior/interior pairs. Training and feature extraction use the complete pairs; the supplied inventory retains the other 29 specimens during clustering.

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

These metrics use the supplied feature banks; a new fine-tuning run may produce different results.
