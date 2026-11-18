# SherdCluster 复现说明

[English](README.md) | 简体中文

本仓库提供用于复现 SherdCluster 考古陶片聚类方法的代码和数据。

原始图片已在 [Hugging Face 图片数据集](https://huggingface.co/datasets/Mwk19990801/SherdCluster_images)公开：共 **35,189 张独立 PNG，约 106.8 GiB**。图片保留原文件名和原始内容，可[逐张浏览和下载](https://huggingface.co/datasets/Mwk19990801/SherdCluster_images/tree/main/images)。模型权重和预计算特征仍通过下方 GitHub Release 下载。

## 快速开始

**1. 下载资源：** 从 [GitHub Release](https://github.com/WenkMa/SherdCluster/releases/tag/reproducibility-assets-v1) 下载模型权重和特征文件：

```bash
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.npz' --dir data/features
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.pth' --pattern '*.pt' --dir weights
```

**2. 配置环境：**

```bash
conda create -n sherdcluster python=3.10 -y
conda activate sherdcluster
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

**3. 使用提供的特征复现结果：**

```bash
python run.py reproduce --device cuda:0
```

结果保存在 `outputs/evaluation/metrics.json`。此流程使用提供的特征，无需下载图片。

## 从图像复现

此流程从提供的 ImageNet 初始化权重开始新一轮微调，随后完成特征提取、陶片聚类和评估。

**1. 准备图像。** 如果 `data/images/` 已包含原始图片，可跳过下面的下载和目录整理步骤；否则，在代码仓库根目录下载公开数据集：

```bash
python -m pip install huggingface_hub
hf download Mwk19990801/SherdCluster_images --repo-type dataset --local-dir data/images/.hf_download
```

数据集按 ID 前缀存放于 18 个子目录，例如 `images/JD00/JD00001_exterior.png`；训练脚本读取平铺的图片目录。运行以下命令整理目录，复用已有目标文件，并优先创建硬链接；无法创建硬链接时会复制文件，需要额外磁盘空间。

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

整理后，图片直接位于 `data/images/` 下，文件命名如下：
```
JD00001_exterior.png
JD00001_interior.png
```

数据集涉及 17,609 件陶片，其中 17,580 件具有完整内外侧图像。训练和特征提取使用完整配对图像；聚类时提供的藏品清单会保留其余 29 件陶片。

**2. 运行完整流程：** 训练 → 特征提取 → 聚类 → 评估：

```bash
python run_fresh.py \
  --image-dir data/images \
  --collection-ids data/collection_ids.txt \
  --output outputs/fresh_run \
  --device cuda:0 \
  --workers 4
```

监控进度：`tail -f outputs/fresh_run/training.log`

## 预期结果

使用提供的特征可得到以下结果：

| 特征 | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: |
| 外侧 | 30.31% | 51.55% | 61.24% |
| 内侧 | 21.24% | 36.49% | 46.80% |
| 内外侧拼接 | 52.78% | 75.67% | 81.86% |

聚类结果：17,609 件陶片得到 8,128 个簇；已知配对保留率为 61.35%。

上述指标基于提供的特征库；新一轮微调的结果可能有所不同。
