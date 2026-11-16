# SherdCluster 复现说明

[English](README.md) | 简体中文

本仓库提供用于复现 SherdCluster 考古陶片聚类方法的代码和数据。

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

结果保存在 [outputs/evaluation/metrics.json](outputs/evaluation/metrics.json)。

## 从图像复现

如需使用自己的图像从头复现：

**1. 准备图像：** 将图像放入 `data/images/`，文件命名如下：
```
JD00001_exterior.png
JD00001_interior.png
```

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
