# SherdCluster 使用说明

本目录实现双表面 BYOL 自监督特征学习、特征检索评估、一致性约束聚类和聚类评估。
命令均在本目录执行；迁移到其他机器时，将下面的 cd 路径换成实际存放位置。

GitHub 仓库不包含原始图像。代码、配置、参考数据和已有验证结果位于 `main` 分支；
五个模型与特征文件保存在 [GitHub Release](https://github.com/WenkMa/SherdCluster/releases/tag/reproducibility-assets-v1)。
克隆仓库后，在仓库根目录执行以下 GitHub CLI 命令，下载需要的文件：

```bash
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.npz' --dir data/features
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.pth' --pattern '*.pt' --dir weights
```

也可以从 Release 页面手动下载，并放到下表所列路径：

| Release 文件 | 本地存放路径 | 用途 |
| --- | --- | --- |
| `exterior.npz` | `data/features/exterior.npz` | 外侧表示的检索评估 |
| `interior.npz` | `data/features/interior.npz` | 内侧表示的检索评估 |
| `both_concat.npz` | `data/features/both_concat.npz` | 融合表示的检索与聚类 |
| `resnet50_imagenet1k_v2.pth` | `weights/resnet50_imagenet1k_v2.pth` | 从 ImageNet 权重开始新微调 |
| `sherdcluster_epoch15.pt` | `weights/sherdcluster_epoch15.pt` | 使用已有陶片模型重新提取特征 |

只验证已有特征的检索和聚类时，下载三个 `.npz` 文件即可。
如需从图像提取特征或微调，请下载相应权重，并另行将成对 PNG 图像放入 `data/images/`。
本地已有这些文件时可直接使用，无需重复下载。下载目录由命令自动创建。

```bash
cd /home/wenkang/SherdCluster_reproducibility
```

根据目的选择操作：

- 只验证已有特征的检索表现：第 3 节。
- 重新聚类并验证已知配对是否保留：第 4 节。
- 从已有陶片模型重新提取特征再验证：第 5 节。
- 从 ImageNet 权重开始一次新的微调：第 6 节。
- 分步训练、提取、聚类、评估或断点续训：第 7 节。

## 1. 环境与设备

依赖版本以 requirements.txt 为准。建议在独立环境中安装：

```bash
conda create -n sherdcluster python=3.10 -y
conda activate sherdcluster
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

上述 PyTorch 安装命令使用 CUDA 12.4 wheel，需要兼容的 NVIDIA 驱动。
已有可用环境时可直接使用；安装步骤无需每次重复执行。
原记录的训练环境为 Python 3.10、PyTorch 2.5.0+cu124、torchvision 0.20.0、
NumPy 2.2.6、Pillow 12.2.0 和 RTX 4090（24 GB）。

以下示例使用一张 GPU。先查看设备，并将 1 换成实际要使用的物理 GPU 编号：

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=1
```

设置后，这张物理 GPU 在当前终端启动的程序中对应 --device cuda:0。
各命令也支持 --device cpu；完整训练建议使用 GPU。
所有输出目录必须是尚不存在的新目录。重复实验时更换 --output 路径。
run.py reproduce 固定写入 outputs/clusters/ 和 outputs/evaluation/；如它们已存在，
请使用第 3、4 节中可指定新输出路径的分步命令。

## 2. 输入文件与数据准备

主要文件：

| 文件 | 用途 |
| --- | --- |
| `train_byol_global_sides.py` | 双表面三视图 BYOL 微调 |
| `global_sides_transforms.py` | 训练增强与推理预处理 |
| `extract_byol_global_sides.py` | 从 online encoder 提取特征 |
| `cluster.py` | 相互近邻与全成员相似度约束聚类 |
| `evaluate.py` | 特征检索评估、参考配对同簇保留率评估 |
| `run.py` | 各阶段入口，读取 configs/ 中的对应配置 |
| `run_fresh.py` | 新微调 -> 提取 -> 聚类 -> 评估的一键流程 |
| `configs/train.json` | 微调配置 |
| `configs/extract.json` | 特征提取配置 |
| `configs/cluster.json` | 聚类配置 |
| `weights/resnet50_imagenet1k_v2.pth` | 新微调所需的 ImageNet 初始化权重 |
| `weights/sherdcluster_epoch15.pt` | 已有陶片模型，可用于直接提取特征 |
| `data/features/*.npz` | 已有模型的外侧、内侧及融合特征 |
| `data/reference_pairs.csv` | 评估用的 251 对已知陶片关系 |
| `data/collection_ids.txt` | 完整藏品清单，共 17,609 个 ID |
| `data/images/` | 原始前景图像，需单独准备，不随 GitHub 仓库上传 |
| `outputs/` | 每次实验生成的结果 |

直接使用已有特征进行第 3、4 节的验证，只需对应脚本、特征、参考配对和清单。
从图像重新提取或微调还需要图像及相应权重。迁移项目时需一并传输所需文件。

图像必须直接放在 data/images/ 下，使用区分大小写的 PNG 文件名：

```text
JD00001_exterior.png
JD00001_interior.png
```

同一件陶片使用相同 ID；不扫描子目录。训练和提取仅使用内外侧均存在的陶片。
当前数据有 17,580 件完整配对陶片，另有 29 件缺少一侧图像。
图像按完整范围保留长宽比，白色补边并缩放至 320 x 320；透明区域合成白色。
程序不执行前景分割或字迹去除，验证原方法时应使用相同的输入照片。

特征 .npz 文件包含：

- sample_ids: 陶片 ID，长度 N。
- features: float32 特征矩阵，形状 N x D。
- metadata_json: 特征来源和预处理信息。

exterior.npz、interior.npz 的 D=2048，both_concat.npz 的 D=4096。
内外侧特征分别归一化后直接拼接，检索和聚类时再对整条向量归一化。
推理使用共享 ResNet50 online encoder 的输出，不使用投影头或额外学习的融合模块。

reference_pairs.csv 的表头为 id_a,id_b，每行一对已知关系；不得重复或自身配对。
collection_ids.txt 每行一个唯一 ID；清单必须包含当前特征中的全部 ID。
参考配对只供 evaluate.py 读取，训练和聚类均不读取这些关系。

## 3. 验证特征表达：检索评估

用已有的三套特征分别验证外侧、内侧、内外侧融合表示：

```bash
python evaluate.py \
  --features data/features/exterior.npz data/features/interior.npz data/features/both_concat.npz \
  --pairs data/reference_pairs.csv \
  --output outputs/paper_retrieval \
  --device cuda:0
```

这一步无需训练、加载陶片模型或读取图像，也不执行聚类。
以参考关系涉及的 485 个不同陶片 ID 为查询，在 17,580 件陶片中按余弦相似度排序。
每个查询排除自身；相似度相同时按陶片 ID 字典序排序。
Hit@K = 前 K 个结果中至少有一个已知配对陶片的查询数 / 查询总数 x 100%。
报告 K=1、5、10。该指标衡量已知关系的检索命中率，不是所有相关陶片的完整召回率。
参考配对中的全部 ID 都必须存在于每一套被评估的特征中。

查看：

- `outputs/paper_retrieval/metrics.json`
  representation 中的 hit_counts、hit_percent、query_count、gallery_size。
- `outputs/paper_retrieval/exterior_queries.csv`
- `outputs/paper_retrieval/interior_queries.csv`
- `outputs/paper_retrieval/both_concat_queries.csv`
  每个查询的已知配对、最佳配对排名和 Top-10 检索 ID。

只评估融合特征时，--features 后只保留 data/features/both_concat.npz，
并使用新的输出目录。评估新提取的特征时，将三个输入路径换成新特征目录。

## 4. 验证聚类：先重新聚类，再评估

第一步：使用已有融合特征计算相互近邻图并执行约束合并：

```bash
python cluster.py \
  --features data/features/both_concat.npz \
  --collection-ids data/collection_ids.txt \
  --k 10 --eta 0.50 --rho 0.55 \
  --output outputs/paper_clusters \
  --device cuda:0
```

默认采用 mutual top-10 近邻；候选边相似度至少为 eta=0.50。
按相似度从高到低处理候选边，每次合并要求两个簇之间的所有成员相似度均不低于
rho=0.55。没有预设簇数或簇大小上限，参考配对不参与建图或合并。
提供完整清单后，29 件没有双表面特征的陶片各自保留为单件簇。
若省略 --collection-ids，则仅对有特征的 17,580 件陶片输出分组。
完整相似度矩阵会临时占用约 1.24 GB 磁盘空间，聚类成功结束后自动删除。

聚类输出：

- `outputs/paper_clusters/assignments.csv`
  sample_id、cluster_id、cluster_size、status，每件陶片一行。
- `outputs/paper_clusters/clusters.csv`
  每个簇的大小、内部配对数及最小相似度。
- `outputs/paper_clusters/summary.json`
  聚类参数、簇数、单件簇数、最大簇大小及合并统计。
- `outputs/paper_clusters/graph.npz`
  有特征陶片的近邻索引、候选边及其相似度。

第二步：独立评估输出分组：

```bash
python evaluate.py \
  --assignments outputs/paper_clusters/assignments.csv \
  --pairs data/reference_pairs.csv \
  --output outputs/paper_partition_eval \
  --device cpu
```

这一步只读取分组和参考配对，不需要图像、特征或 GPU。
查看 outputs/paper_partition_eval/metrics.json 中的 clustering：

- retained_pairs: 已知配对中两个陶片被分入同一簇的数量。
- retention_percent: retained_pairs / reference_pairs x 100%。
- total_sherds、clusters、singletons、max_cluster_size: 分组规模统计。

逐对结果在 outputs/paper_partition_eval/reference_pair_outcomes.csv。
参考配对同簇保留率不能单独表示整体聚类准确率；当前参考关系未提供完整分组真值，
此评估不报告全体陶片的聚类准确率、ARI 或 NMI。

也可以一条命令重算已有融合特征的聚类，并评估三套特征的检索和聚类结果：

```bash
python run.py reproduce --device cuda:0
```

该入口使用 data/features/、configs/cluster.json 和默认参考配对，
结果固定写入 outputs/clusters/ 与 outputs/evaluation/，不执行训练或特征提取。

## 5. 从已有陶片模型重新提取特征，再验证

如果希望验证“图像 -> 模型 -> 特征 -> 检索/聚类”的完整推理过程：

```bash
python extract_byol_global_sides.py \
  --checkpoint weights/sherdcluster_epoch15.pt \
  --image-dir data/images \
  --output outputs/paper_reextracted \
  --batch-size 16 --workers 4 \
  --device cuda:0
```

输出 exterior.npz、interior.npz、both_concat.npz 和自动生成的来源说明 metadata.json。
随后执行第 3、4 节的命令，将 data/features/ 替换成 outputs/paper_reextracted/，
并为检索、聚类和聚类评估分别指定新的输出目录。
该流程使用已有陶片模型的固定权重，不更新模型参数。

## 6. 从 0 开始新的微调：一键完成全部阶段

这里的“从 0 微调”指从 ImageNet IMAGENET1K_V2 预训练 ResNet50 开始新的陶片训练。
不加载已有陶片模型、历史特征或历史聚类结果；当前入口不提供随机初始化骨干的训练方式。
需要完整配对图像和 weights/resnet50_imagenet1k_v2.pth。

```bash
python run_fresh.py \
  --image-dir data/images \
  --collection-ids data/collection_ids.txt \
  --output outputs/fresh_run \
  --device cuda:0 \
  --workers 4
```

执行顺序：

1. 读取 configs/train.json，从本地 ImageNet 权重开始双表面 BYOL 自监督微调。
2. 使用此次生成的 train/checkpoint_epoch15.pt 提取全部配对图像的新特征。
3. 使用此次生成的 features/both_concat.npz 进行约束聚类。
4. 使用此次生成的三套特征和聚类分组，分别计算检索指标与参考配对同簇保留率。

训练配置：

- 实际训练 15 个 epoch，余弦学习率调度的总周期为 100 个 epoch。
- --epochs 100 --stop-after-epoch 15 保留原方法的前 15 轮学习率轨迹；
- 将 --epochs 改成 15 会改变学习率轨迹。
- batch size=32 件陶片，每件陶片内外侧各生成 3 个视图。
- 共享编码器，每侧独立计算 6 个有向跨视图预测损失，再将两侧损失相加更新一次。
- 不使用跨表面正样本、跨陶片配对标注或聚类结果作为训练监督。
- AdamW；骨干学习率 1e-4，投影/预测头学习率 3e-4；weight decay=1e-4。
- 图像 320 x 320，FP32，梯度裁剪 1.0，EMA 动量 0.996，seed=42。
- 全部骨干层参与微调；MLP 隐藏/输出维度为 4096/256。

训练开始前会扫描配对图像并读取文件内容记录输入签名，初始化阶段可能需要等待。
日志和主要结果：

| 文件 | 用途 |
| --- | --- |
| `outputs/fresh_run/status.json` | 流程状态和已完成阶段 |
| `outputs/fresh_run/training.log` | 训练日志 |
| `outputs/fresh_run/extraction.log` | 提取日志 |
| `outputs/fresh_run/clustering.log` | 聚类日志 |
| `outputs/fresh_run/evaluation.log` | 评估日志 |
| `outputs/fresh_run/train/config.json` | 实际训练配置 |
| `outputs/fresh_run/train/history.jsonl` | 各 epoch 的损失及特征监控信息 |
| `outputs/fresh_run/train/last.pt` | 最近完成 epoch 的完整训练断点 |
| `outputs/fresh_run/train/checkpoint_epoch15.pt` | 第 15 轮模型 |
| `outputs/fresh_run/features/*.npz` | 本次新模型的特征 |
| `outputs/fresh_run/clusters/assignments.csv` | 本次聚类分组 |
| `outputs/fresh_run/clusters/summary.json` | 本次聚类统计 |
| `outputs/fresh_run/evaluation/metrics.json` | 本次检索与聚类评估指标 |

在另一终端查看进度：

```bash
tail -f outputs/fresh_run/training.log
```

全部成功时 status.json 的 state 为 complete，且 evaluation/metrics.json 已生成。
只看到训练完成并不表示后续提取、聚类和评估均已完成。
该一键流程固定按第 15 轮模型继续提取。若要改变训练轮数，使用第 7 节分步流程，
并将提取步骤的 checkpoint 路径改成实际保存的轮数。
训练固定种子但未强制完全确定性的 GPU 运算，新微调的指标可能与已有模型不同。
本目录已实际跑通已有特征的检索和聚类评估；本次文档更新未重新执行完整微调。

## 7. 分步微调与断点续训

希望分别控制各阶段时，按以下顺序执行。下面所有命令使用同一套新模型及新特征：

```bash
python run.py train --output outputs/manual_run/train --device cuda:0
```

```bash
python run.py extract \
  --checkpoint outputs/manual_run/train/checkpoint_epoch15.pt \
  --output outputs/manual_run/features \
  --device cuda:0
```

```bash
python run.py cluster \
  --features outputs/manual_run/features/both_concat.npz \
  --output outputs/manual_run/clusters \
  --device cuda:0
```

```bash
python evaluate.py \
  --features outputs/manual_run/features/exterior.npz outputs/manual_run/features/interior.npz outputs/manual_run/features/both_concat.npz \
  --assignments outputs/manual_run/clusters/assignments.csv \
  --pairs data/reference_pairs.csv \
  --output outputs/manual_run/evaluation \
  --device cuda:0
```

run.py train/extract/cluster 分别读取 configs/train.json、extract.json、cluster.json。
上面的命令覆盖其中的输出、模型或特征路径，其他参数保持原配置。
更换图像目录时，训练和提取两步都加 --image-dir 实际目录。
更换藏品清单时，聚类加 --collection-ids 实际清单。
可通过对应阶段的 --config 指定另一份 JSON 配置，或通过命令行覆盖参数。
run.py evaluate 默认只评估 outputs/features/ 的检索；若还要评估聚类，
需额外提供 --assignments，或像上面一样直接调用 evaluate.py。

如新微调在第 15 轮之前中断，可从最近完整 epoch 的 last.pt 恢复：

```bash
python run.py train \
  --resume outputs/fresh_run/train/last.pt \
  --output outputs/resumed_train \
  --device cuda:0
```

此示例适用于第 6 节的默认训练配置；恢复时须保留原图像内容、训练超参数及
100 轮调度总周期。若原来改过配置，恢复时也要提供相同配置。
--resume 恢复模型、优化器、scaler 和随机数状态，从已完成 epoch 之后继续；
不恢复被中断 epoch 的批次进度，输出目录仍须使用新路径。
已完成第 15 轮的断点不能以相同的停止轮数继续训练。
恢复完成后，使用 outputs/resumed_train/checkpoint_epoch15.pt 提取特征，
再按第 3、4 节或本节分步命令执行检索、聚类和评估。
run_fresh.py 本身不提供 --resume，也不会自动跳过已完成阶段。
如需从已有 BYOL 模型启动另一轮微调，训练脚本另支持 --init-checkpoint；
该选项仅加载模型参数并重新创建优化器，区别于保留训练进度的 --resume。

## 8. 已有特征的实际验证结果

以下为本目录已有特征的实际运行结果，可用来核对操作是否正确。
不作为新微调必须达到的数值，也不表示模型在独立留出测试集上的泛化表现。

| 特征 | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: |
| 外侧 | 30.31% | 51.55% | 61.24% |
| 内侧 | 21.24% | 36.49% | 46.80% |
| 内外侧融合 | 52.78% | 75.67% | 81.86% |

聚类参数 k=10、eta=0.50、rho=0.55、mutual kNN，并包含完整藏品清单：

- 总陶片 17,609；总簇数 8,128；单件簇 2,877；最大簇 16 件。
- 251 对已知关系中 154 对同簇，保留率 61.35%。

已保存的结果：

- `outputs/validation_20261115_112205/evaluation/metrics.json`
- `outputs/validation_20261115_112205/clusters/assignments.csv`
- `outputs/validation_20261115_112205/clusters/summary.json`

这次已有特征验证使用 Python 3.13.13、PyTorch 2.6.0+cu124、NumPy 2.5.1；
它仅确认此环境下检索与聚类可运行，不替代完整训练环境的验证。

## 9. 其他数据与方法细节

用于其他批次陶片时，提供该批次的成对图像和完整 ID 清单。
训练、提取和聚类不需要参考配对；若要定量评估检索或配对保留率，
还须提供该批次自己的 id_a,id_b CSV，并在 evaluate.py 中通过 --pairs 指定。
run_fresh.py 的最终评估固定使用 data/reference_pairs.csv；其他数据的独立实验
建议采用第 7 节的分步命令，并在评估时显式设置 --pairs。

训练视图：

- 视图 0：完整范围，水平翻转概率 0.5。
- 视图 1：宽高保留 95%-100%，水平翻转概率 0.5，旋转 +/-10 度。
- 视图 2：宽高保留 97%-100%，透视概率 0.3、强度 0.05，旋转 +/-5 度，
  垂直翻转概率 0.3。

全部视图保留长宽比，每侧白边 4%，使用 ImageNet 归一化；无随机光度扰动。
提取时关闭随机增强。target 网络的参数使用 EMA，BN 运行统计由其自身前向更新。

聚类消融参数：

- --union-knn: 使用任一方向近邻关系，替代 mutual kNN。
- --no-all-pairs: 关闭所有跨簇成员相似度的合并约束。

这些参数会改变原方法的默认行为，应使用独立输出目录保存结果。
