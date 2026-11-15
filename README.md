# SherdCluster User Guide

English | [简体中文](README.zh-CN.md)

This repository implements dual-surface BYOL self-supervised representation learning,
retrieval evaluation, consistency-constrained clustering, and clustering evaluation
for archaeological pottery sherds. Run commands from the repository root. When moving
the project to another machine, replace the example `cd` path with its actual location.

Original images are excluded from GitHub. The `main` branch contains code,
configurations, reference data, and saved validation results. The five model and
feature files are available in the [GitHub Release](https://github.com/WenkMa/SherdCluster/releases/tag/reproducibility-assets-v1).
After cloning, run these GitHub CLI commands from the repository root:

```bash
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.npz' --dir data/features
gh release download reproducibility-assets-v1 --repo WenkMa/SherdCluster --pattern '*.pth' --pattern '*.pt' --dir weights
```

Alternatively, download the assets manually and place them at these paths:

| Release asset | Local path | Purpose |
| --- | --- | --- |
| `exterior.npz` | `data/features/exterior.npz` | Retrieval evaluation of exterior descriptors |
| `interior.npz` | `data/features/interior.npz` | Retrieval evaluation of interior descriptors |
| `both_concat.npz` | `data/features/both_concat.npz` | Retrieval and clustering with concatenated descriptors |
| `resnet50_imagenet1k_v2.pth` | `weights/resnet50_imagenet1k_v2.pth` | Start fine-tuning from ImageNet weights |
| `sherdcluster_epoch15.pt` | `weights/sherdcluster_epoch15.pt` | Extract features using the supplied sherd-domain model |

For retrieval and clustering evaluation using the supplied features, download the
three `.npz` files. Feature extraction and fine-tuning additionally require the
corresponding weights and paired PNG images in `data/images/`. Existing local files
can be used directly. The download commands create the destination directories.

```bash
cd /home/wenkang/SherdCluster_reproducibility
```

Choose the workflow for your task:

- Evaluate retrieval using the supplied features: Section 3.
- Recompute clusters and measure known-pair retention: Section 4.
- Extract features using the supplied sherd-domain model, then evaluate: Section 5.
- Start a new fine-tuning run from ImageNet weights: Section 6.
- Run individual stages or resume interrupted training: Section 7.

## 1. Environment and device

Dependency versions are specified in `requirements.txt`. An isolated environment is recommended:

```bash
conda create -n sherdcluster python=3.10 -y
conda activate sherdcluster
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The PyTorch command uses CUDA 12.4 wheels and requires a compatible NVIDIA driver.
You can use an existing working environment; installation need not be repeated for
each experiment. The recorded training environment used Python 3.10, PyTorch
2.5.0+cu124, torchvision 0.20.0, NumPy 2.2.6, Pillow 12.2.0, and an RTX 4090 (24 GB).

The examples use one GPU. Inspect the devices and replace `1` with the physical GPU
index you want to use:

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=1
```

The selected physical GPU is exposed as `--device cuda:0` to programs started from
that terminal. Commands also accept `--device cpu`; a GPU is recommended for full training.
Each output directory must be a new, nonexistent directory. Change `--output` for
repeated experiments. `run.py reproduce` always writes to `outputs/clusters/` and
`outputs/evaluation/`. If those directories already exist, use Sections 3 and 4 to
select different output paths.

## 2. Input files and data preparation

Main files:

| File | Purpose |
| --- | --- |
| `train_byol_global_sides.py` | Dual-surface, three-view BYOL fine-tuning |
| `global_sides_transforms.py` | Training augmentation and inference preprocessing |
| `extract_byol_global_sides.py` | Extract features from the online encoder |
| `cluster.py` | Clustering with mutual neighbors and all-member similarity constraints |
| `evaluate.py` | Retrieval evaluation and known-pair cluster retention evaluation |
| `run.py` | Stage entry point using corresponding configurations in `configs/` |
| `run_fresh.py` | New fine-tuning → extraction → clustering → evaluation |
| `configs/train.json` | Fine-tuning configuration |
| `configs/extract.json` | Feature extraction configuration |
| `configs/cluster.json` | Clustering configuration |
| `weights/resnet50_imagenet1k_v2.pth` | ImageNet initialization for new fine-tuning |
| `weights/sherdcluster_epoch15.pt` | Supplied sherd-domain model for feature extraction |
| `data/features/*.npz` | Supplied exterior, interior, and concatenated features |
| `data/reference_pairs.csv` | 251 known sherd pairs used for evaluation |
| `data/collection_ids.txt` | Complete inventory of 17,609 specimen IDs |
| `data/images/` | Original foreground images; prepare separately, excluded from GitHub |
| `outputs/` | Results generated by experiments |

Sections 3 and 4 require the relevant scripts, feature banks, reference pairs, and
inventory. Extracting features or fine-tuning also requires images and the corresponding
weights. Transfer the required files when moving the project.

Place images directly in `data/images/` using these case-sensitive PNG filenames:

```text
JD00001_exterior.png
JD00001_interior.png
```

Use the same specimen ID for both surfaces of a sherd. Subdirectories are not scanned.
Training and extraction use only specimens with both surfaces available. The supplied
data covers 17,580 paired specimens and 29 additional specimens missing one surface image.
Preprocessing preserves the full image extent and aspect ratio, adds white padding,
and produces 320 × 320 images. Transparent areas are composited on white. The program
does not segment foregrounds or remove specimen writing. Use the same photographs
when validating the original method.

Each feature `.npz` file contains:

- `sample_ids`: an array of N specimen IDs.
- `features`: a float32 feature matrix with shape N × D.
- `metadata_json`: feature provenance and preprocessing information.

For `exterior.npz` and `interior.npz`, D=2048; for `both_concat.npz`, D=4096.
Surface descriptors are individually normalized and directly concatenated; retrieval
and clustering normalize the concatenated vector again. Inference uses the shared
ResNet50 online encoder output, without a projection head or an additional learned fusion module.

`reference_pairs.csv` has the header `id_a,id_b`, with one known relation per row.
Duplicate undirected pairs and self-pairs are invalid. `collection_ids.txt` contains
one unique ID per line and must include every ID in the feature bank. Only
`evaluate.py` reads the reference pairs; training and clustering do not use them.

## 3. Validate feature representations: retrieval evaluation

Evaluate the supplied exterior, interior, and concatenated feature banks:

```bash
python evaluate.py \
  --features data/features/exterior.npz data/features/interior.npz data/features/both_concat.npz \
  --pairs data/reference_pairs.csv \
  --output outputs/paper_retrieval \
  --device cuda:0
```

This step operates directly on feature banks, without training, loading model weights,
reading images, or running clustering. The 485 distinct specimen IDs involved in the
reference pairs are queries against a gallery of 17,580 specimens ranked by cosine
similarity. Each query excludes itself; ties are resolved by lexical specimen ID order.

**Hit@K = queries with at least one known partner in the first K results / total queries × 100%.**

The program reports K=1, 5, and 10. This measures retrieval hits for known relations;
it does not measure complete recall of every relevant sherd. Every reference-pair ID
must be present in each feature bank being evaluated.

Inspect these outputs:

- `outputs/paper_retrieval/metrics.json`: `hit_counts`, `hit_percent`, `query_count`,
  and `gallery_size` under `representation`.
- `outputs/paper_retrieval/exterior_queries.csv`
- `outputs/paper_retrieval/interior_queries.csv`
- `outputs/paper_retrieval/both_concat_queries.csv`

The query CSV files contain known partners, the best partner rank, and Top-10 retrieved
IDs for each query. To evaluate concatenated features alone, keep only
`data/features/both_concat.npz` after `--features` and select a new output directory.
To evaluate new features, replace the three input paths with the new feature bank paths.

## 4. Validate clustering: recompute clusters, then evaluate

First, build the mutual-neighbor graph and run constrained cluster merging using
the supplied concatenated features:

```bash
python cluster.py \
  --features data/features/both_concat.npz \
  --collection-ids data/collection_ids.txt \
  --k 10 --eta 0.50 --rho 0.55 \
  --output outputs/paper_clusters \
  --device cuda:0
```

The default uses mutual top-10 neighbors and candidate edges with similarity at least
eta=0.50. Edges are processed in descending similarity order. A merge requires every
cross-cluster member similarity to be at least rho=0.55. There is no prescribed cluster
count or cluster-size cap. Reference pairs do not enter graph construction or merging.

With the complete inventory, the 29 specimens lacking dual-surface descriptors remain
individual singleton clusters. Omitting `--collection-ids` produces assignments only
for the 17,580 specimens with features. The full similarity matrix temporarily occupies
approximately 1.24 GB of disk space and is deleted after successful clustering.

Clustering outputs:

- `outputs/paper_clusters/assignments.csv`: one row per specimen with `sample_id`,
  `cluster_id`, `cluster_size`, and `status`.
- `outputs/paper_clusters/clusters.csv`: cluster sizes, within-cluster pair counts,
  and minimum similarities.
- `outputs/paper_clusters/summary.json`: parameters, cluster counts, singleton counts,
  maximum cluster size, and merge statistics.
- `outputs/paper_clusters/graph.npz`: neighbor indices, candidate edges, and similarities
  for specimens with features.

Second, independently evaluate the resulting partition:

```bash
python evaluate.py \
  --assignments outputs/paper_clusters/assignments.csv \
  --pairs data/reference_pairs.csv \
  --output outputs/paper_partition_eval \
  --device cpu
```

This step reads only assignments and reference pairs; it requires no images, features,
or GPU. Inspect `clustering` in `outputs/paper_partition_eval/metrics.json`:

- `retained_pairs`: known pairs whose specimens are assigned to the same cluster.
- `retention_percent`: `retained_pairs / reference_pairs × 100%`.
- `total_sherds`, `clusters`, `singletons`, and `max_cluster_size`: partition statistics.

Per-pair results are saved to `outputs/paper_partition_eval/reference_pair_outcomes.csv`.
Known-pair retention alone does not establish overall clustering accuracy. The reference
relations do not provide complete partition ground truth, so this evaluation does not
report overall clustering accuracy, ARI, or NMI.

Alternatively, use one command to recompute clustering from supplied concatenated
features and evaluate retrieval for all three feature banks and known-pair retention:

```bash
python run.py reproduce --device cuda:0
```

This uses `data/features/`, `configs/cluster.json`, and the default reference pairs.
It writes to `outputs/clusters/` and `outputs/evaluation/`, without training or extraction.

## 5. Extract features from the supplied sherd-domain model, then evaluate

To validate the full inference path from images through the model and descriptors
to retrieval and clustering:

```bash
python extract_byol_global_sides.py \
  --checkpoint weights/sherdcluster_epoch15.pt \
  --image-dir data/images \
  --output outputs/paper_reextracted \
  --batch-size 16 --workers 4 \
  --device cuda:0
```

Outputs are `exterior.npz`, `interior.npz`, `both_concat.npz`, and the automatically
generated provenance file `metadata.json`. Then run Sections 3 and 4, replacing
`data/features/` with `outputs/paper_reextracted/`. Select new output directories
for retrieval, clustering, and partition evaluation. Model weights remain fixed.

## 6. Start a new fine-tuning run: the full pipeline

A new fine-tuning run initializes ResNet50 with ImageNet IMAGENET1K_V2 pretrained
weights. It does not load the supplied sherd-domain checkpoint, historical features,
or historical clustering results. The current entry point does not provide training
with a randomly initialized backbone. Paired images and
`weights/resnet50_imagenet1k_v2.pth` are required.

```bash
python run_fresh.py \
  --image-dir data/images \
  --collection-ids data/collection_ids.txt \
  --output outputs/fresh_run \
  --device cuda:0 \
  --workers 4
```

The pipeline performs these stages:

1. Read `configs/train.json` and fine-tune dual-surface BYOL from local ImageNet weights.
2. Extract features from all paired images using this run's `train/checkpoint_epoch15.pt`.
3. Cluster this run's `features/both_concat.npz` using the consistency constraints.
4. Evaluate retrieval and known-pair retention using the new features and assignments.

Training configuration:

- Train for 15 actual epochs with a cosine learning-rate schedule spanning 100 epochs.
  `--epochs 100 --stop-after-epoch 15` preserves the original method's first 15 epochs
  of learning rates; setting `--epochs 15` changes the schedule.
- Batch size: 32 sherds, with three views for each surface.
- A shared encoder computes six directed cross-view prediction losses independently
  for each surface; the two surface losses are summed for one update.
- No cross-surface positives, cross-sherd relation annotations, or clustering results
  are used as training supervision.
- AdamW; backbone learning rate 1e-4, projection/prediction head learning rate 3e-4,
  and weight decay 1e-4.
- 320 × 320 images, FP32, gradient clipping at 1.0, EMA momentum 0.996, and seed 42.
- All backbone layers are fine-tuned. MLP hidden/output dimensions are 4096/256.

Before training, the program scans paired images and reads their contents to record
an input signature. Initialization can therefore take some time. Logs and main outputs:

| File | Purpose |
| --- | --- |
| `outputs/fresh_run/status.json` | Pipeline state and completed stages |
| `outputs/fresh_run/training.log` | Training log |
| `outputs/fresh_run/extraction.log` | Extraction log |
| `outputs/fresh_run/clustering.log` | Clustering log |
| `outputs/fresh_run/evaluation.log` | Evaluation log |
| `outputs/fresh_run/train/config.json` | Actual training configuration |
| `outputs/fresh_run/train/history.jsonl` | Per-epoch losses and feature monitoring |
| `outputs/fresh_run/train/last.pt` | Full checkpoint from the latest completed epoch |
| `outputs/fresh_run/train/checkpoint_epoch15.pt` | Epoch-15 model |
| `outputs/fresh_run/features/*.npz` | Features from the new model |
| `outputs/fresh_run/clusters/assignments.csv` | New cluster assignments |
| `outputs/fresh_run/clusters/summary.json` | New clustering statistics |
| `outputs/fresh_run/evaluation/metrics.json` | New retrieval and clustering metrics |

Monitor progress from another terminal:

```bash
tail -f outputs/fresh_run/training.log
```

The full pipeline succeeds when `state` in `status.json` is `complete` and
`evaluation/metrics.json` exists. Finishing training alone does not mean extraction,
clustering, and evaluation have also finished. The one-command pipeline extracts from
the epoch-15 checkpoint. To change the training duration, use Section 7 and point
extraction to the checkpoint for the actual saved epoch.

Training uses a fixed seed but does not enforce fully deterministic GPU computation,
so new runs can produce different metrics from the supplied model. Retrieval and
clustering with the supplied features have been run successfully; this documentation
update did not rerun full fine-tuning.

## 7. Individual stages and resuming training

To control each stage separately, run these commands in order. All stages below use
the same new model and its newly extracted features:

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

`run.py train`, `run.py extract`, and `run.py cluster` read `configs/train.json`,
`configs/extract.json`, and `configs/cluster.json`, respectively. These commands
override output, model, or feature paths while retaining other configured values.
For a different image directory, add `--image-dir YOUR_DIRECTORY` to both training
and extraction. For a different inventory, add `--collection-ids YOUR_INVENTORY`
to clustering. Use `--config` to select another JSON configuration for the stage,
or override parameters on the command line.

`run.py evaluate` evaluates retrieval from `outputs/features/` by default. To also
evaluate clustering, supply `--assignments`, or call `evaluate.py` directly as above.

If training is interrupted before epoch 15, resume from the latest completed epoch's `last.pt`:

```bash
python run.py train \
  --resume outputs/fresh_run/train/last.pt \
  --output outputs/resumed_train \
  --device cuda:0
```

This example assumes the default configuration in Section 6. Preserve input image
contents, training hyperparameters, and the 100-epoch schedule horizon. If you
changed the original configuration, provide the same settings when resuming.
`--resume` restores the model, optimizer, scaler, and random number generator states,
then continues after the last completed epoch. Batch progress within an interrupted
epoch is not restored. The resumed run requires a new output directory. A checkpoint
already at epoch 15 cannot continue with the same stopping epoch.

After resuming, extract features using `outputs/resumed_train/checkpoint_epoch15.pt`,
then run retrieval, clustering, and evaluation using Sections 3 and 4 or the commands
above. `run_fresh.py` has no `--resume` option and does not automatically skip completed
stages. To start another fine-tuning run from an existing BYOL model, the training
script also supports `--init-checkpoint`. This loads model parameters and creates
a new optimizer, whereas `--resume` preserves training progress.

## 8. Measured results using the supplied features

These results were obtained by running the supplied feature banks and can help check
that evaluation was executed correctly. They are not required targets for new runs
and do not establish generalization on an independent held-out test set.

| Features | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: |
| Exterior | 30.31% | 51.55% | 61.24% |
| Interior | 21.24% | 36.49% | 46.80% |
| Exterior + interior concatenation | 52.78% | 75.67% | 81.86% |

Clustering used k=10, eta=0.50, rho=0.55, mutual kNN, and the complete inventory:

- 17,609 specimens; 8,128 clusters; 2,877 singleton clusters; maximum cluster size 16.
- 154 of the 251 known pairs were retained within clusters: 61.35% retention.

Saved results:

- `outputs/validation_20261115_112205/evaluation/metrics.json`
- `outputs/validation_20261115_112205/clusters/assignments.csv`
- `outputs/validation_20261115_112205/clusters/summary.json`

This supplied-feature validation used Python 3.13.13, PyTorch 2.6.0+cu124, and NumPy 2.5.1.
It confirms retrieval and clustering run in that environment; it does not replace
validation of the full training environment.

## 9. Other datasets and method details

For another assemblage, provide its paired images and complete ID inventory.
Training, extraction, and clustering do not require reference pairs. Quantitative
retrieval or pair-retention evaluation requires that assemblage's own `id_a,id_b`
CSV, supplied through `evaluate.py --pairs`. The final evaluation in `run_fresh.py`
uses `data/reference_pairs.csv`. For independent experiments on another dataset,
use Section 7 and explicitly set `--pairs` during evaluation.

Training views:

- View 0: full image extent; horizontal flip probability 0.5.
- View 1: retain 95%–100% of width and height; horizontal flip probability 0.5;
  rotation ±10 degrees.
- View 2: retain 97%–100%; perspective probability 0.3 with strength 0.05;
  rotation ±5 degrees; vertical flip probability 0.3.

All views preserve aspect ratio, use 4% white padding per side and ImageNet
normalization, and apply no random photometric perturbation. Random augmentation
is disabled during extraction. Target network parameters use EMA; batch-normalization
running statistics are updated by the target network's own forward passes.

Clustering ablations:

- `--union-knn`: use a neighbor relation in either direction instead of mutual kNN.
- `--no-all-pairs`: disable the cross-cluster all-member similarity constraint.

These options change the default method. Save each ablation in its own output directory.
