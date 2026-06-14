"""Default paths and parameters for the sherd pairing agents."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/home/wenkang/SherdPair_Agent")
CLUSTER_ROOT = Path("/home/wenkang/SherdCluster1")

DEFAULT_CLUSTERS_CSV = (
    CLUSTER_ROOT
    / "outputs/kmeans_dino-color-texture-resgcn_2side_pair_fused_concat_k6000"
    / "clusters_singleton_merged.csv"
)
DEFAULT_EMBEDDINGS_NPZ = (
    CLUSTER_ROOT / "outputs/dino-color-texture-resgcn_2side_20260613-013127.npz"
)
DEFAULT_MATCH_FILE = Path("/home/wenkang/datasets/match_all.xlsx")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_FEATURE = "pair_fused_concat"

