"""Data loading helpers for sherd clusters, embeddings, and known pairs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class SherdRecord:
    sherd_id: str
    cluster_id: int
    distance_to_center: float
    exterior_path: str
    interior_path: str


def normalize_id(value: object) -> str:
    item = str(value).strip()
    if item.endswith(".png"):
        item = Path(item).stem
    for suffix in ("_exterior", "_interior"):
        if item.endswith(suffix):
            item = item[: -len(suffix)]
    return item


def read_clusters(path: Path) -> list[SherdRecord]:
    records: list[SherdRecord] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cluster_col = "cluster_id" if "cluster_id" in reader.fieldnames else "cluster"
        for row in reader:
            records.append(
                SherdRecord(
                    sherd_id=normalize_id(row["sherd_id"]),
                    cluster_id=int(row[cluster_col]),
                    distance_to_center=float(row.get("distance_to_center") or 0.0),
                    exterior_path=row["exterior_path"],
                    interior_path=row["interior_path"],
                )
            )
    return records


def records_by_id(records: list[SherdRecord]) -> dict[str, SherdRecord]:
    return {record.sherd_id: record for record in records}


def cluster_members(records: list[SherdRecord]) -> dict[int, list[SherdRecord]]:
    grouped: dict[int, list[SherdRecord]] = {}
    for record in records:
        grouped.setdefault(record.cluster_id, []).append(record)
    for members in grouped.values():
        members.sort(key=lambda r: (r.distance_to_center, r.sherd_id))
    return grouped


def load_feature_arrays(
    embeddings_path: Path,
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(embeddings_path, allow_pickle=True)
    sherd_ids = np.asarray([normalize_id(x) for x in data["sherd_ids"]])
    required = {
        "pair_fused_concat": ["exterior_fused", "interior_fused"],
        "pair_dino_concat": ["exterior_dino", "interior_dino"],
        "pair_color_concat": ["exterior_color", "interior_color"],
        "pair_texture_concat": ["exterior_texture", "interior_texture"],
        "averaged_fused": ["averaged_fused"],
        "averaged_dino": ["averaged_dino"],
        "averaged_color": ["averaged_color"],
        "averaged_texture": ["averaged_texture"],
    }
    if feature not in required:
        raise ValueError(f"Unsupported feature: {feature}")
    missing = [key for key in required[feature] if key not in data.files]
    if missing:
        raise ValueError(f"Missing arrays for {feature}: {missing}")
    parts = [data[key] for key in required[feature]]
    vectors = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    return sherd_ids, vectors.astype(np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def preprocess_vectors(
    vectors: np.ndarray,
    pca_dim: int = 0,
    random_state: int = 42,
) -> np.ndarray:
    vectors = l2_normalize(vectors)
    if pca_dim > 0 and pca_dim < vectors.shape[1]:
        pca_dim = min(pca_dim, vectors.shape[0])
        vectors = PCA(n_components=pca_dim, random_state=random_state).fit_transform(vectors)
        vectors = l2_normalize(vectors.astype(np.float32))
    return vectors.astype(np.float32)


def align_vectors_to_records(
    records: list[SherdRecord],
    embedding_ids: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    id_to_index = {str(sherd_id): i for i, sherd_id in enumerate(embedding_ids)}
    missing = [record.sherd_id for record in records if record.sherd_id not in id_to_index]
    if missing:
        raise ValueError(f"{len(missing)} cluster sherds are missing from embeddings; first={missing[:5]}")
    return np.stack([vectors[id_to_index[record.sherd_id]] for record in records]).astype(np.float32)


def read_known_pairs(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if not {"ID1", "ID2"} <= set(df.columns):
        return set()
    pairs: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        a = normalize_id(row["ID1"])
        b = normalize_id(row["ID2"])
        if a == b:
            continue
        pairs.add(tuple(sorted((a, b))))
    return pairs

