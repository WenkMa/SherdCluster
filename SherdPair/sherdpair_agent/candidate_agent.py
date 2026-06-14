"""Candidate-set agent for filtering noisy cluster members and expanding neighbors."""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .data import SherdRecord, cluster_members


@dataclass(frozen=True)
class CandidateConfig:
    top_k_external: int = 10
    max_intra_pairs_per_cluster: int = 30
    min_cluster_size_for_filter: int = 3
    outlier_mad_z: float = 3.5
    outlier_quantile: float = 0.95
    external_search_pool: int = 80
    include_known_pairs: bool = True


def robust_outlier_flags(distances: np.ndarray, mad_z: float, quantile: float) -> np.ndarray:
    if len(distances) == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    if mad > 1e-9:
        z = 0.6745 * (distances - median) / mad
        return z > mad_z
    threshold = float(np.quantile(distances, quantile))
    return distances > threshold


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sim = vectors @ vectors.T
    return 1.0 - sim


def build_candidate_sets(
    records: list[SherdRecord],
    vectors: np.ndarray,
    known_pairs: set[tuple[str, str]],
    output_dir: Path,
    config: CandidateConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = cluster_members(records)
    id_to_index = {record.sherd_id: i for i, record in enumerate(records)}

    kept_ids: set[str] = set()
    outlier_ids: set[str] = set()
    member_rows: list[dict[str, object]] = []

    for cluster_id, members in grouped.items():
        indices = np.array([id_to_index[m.sherd_id] for m in members], dtype=np.int32)
        cluster_vectors = vectors[indices]
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid[None, :], axis=1)
        flags = np.zeros(len(members), dtype=bool)
        if len(members) >= config.min_cluster_size_for_filter:
            flags = robust_outlier_flags(
                distances,
                mad_z=config.outlier_mad_z,
                quantile=config.outlier_quantile,
            )
            # Do not empty a cluster by filtering too aggressively.
            if np.count_nonzero(~flags) < 2:
                order = np.argsort(distances)
                flags[:] = True
                flags[order[: min(2, len(order))]] = False
        for member, distance, is_outlier in zip(members, distances, flags):
            if is_outlier:
                outlier_ids.add(member.sherd_id)
            else:
                kept_ids.add(member.sherd_id)
            member_rows.append(
                {
                    "cluster_id": cluster_id,
                    "sherd_id": member.sherd_id,
                    "cluster_size": len(members),
                    "distance_to_cluster_centroid": f"{float(distance):.6f}",
                    "candidate_status": "outlier" if is_outlier else "kept",
                    "exterior_path": member.exterior_path,
                    "interior_path": member.interior_path,
                }
            )

    candidates: dict[tuple[str, str], dict[str, object]] = {}

    def add_pair(
        a: str,
        b: str,
        source: str,
        priority: float,
        cluster_a: int,
        cluster_b: int,
        embedding_distance: float,
        rank: int | str,
        known_match: bool = False,
    ) -> None:
        if a == b:
            return
        key = tuple(sorted((a, b)))
        row = candidates.get(key)
        if row is None or priority > float(row["priority"]):
            candidates[key] = {
                "sherd_id_a": key[0],
                "sherd_id_b": key[1],
                "cluster_a": cluster_a if key[0] == a else cluster_b,
                "cluster_b": cluster_b if key[1] == b else cluster_a,
                "source": source,
                "priority": f"{priority:.6f}",
                "embedding_distance": f"{embedding_distance:.6f}",
                "rank": rank,
                "known_match": int(known_match),
            }

    for cluster_id, members in grouped.items():
        kept_members = [m for m in members if m.sherd_id in kept_ids]
        if len(kept_members) < 2:
            continue
        pairs = list(itertools.combinations(kept_members, 2))
        pairs.sort(
            key=lambda pair: (
                np.linalg.norm(
                    vectors[id_to_index[pair[0].sherd_id]]
                    - vectors[id_to_index[pair[1].sherd_id]]
                ),
                pair[0].distance_to_center + pair[1].distance_to_center,
            )
        )
        if config.max_intra_pairs_per_cluster > 0:
            pairs = pairs[: config.max_intra_pairs_per_cluster]
        for rank, (a, b) in enumerate(pairs, start=1):
            dist = float(np.linalg.norm(vectors[id_to_index[a.sherd_id]] - vectors[id_to_index[b.sherd_id]]))
            add_pair(
                a.sherd_id,
                b.sherd_id,
                "intra_cluster",
                priority=1.0 / (1.0 + dist),
                cluster_a=a.cluster_id,
                cluster_b=b.cluster_id,
                embedding_distance=dist,
                rank=rank,
                known_match=tuple(sorted((a.sherd_id, b.sherd_id))) in known_pairs,
            )

    nn = NearestNeighbors(
        n_neighbors=min(config.external_search_pool + 1, len(records)),
        metric="euclidean",
    )
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)
    for source_index, record in enumerate(records):
        if record.sherd_id not in kept_ids:
            continue
        added = 0
        rank = 0
        for distance, target_index in zip(distances[source_index][1:], indices[source_index][1:]):
            target = records[int(target_index)]
            if target.sherd_id not in kept_ids:
                continue
            if target.cluster_id == record.cluster_id:
                continue
            rank += 1
            add_pair(
                record.sherd_id,
                target.sherd_id,
                "external_topk",
                priority=0.8 / (1.0 + float(distance)),
                cluster_a=record.cluster_id,
                cluster_b=target.cluster_id,
                embedding_distance=float(distance),
                rank=rank,
                known_match=tuple(sorted((record.sherd_id, target.sherd_id))) in known_pairs,
            )
            added += 1
            if added >= config.top_k_external:
                break

    if config.include_known_pairs:
        record_by_id = {record.sherd_id: record for record in records}
        for a, b in known_pairs:
            if a in record_by_id and b in record_by_id:
                ia = id_to_index[a]
                ib = id_to_index[b]
                dist = float(np.linalg.norm(vectors[ia] - vectors[ib]))
                add_pair(
                    a,
                    b,
                    "known_pair_seed",
                    priority=2.0,
                    cluster_a=record_by_id[a].cluster_id,
                    cluster_b=record_by_id[b].cluster_id,
                    embedding_distance=dist,
                    rank="known",
                    known_match=True,
                )

    members_path = output_dir / "candidate_members.csv"
    pairs_path = output_dir / "candidate_pairs.csv"
    summary_path = output_dir / "candidate_summary.csv"

    with members_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "sherd_id",
                "cluster_size",
                "distance_to_cluster_centroid",
                "candidate_status",
                "exterior_path",
                "interior_path",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(member_rows, key=lambda r: (int(r["cluster_id"]), str(r["sherd_id"]))))

    with pairs_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sherd_id_a",
            "sherd_id_b",
            "cluster_a",
            "cluster_b",
            "source",
            "priority",
            "embedding_distance",
            "rank",
            "known_match",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(
                candidates.values(),
                key=lambda r: (-float(r["priority"]), float(r["embedding_distance"]), r["sherd_id_a"], r["sherd_id_b"]),
            )
        )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["records", len(records)])
        writer.writerow(["clusters", len(grouped)])
        writer.writerow(["kept_members", len(kept_ids)])
        writer.writerow(["outlier_members", len(outlier_ids)])
        writer.writerow(["candidate_pairs", len(candidates)])
        writer.writerow(["known_pairs_available", len(known_pairs)])

    return {
        "members": members_path,
        "pairs": pairs_path,
        "summary": summary_path,
    }

