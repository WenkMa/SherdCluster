"""Judge agent for accepting, rejecting, and feeding back pairing results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgeConfig:
    accept_score: float = 2.25
    retry_score: float = 1.25
    max_overlap_ratio: float = 0.08
    retry_overlap_ratio: float = 0.18
    min_contact_ratio: float = 0.006
    max_gap_px: float = 6.0
    min_edge_color_score: float = 0.35
    min_texture_orientation_score: float = 0.25
    min_tangent_opposition_score: float = 0.35
    require_segment_match: bool = True
    max_segment_rms_px: float = 18.0
    max_segment_length_ratio: float = 1.35
    max_segment_angle_diff_deg: float = 35.0
    max_segment_roughness_diff: float = 0.45
    min_segment_shape_score: float = 0.35
    min_segment_color_score: float = 0.35
    min_segment_texture_score: float = 0.25
    min_segment_tangent_score: float = 0.35
    accept_known_pairs: bool = True


def judge_row(row: dict[str, str], config: JudgeConfig) -> tuple[str, float, str]:
    if row.get("operation") in {"load_failed", "no_feasible_transform"}:
        return "reject", -999.0, row.get("error", "load_failed")

    score = float(row.get("pair_score") or -999.0)
    overlap = float(row.get("overlap_ratio") or 1.0)
    contact = float(row.get("contact_ratio") or 0.0)
    gap = float(row.get("gap_px") or 9999.0)
    edge_color = float(row.get("edge_color_score") or 0.0)
    texture = float(row.get("texture_orientation_score") or 0.0)
    tangent = float(row.get("tangent_opposition_score") or 0.0)
    segment_rms = float(row.get("segment_rms_px") or 9999.0)
    segment_length_ratio = float(row.get("segment_length_ratio") or 9999.0)
    segment_angle_diff = float(row.get("segment_angle_diff_deg") or 9999.0)
    segment_roughness_diff = float(row.get("segment_roughness_diff") or 9999.0)
    segment_shape = float(row.get("segment_shape_score") or 0.0)
    segment_color = float(row.get("segment_color_score") or 0.0)
    segment_texture = float(row.get("segment_texture_score") or 0.0)
    segment_tangent = float(row.get("segment_tangent_score") or 0.0)
    known = str(row.get("known_match", "0")) in {"1", "True", "true"}

    reasons: list[str] = []
    if known and config.accept_known_pairs:
        score += 0.50
        reasons.append("known_pair_bonus")
    if overlap > config.retry_overlap_ratio:
        reasons.append("overlap_too_high")
        return "reject", score, ";".join(reasons)
    if overlap > config.max_overlap_ratio:
        reasons.append("overlap_needs_retry")
        return "retry", score, ";".join(reasons)
    if contact < config.min_contact_ratio and gap > config.max_gap_px:
        reasons.append("weak_contact_large_gap")
        return "retry" if score >= config.retry_score else "reject", score, ";".join(reasons)
    if config.require_segment_match and not row.get("segment_rms_px"):
        reasons.append("missing_long_segment_match")
        return "retry" if score >= config.retry_score else "reject", score, ";".join(reasons)
    segment_weak = 0
    if segment_rms > config.max_segment_rms_px:
        segment_weak += 1
        reasons.append("segment_rms_high")
    if segment_length_ratio > config.max_segment_length_ratio:
        segment_weak += 1
        reasons.append("segment_length_mismatch")
    if segment_angle_diff > config.max_segment_angle_diff_deg:
        segment_weak += 1
        reasons.append("segment_angle_mismatch")
    if segment_roughness_diff > config.max_segment_roughness_diff:
        segment_weak += 1
        reasons.append("segment_roughness_mismatch")
    if segment_shape < config.min_segment_shape_score:
        segment_weak += 1
        reasons.append("segment_shape_weak")
    if segment_color < config.min_segment_color_score:
        segment_weak += 1
        reasons.append("segment_color_weak")
    if segment_texture < config.min_segment_texture_score:
        segment_weak += 1
        reasons.append("segment_texture_weak")
    if segment_tangent < config.min_segment_tangent_score:
        segment_weak += 1
        reasons.append("segment_tangent_weak")
    if segment_weak >= 2:
        return "retry" if score >= config.retry_score else "reject", score, ";".join(reasons)
    weak_evidence = 0
    if edge_color < config.min_edge_color_score:
        weak_evidence += 1
        reasons.append("edge_color_weak")
    if texture < config.min_texture_orientation_score:
        weak_evidence += 1
        reasons.append("texture_orientation_weak")
    if tangent < config.min_tangent_opposition_score:
        weak_evidence += 1
        reasons.append("tangent_opposition_weak")
    if weak_evidence >= 2:
        return "retry" if score >= config.retry_score else "reject", score, ";".join(reasons)
    if score >= config.accept_score:
        reasons.append("score_accept")
        return "accept", score, ";".join(reasons)
    if score >= config.retry_score:
        reasons.append("score_retry")
        return "retry", score, ";".join(reasons)
    reasons.append("score_reject")
    return "reject", score, ";".join(reasons)


def run_judge_agent(
    pairing_results_csv: Path,
    output_dir: Path,
    config: JudgeConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    judged_path = output_dir / "judged_pairing_results.csv"
    accepted_path = output_dir / "accepted_pairs.csv"
    feedback_path = output_dir / "feedback_candidates.csv"
    summary_path = output_dir / "judge_summary.csv"

    with pairing_results_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    judged_rows: list[dict[str, object]] = []
    best_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    feedback_rows: list[dict[str, object]] = []
    counts = {"accept": 0, "retry": 0, "reject": 0}

    for row in rows:
        decision, calibrated_score, reason = judge_row(row, config)
        counts[decision] += 1
        judged = dict(row)
        judged["judge_decision"] = decision
        judged["calibrated_score"] = f"{calibrated_score:.6f}"
        judged["judge_reason"] = reason
        judged_rows.append(judged)

        key = tuple(sorted((row["sherd_id_a"], row["sherd_id_b"])))
        prev = best_by_pair.get(key)
        if decision == "accept" and (
            prev is None or calibrated_score > float(prev["calibrated_score"])
        ):
            best_by_pair[key] = judged
        elif decision == "retry":
            feedback_rows.append(
                {
                    "candidate_index": row.get("candidate_index", ""),
                    "sherd_id_a": key[0],
                    "sherd_id_b": key[1],
                    "cluster_a": row.get("cluster_a", ""),
                    "cluster_b": row.get("cluster_b", ""),
                    "source": row.get("source", ""),
                    "embedding_distance": row.get("embedding_distance", ""),
                    "known_match": row.get("known_match", "0"),
                    "feedback_action": "retry_with_wider_search",
                    "previous_score": f"{calibrated_score:.6f}",
                    "previous_reason": reason,
                    "previous_preview_path": row.get("preview_path", ""),
                }
            )

    judged_fieldnames = list(rows[0].keys()) + ["judge_decision", "calibrated_score", "judge_reason"] if rows else []
    with judged_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=judged_fieldnames)
        writer.writeheader()
        writer.writerows(judged_rows)

    accepted = sorted(
        best_by_pair.values(),
        key=lambda r: (-float(r["calibrated_score"]), r["sherd_id_a"], r["sherd_id_b"]),
    )
    with accepted_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = judged_fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)

    with feedback_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "candidate_index",
            "sherd_id_a",
            "sherd_id_b",
            "cluster_a",
            "cluster_b",
            "source",
            "embedding_distance",
            "known_match",
            "feedback_action",
            "previous_score",
            "previous_reason",
            "previous_preview_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feedback_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["input_results", len(rows)])
        writer.writerow(["accepted_transforms", counts["accept"]])
        writer.writerow(["accepted_unique_pairs", len(accepted)])
        writer.writerow(["retry_transforms", counts["retry"]])
        writer.writerow(["rejected_transforms", counts["reject"]])
        writer.writerow(["feedback_pairs", len(feedback_rows)])

    return {
        "judged": judged_path,
        "accepted": accepted_path,
        "feedback": feedback_path,
        "summary": summary_path,
    }
