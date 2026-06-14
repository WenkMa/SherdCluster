"""Command line entry point for sherd pairing agents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .candidate_agent import CandidateConfig, build_candidate_sets
from .config import (
    DEFAULT_CLUSTERS_CSV,
    DEFAULT_EMBEDDINGS_NPZ,
    DEFAULT_FEATURE,
    DEFAULT_MATCH_FILE,
    DEFAULT_OUTPUT_DIR,
)
from .data import (
    SherdRecord,
    align_vectors_to_records,
    load_feature_arrays,
    preprocess_vectors,
    read_clusters,
    read_known_pairs,
)
from .judge_agent import JudgeConfig, run_judge_agent
from .legacy_refine_agent import DEFAULT_LEGACY_MODEL, LegacyRefineConfig, run_legacy_refine_agent
from .pair_agent import PairingConfig, run_pairing_agent
from .visualize import build_pairing_contact_sheet


def add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_NPZ)
    parser.add_argument("--feature", default=DEFAULT_FEATURE)
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--match-file", type=Path, default=DEFAULT_MATCH_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cluster-limit",
        type=int,
        default=0,
        help="Only use the first N cluster IDs, sorted ascending. 0 means all clusters.",
    )


def limit_records_by_clusters(records: list[SherdRecord], cluster_limit: int) -> list[SherdRecord]:
    if cluster_limit <= 0:
        return records
    cluster_ids = sorted({record.cluster_id for record in records})[:cluster_limit]
    allowed = set(cluster_ids)
    return [record for record in records if record.cluster_id in allowed]


def load_records_and_vectors(args: argparse.Namespace):
    records = read_clusters(args.clusters)
    records = limit_records_by_clusters(records, args.cluster_limit)
    embedding_ids, raw_vectors = load_feature_arrays(args.embeddings, args.feature)
    vectors = preprocess_vectors(raw_vectors, pca_dim=args.pca_dim)
    vectors = align_vectors_to_records(records, embedding_ids, vectors)
    known_pairs = read_known_pairs(args.match_file)
    return records, vectors, known_pairs


def run_candidates(args: argparse.Namespace) -> Path:
    records, vectors, known_pairs = load_records_and_vectors(args)
    paths = build_candidate_sets(
        records=records,
        vectors=vectors,
        known_pairs=known_pairs,
        output_dir=args.output_dir / "candidates",
        config=CandidateConfig(
            top_k_external=args.top_k_external,
            max_intra_pairs_per_cluster=args.max_intra_pairs_per_cluster,
            min_cluster_size_for_filter=args.min_cluster_size_for_filter,
            outlier_mad_z=args.outlier_mad_z,
            outlier_quantile=args.outlier_quantile,
            external_search_pool=args.external_search_pool,
            include_known_pairs=not args.no_known_pair_seed,
        ),
    )
    print(f"candidate members: {paths['members']}")
    print(f"candidate pairs: {paths['pairs']}")
    print(f"candidate summary: {paths['summary']}")
    return paths["pairs"]


def run_pairing(args: argparse.Namespace) -> Path:
    records = read_clusters(args.clusters)
    records = limit_records_by_clusters(records, args.cluster_limit)
    output = run_pairing_agent(
        candidates_csv=args.candidates,
        records=records,
        output_dir=args.output_dir / "pairing",
        config=PairingConfig(
            max_pairs=args.max_pairs,
            contour_points=args.contour_points,
            angle_step_deg=args.angle_step_deg,
            top_transforms_per_pair=args.top_transforms_per_pair,
            refine_top_n=args.refine_top_n,
            normal_push_px=args.normal_push_px,
            max_overlap_ratio=args.max_overlap_ratio,
            min_contact_ratio=args.min_contact_ratio,
            same_side_only=not args.allow_cross_side,
            use_segment_matching=not args.disable_segment_matching,
            segment_sample_points=args.segment_sample_points,
            segment_points=args.segment_points,
            segment_length_px=args.segment_length_px,
            min_segment_length_px=args.min_segment_length_px,
            max_segment_length_ratio=args.max_segment_length_ratio,
            max_segment_angle_diff_deg=args.max_segment_angle_diff_deg,
            max_segment_roughness_diff=args.max_segment_roughness_diff,
            curvature_weight=args.curvature_weight,
            segment_stride=args.segment_stride,
            max_segment_pairs=args.max_segment_pairs,
            max_segment_rms_px=args.max_segment_rms_px,
            max_segment_max_px=args.max_segment_max_px,
            enable_point_fallback=args.enable_point_fallback,
            edge_patch_radius=args.edge_patch_radius,
            edge_color_weight=args.edge_color_weight,
            texture_weight=args.texture_weight,
            tangent_weight=args.tangent_weight,
            segment_shape_weight=args.segment_shape_weight,
            segment_param_weight=args.segment_param_weight,
            segment_color_weight=args.segment_color_weight,
            segment_texture_weight=args.segment_texture_weight,
            segment_tangent_weight=args.segment_tangent_weight,
            texture_patch_radius=args.texture_patch_radius,
            texture_samples_per_segment=args.texture_samples_per_segment,
            min_texture_strength=args.min_texture_strength,
        ),
    )
    print(f"pairing results: {output}")
    return output


def run_judge(args: argparse.Namespace) -> dict[str, Path]:
    paths = run_judge_agent(
        pairing_results_csv=args.pairing_results,
        output_dir=args.output_dir / "judge",
        config=JudgeConfig(
            accept_score=args.accept_score,
            retry_score=args.retry_score,
            max_overlap_ratio=args.judge_max_overlap_ratio,
            retry_overlap_ratio=args.retry_overlap_ratio,
            min_contact_ratio=args.judge_min_contact_ratio,
            max_gap_px=args.max_gap_px,
            min_edge_color_score=args.min_edge_color_score,
            min_texture_orientation_score=args.min_texture_orientation_score,
            min_tangent_opposition_score=args.min_tangent_opposition_score,
            require_segment_match=not args.no_require_segment_match,
            max_segment_rms_px=args.judge_max_segment_rms_px,
            max_segment_length_ratio=args.judge_max_segment_length_ratio,
            max_segment_angle_diff_deg=args.judge_max_segment_angle_diff_deg,
            max_segment_roughness_diff=args.judge_max_segment_roughness_diff,
            min_segment_shape_score=args.min_segment_shape_score,
            min_segment_color_score=args.min_segment_color_score,
            min_segment_texture_score=args.min_segment_texture_score,
            min_segment_tangent_score=args.min_segment_tangent_score,
            accept_known_pairs=not args.no_accept_known_pairs,
        ),
    )
    print(f"judged results: {paths['judged']}")
    print(f"accepted pairs: {paths['accepted']}")
    print(f"feedback candidates: {paths['feedback']}")
    print(f"judge summary: {paths['summary']}")
    return paths


def cmd_candidates(args: argparse.Namespace) -> None:
    run_candidates(args)


def cmd_pair(args: argparse.Namespace) -> None:
    run_pairing(args)


def cmd_judge(args: argparse.Namespace) -> None:
    run_judge(args)


def cmd_legacy_refine(args: argparse.Namespace) -> None:
    records = read_clusters(args.clusters)
    records = limit_records_by_clusters(records, args.cluster_limit)
    output = run_legacy_refine_agent(
        pairs_csv=args.pairs,
        records=records,
        output_dir=args.output_dir / "legacy_refine",
        config=LegacyRefineConfig(
            max_pairs=args.legacy_max_pairs,
            num_repeat=args.legacy_num_repeat,
            contour_spacing=args.legacy_contour_spacing,
            patch_size=args.legacy_patch_size,
            model_path=args.legacy_model_path,
        ),
    )
    print(f"legacy refine results: {output}")


def cmd_visualize(args: argparse.Namespace) -> None:
    output = build_pairing_contact_sheet(
        pairing_results_csv=args.pairing_results,
        output_path=args.output,
        max_items=args.max_items,
        thumb_width=args.thumb_width,
    )
    print(f"contact sheet: {output}")


def cmd_run(args: argparse.Namespace) -> None:
    candidates = run_candidates(args)
    args.candidates = candidates
    pairing_results = run_pairing(args)
    args.pairing_results = pairing_results
    judge_paths = run_judge(args)
    if args.legacy_refine:
        records = read_clusters(args.clusters)
        records = limit_records_by_clusters(records, args.cluster_limit)
        output = run_legacy_refine_agent(
            pairs_csv=judge_paths["accepted"],
            records=records,
            output_dir=args.output_dir / "legacy_refine",
            config=LegacyRefineConfig(
                max_pairs=args.legacy_max_pairs,
                num_repeat=args.legacy_num_repeat,
                contour_spacing=args.legacy_contour_spacing,
                patch_size=args.legacy_patch_size,
                model_path=args.legacy_model_path,
            ),
        )
        print(f"legacy refine results: {output}")
    if args.feedback_rounds > 0:
        run_feedback_rounds(args, judge_paths["feedback"])


def run_feedback_rounds(args: argparse.Namespace, feedback_csv: Path) -> None:
    records = read_clusters(args.clusters)
    records = limit_records_by_clusters(records, args.cluster_limit)
    current_feedback = feedback_csv
    for round_index in range(1, args.feedback_rounds + 1):
        if count_csv_rows(current_feedback) == 0:
            print(f"feedback round {round_index}: no retry candidates")
            break
        round_output_dir = args.output_dir / f"feedback_round_{round_index}"
        pairing_results = run_pairing_agent(
            candidates_csv=current_feedback,
            records=records,
            output_dir=round_output_dir / "pairing",
            config=PairingConfig(
                max_pairs=args.feedback_max_pairs,
                contour_points=max(args.contour_points, args.feedback_contour_points),
                angle_step_deg=min(args.angle_step_deg, args.feedback_angle_step_deg),
                top_transforms_per_pair=args.top_transforms_per_pair,
                refine_top_n=max(args.refine_top_n, args.feedback_refine_top_n),
                normal_push_px=args.normal_push_px,
                max_overlap_ratio=args.max_overlap_ratio,
                min_contact_ratio=args.min_contact_ratio,
                same_side_only=not args.allow_cross_side,
                use_segment_matching=not args.disable_segment_matching,
                segment_sample_points=args.segment_sample_points,
                segment_points=args.segment_points,
                segment_length_px=args.segment_length_px,
                min_segment_length_px=args.min_segment_length_px,
                max_segment_length_ratio=args.max_segment_length_ratio,
                max_segment_angle_diff_deg=args.max_segment_angle_diff_deg,
                max_segment_roughness_diff=args.max_segment_roughness_diff,
                curvature_weight=args.curvature_weight,
                segment_stride=args.segment_stride,
                max_segment_pairs=args.max_segment_pairs,
                max_segment_rms_px=args.max_segment_rms_px,
                max_segment_max_px=args.max_segment_max_px,
                enable_point_fallback=args.enable_point_fallback,
                edge_patch_radius=args.edge_patch_radius,
                edge_color_weight=args.edge_color_weight,
                texture_weight=args.texture_weight,
                tangent_weight=args.tangent_weight,
                segment_shape_weight=args.segment_shape_weight,
                segment_param_weight=args.segment_param_weight,
                segment_color_weight=args.segment_color_weight,
                segment_texture_weight=args.segment_texture_weight,
                segment_tangent_weight=args.segment_tangent_weight,
                texture_patch_radius=args.texture_patch_radius,
                texture_samples_per_segment=args.texture_samples_per_segment,
                min_texture_strength=args.min_texture_strength,
            ),
        )
        paths = run_judge_agent(
            pairing_results_csv=pairing_results,
            output_dir=round_output_dir / "judge",
            config=JudgeConfig(
                accept_score=args.accept_score,
                retry_score=args.retry_score,
                max_overlap_ratio=args.judge_max_overlap_ratio,
                retry_overlap_ratio=args.retry_overlap_ratio,
                min_contact_ratio=args.judge_min_contact_ratio,
                max_gap_px=args.max_gap_px,
                min_edge_color_score=args.min_edge_color_score,
                min_texture_orientation_score=args.min_texture_orientation_score,
                min_tangent_opposition_score=args.min_tangent_opposition_score,
                require_segment_match=not args.no_require_segment_match,
                max_segment_rms_px=args.judge_max_segment_rms_px,
                max_segment_length_ratio=args.judge_max_segment_length_ratio,
                max_segment_angle_diff_deg=args.judge_max_segment_angle_diff_deg,
                max_segment_roughness_diff=args.judge_max_segment_roughness_diff,
                min_segment_shape_score=args.min_segment_shape_score,
                min_segment_color_score=args.min_segment_color_score,
                min_segment_texture_score=args.min_segment_texture_score,
                min_segment_tangent_score=args.min_segment_tangent_score,
                accept_known_pairs=not args.no_accept_known_pairs,
            ),
        )
        print(f"feedback round {round_index} accepted: {paths['accepted']}")
        current_feedback = paths["feedback"]


def count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sherd candidate, pairing, and judge agents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates = subparsers.add_parser("candidates", help="Build filtered and expanded candidate pairs.")
    add_common_inputs(candidates)
    add_candidate_args(candidates)
    candidates.set_defaults(func=cmd_candidates)

    pair = subparsers.add_parser("pair", help="Search rigid joins for candidate pairs.")
    add_common_inputs(pair)
    pair.add_argument("--candidates", type=Path, required=True)
    add_pair_args(pair)
    pair.set_defaults(func=cmd_pair)

    judge = subparsers.add_parser("judge", help="Judge pairing results and produce feedback.")
    judge.add_argument("--pairing-results", type=Path, required=True)
    judge.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_judge_args(judge)
    judge.set_defaults(func=cmd_judge)

    legacy_refine = subparsers.add_parser(
        "legacy-refine",
        help="Refine accepted pairs with /home/wenkang/SherdPair DINO local features and diffusion model.",
    )
    legacy_refine.add_argument("--pairs", type=Path, required=True)
    legacy_refine.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS_CSV)
    legacy_refine.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    legacy_refine.add_argument(
        "--cluster-limit",
        type=int,
        default=0,
        help="Only use the first N cluster IDs, sorted ascending. 0 means all clusters.",
    )
    add_legacy_refine_args(legacy_refine)
    legacy_refine.set_defaults(func=cmd_legacy_refine)

    visualize = subparsers.add_parser("visualize", help="Build an exterior rank-1 pairing contact sheet.")
    visualize.add_argument("--pairing-results", type=Path, required=True)
    visualize.add_argument("--output", type=Path, required=True)
    visualize.add_argument("--max-items", type=int, default=80)
    visualize.add_argument("--thumb-width", type=int, default=360)
    visualize.set_defaults(func=cmd_visualize)

    run = subparsers.add_parser("run", help="Run candidates -> pairing -> judge.")
    add_common_inputs(run)
    add_candidate_args(run)
    add_pair_args(run)
    add_judge_args(run)
    add_legacy_refine_args(run)
    run.add_argument("--legacy-refine", action="store_true")
    run.add_argument("--feedback-rounds", type=int, default=1)
    run.add_argument("--feedback-max-pairs", type=int, default=100)
    run.add_argument("--feedback-contour-points", type=int, default=32)
    run.add_argument("--feedback-angle-step-deg", type=int, default=15)
    run.add_argument("--feedback-refine-top-n", type=int, default=48)
    run.set_defaults(func=cmd_run)
    return parser


def add_candidate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k-external", type=int, default=10)
    parser.add_argument("--max-intra-pairs-per-cluster", type=int, default=30)
    parser.add_argument("--min-cluster-size-for-filter", type=int, default=3)
    parser.add_argument("--outlier-mad-z", type=float, default=3.5)
    parser.add_argument("--outlier-quantile", type=float, default=0.95)
    parser.add_argument("--external-search-pool", type=int, default=80)
    parser.add_argument("--no-known-pair-seed", action="store_true")


def add_pair_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--contour-points", type=int, default=20)
    parser.add_argument("--angle-step-deg", type=int, default=30)
    parser.add_argument("--top-transforms-per-pair", type=int, default=1)
    parser.add_argument("--refine-top-n", type=int, default=24)
    parser.add_argument("--normal-push-px", type=float, default=2.0)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--min-contact-ratio", type=float, default=0.002)
    parser.add_argument(
        "--allow-cross-side",
        action="store_true",
        help="Also try exterior-interior and interior-exterior. Default only tries same-side pairs.",
    )
    parser.add_argument(
        "--disable-segment-matching",
        action="store_true",
        help="Use the old single-point edge matching instead of long boundary segment matching.",
    )
    parser.add_argument("--segment-sample-points", type=int, default=128)
    parser.add_argument("--segment-points", type=int, default=24)
    parser.add_argument("--segment-length-px", type=float, default=180.0)
    parser.add_argument("--min-segment-length-px", type=float, default=80.0)
    parser.add_argument("--max-segment-length-ratio", type=float, default=1.35)
    parser.add_argument("--max-segment-angle-diff-deg", type=float, default=35.0)
    parser.add_argument("--max-segment-roughness-diff", type=float, default=0.45)
    parser.add_argument("--curvature-weight", type=float, default=0.85)
    parser.add_argument("--segment-stride", type=int, default=4)
    parser.add_argument("--max-segment-pairs", type=int, default=600)
    parser.add_argument("--max-segment-rms-px", type=float, default=18.0)
    parser.add_argument("--max-segment-max-px", type=float, default=45.0)
    parser.add_argument(
        "--enable-point-fallback",
        action="store_true",
        help="After long-segment matching, also run the old single-point hypotheses.",
    )
    parser.add_argument("--edge-patch-radius", type=int, default=10)
    parser.add_argument("--edge-color-weight", type=float, default=0.65)
    parser.add_argument("--texture-weight", type=float, default=0.75)
    parser.add_argument("--tangent-weight", type=float, default=0.45)
    parser.add_argument("--segment-shape-weight", type=float, default=1.20)
    parser.add_argument("--segment-param-weight", type=float, default=1.10)
    parser.add_argument("--segment-color-weight", type=float, default=0.85)
    parser.add_argument("--segment-texture-weight", type=float, default=1.45)
    parser.add_argument("--segment-tangent-weight", type=float, default=0.65)
    parser.add_argument("--texture-patch-radius", type=int, default=14)
    parser.add_argument("--texture-samples-per-segment", type=int, default=32)
    parser.add_argument("--min-texture-strength", type=float, default=3.0)


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--accept-score", type=float, default=2.25)
    parser.add_argument("--retry-score", type=float, default=1.25)
    parser.add_argument("--judge-max-overlap-ratio", type=float, default=0.08)
    parser.add_argument("--retry-overlap-ratio", type=float, default=0.18)
    parser.add_argument("--judge-min-contact-ratio", type=float, default=0.006)
    parser.add_argument("--max-gap-px", type=float, default=6.0)
    parser.add_argument("--min-edge-color-score", type=float, default=0.35)
    parser.add_argument("--min-texture-orientation-score", type=float, default=0.25)
    parser.add_argument("--min-tangent-opposition-score", type=float, default=0.35)
    parser.add_argument("--no-require-segment-match", action="store_true")
    parser.add_argument("--judge-max-segment-rms-px", type=float, default=18.0)
    parser.add_argument("--judge-max-segment-length-ratio", type=float, default=1.35)
    parser.add_argument("--judge-max-segment-angle-diff-deg", type=float, default=35.0)
    parser.add_argument("--judge-max-segment-roughness-diff", type=float, default=0.45)
    parser.add_argument("--min-segment-shape-score", type=float, default=0.35)
    parser.add_argument("--min-segment-color-score", type=float, default=0.35)
    parser.add_argument("--min-segment-texture-score", type=float, default=0.25)
    parser.add_argument("--min-segment-tangent-score", type=float, default=0.35)
    parser.add_argument("--no-accept-known-pairs", action="store_true")


def add_legacy_refine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--legacy-max-pairs", type=int, default=10)
    parser.add_argument("--legacy-num-repeat", type=int, default=3)
    parser.add_argument("--legacy-contour-spacing", type=int, default=35)
    parser.add_argument("--legacy-patch-size", type=int, default=48)
    parser.add_argument("--legacy-model-path", type=Path, default=DEFAULT_LEGACY_MODEL)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
