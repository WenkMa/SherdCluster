"""Pairing agent that searches rigid 2D sherd joins for candidate pairs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import SherdRecord, records_by_id
from .image_geometry import (
    ShapeSide,
    TransformResult,
    compose_transform,
    color_distance_score,
    contour_tangent,
    load_shape,
    local_edge_descriptor,
    orientation_compatibility,
    render_pair,
    resample_closed_contour_by_arclength,
    sample_contour_points,
    quick_score_masks,
    score_masks,
    tangent_opposition,
    warp_pair_masks,
)

_TEXTURE_AXIS_CACHE: dict[tuple[str, int, int, int], tuple[float, float]] = {}


@dataclass(frozen=True)
class PairingConfig:
    max_pairs: int = 2000
    contour_points: int = 24
    angle_step_deg: int = 30
    top_transforms_per_pair: int = 1
    refine_top_n: int = 24
    normal_push_px: float = 2.0
    max_overlap_ratio: float = 0.20
    min_contact_ratio: float = 0.002
    same_side_only: bool = True
    use_segment_matching: bool = True
    segment_sample_points: int = 128
    segment_points: int = 24
    segment_length_px: float = 180.0
    min_segment_length_px: float = 80.0
    max_segment_length_ratio: float = 1.35
    max_segment_angle_diff_deg: float = 35.0
    max_segment_roughness_diff: float = 0.45
    curvature_weight: float = 0.85
    segment_stride: int = 4
    max_segment_pairs: int = 600
    max_segment_rms_px: float = 18.0
    max_segment_max_px: float = 45.0
    enable_point_fallback: bool = False
    edge_patch_radius: int = 10
    edge_color_weight: float = 0.65
    texture_weight: float = 0.75
    tangent_weight: float = 0.45
    segment_shape_weight: float = 1.20
    segment_param_weight: float = 1.10
    segment_color_weight: float = 0.85
    segment_texture_weight: float = 1.45
    segment_tangent_weight: float = 0.65
    texture_patch_radius: int = 14
    texture_samples_per_segment: int = 32
    min_texture_strength: float = 3.0
    preview_dir_name: str = "pair_previews"


@dataclass(frozen=True)
class BreakSegment:
    start: int
    points: np.ndarray
    arc_length: float
    chord_length: float
    angle_deg: float
    roughness: float
    curvature: np.ndarray


def _side_pair_options(
    record_a: SherdRecord,
    record_b: SherdRecord,
    same_side_only: bool,
) -> list[tuple[str, str, str, str]]:
    # Current production mode evaluates the true visible/front surface only.
    # Interior/cross-side hypotheses created many visually plausible but physically
    # invalid joins for this dataset.
    return [("exterior", record_a.exterior_path, "exterior", record_b.exterior_path)]

    same_side_pairs = [
        ("exterior", record_a.exterior_path, "exterior", record_b.exterior_path),
        ("interior", record_a.interior_path, "interior", record_b.interior_path),
    ]
    if same_side_only:
        return same_side_pairs
    return same_side_pairs + [
        ("exterior", record_a.exterior_path, "interior", record_b.interior_path),
        ("interior", record_a.interior_path, "exterior", record_b.exterior_path),
    ]


def _area_balance(a: ShapeSide, b: ShapeSide) -> float:
    return min(a.area, b.area) / max(1, max(a.area, b.area))


def transform_score(
    overlap_ratio: float,
    contact_ratio: float,
    gap_px: float,
    area_balance: float,
    embedding_distance: float,
    edge_color_score: float,
    texture_orientation_score: float,
    tangent_opposition_score: float,
    segment_shape_score: float,
    segment_param_score: float,
    segment_color_score: float,
    segment_texture_score: float,
    segment_tangent_score: float,
    config: PairingConfig,
) -> float:
    score = 0.0
    score += 5.0 * contact_ratio
    score += 1.5 * area_balance
    score += 1.0 / (1.0 + max(0.0, gap_px))
    score += 1.0 / (1.0 + embedding_distance)
    score += config.edge_color_weight * edge_color_score
    score += config.texture_weight * texture_orientation_score
    score += config.tangent_weight * tangent_opposition_score
    score += config.segment_shape_weight * segment_shape_score
    score += config.segment_param_weight * segment_param_score
    score += config.segment_color_weight * segment_color_score
    score += config.segment_texture_weight * segment_texture_score
    score += config.segment_tangent_weight * segment_tangent_score
    score -= 8.0 * overlap_ratio
    return float(score)


def local_match_evidence(
    shape_a: ShapeSide,
    shape_b: ShapeSide,
    a_point: np.ndarray,
    b_point: np.ndarray,
    angle_deg: float,
    config: PairingConfig,
) -> tuple[float, float, float, float]:
    color_a, orientation_a, strength_a = local_edge_descriptor(
        shape_a, a_point, patch_radius=config.edge_patch_radius
    )
    color_b, orientation_b, strength_b = local_edge_descriptor(
        shape_b, b_point, patch_radius=config.edge_patch_radius
    )
    color_distance, edge_color_score = color_distance_score(color_a, color_b)
    texture_score = orientation_compatibility(orientation_a, orientation_b, angle_deg)
    if strength_a < 2.0 or strength_b < 2.0:
        texture_score *= 0.5
    tangent_a = contour_tangent(shape_a.contour, a_point)
    tangent_b = contour_tangent(shape_b.contour, b_point)
    tangent_score = tangent_opposition(tangent_a, tangent_b, angle_deg)
    return color_distance, edge_color_score, texture_score, tangent_score


def _cyclic_segments(points: np.ndarray, segment_points: int, stride: int) -> list[tuple[int, np.ndarray]]:
    if len(points) < segment_points:
        return []
    segments = []
    indices = np.arange(segment_points)
    for start in range(0, len(points), max(1, stride)):
        segment = points[(start + indices) % len(points)]
        segments.append((start, segment.astype(np.float32)))
    return segments


def _angle_difference_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(min(diff, abs(180.0 - diff)))


def _segment_curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float32)
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    v1 = points - prev_points
    v2 = next_points - points
    a1 = np.arctan2(v1[:, 1], v1[:, 0])
    a2 = np.arctan2(v2[:, 1], v2[:, 0])
    curvature = np.angle(np.exp(1j * (a2 - a1)))
    return curvature.astype(np.float32)


def _break_segment(start: int, points: np.ndarray) -> BreakSegment:
    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    arc_length = float(seg_lengths.sum())
    chord_vec = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord_vec))
    angle_deg = float(np.rad2deg(np.arctan2(chord_vec[1], chord_vec[0])) % 360.0)
    roughness = float(arc_length / max(chord_length, 1e-6))
    curvature = _segment_curvature(points)
    return BreakSegment(
        start=start,
        points=points.astype(np.float32),
        arc_length=arc_length,
        chord_length=chord_length,
        angle_deg=angle_deg,
        roughness=roughness,
        curvature=curvature,
    )


def _resample_open_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == num_points:
        return points.copy()
    if len(points) < 2:
        return np.repeat(points, num_points, axis=0) if len(points) else np.zeros((num_points, 2), dtype=np.float32)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(points[:1], num_points, axis=0)
    targets = np.linspace(0.0, total, num_points)
    sampled = []
    seg_idx = 0
    for target in targets:
        while seg_idx < len(lengths) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        start_len = cumulative[seg_idx]
        end_len = cumulative[seg_idx + 1]
        alpha = 0.0 if end_len == start_len else (target - start_len) / (end_len - start_len)
        sampled.append((1.0 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1])
    return np.asarray(sampled, dtype=np.float32)


def _cyclic_break_segments_by_length(
    points: np.ndarray,
    segment_length_px: float,
    min_segment_length_px: float,
    stride: int,
) -> list[BreakSegment]:
    if len(points) < 8:
        return []
    closed = np.vstack([points, points[0]])
    edge_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    mean_step = float(np.mean(edge_lengths)) if len(edge_lengths) else 1.0
    count = max(8, int(round(segment_length_px / max(mean_step, 1e-6))))
    count = min(count, len(points) - 1)
    segments = []
    offsets = np.arange(count)
    for start in range(0, len(points), max(1, stride)):
        seg_points = points[(start + offsets) % len(points)]
        segment = _break_segment(start, seg_points)
        if segment.arc_length >= min_segment_length_px:
            segments.append(segment)
    return segments


def _segment_parameter_compatibility(
    segment_a: BreakSegment,
    segment_b: BreakSegment,
    config: PairingConfig,
) -> tuple[bool, float, float, float, float]:
    length_ratio = max(segment_a.arc_length, segment_b.arc_length) / max(
        1e-6, min(segment_a.arc_length, segment_b.arc_length)
    )
    # Reverse B before fitting, so opposite chord directions should differ by roughly 180;
    # using axial difference makes 0/180 both acceptable after reversal.
    angle_diff = _angle_difference_deg(segment_a.angle_deg, (segment_b.angle_deg + 180.0) % 360.0)
    roughness_diff = abs(segment_a.roughness - segment_b.roughness)
    n = min(len(segment_a.curvature), len(segment_b.curvature))
    if n:
        curv_a = segment_a.curvature[:n]
        curv_b = -segment_b.curvature[::-1][:n]
        curvature_distance = float(np.mean(np.abs(curv_a - curv_b)))
    else:
        curvature_distance = 999.0
    ok = (
        length_ratio <= config.max_segment_length_ratio
        and angle_diff <= config.max_segment_angle_diff_deg
        and roughness_diff <= config.max_segment_roughness_diff
    )
    return ok, length_ratio, angle_diff, roughness_diff, curvature_distance


def _rigid_transform_from_segments(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)
    src_centered = source - src_centroid[None, :]
    dst_centered = target - dst_centroid[None, :]
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    transformed = src_centered @ rot.T + dst_centroid[None, :]
    residuals = np.linalg.norm(transformed - target, axis=1)
    mat = np.eye(3, dtype=np.float32)
    mat[:2, :2] = rot.astype(np.float32)
    mat[:2, 2] = (dst_centroid - rot @ src_centroid).astype(np.float32)
    return mat, float(np.sqrt(np.mean(residuals**2))), float(residuals.max())


def _patch_texture_axis(
    shape: ShapeSide,
    point: np.ndarray,
    patch_radius: int,
) -> tuple[float, float]:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    cache_key = (shape.path, x, y, patch_radius)
    cached = _TEXTURE_AXIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    h, w = shape.mask.shape
    x0 = max(0, x - patch_radius)
    x1 = min(w, x + patch_radius + 1)
    y0 = max(0, y - patch_radius)
    y1 = min(h, y + patch_radius + 1)
    if x0 >= x1 or y0 >= y1:
        _TEXTURE_AXIS_CACHE[cache_key] = (0.0, 0.0)
        return _TEXTURE_AXIS_CACHE[cache_key]

    patch_mask = shape.mask[y0:y1, x0:x1] > 0
    if not np.any(patch_mask):
        _TEXTURE_AXIS_CACHE[cache_key] = (0.0, 0.0)
        return _TEXTURE_AXIS_CACHE[cache_key]
    if patch_mask.shape[0] >= 5 and patch_mask.shape[1] >= 5:
        patch_mask = cv2.erode(patch_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if not np.any(patch_mask):
            patch_mask = shape.mask[y0:y1, x0:x1] > 0

    strengths = shape.gradient_strength[y0:y1, x0:x1][patch_mask].astype(np.float32)
    angles = shape.gradient_angle[y0:y1, x0:x1][patch_mask].astype(np.float32)
    valid = strengths > 1.0
    if not np.any(valid):
        _TEXTURE_AXIS_CACHE[cache_key] = (0.0, 0.0)
        return _TEXTURE_AXIS_CACHE[cache_key]

    strengths = strengths[valid]
    angles = angles[valid]
    # Use doubled angles for axial texture direction: theta and theta+pi are identical.
    sin2 = float(np.average(np.sin(2.0 * angles), weights=strengths))
    cos2 = float(np.average(np.cos(2.0 * angles), weights=strengths))
    coherence = float(np.hypot(sin2, cos2))
    axis = 0.5 * float(np.arctan2(sin2, cos2))
    strength = float(strengths.mean() * coherence)
    _TEXTURE_AXIS_CACHE[cache_key] = (axis, strength)
    return _TEXTURE_AXIS_CACHE[cache_key]


def _segment_texture_direction_score(
    shape_a: ShapeSide,
    shape_b: ShapeSide,
    segment_a: np.ndarray,
    segment_b_reversed: np.ndarray,
    angle_deg: float,
    config: PairingConfig,
) -> float:
    sample_count = min(len(segment_a), len(segment_b_reversed))
    if sample_count <= 0:
        return 0.0
    take = min(config.texture_samples_per_segment, sample_count)
    sample_indices = np.linspace(0, sample_count - 1, take).astype(np.int32)

    scores = []
    weights = []
    rot_rad = np.deg2rad(angle_deg)
    for idx in sample_indices:
        axis_a, strength_a = _patch_texture_axis(shape_a, segment_a[idx], config.texture_patch_radius)
        axis_b, strength_b = _patch_texture_axis(shape_b, segment_b_reversed[idx], config.texture_patch_radius)
        strength = min(strength_a, strength_b)
        if strength < config.min_texture_strength:
            continue
        diff = abs(float(np.angle(np.exp(1j * (axis_a - (axis_b + rot_rad))))))
        diff = min(diff, abs(np.pi - diff))
        scores.append(max(0.0, np.cos(diff)))
        weights.append(strength)

    if not scores:
        return 0.0
    return float(np.average(np.asarray(scores, dtype=np.float32), weights=np.asarray(weights, dtype=np.float32)))


def _segment_local_evidence(
    shape_a: ShapeSide,
    shape_b: ShapeSide,
    segment_a: np.ndarray,
    segment_b_reversed: np.ndarray,
    angle_deg: float,
    config: PairingConfig,
) -> tuple[float, float, float, float, float, float, float]:
    color_distances = []
    color_scores = []
    texture_scores = []
    tangent_scores = []
    sample_count = min(len(segment_a), len(segment_b_reversed))
    sample_indices = np.linspace(0, sample_count - 1, min(12, sample_count)).astype(np.int32)
    for idx in sample_indices:
        (
            color_distance,
            color_score,
            texture_score,
            tangent_score,
        ) = local_match_evidence(
            shape_a=shape_a,
            shape_b=shape_b,
            a_point=segment_a[idx],
            b_point=segment_b_reversed[idx],
            angle_deg=angle_deg,
            config=config,
        )
        color_distances.append(color_distance)
        color_scores.append(color_score)
        texture_scores.append(texture_score)
        tangent_scores.append(tangent_score)
    edge_color_distance = float(np.mean(color_distances)) if color_distances else 999.0
    edge_color_score = float(np.mean(color_scores)) if color_scores else 0.0
    texture_score = float(np.mean(texture_scores)) if texture_scores else 0.0
    segment_texture_direction = _segment_texture_direction_score(
        shape_a=shape_a,
        shape_b=shape_b,
        segment_a=segment_a,
        segment_b_reversed=segment_b_reversed,
        angle_deg=angle_deg,
        config=config,
    )
    texture_score = 0.35 * texture_score + 0.65 * segment_texture_direction
    tangent_score = float(np.mean(tangent_scores)) if tangent_scores else 0.0
    return (
        edge_color_distance,
        edge_color_score,
        texture_score,
        tangent_score,
        edge_color_score,
        texture_score,
        tangent_score,
    )


def _lab_strip_score(
    shape_a: ShapeSide,
    shape_b: ShapeSide,
    segment_a: np.ndarray,
    segment_b_reversed: np.ndarray,
) -> tuple[float, float]:
    rgb_a = []
    rgb_b = []
    for point_a, point_b in zip(segment_a, segment_b_reversed):
        xa, ya = int(round(point_a[0])), int(round(point_a[1]))
        xb, yb = int(round(point_b[0])), int(round(point_b[1]))
        if 0 <= ya < shape_a.color.shape[0] and 0 <= xa < shape_a.color.shape[1]:
            rgb_a.append(shape_a.color[ya, xa])
        if 0 <= yb < shape_b.color.shape[0] and 0 <= xb < shape_b.color.shape[1]:
            rgb_b.append(shape_b.color[yb, xb])
    if not rgb_a or not rgb_b:
        return 999.0, 0.0
    n = min(len(rgb_a), len(rgb_b))
    arr_a = np.asarray(rgb_a[:n], dtype=np.uint8)[None, :, :]
    arr_b = np.asarray(rgb_b[:n], dtype=np.uint8)[None, :, :]
    lab_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2LAB).astype(np.float32)[0]
    lab_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2LAB).astype(np.float32)[0]
    diff = np.linalg.norm(lab_a - lab_b, axis=1)
    distance = float(np.mean(diff))
    return distance, float(max(0.0, 1.0 - distance / 80.0))


def _segment_match_search(
    record_a: SherdRecord,
    record_b: SherdRecord,
    side_a: str,
    side_b: str,
    shape_a: ShapeSide,
    shape_b: ShapeSide,
    embedding_distance: float,
    config: PairingConfig,
) -> list[tuple[float, dict[str, object], np.ndarray, ShapeSide, ShapeSide]]:
    points_a = resample_closed_contour_by_arclength(shape_a.contour, config.segment_sample_points)
    points_b = resample_closed_contour_by_arclength(shape_b.contour, config.segment_sample_points)
    segments_a = _cyclic_break_segments_by_length(
        points_a,
        segment_length_px=config.segment_length_px,
        min_segment_length_px=config.min_segment_length_px,
        stride=config.segment_stride,
    )
    segments_b = _cyclic_break_segments_by_length(
        points_b,
        segment_length_px=config.segment_length_px,
        min_segment_length_px=config.min_segment_length_px,
        stride=config.segment_stride,
    )
    coarse: list[tuple[float, dict[str, object], np.ndarray, ShapeSide, ShapeSide]] = []
    candidates_checked = 0

    for segment_a in segments_a:
        for segment_b in segments_b:
            (
                param_ok,
                segment_length_ratio,
                segment_angle_diff_deg,
                segment_roughness_diff,
                segment_curvature_distance,
            ) = _segment_parameter_compatibility(segment_a, segment_b, config)
            if not param_ok:
                continue
            fit_points = max(12, min(len(segment_a.points), len(segment_b.points), config.segment_points))
            segment_a_fit = _resample_open_polyline(segment_a.points, fit_points)
            segment_b_fit = _resample_open_polyline(segment_b.points, fit_points)
            segment_b_reversed = segment_b_fit[::-1].copy()
            transform, rms_px, max_px = _rigid_transform_from_segments(segment_b_reversed, segment_a_fit)
            if rms_px > config.max_segment_rms_px or max_px > config.max_segment_max_px:
                continue
            a_mask, b_mask = warp_pair_masks(shape_a, shape_b, transform)
            overlap_ratio, contact_ratio = quick_score_masks(a_mask, b_mask)
            if overlap_ratio > config.max_overlap_ratio:
                continue
            if contact_ratio < config.min_contact_ratio:
                continue

            angle_deg = float(np.rad2deg(np.arctan2(transform[1, 0], transform[0, 0])) % 360.0)
            (
                edge_color_distance,
                edge_color_score,
                texture_orientation_score,
                tangent_opposition_score,
                segment_color_score,
                segment_texture_score,
                segment_tangent_score,
            ) = _segment_local_evidence(
                shape_a=shape_a,
                shape_b=shape_b,
                segment_a=segment_a_fit,
                segment_b_reversed=segment_b_reversed,
                angle_deg=angle_deg,
                config=config,
            )
            lab_distance, lab_score = _lab_strip_score(shape_a, shape_b, segment_a_fit, segment_b_reversed)
            segment_color_score = 0.5 * segment_color_score + 0.5 * lab_score
            segment_shape_score = float(np.exp(-rms_px / max(1e-6, config.max_segment_rms_px)))
            segment_param_score = float(
                np.exp(-(segment_length_ratio - 1.0))
                * np.exp(-segment_angle_diff_deg / max(1e-6, config.max_segment_angle_diff_deg))
                * np.exp(-segment_roughness_diff)
                * np.exp(-config.curvature_weight * segment_curvature_distance)
            )
            segment_shape_score = 0.65 * segment_shape_score + 0.35 * segment_param_score
            area_balance = _area_balance(shape_a, shape_b)
            coarse_score = transform_score(
                overlap_ratio=overlap_ratio,
                contact_ratio=contact_ratio,
                gap_px=1.0 / max(contact_ratio, 1e-6),
                area_balance=area_balance,
                embedding_distance=embedding_distance,
                edge_color_score=edge_color_score,
                texture_orientation_score=texture_orientation_score,
                tangent_opposition_score=tangent_opposition_score,
                segment_shape_score=segment_shape_score,
                segment_param_score=segment_param_score,
                segment_color_score=segment_color_score,
                segment_texture_score=segment_texture_score,
                segment_tangent_score=segment_tangent_score,
                config=config,
            )
            row = {
                "sherd_id_a": record_a.sherd_id,
                "sherd_id_b": record_b.sherd_id,
                "side_a": side_a,
                "side_b": side_b,
                "operation": (
                    f"用连续断边片段匹配: {record_b.sherd_id} 的 {side_b} 面片段 "
                    f"start={segment_b.start} 反向对齐到 {record_a.sherd_id} 的 {side_a} 面片段 start={segment_a.start}; "
                    f"长度 {segment_b.arc_length:.1f}px->{segment_a.arc_length:.1f}px, "
                    f"旋转 {angle_deg:.1f} 度, 平移 tx={float(transform[0, 2]):.1f}, ty={float(transform[1, 2]):.1f}"
                ),
                "angle_deg": f"{angle_deg:.1f}",
                "tx": f"{float(transform[0, 2]):.3f}",
                "ty": f"{float(transform[1, 2]):.3f}",
                "overlap_ratio": f"{overlap_ratio:.6f}",
                "contact_ratio": f"{contact_ratio:.6f}",
                "gap_px": "",
                "area_balance": f"{area_balance:.6f}",
                "edge_color_distance": f"{edge_color_distance:.6f}",
                "edge_color_score": f"{edge_color_score:.6f}",
                "texture_orientation_score": f"{texture_orientation_score:.6f}",
                "tangent_opposition_score": f"{tangent_opposition_score:.6f}",
                "segment_rms_px": f"{rms_px:.6f}",
                "segment_max_px": f"{max_px:.6f}",
                "segment_shape_score": f"{segment_shape_score:.6f}",
                "segment_param_score": f"{segment_param_score:.6f}",
                "segment_length_a": f"{segment_a.arc_length:.6f}",
                "segment_length_b": f"{segment_b.arc_length:.6f}",
                "segment_length_ratio": f"{segment_length_ratio:.6f}",
                "segment_chord_a": f"{segment_a.chord_length:.6f}",
                "segment_chord_b": f"{segment_b.chord_length:.6f}",
                "segment_angle_a": f"{segment_a.angle_deg:.6f}",
                "segment_angle_b": f"{segment_b.angle_deg:.6f}",
                "segment_angle_diff_deg": f"{segment_angle_diff_deg:.6f}",
                "segment_roughness_a": f"{segment_a.roughness:.6f}",
                "segment_roughness_b": f"{segment_b.roughness:.6f}",
                "segment_roughness_diff": f"{segment_roughness_diff:.6f}",
                "segment_curvature_distance": f"{segment_curvature_distance:.6f}",
                "segment_lab_distance": f"{lab_distance:.6f}",
                "segment_color_score": f"{segment_color_score:.6f}",
                "segment_texture_score": f"{segment_texture_score:.6f}",
                "segment_tangent_score": f"{segment_tangent_score:.6f}",
                "segment_start_a": segment_a.start,
                "segment_start_b": segment_b.start,
                "segment_points": fit_points,
                "embedding_distance": f"{embedding_distance:.6f}",
                "pair_score": f"{coarse_score:.6f}",
                "preview_path": "",
            }
            coarse.append((coarse_score, row, transform, shape_a, shape_b))
            candidates_checked += 1
            if 0 < config.max_segment_pairs <= candidates_checked:
                # Keep deterministic coverage by stopping after enough feasible long-segment hypotheses.
                return coarse
    return coarse


def search_pair_transform(
    record_a: SherdRecord,
    record_b: SherdRecord,
    embedding_distance: float,
    output_dir: Path,
    config: PairingConfig,
) -> list[dict[str, object]]:
    coarse: list[tuple[float, dict[str, object], np.ndarray, ShapeSide, ShapeSide]] = []
    results: list[tuple[float, dict[str, object], TransformResult]] = []
    angles = list(range(0, 360, config.angle_step_deg))

    for side_a, path_a, side_b, path_b in _side_pair_options(
        record_a, record_b, same_side_only=config.same_side_only
    ):
            try:
                shape_a = load_shape(path_a)
                shape_b = load_shape(path_b)
            except Exception as exc:
                results.append(
                    (
                        -999.0,
                        {
                            "sherd_id_a": record_a.sherd_id,
                            "sherd_id_b": record_b.sherd_id,
                            "side_a": side_a,
                            "side_b": side_b,
                            "operation": "load_failed",
                            "error": str(exc),
                        },
                        TransformResult(
                            angle_deg=0.0,
                            tx=0.0,
                            ty=0.0,
                            overlap_ratio=1.0,
                            contact_ratio=0.0,
                            gap_px=9999.0,
                            area_balance=0.0,
                            edge_color_distance=999.0,
                            texture_orientation_score=0.0,
                            tangent_opposition_score=0.0,
                            segment_shape_score=0.0,
                            segment_param_score=0.0,
                            segment_color_score=0.0,
                            segment_texture_score=0.0,
                            segment_tangent_score=0.0,
                            score=-999.0,
                            transform=np.eye(3),
                            canvas_path="",
                        ),
                    )
                )
                continue

            points_a = sample_contour_points(shape_a.contour, config.contour_points)
            points_b = sample_contour_points(shape_b.contour, config.contour_points)
            area_balance = _area_balance(shape_a, shape_b)

            if config.use_segment_matching:
                coarse.extend(
                    _segment_match_search(
                        record_a=record_a,
                        record_b=record_b,
                        side_a=side_a,
                        side_b=side_b,
                        shape_a=shape_a,
                        shape_b=shape_b,
                        embedding_distance=embedding_distance,
                        config=config,
                    )
                )
                if not config.enable_point_fallback:
                    continue

            for a_point in points_a:
                for b_point in points_b:
                    for angle in angles:
                        (
                            edge_color_distance,
                            edge_color_score,
                            texture_orientation_score,
                            tangent_opposition_score,
                        ) = local_match_evidence(
                            shape_a=shape_a,
                            shape_b=shape_b,
                            a_point=a_point,
                            b_point=b_point,
                            angle_deg=float(angle),
                            config=config,
                        )
                        transform = compose_transform(
                            b_centroid=shape_b.centroid,
                            a_point=a_point,
                            b_point=b_point,
                            angle_deg=float(angle),
                            normal_push_px=config.normal_push_px,
                        )
                        a_mask, b_mask = warp_pair_masks(shape_a, shape_b, transform)
                        overlap_ratio, contact_ratio = quick_score_masks(a_mask, b_mask)
                        if overlap_ratio > config.max_overlap_ratio:
                            continue
                        if contact_ratio < config.min_contact_ratio:
                            continue
                        coarse_score = transform_score(
                            overlap_ratio=overlap_ratio,
                            contact_ratio=contact_ratio,
                            gap_px=1.0 / max(contact_ratio, 1e-6),
                            area_balance=area_balance,
                            embedding_distance=embedding_distance,
                            edge_color_score=edge_color_score,
                            texture_orientation_score=texture_orientation_score,
                            tangent_opposition_score=tangent_opposition_score,
                            segment_shape_score=0.0,
                            segment_param_score=0.0,
                            segment_color_score=0.0,
                            segment_texture_score=0.0,
                            segment_tangent_score=0.0,
                            config=config,
                        )
                        row = {
                            "sherd_id_a": record_a.sherd_id,
                            "sherd_id_b": record_b.sherd_id,
                            "side_a": side_a,
                            "side_b": side_b,
                            "operation": (
                                f"把 {record_b.sherd_id} 的 {side_b} 面旋转 {angle:.1f} 度, "
                                f"再平移 tx={float(transform[0, 2]):.1f}, ty={float(transform[1, 2]):.1f}, "
                                f"贴合到 {record_a.sherd_id} 的 {side_a} 面边缘"
                            ),
                            "angle_deg": f"{float(angle):.1f}",
                            "tx": f"{float(transform[0, 2]):.3f}",
                            "ty": f"{float(transform[1, 2]):.3f}",
                            "overlap_ratio": f"{overlap_ratio:.6f}",
                            "contact_ratio": f"{contact_ratio:.6f}",
                            "gap_px": "",
                            "area_balance": f"{area_balance:.6f}",
                            "edge_color_distance": f"{edge_color_distance:.6f}",
                            "edge_color_score": f"{edge_color_score:.6f}",
                            "texture_orientation_score": f"{texture_orientation_score:.6f}",
                            "tangent_opposition_score": f"{tangent_opposition_score:.6f}",
                            "segment_rms_px": "",
                            "segment_max_px": "",
                            "segment_shape_score": "",
                            "segment_param_score": "",
                            "segment_length_a": "",
                            "segment_length_b": "",
                            "segment_length_ratio": "",
                            "segment_chord_a": "",
                            "segment_chord_b": "",
                            "segment_angle_a": "",
                            "segment_angle_b": "",
                            "segment_angle_diff_deg": "",
                            "segment_roughness_a": "",
                            "segment_roughness_b": "",
                            "segment_roughness_diff": "",
                            "segment_curvature_distance": "",
                            "segment_lab_distance": "",
                            "segment_color_score": "",
                            "segment_texture_score": "",
                            "segment_tangent_score": "",
                            "segment_start_a": "",
                            "segment_start_b": "",
                            "segment_points": "",
                            "embedding_distance": f"{embedding_distance:.6f}",
                            "pair_score": f"{coarse_score:.6f}",
                            "preview_path": "",
                        }
                        coarse.append((coarse_score, row, transform, shape_a, shape_b))

    coarse.sort(key=lambda item: item[0], reverse=True)
    for _, row, transform, shape_a, shape_b in coarse[: config.refine_top_n]:
        a_mask, b_mask = warp_pair_masks(shape_a, shape_b, transform)
        overlap_ratio, contact_ratio, gap_px = score_masks(a_mask, b_mask)
        area_balance = float(row["area_balance"])
        edge_color_score = float(row["edge_color_score"])
        texture_orientation_score = float(row["texture_orientation_score"])
        tangent_opposition_score = float(row["tangent_opposition_score"])
        segment_shape_score = float(row.get("segment_shape_score") or 0.0)
        segment_param_score = float(row.get("segment_param_score") or 0.0)
        segment_color_score = float(row.get("segment_color_score") or 0.0)
        segment_texture_score = float(row.get("segment_texture_score") or 0.0)
        segment_tangent_score = float(row.get("segment_tangent_score") or 0.0)
        embedding_distance = float(row["embedding_distance"])
        score = transform_score(
            overlap_ratio=overlap_ratio,
            contact_ratio=contact_ratio,
            gap_px=gap_px,
            area_balance=area_balance,
            embedding_distance=embedding_distance,
            edge_color_score=edge_color_score,
            texture_orientation_score=texture_orientation_score,
            tangent_opposition_score=tangent_opposition_score,
            segment_shape_score=segment_shape_score,
            segment_param_score=segment_param_score,
            segment_color_score=segment_color_score,
            segment_texture_score=segment_texture_score,
            segment_tangent_score=segment_tangent_score,
            config=config,
        )
        row["overlap_ratio"] = f"{overlap_ratio:.6f}"
        row["contact_ratio"] = f"{contact_ratio:.6f}"
        row["gap_px"] = f"{gap_px:.3f}"
        row["pair_score"] = f"{score:.6f}"
        results.append(
            (
                score,
                row,
                TransformResult(
                    angle_deg=float(row["angle_deg"]),
                    tx=float(transform[0, 2]),
                    ty=float(transform[1, 2]),
                    overlap_ratio=overlap_ratio,
                    contact_ratio=contact_ratio,
                    gap_px=gap_px,
                    area_balance=area_balance,
                    edge_color_distance=float(row["edge_color_distance"]),
                    texture_orientation_score=texture_orientation_score,
                    tangent_opposition_score=tangent_opposition_score,
                    segment_shape_score=segment_shape_score,
                    segment_param_score=segment_param_score,
                    segment_color_score=segment_color_score,
                    segment_texture_score=segment_texture_score,
                    segment_tangent_score=segment_tangent_score,
                    score=score,
                    transform=transform,
                    canvas_path="",
                ),
            )
        )

    results.sort(key=lambda item: item[0], reverse=True)
    selected = results[: config.top_transforms_per_pair]
    rows: list[dict[str, object]] = []
    if not selected:
        return [
            {
                "sherd_id_a": record_a.sherd_id,
                "sherd_id_b": record_b.sherd_id,
                "side_a": "",
                "side_b": "",
                "rank": 1,
                "operation": "no_feasible_transform",
                "angle_deg": "",
                "tx": "",
                "ty": "",
                "overlap_ratio": "",
                "contact_ratio": "",
                "gap_px": "",
                "area_balance": "",
                "edge_color_distance": "",
                "edge_color_score": "",
                "texture_orientation_score": "",
                "tangent_opposition_score": "",
                "segment_rms_px": "",
                "segment_max_px": "",
                "segment_shape_score": "",
                "segment_param_score": "",
                "segment_length_a": "",
                "segment_length_b": "",
                "segment_length_ratio": "",
                "segment_chord_a": "",
                "segment_chord_b": "",
                "segment_angle_a": "",
                "segment_angle_b": "",
                "segment_angle_diff_deg": "",
                "segment_roughness_a": "",
                "segment_roughness_b": "",
                "segment_roughness_diff": "",
                "segment_curvature_distance": "",
                "segment_lab_distance": "",
                "segment_color_score": "",
                "segment_texture_score": "",
                "segment_tangent_score": "",
                "segment_start_a": "",
                "segment_start_b": "",
                "segment_points": "",
                "embedding_distance": f"{embedding_distance:.6f}",
                "pair_score": "-999.000000",
                "preview_path": "",
                "error": "No transform passed the overlap/contact filters.",
            }
        ]
    for rank, (_, row, result) in enumerate(selected, start=1):
        if row.get("operation") == "load_failed":
            row["rank"] = rank
            rows.append(row)
            continue
        preview_path = (
            output_dir
            / config.preview_dir_name
            / f"{record_a.sherd_id}_{record_b.sherd_id}_rank{rank}.png"
        )
        side_a = str(row["side_a"])
        side_b = str(row["side_b"])
        shape_a = load_shape(record_a.exterior_path if side_a == "exterior" else record_a.interior_path)
        shape_b = load_shape(record_b.exterior_path if side_b == "exterior" else record_b.interior_path)
        render_pair(shape_a, shape_b, result.transform, preview_path)
        row["rank"] = rank
        row["preview_path"] = str(preview_path)
        rows.append(row)
    return rows


def run_pairing_agent(
    candidates_csv: Path,
    records: list[SherdRecord],
    output_dir: Path,
    config: PairingConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_map = records_by_id(records)
    output_path = output_dir / "pairing_results.csv"

    with candidates_csv.open("r", newline="", encoding="utf-8-sig") as f:
        candidates = list(csv.DictReader(f))
    candidates.sort(key=lambda r: (-float(r.get("priority", 0.0)), float(r.get("embedding_distance", 999.0))))
    if config.max_pairs > 0:
        candidates = candidates[: config.max_pairs]

    fieldnames = [
        "candidate_index",
        "sherd_id_a",
        "sherd_id_b",
        "side_a",
        "side_b",
        "rank",
        "operation",
        "angle_deg",
        "tx",
        "ty",
        "overlap_ratio",
        "contact_ratio",
        "gap_px",
        "area_balance",
        "edge_color_distance",
        "edge_color_score",
        "texture_orientation_score",
        "tangent_opposition_score",
        "segment_rms_px",
        "segment_max_px",
        "segment_shape_score",
        "segment_param_score",
        "segment_length_a",
        "segment_length_b",
        "segment_length_ratio",
        "segment_chord_a",
        "segment_chord_b",
        "segment_angle_a",
        "segment_angle_b",
        "segment_angle_diff_deg",
        "segment_roughness_a",
        "segment_roughness_b",
        "segment_roughness_diff",
        "segment_curvature_distance",
        "segment_lab_distance",
        "segment_color_score",
        "segment_texture_score",
        "segment_tangent_score",
        "segment_start_a",
        "segment_start_b",
        "segment_points",
        "embedding_distance",
        "pair_score",
        "preview_path",
        "source",
        "cluster_a",
        "cluster_b",
        "known_match",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, candidate in enumerate(candidates, start=1):
            sherd_a = candidate["sherd_id_a"]
            sherd_b = candidate["sherd_id_b"]
            if sherd_a not in record_map or sherd_b not in record_map:
                continue
            rows = search_pair_transform(
                record_map[sherd_a],
                record_map[sherd_b],
                embedding_distance=float(candidate.get("embedding_distance", 999.0)),
                output_dir=output_dir,
                config=config,
            )
            for row in rows:
                row.update(
                    {
                        "candidate_index": i,
                        "source": candidate.get("source", ""),
                        "cluster_a": candidate.get("cluster_a", ""),
                        "cluster_b": candidate.get("cluster_b", ""),
                        "known_match": candidate.get("known_match", 0),
                    }
                )
                writer.writerow(row)
            if i % 25 == 0:
                print(f"pairing progress: {i}/{len(candidates)} candidates")

    return output_path
