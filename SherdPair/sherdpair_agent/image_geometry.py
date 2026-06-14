"""Mask, contour, and transform scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ShapeSide:
    path: str
    mask: np.ndarray
    color: np.ndarray
    contour: np.ndarray
    centroid: np.ndarray
    radius: float
    area: int
    bbox: tuple[int, int, int, int]
    boundary: np.ndarray
    color_mean: np.ndarray
    gray: np.ndarray
    gradient_angle: np.ndarray
    gradient_strength: np.ndarray


@dataclass(frozen=True)
class TransformResult:
    angle_deg: float
    tx: float
    ty: float
    overlap_ratio: float
    contact_ratio: float
    gap_px: float
    area_balance: float
    edge_color_distance: float
    texture_orientation_score: float
    tangent_opposition_score: float
    segment_shape_score: float
    segment_param_score: float
    segment_color_score: float
    segment_texture_score: float
    segment_tangent_score: float
    score: float
    transform: np.ndarray
    canvas_path: str


def _mask_from_rgba(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    not_white = np.any(rgb < 245, axis=2)
    mask = ((alpha > 15) & not_white).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        mask = (alpha > 15).astype(np.uint8) * 255
    return mask


@lru_cache(maxsize=4096)
def load_shape(path: str) -> ShapeSide:
    image = Image.open(path).convert("RGBA")
    rgba = np.asarray(image)
    mask = _mask_from_rgba(image)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No foreground contour found: {path}")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        centroid = contour.mean(axis=0)
    else:
        centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    radius = float(np.linalg.norm(contour - centroid[None, :], axis=1).max())
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    area = int(np.count_nonzero(mask))
    eroded = cv2.erode(mask, kernel)
    boundary = cv2.subtract(mask, eroded)
    pixels = rgba[:, :, :3][mask > 0]
    color_mean = pixels.mean(axis=0).astype(np.float32) if len(pixels) else np.zeros(3, dtype=np.float32)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_angle = np.arctan2(grad_y, grad_x).astype(np.float32)
    gradient_strength = cv2.magnitude(grad_x, grad_y).astype(np.float32)
    return ShapeSide(
        path=path,
        mask=mask,
        color=rgba[:, :, :3],
        contour=contour,
        centroid=centroid,
        radius=radius,
        area=area,
        bbox=(x, y, w, h),
        boundary=boundary,
        color_mean=color_mean,
        gray=gray,
        gradient_angle=gradient_angle,
        gradient_strength=gradient_strength,
    )


def sample_contour_points(contour: np.ndarray, max_points: int) -> np.ndarray:
    if len(contour) <= max_points:
        return contour.copy()
    indices = np.linspace(0, len(contour) - 1, max_points).astype(np.int32)
    return contour[indices]


def resample_closed_contour_by_arclength(contour: np.ndarray, num_points: int) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float32)
    if len(contour) < 3:
        return contour.copy()
    closed = np.vstack([contour, contour[0]])
    deltas = np.diff(closed, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total = float(cumulative[-1])
    if total <= 1e-6:
        return sample_contour_points(contour, num_points)

    targets = np.linspace(0.0, total, num_points, endpoint=False)
    sampled = []
    seg_idx = 0
    for target in targets:
        while seg_idx < len(seg_lengths) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        start_len = cumulative[seg_idx]
        end_len = cumulative[seg_idx + 1]
        alpha = 0.0 if end_len == start_len else (target - start_len) / (end_len - start_len)
        point = (1.0 - alpha) * closed[seg_idx] + alpha * closed[seg_idx + 1]
        sampled.append(point)
    return np.asarray(sampled, dtype=np.float32)


def closed_contour_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0
    closed = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def contour_tangent(contour: np.ndarray, point: np.ndarray, window: int = 4) -> np.ndarray:
    if len(contour) < 3:
        return np.array([1.0, 0.0], dtype=np.float32)
    index = int(np.argmin(np.linalg.norm(contour - point[None, :], axis=1)))
    prev_point = contour[(index - window) % len(contour)]
    next_point = contour[(index + window) % len(contour)]
    tangent = next_point - prev_point
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (tangent / norm).astype(np.float32)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def pair_bbox_after_transform(
    a: ShapeSide,
    b: ShapeSide,
    transform_b: np.ndarray,
    margin: int = 24,
) -> tuple[int, int, int, int]:
    a_corners = np.array(
        [[0, 0], [a.mask.shape[1], 0], [a.mask.shape[1], a.mask.shape[0]], [0, a.mask.shape[0]]],
        dtype=np.float32,
    )
    b_corners = np.array(
        [[0, 0], [b.mask.shape[1], 0], [b.mask.shape[1], b.mask.shape[0]], [0, b.mask.shape[0]]],
        dtype=np.float32,
    )
    b_warped = cv2.transform(b_corners[None, :, :], transform_b[:2, :])[0]
    pts = np.vstack([a_corners, b_warped])
    min_xy = np.floor(pts.min(axis=0) - margin).astype(int)
    max_xy = np.ceil(pts.max(axis=0) + margin).astype(int)
    return int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])


def render_pair(
    a: ShapeSide,
    b: ShapeSide,
    transform_b: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = pair_bbox_after_transform(a, b, transform_b)
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    offset = np.array([[1, 0, -min_x], [0, 1, -min_y]], dtype=np.float32)
    b_mat = offset @ transform_b
    a_mat = offset

    a_mask = cv2.warpAffine(a.mask, a_mat, (width, height), flags=cv2.INTER_NEAREST)
    b_mask = cv2.warpAffine(b.mask, b_mat[:2, :], (width, height), flags=cv2.INTER_NEAREST)
    a_color = cv2.warpAffine(a.color, a_mat, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    b_color = cv2.warpAffine(b.color, b_mat[:2, :], (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[a_mask > 0] = a_color[a_mask > 0]
    b_only = b_mask > 0
    overlap = (a_mask > 0) & b_only
    canvas[b_only] = b_color[b_only]
    canvas[overlap] = np.array([235, 40, 40], dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    return a_mask, b_mask


def warp_pair_masks(
    a: ShapeSide,
    b: ShapeSide,
    transform_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = pair_bbox_after_transform(a, b, transform_b)
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    offset = np.array([[1, 0, -min_x], [0, 1, -min_y]], dtype=np.float32)
    b_mat = offset @ transform_b
    a_mask = cv2.warpAffine(a.mask, offset, (width, height), flags=cv2.INTER_NEAREST)
    b_mask = cv2.warpAffine(b.mask, b_mat[:2, :], (width, height), flags=cv2.INTER_NEAREST)
    return a_mask, b_mask


def score_masks(a_mask: np.ndarray, b_mask: np.ndarray) -> tuple[float, float, float]:
    overlap_ratio, contact_ratio = quick_score_masks(a_mask, b_mask)

    a_points = np.column_stack(np.where(a_mask > 0)).astype(np.float32)
    b_points = np.column_stack(np.where(b_mask > 0)).astype(np.float32)
    if len(a_points) == 0 or len(b_points) == 0:
        gap_px = 9999.0
    else:
        # Distance transform is cheaper than a full pairwise point distance.
        inv_b = (b_mask == 0).astype(np.uint8)
        dist_to_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
        boundary_a = cv2.subtract(a_mask, cv2.erode(a_mask, np.ones((3, 3), np.uint8)))
        vals = dist_to_b[boundary_a > 0]
        gap_px = float(vals.min()) if len(vals) else 9999.0
    return overlap_ratio, contact_ratio, gap_px


def quick_score_masks(a_mask: np.ndarray, b_mask: np.ndarray) -> tuple[float, float]:
    a_bool = a_mask > 0
    b_bool = b_mask > 0
    overlap = int(np.count_nonzero(a_bool & b_bool))
    min_area = max(1, min(int(np.count_nonzero(a_bool)), int(np.count_nonzero(b_bool))))
    overlap_ratio = overlap / min_area

    kernel = np.ones((5, 5), np.uint8)
    a_dil = cv2.dilate(a_mask, kernel)
    b_dil = cv2.dilate(b_mask, kernel)
    near = int(np.count_nonzero((a_dil > 0) & (b_dil > 0) & ~(a_bool & b_bool)))
    contact_ratio = near / max(1, min_area)
    return overlap_ratio, contact_ratio


def local_edge_descriptor(
    shape: ShapeSide,
    point: np.ndarray,
    patch_radius: int = 10,
) -> tuple[np.ndarray, float, float]:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    h, w = shape.mask.shape
    x0 = max(0, x - patch_radius)
    x1 = min(w, x + patch_radius + 1)
    y0 = max(0, y - patch_radius)
    y1 = min(h, y + patch_radius + 1)
    if x0 >= x1 or y0 >= y1:
        return shape.color_mean, 0.0, 0.0

    patch_mask = shape.mask[y0:y1, x0:x1] > 0
    patch_color = shape.color[y0:y1, x0:x1]
    patch_strength = shape.gradient_strength[y0:y1, x0:x1]
    patch_angle = shape.gradient_angle[y0:y1, x0:x1]
    if not np.any(patch_mask):
        return shape.color_mean, 0.0, 0.0

    weights = patch_strength[patch_mask].astype(np.float32)
    colors = patch_color[patch_mask].astype(np.float32)
    if float(weights.sum()) > 1e-6:
        color_mean = np.average(colors, axis=0, weights=weights)
        sin_mean = float(np.average(np.sin(patch_angle[patch_mask]), weights=weights))
        cos_mean = float(np.average(np.cos(patch_angle[patch_mask]), weights=weights))
        strength = float(weights.mean())
    else:
        color_mean = colors.mean(axis=0)
        sin_mean = float(np.sin(patch_angle[patch_mask]).mean())
        cos_mean = float(np.cos(patch_angle[patch_mask]).mean())
        strength = 0.0
    orientation = float(np.arctan2(sin_mean, cos_mean))
    return color_mean.astype(np.float32), orientation, strength


def color_distance_score(color_a: np.ndarray, color_b: np.ndarray) -> tuple[float, float]:
    distance = float(np.linalg.norm(color_a.astype(np.float32) - color_b.astype(np.float32)))
    # RGB max distance is about 441. This maps close colors toward 1.
    score = max(0.0, 1.0 - distance / 120.0)
    return distance, score


def orientation_compatibility(angle_a: float, angle_b: float, rot_angle_deg: float) -> float:
    rotated_b = angle_b + np.deg2rad(rot_angle_deg)
    diff = abs(float(np.angle(np.exp(1j * (angle_a - rotated_b)))))
    # Texture/stripe direction is axial, so 180 degrees is equivalent.
    diff = min(diff, abs(np.pi - diff))
    return float(max(0.0, np.cos(diff)))


def tangent_opposition(tangent_a: np.ndarray, tangent_b: np.ndarray, rot_angle_deg: float) -> float:
    rot = rotation_matrix(rot_angle_deg)
    tangent_b_rot = rot @ tangent_b
    # Matched break edges usually run in opposite traversal directions.
    return float((1.0 - np.dot(tangent_a, tangent_b_rot)) / 2.0)


def compose_transform(
    b_centroid: np.ndarray,
    a_point: np.ndarray,
    b_point: np.ndarray,
    angle_deg: float,
    normal_push_px: float,
) -> np.ndarray:
    rot = rotation_matrix(angle_deg)
    b_rotated_point = rot @ (b_point - b_centroid)
    desired = a_point - b_rotated_point
    direction = a_point - desired
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        desired = desired - direction / norm * normal_push_px
    mat = np.eye(3, dtype=np.float32)
    mat[:2, :2] = rot
    mat[:2, 2] = desired - rot @ b_centroid
    return mat
