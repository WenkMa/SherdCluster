"""Optional refinement agent using the legacy SherdPair diffusion model."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import SherdRecord, records_by_id


LEGACY_DIR = Path("/home/wenkang/SherdPair")
DEFAULT_LEGACY_MODEL = LEGACY_DIR / "dinov3_epoch_500.pth"


@dataclass(frozen=True)
class LegacyRefineConfig:
    max_pairs: int = 20
    num_repeat: int = 3
    contour_spacing: int = 35
    patch_size: int = 48
    model_path: Path = DEFAULT_LEGACY_MODEL


class LegacyImports:
    def __init__(self) -> None:
        if str(LEGACY_DIR) not in sys.path:
            sys.path.insert(0, str(LEGACY_DIR))
        from diffusion import run_diffusion_refinement  # type: ignore
        from utils import build_feature_model, extract_features_from_image  # type: ignore

        self.run_diffusion_refinement = run_diffusion_refinement
        self.build_feature_model = build_feature_model
        self.extract_features_from_image = extract_features_from_image


class LegacyFeatureCache:
    def __init__(self, legacy: LegacyImports, model_path: Path, device: torch.device, config: LegacyRefineConfig) -> None:
        self.legacy = legacy
        self.device = device
        self.config = config
        self.model = legacy.build_feature_model(str(model_path), device=device, proj_dim=128)
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(self, sherd_id: str, side: str, image_path: str) -> dict[str, Any]:
        key = (sherd_id, side)
        if key not in self.cache:
            self.cache[key] = self.legacy.extract_features_from_image(
                image_path=image_path,
                feature_model=self.model,
                device=self.device,
                contour_spacing=self.config.contour_spacing,
                patch_size=self.config.patch_size,
            )
        return self.cache[key]


def record_path_for_side(record: SherdRecord, side: str) -> str:
    if side == "exterior":
        return record.exterior_path
    if side == "interior":
        return record.interior_path
    raise ValueError(f"Unsupported side: {side}")


def pred_to_jsonable(pred: Any) -> str:
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    return json.dumps(np.asarray(pred).round(6).tolist(), ensure_ascii=False)


def run_legacy_refine_agent(
    pairs_csv: Path,
    records: list[SherdRecord],
    output_dir: Path,
    config: LegacyRefineConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "legacy_refine_results.csv"
    image_dir = output_dir / "diffusion_previews"

    if not config.model_path.exists():
        raise FileNotFoundError(f"Legacy DINO model not found: {config.model_path}")

    with pairs_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [
        row
        for row in rows
        if row.get("sherd_id_a") and row.get("sherd_id_b") and row.get("side_a") == row.get("side_b")
    ]
    rows.sort(key=lambda row: -float(row.get("calibrated_score") or row.get("pair_score") or 0.0))
    if config.max_pairs > 0:
        rows = rows[: config.max_pairs]

    legacy = LegacyImports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = LegacyFeatureCache(legacy, config.model_path, device, config)
    record_map = records_by_id(records)

    fieldnames = [
        "rank",
        "sherd_id_a",
        "sherd_id_b",
        "side",
        "source_pair_score",
        "source_preview_path",
        "repeat_index",
        "diffusion_preview_path",
        "pred",
        "status",
        "error",
    ]
    with results_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for pair_rank, row in enumerate(rows, start=1):
            sherd_a = row["sherd_id_a"]
            sherd_b = row["sherd_id_b"]
            side = row["side_a"]
            if sherd_a not in record_map or sherd_b not in record_map:
                writer.writerow(
                    {
                        "rank": pair_rank,
                        "sherd_id_a": sherd_a,
                        "sherd_id_b": sherd_b,
                        "side": side,
                        "status": "missing_record",
                        "error": "Sherd ID not found in clusters CSV.",
                    }
                )
                continue

            try:
                record_a = record_map[sherd_a]
                record_b = record_map[sherd_b]
                feat_a = cache.get(sherd_a, side, record_path_for_side(record_a, side))
                feat_b = cache.get(sherd_b, side, record_path_for_side(record_b, side))
                for repeat_index in range(config.num_repeat):
                    save_tag = f"{pair_rank:04d}_{sherd_a}_{sherd_b}_{side}_sample{repeat_index}"
                    pred = legacy.run_diffusion_refinement(
                        query_coords=feat_a["coords"],
                        target_coords=feat_b["coords"],
                        query_feat=feat_a["feat"],
                        target_feat=feat_b["feat"],
                        query_global_feat=feat_a["global_feat"],
                        target_global_feat=feat_b["global_feat"],
                        query_image=feat_a["image"],
                        target_image=feat_b["image"],
                        save_tag=save_tag,
                        save_dir=str(image_dir),
                    )
                    writer.writerow(
                        {
                            "rank": pair_rank,
                            "sherd_id_a": sherd_a,
                            "sherd_id_b": sherd_b,
                            "side": side,
                            "source_pair_score": row.get("calibrated_score") or row.get("pair_score", ""),
                            "source_preview_path": row.get("preview_path", ""),
                            "repeat_index": repeat_index,
                            "diffusion_preview_path": str(image_dir / f"{save_tag}.png"),
                            "pred": pred_to_jsonable(pred),
                            "status": "ok",
                            "error": "",
                        }
                    )
            except Exception as exc:
                writer.writerow(
                    {
                        "rank": pair_rank,
                        "sherd_id_a": sherd_a,
                        "sherd_id_b": sherd_b,
                        "side": side,
                        "source_pair_score": row.get("calibrated_score") or row.get("pair_score", ""),
                        "source_preview_path": row.get("preview_path", ""),
                        "repeat_index": "",
                        "diffusion_preview_path": "",
                        "pred": "",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            if pair_rank % 5 == 0:
                print(f"legacy refine progress: {pair_rank}/{len(rows)} pairs")

    return results_path

