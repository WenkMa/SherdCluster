"""Visualization helpers for pairing outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_pairing_contact_sheet(
    pairing_results_csv: Path,
    output_path: Path,
    *,
    max_items: int = 80,
    thumb_width: int = 360,
    label_height: int = 54,
) -> Path:
    with pairing_results_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if row.get("preview_path")]

    rows = [row for row in rows if str(row.get("side_a")) == "exterior" and str(row.get("side_b")) == "exterior"]
    rows = [row for row in rows if str(row.get("rank")) in {"", "1"}]
    rows = rows[:max_items]
    if not rows:
        raise ValueError(f"No exterior rank-1 previews found in {pairing_results_csv}")

    thumbs = []
    for row in rows:
        image = Image.open(row["preview_path"]).convert("RGB")
        scale = thumb_width / max(1, image.width)
        thumb_height = max(1, int(round(image.height * scale)))
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumbs.append((row, image))

    cols = 2 if len(thumbs) > 1 else 1
    rows_count = (len(thumbs) + cols - 1) // cols
    cell_height = max(image.height for _, image in thumbs) + label_height
    sheet = Image.new("RGB", (cols * thumb_width, rows_count * cell_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (row, image) in enumerate(thumbs):
        col = idx % cols
        row_idx = idx // cols
        x = col * thumb_width
        y = row_idx * cell_height
        sheet.paste(image, (x, y))
        label = (
            f"{row.get('sherd_id_a')} + {row.get('sherd_id_b')}  "
            f"score={row.get('pair_score')} tex={row.get('segment_texture_score')} "
            f"rms={row.get('segment_rms_px')}"
        )
        draw.rectangle([x, y + image.height, x + thumb_width, y + cell_height], fill=(255, 255, 255))
        draw.text((x + 8, y + image.height + 8), label[:92], fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
