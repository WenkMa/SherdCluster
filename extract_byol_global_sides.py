#!/usr/bin/env python3
"""Extract whole-image BYOL descriptors from both surfaces, then concatenate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from global_sides_transforms import evaluation_transform, preprocessing_description

PACKAGE_ROOT = Path(__file__).resolve().parent


class PairedSurfaces(Dataset):
    def __init__(self, image_dir: Path, image_size=320, require_complete=True, preprocessing="legacy"):
        exterior_ids = {p.stem.removesuffix("_exterior") for p in image_dir.glob("*_exterior.png")}
        interior_ids = {p.stem.removesuffix("_interior") for p in image_dir.glob("*_interior.png")}
        self.excluded_unpaired = {"missing_interior": sorted(exterior_ids - interior_ids),
                                 "missing_exterior": sorted(interior_ids - exterior_ids)}
        self.ids = sorted(exterior_ids & interior_ids)
        if not self.ids or (require_complete and exterior_ids != interior_ids):
            raise ValueError("Image directory must contain complete exterior/interior pairs")
        self.image_dir = image_dir
        self.preprocessing = preprocessing
        self.transform = evaluation_transform(image_size, preprocessing)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        views = []
        for side in ("exterior", "interior"):
            with Image.open(self.image_dir / f"{sample_id}_{side}.png") as image:
                views.append(self.transform(image.convert("RGB") if self.preprocessing == "legacy" else image))
        return torch.stack(views), sample_id


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/sherdcluster_epoch15.pt"))
    parser.add_argument("--image-dir", default=Path("data/images"), type=Path, help="Directory containing ID_exterior.png / ID_interior.png")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320,
                        help="Fixed full-image 320 protocol used by the delivered model")
    parser.add_argument("--preprocessing", choices=("full_global",), default="full_global")
    parser.add_argument("--ids", type=Path, help="Optional text file of IDs to extract")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or worker count")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_epoch = checkpoint.get("epoch")
    config = checkpoint.get("config") or {}
    image_dir = args.image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    image_size = args.image_size
    preprocessing = args.preprocessing
    if image_size < 64:
        raise ValueError("Image size must be at least 64")
    state = checkpoint["model_state_dict"]
    encoder_state = {k.removeprefix("online_encoder."): v for k, v in state.items() if k.startswith("online_encoder.")}
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    device = torch.device(args.device)
    encoder = encoder.to(device).eval()
    del checkpoint, state, encoder_state
    dataset = PairedSurfaces(image_dir, image_size, require_complete=False, preprocessing=preprocessing)
    if args.ids:
        wanted = args.ids.read_text().splitlines()
        if not set(wanted) <= set(dataset.ids):
            raise ValueError("Requested extraction IDs lack paired images")
        dataset.ids = sorted(set(wanted))
    print(f"Paired sherds: {len(dataset)}; excluded unpaired views: "
          f"{ {k: len(v) for k, v in dataset.excluded_unpaired.items()} }", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    descriptions, sample_ids = [], []
    with torch.inference_mode():
        for step, (images, ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            batch = len(images)
            features = F.normalize(encoder(images.flatten(0, 1)), dim=1).reshape(batch, 2, 2048)
            if not torch.isfinite(features).all() or torch.any(torch.linalg.vector_norm(features, dim=2) == 0):
                raise ValueError("Invalid BYOL descriptor")
            descriptions.append(features.cpu().numpy())
            sample_ids.extend(ids)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"Extracted {len(sample_ids)}/{len(dataset)} paired sherds", flush=True)
    features = np.concatenate(descriptions).astype(np.float32, copy=False)
    del descriptions
    metadata = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "encoder": "frozen BYOL online ResNet50 backbone; no projection head",
        "image_dir": str(image_dir), "samples": len(sample_ids),
        "excluded_unpaired": dataset.excluded_unpaired,
        "preprocessing": preprocessing_description(image_size, preprocessing),
        "feature_preprocessing": preprocessing,
        "image_size": image_size,
        "training_protocol": config.get("training_protocol", "historical BYOL"),
        "paired_sides_training": bool(config.get("paired_sides_training", False)),
        "precision": "FP32, TF32 disabled", "labels_read": False,
        "fusion": "direct concatenation of normalized exterior and interior global descriptors",
        "edge_features_used": False,
        "limitation": ("Dual-side training performance requires matched downstream evaluation."
                       if config.get("paired_sides_training") else
                       "Historical checkpoint was trained on exterior images; interior generalization is tested by this experiment."),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, array in (
        ("exterior", features[:, 0]), ("interior", features[:, 1]),
        ("both_concat", np.concatenate([features[:, 0], features[:, 1]], axis=1)),
    ):
        np.savez_compressed(args.output / f"{name}.npz", sample_ids=np.asarray(sample_ids),
                            features=array, metadata_json=json.dumps({**metadata, "view": name}))
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved global feature banks:", args.output, flush=True)


if __name__ == "__main__":
    main()
