#!/usr/bin/env python3
"""Pure BYOL: three independent geometric views per side, two separate losses."""
from __future__ import annotations


import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

from extract_byol_global_sides import PairedSurfaces
from global_sides_transforms import FullGlobalViews, evaluation_transform, preprocessing_description


class ColorPreservingViews:
    """Geometry only; avoid deliberate color, brightness or texture corruption."""
    def __init__(self, size=320):
        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        interpolation = transforms.InterpolationMode.BILINEAR
        self.views = [
            transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(.7, 1.), interpolation=interpolation),
                transforms.RandomHorizontalFlip(), tensor,
            ]),
            transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.RandomRotation(20, interpolation=interpolation),
                transforms.CenterCrop(size), transforms.RandomVerticalFlip(), tensor,
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(.75, 1.), interpolation=interpolation),
                transforms.RandomApply([
                    transforms.RandomPerspective(.15, p=1., interpolation=interpolation),
                ], p=.5),
                tensor,
            ]),
        ]

    def __call__(self, image):
        return torch.stack([view(image) for view in self.views])


class PairedTrainingSurfaces(PairedSurfaces):
    def __init__(self, image_dir, size=320, limit=0, view_policy="legacy"):
        if view_policy not in ("legacy", "full_global"):
            raise ValueError(f"Unknown view policy: {view_policy}")
        super().__init__(Path(image_dir), require_complete=False, preprocessing=view_policy)
        self.transform = ColorPreservingViews(size) if view_policy == "legacy" else FullGlobalViews(size)
        if limit:
            self.ids = self.ids[:limit]
        self.size = size


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def separate_side_losses(prediction, target):
    """B x 2 sides x 3 views x D; six directed pairs on each physical side."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1:3] != (2, 3):
        raise ValueError("Expected matching B x 2 x 3 x D predictions and targets")
    # All reductions stay in FP32, including when the encoder uses autocast.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p = F.normalize(prediction.float(), dim=-1)
        t = F.normalize(target.detach().float(), dim=-1)
        distances = 2 - 2 * (p @ t.transpose(-1, -2))
        off_diagonal = ~torch.eye(3, device=p.device, dtype=torch.bool)
        losses = distances[:, :, off_diagonal].mean(dim=(0, 2))
    return {"loss": losses.sum(), "exterior_loss": losses[0], "interior_loss": losses[1]}


class PairedGlobalBYOL(nn.Module):
    def __init__(self, hidden_dim=4096, projection_dim=256, imagenet=False):
        super().__init__()
        self.online_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if imagenet else None)
        self.online_encoder.fc = nn.Identity()
        self.online_projector = MLP(2048, hidden_dim, projection_dim)
        self.online_predictor = MLP(projection_dim, hidden_dim, projection_dim)
        self.target_encoder = deepcopy(self.online_encoder).requires_grad_(False)
        self.target_projector = deepcopy(self.online_projector).requires_grad_(False)
        self.frozen_names = []

    def freeze_through(self, stage):
        self.frozen_names = [] if stage == "none" else ["conv1", "bn1"]
        if stage not in ("none", "stem"):
            self.frozen_names += [f"layer{i}" for i in range(1, int(stage[-1]) + 1)]
        for encoder in (self.online_encoder, self.target_encoder):
            for name in self.frozen_names:
                getattr(encoder, name).requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Retain historical BYOL behavior: each target maintains its own BN
        # statistics on augmented images; EMA applies to parameters only.
        for encoder in (self.online_encoder, self.target_encoder):
            for name in self.frozen_names:
                getattr(encoder, name).eval()
        return self

    def side_predictions(self, views):
        if views.ndim != 5 or views.shape[1:3] != (3, 3):
            raise ValueError("Expected B x 3 views x 3 channels x H x W")
        b = len(views)
        flat = views.flatten(0, 1)
        features = self.online_encoder(flat).reshape(b, 3, -1)
        prediction = torch.stack([self.online_predictor(self.online_projector(features[:, v]))
                                  for v in range(3)], dim=1)
        with torch.no_grad():
            target_features = self.target_encoder(flat).reshape(b, 3, -1)
            target = torch.stack([self.target_projector(target_features[:, v]) for v in range(3)], dim=1)
        return prediction, target

    def forward_side(self, views):
        prediction, target = self.side_predictions(views)
        # Duplicate only the small head outputs to reuse the checked side-loss
        # reduction. Return one side's six directed-pair mean, not the sum.
        return separate_side_losses(prediction[:, None].expand(-1, 2, -1, -1),
                                    target[:, None].expand(-1, 2, -1, -1))["exterior_loss"]

    def forward(self, views):
        if views.ndim != 6 or views.shape[1:4] != (2, 3, 3):
            raise ValueError("Expected B x 2 sides x 3 views x 3 channels x H x W")
        pairs = [self.side_predictions(views[:, side]) for side in range(2)]
        prediction = torch.stack([p for p, _ in pairs], dim=1)
        target = torch.stack([t for _, t in pairs], dim=1)
        return separate_side_losses(prediction, target)

    @torch.no_grad()
    def update_target(self, momentum=.996):
        for online, target in ((self.online_encoder, self.target_encoder),
                               (self.online_projector, self.target_projector)):
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.lerp_(op, 1 - momentum)


def build_optimizer(model, lr, weight_decay, backbone_lr=None):
    """Keep the legacy single group unless separate backbone LR is requested."""
    if backbone_lr is None:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "name": "online", "base_lr": lr, "lr": lr}]
    else:
        groups = []
        for name, modules, rate in (
            ("backbone", (model.online_encoder,), backbone_lr),
            ("heads", (model.online_projector, model.online_predictor), lr),
        ):
            parameters = [p for module in modules for p in module.parameters() if p.requires_grad]
            if parameters:
                groups.append({"params": parameters, "name": name, "base_lr": rate, "lr": rate})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


@torch.inference_mode()
def monitor(model, dataset, device, count):
    was_training = model.training
    model.eval()
    transform = evaluation_transform(dataset.size, dataset.preprocessing)
    count = min(count, len(dataset))
    # Legacy keeps its historical sample order. New runs spread monitoring over
    # the gallery rather than choosing only the first lexicographic IDs.
    sample_ids = dataset.ids[:count] if dataset.preprocessing == "legacy" else [
        dataset.ids[index] for index in np.linspace(0, len(dataset) - 1, count, dtype=int)
    ]
    outputs = []
    for start in range(0, count, 8):
        images = []
        for sample_id in sample_ids[start:start + 8]:
            for side in ("exterior", "interior"):
                with Image.open(dataset.image_dir / f"{sample_id}_{side}.png") as image:
                    images.append(transform(image.convert("RGB") if dataset.preprocessing == "legacy" else image))
        feature = F.normalize(model.online_encoder(torch.stack(images).to(device)).float(), dim=-1)
        outputs.append(feature.reshape(-1, 2, 2048).cpu())
    features = torch.cat(outputs)
    if not torch.isfinite(features).all() or torch.any(features.norm(dim=-1) < .99):
        raise FloatingPointError("Invalid monitored global features")
    result = {f"{side}_feature_std": float(features[:, index].std(0, unbiased=True).mean())
              for index, side in enumerate(("exterior", "interior"))}
    result["samples"] = len(features)
    if dataset.preprocessing == "full_global":
        for index, side in enumerate(("exterior", "interior")):
            vectors = features[:, index]
            n = len(vectors)
            # Exact mean off-diagonal cosine, without constructing an N x N matrix.
            mean_cosine = (vectors.sum(0).square().sum() - vectors.square().sum()) / (n * (n - 1))
            result[f"{side}_mean_pair_cosine"] = float(mean_cosine)
            result[f"{side}_feature_std_p10"] = float(vectors.std(0).quantile(.1))
    model.train(was_training)
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_signature(dataset):
    """Detect changed training files on resume without decoding all images."""
    digest = hashlib.sha256()
    for sample_id in dataset.ids:
        for side in ("exterior", "interior"):
            path = dataset.image_dir / f"{sample_id}_{side}.png"
            digest.update(json.dumps([path.name, file_hash(path)]).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def restore_initialization_metadata(config, saved_config):
    """Resume uses the saved model, so does not reopen its original init file."""
    for key in ("init_checkpoint", "init_checkpoint_sha256", "initialization"):
        config[key] = saved_config[key]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image-dir", type=Path, default=Path("data/images"))
    p.add_argument("--init-checkpoint", type=Path, help="Optional continued training; default starts from ImageNet IMAGENET1K_V2")
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32, help="Sherds per batch; six augmented images per sherd")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-4, help="Separate backbone LR; --lr then sets the MLP LR")
    p.add_argument("--view-policy", choices=("legacy", "full_global"), default="full_global",
                   help="full_global: one complete-image anchor plus two mild geometric views")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=.996)
    p.add_argument("--freeze-through", choices=("none", "stem", "layer1", "layer2", "layer3", "layer4"), default="none")
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--monitor-size", type=int, default=1600, help="Old successful run: 50 batches of 32 sherds")
    p.add_argument("--monitor-every", type=int, default=5)
    p.add_argument("--min-feature-std", type=float, default=.01)
    p.add_argument("--limit", type=int, default=0, help="Small smoke tests only; 0 uses all paired sherds")
    p.add_argument("--stop-after-epoch", type=int, default=15,
                   help="Stop at this epoch while retaining the --epochs cosine schedule (historical horizon: 100)")
    p.add_argument("--imagenet-weights", type=Path, default=Path("weights/resnet50_imagenet1k_v2.pth"))
    args = p.parse_args()
    if not 0 < args.stop_after_epoch <= args.epochs:
        raise ValueError("stop-after-epoch must lie within the cosine schedule horizon")
    if (args.epochs < 1 or args.batch_size < 2 or args.image_size < 64 or args.workers < 0
        or args.lr <= 0 or args.weight_decay < 0
        or (args.backbone_lr is not None and args.backbone_lr <= 0)
        or not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.momentum,
                                             args.min_feature_std, args.backbone_lr or args.lr))
        or not 0 <= args.momentum < 1 or args.checkpoint_every < 1 or args.monitor_size < 2
        or args.monitor_every < 1 or args.min_feature_std < 0 or args.limit < 0):
        raise ValueError("Invalid training configuration")
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    dataset = PairedTrainingSurfaces(args.image_dir, args.image_size, args.limit, args.view_policy)
    if len(dataset) < args.batch_size:
        raise ValueError("At least one complete batch is required")
    source = args.resume or args.init_checkpoint
    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True) if source else None
    protocol = "paired-global-legacy-byol-v1" if args.view_policy == "legacy" else "paired-global-full-view-byol-v2"
    if args.resume and (checkpoint.get("config") or {}).get("training_protocol") != protocol:
        raise ValueError("--resume requires a checkpoint with the same paired BYOL view policy")
    # Initialize from the bundled weights, without downloading or using external caches.
    model = PairedGlobalBYOL(imagenet=False)
    if source is None:
        init = torch.load(args.imagenet_weights, map_location="cpu", weights_only=True)
        init = {k: v for k, v in init.items() if not k.startswith("fc.")}
        model.online_encoder.load_state_dict(init, strict=True)
        model.target_encoder.load_state_dict(init, strict=True)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.freeze_through(args.freeze_through)
    model.to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, args.backbone_lr)
    dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    amp_enabled = dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and dtype == torch.float16)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(training_protocol=protocol, paired_sides_training=True,
                  feature_preprocessing=dataset.preprocessing,
                  evaluation_preprocessing=preprocessing_description(args.image_size, dataset.preprocessing),
                  input_signature_format="sha256_file_contents_v1", labels_read=False, edge_features_used=False, feature_fusion="direct normalized-block concatenation",
                  loss="exterior six-pair mean + interior six-pair mean; no cross-side positives",
                  target_bn="own running statistics in train mode; parameter EMA only",
                  initialization="continued BYOL checkpoint" if args.init_checkpoint else "ImageNet IMAGENET1K_V2",
                  init_checkpoint_sha256=(file_hash(args.init_checkpoint) if args.init_checkpoint and not args.resume else None),
                  train_ids_sha256=hashlib.sha256("\n".join(dataset.ids).encode()).hexdigest(),
                  train_input_signature_sha256=input_signature(dataset))
    start = 0
    saved_initial_monitor = None
    if args.resume:
        old_config = checkpoint["config"]
        restore_initialization_metadata(config, old_config)
        saved_initial_monitor = old_config.get("initial_monitor")
        legacy_defaults = {"view_policy": "legacy", "backbone_lr": None,
                           "feature_preprocessing": "legacy",
                           "evaluation_preprocessing": preprocessing_description(args.image_size, "legacy")}
        for key, value in config.items():
            if key == "train_input_signature_sha256" and old_config.get("input_signature_format") == "historical_mtime_not_portable":
                continue
            if key not in ("output", "resume", "workers", "device", "image_dir", "stop_after_epoch", "imagenet_weights", "input_signature_format") and value != old_config.get(key, legacy_defaults.get(key)):
                raise ValueError(f"Resume configuration mismatch: {key}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["rng"]["torch"])
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        if device.type == "cuda":
            saved_device = torch.device(old_config.get("device", "cuda:0"))
            saved_index = saved_device.index or 0
            states = checkpoint["rng"]["cuda"]
            if saved_index >= len(states):
                raise ValueError("Saved CUDA RNG state does not cover the training device")
            torch.cuda.set_rng_state(states[saved_index], device=device)
    if start >= args.stop_after_epoch:
        raise ValueError("Checkpoint already completed requested epochs")
    del checkpoint
    initial_monitor = monitor(model, dataset, device, args.monitor_size)
    config["initial_monitor"] = saved_initial_monitor if args.resume and saved_initial_monitor is not None else initial_monitor
    if args.resume:
        config["resume_monitor"] = initial_monitor
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "train_ids.txt").write_text("\n".join(dataset.ids) + "\n")
    (args.output / "excluded_unpaired.json").write_text(json.dumps(dataset.excluded_unpaired, indent=2))
    print("Initialization:", config["initialization"], flush=True)
    print("View policy:", args.view_policy, "| Extraction:", config["evaluation_preprocessing"], flush=True)
    print("Learning rates:", {group.get("name", "online"): group.get("base_lr", args.lr)
                               for group in optimizer.param_groups}, flush=True)
    print("Excluded unpaired views:", {k: len(v) for k, v in dataset.excluded_unpaired.items()}, flush=True)
    print(f"Paired sherds: {len(dataset)}; six images per sherd; {config['loss']}", flush=True)
    print("Initial features:", initial_monitor, flush=True)
    for epoch in range(start, args.stop_after_epoch):
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        for group in optimizer.param_groups:
            group["lr"] = group.get("base_lr", args.lr) * .5 * (1 + math.cos(math.pi * epoch / args.epochs))
        model.train()
        sums = {"loss": 0., "exterior_loss": 0., "interior_loss": 0.}
        steps = updates = 0
        for views, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            views = views.to(device, non_blocking=True)
            # Accumulate the two unweighted losses before ONE optimizer/EMA
            # update; retain only one side's activations at a time.
            metrics = {}
            for side, name in enumerate(("exterior_loss", "interior_loss")):
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                    side_loss = model.forward_side(views[:, side])
                if not torch.isfinite(side_loss):
                    raise FloatingPointError(f"Nonfinite {name} at epoch {epoch + 1}")
                scaler.scale(side_loss).backward()
                metrics[name] = side_loss.detach()
                del side_loss
            metrics["loss"] = metrics["exterior_loss"] + metrics["interior_loss"]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                model.update_target(args.momentum)
                updates += 1
            steps += 1
            for key in sums:
                sums[key] += float(metrics[key].detach())
            if steps == 1 or steps % 50 == 0 or steps == len(loader):
                print(f"Epoch {epoch+1}/{args.epochs} [{steps}/{len(loader)}] "
                      f"Loss {sums['loss']/steps:.4f} Exterior {sums['exterior_loss']/steps:.4f} "
                      f"Interior {sums['interior_loss']/steps:.4f}", flush=True)
        if updates == 0:
            raise FloatingPointError("All optimizer updates were skipped")
        feature_monitor = {"exterior_feature_std": None, "interior_feature_std": None}
        if (epoch + 1) % args.monitor_every == 0 or epoch + 1 == args.stop_after_epoch:
            feature_monitor = monitor(model, dataset, device, args.monitor_size)
            if min(feature_monitor["exterior_feature_std"], feature_monitor["interior_feature_std"]) < args.min_feature_std:
                raise FloatingPointError(f"Global feature diversity below threshold: {feature_monitor}")
        row = {"epoch": epoch + 1, "optimizer_updates": updates, "skipped_updates": steps - updates,
               "learning_rates": {group.get("name", "online"): group["lr"] for group in optimizer.param_groups},
               **{k: v / steps for k, v in sums.items()}, **feature_monitor}
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "config": config, "metrics": row,
                   "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                           "numpy": np.random.get_state(),
                           "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
        temporary = args.output / "last.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.output / "last.pt")
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.stop_after_epoch:
            import shutil
            shutil.copyfile(args.output / "last.pt", args.output / f"checkpoint_epoch{epoch+1}.pt")
        print("Saved epoch:", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
