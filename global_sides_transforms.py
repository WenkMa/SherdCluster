"""Whole-image transforms shared by paired BYOL training and extraction."""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


class WhiteRGB:
    """Composite actual transparency on white; never infer a mask from color."""
    def __call__(self, image):
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(canvas, rgba).convert("RGB")
        return image.convert("RGB")


class FitWhiteSquare:
    """Resize the entire image without distortion and center on a white square."""
    def __init__(self, size, margin=.04):
        if size < 1 or not 0 <= margin < .5:
            raise ValueError("Invalid square size or margin")
        self.size = size
        self.margin = margin

    def __call__(self, image):
        image = WhiteRGB()(image)
        width, height = image.size
        inner = max(1, round(self.size * (1 - 2 * self.margin)))
        scale = inner / max(width, height)
        resized_width = max(1, min(inner, round(width * scale)))
        resized_height = max(1, min(inner, round(height * scale)))
        image = TF.resize(image, [resized_height, resized_width],
                          interpolation=transforms.InterpolationMode.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        canvas.paste(image, ((self.size - resized_width) // 2,
                            (self.size - resized_height) // 2))
        return canvas


class MildCrop:
    """Retain at least min_fraction of each image dimension; no aspect warping.

    This bounds the removed image area, not the removed foreground area.
    Only the full-image anchor and evaluation transform guarantee no cropping.
    """
    def __init__(self, min_fraction=.95):
        if not 0 < min_fraction <= 1:
            raise ValueError("Invalid crop fraction")
        self.min_fraction = min_fraction

    def __call__(self, image):
        width, height = image.size
        fraction = self.min_fraction + (1 - self.min_fraction) * torch.rand(2)
        crop_width = min(width, max(1, int(torch.ceil(fraction[0] * width))))
        crop_height = min(height, max(1, int(torch.ceil(fraction[1] * height))))
        left = int(torch.randint(width - crop_width + 1, ()).item())
        top = int(torch.randint(height - crop_height + 1, ()).item())
        return TF.crop(image, top, left, crop_height, crop_width)


def tensor_normalization():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def evaluation_transform(size, preprocessing="legacy"):
    if preprocessing == "legacy":
        return transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size), tensor_normalization(),
        ])
    if preprocessing == "full_global":
        return transforms.Compose([WhiteRGB(), FitWhiteSquare(size), tensor_normalization()])
    raise ValueError(f"Unknown preprocessing: {preprocessing}")


class SampleWithoutApplying:
    """Consume a disabled operation's RNG draws, while preserving the input.

    Deleting a random transform changes later random choices. Sampling and
    discarding it keeps the existing random stream for controlled ablations.
    The wrapped torchvision/PIL transforms return new images and do not mutate
    the input. Perspective pixel coordinates can still depend on image size.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        self.transform(image)
        return image


class FullGlobalViews:
    """One whole-image anchor plus two mild geometric views, without color jitter."""
    def __init__(self, size=320, *, crop=True, horizontal_flip=True,
                 vertical_flip=True, rotation=True, perspective=True):
        interpolation = transforms.InterpolationMode.BILINEAR
        def selected(transform, enabled):
            return transform if enabled else SampleWithoutApplying(transform)

        self.views = [
            transforms.Compose([
                WhiteRGB(), selected(transforms.RandomHorizontalFlip(), horizontal_flip),
                FitWhiteSquare(size), tensor_normalization(),
            ]),
            transforms.Compose([
                WhiteRGB(), selected(MildCrop(.95), crop),
                selected(transforms.RandomHorizontalFlip(), horizontal_flip),
                selected(transforms.RandomRotation(10, interpolation=interpolation,
                                                   expand=True, fill=(255, 255, 255)), rotation),
                FitWhiteSquare(size), tensor_normalization(),
            ]),
            transforms.Compose([
                WhiteRGB(), selected(MildCrop(.97), crop),
                selected(transforms.RandomPerspective(.05, p=.3, interpolation=interpolation,
                                                      fill=(255, 255, 255)), perspective),
                selected(transforms.RandomRotation(5, interpolation=interpolation,
                                                   expand=True, fill=(255, 255, 255)), rotation),
                selected(transforms.RandomVerticalFlip(p=.3), vertical_flip),
                FitWhiteSquare(size), tensor_normalization(),
            ]),
        ]

    def __call__(self, image):
        return torch.stack([view(image) for view in self.views])


def preprocessing_description(size, preprocessing):
    if preprocessing == "full_global":
        return (f"Alpha composited on white when present; RGB; complete image resized "
                f"with aspect ratio preserved; centered white square {size}, margin 0.04; "
                "ImageNet normalization; no center crop")
    if preprocessing == "legacy":
        return f"RGB conversion; Resize shorter edge {size}; CenterCrop {size}; ImageNet normalization"
    raise ValueError(f"Unknown preprocessing: {preprocessing}")
