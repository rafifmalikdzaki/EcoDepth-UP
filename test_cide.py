"""CIDE module with flexible ViT/ConvNeXt backbone and classifier‑safe loading.

This version fixes the **IndentationError** and a stray division in `_prep`.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import timm
import numpy as np
from PIL import Image
from einops import repeat
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTConfig

# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------

def pad_to_square(x: torch.Tensor) -> torch.Tensor:
    """Zero‑pad a BCHW tensor (range ‑1…1) to make it square."""
    b, c, h, w = x.shape
    if h == w:
        return x
    if h < w:
        pad = (0, 0, 0, w - h)  # bottom pad
    else:
        pad = (0, 0, h - w, 0)  # right pad
    return TF.pad(x, pad, fill=-1.0)


def load_image(path: str | os.PathLike, image_size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [image_size, image_size], antialias=True)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)


# -----------------------------------------------------------------------------
#  Embedding adapter
# -----------------------------------------------------------------------------
class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return repeat(x + gamma * self.fc(x), "b d -> b 1 d")


# -----------------------------------------------------------------------------
#  CIDE main class
# -----------------------------------------------------------------------------
class CIDE(nn.Module):
    def __init__(
        self,
        args: Any,
        emb_dim: int,
        train_from_scratch: bool = False,
        *,
        vit_type: str = "microsoft/beit-base-patch16-224",
        use_timm: bool = False,
    ) -> None:
        super().__init__()
        self.no_of_classes = getattr(args, "no_of_classes", 100)
        self.image_size = getattr(args, "img_size", 224)

        # Backbone -------------------------------------------------------
        if use_timm:
            self.vit_model = timm.create_model(
                vit_type, pretrained=not train_from_scratch, num_classes=self.no_of_classes
            )
            self.processor = None
        else:
            try:
                self.processor = AutoImageProcessor.from_pretrained(vit_type, resume_download=True)
                self.vit_model = AutoModelForImageClassification.from_pretrained(
                    vit_type,
                    resume_download=True,
                    num_labels=self.no_of_classes,
                    ignore_mismatched_sizes=True,
                )
            except Exception:
                cfg = ViTConfig(num_labels=self.no_of_classes)
                self.vit_model = AutoModelForImageClassification.from_config(cfg)
                self.processor = None

        for p in self.vit_model.parameters():
            p.requires_grad = False

        # Scene‑conditioning --------------------------------------------
        self.softmax = nn.Softmax(dim=1)
        self.embeddings = nn.Parameter(torch.randn(self.no_of_classes, emb_dim))
        self.adapter = EmbeddingAdapter(emb_dim)
        self.gamma = nn.Parameter(torch.full((emb_dim,), 1e-4))

    # ------------------------------------------------------------------
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Pad, resize, and normalise to match backbone expectations."""
        x = pad_to_square(x)

        if self.processor is not None:  # Hugging Face path
            if x.shape[-1] != self.image_size:
                x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
            x = (x + 1.0) / 2.0
            x = x.clamp_(0, 1)
            return self.processor(images=x, return_tensors="pt", do_rescale=False).pixel_values.to(x.device)

        # timm path
        if x.shape[-1] != self.image_size:
            x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pv = self._prep(x)
        logits = self.vit_model(pv).logits  # [B, C]
        probs = self.softmax(logits)
        emb = probs @ self.embeddings  # [B, D]
        return self.adapter(emb, self.gamma)  # [B, 1, D]


# -----------------------------------------------------------------------------
#  CLI for manual test
# -----------------------------------------------------------------------------
class DummyArgs:
    no_of_classes = 100
    img_size = 224


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--vit_type", default="google/vit-base-patch16-224")
    parser.add_argument("--use_timm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="outputs", type=Path)
    opt = parser.parse_args()

    x = load_image(opt.image, 256).to(opt.device)
    model = CIDE(DummyArgs(), emb_dim=768, vit_type=opt.vit_type, use_timm=opt.use_timm).to(opt.device)
    with torch.no_grad():
        emb = model(x)
    print("Scene embedding", emb.shape)

    opt.out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{opt.vit_type.replace('/', '_')}_scene_emb.npy"
    np.save(opt.out_dir / fn, emb.squeeze(1).cpu().numpy())
    print("✅ saved to", opt.out_dir / fn)


if __name__ == "__main__":
    _cli()
