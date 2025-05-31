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
from timm.layers import Swish
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTConfig
from swish_act import *
from BiFPN import *

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
        emb_transformed = self.fc(x)
        texts = x + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts


# -----------------------------------------------------------------------------
#  CIDE main class
# -----------------------------------------------------------------------------
class CIDE(nn.Module):
    def __init__(
        self,
        args: Any,
        emb_dim: int,
        *,
        vit_type: str = "microsoft/swinv2-base-patch4-window16-256",
        activations: str = "sigmoid",
        output_vit_layers_indices: list[int] = [0,1,2],
        bifpn_layers: int = 2,
        feature_size: int = 128,
        testing: bool = False,
    ) -> None:
        super().__init__()
        self.testing = testing
        self.bifpn_layers = bifpn_layers
        self.bifpn_features_size = feature_size
        self.bifpn = BiFPN([128,256,512], self.bifpn_features_size, self.bifpn_layers)
        act_lower = activations.lower()
        self.softmax = nn.Softmax(dim=1)
        self.activations = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "swisht": SwishT(),
            "swishta": SwishT_A(),
            "swishtb": SwishT_B(),
            "swishtc": SwishT_C(),
            "softmax": nn.Softmax(dim=1)
        }.get(act_lower, nn.Sigmoid())

        self.no_of_classes = getattr(args, "no_of_classes", 100)
        self.image_size = getattr(args, "img_size", 224)
        self.intermediate_embeddings = 512
        self.last_embedding_size = 512 + 640

        self.output_vit_layers_indices = output_vit_layers_indices
        # Determine if we need hidden states based on output_vit_layers_indices
        self._output_hidden_states = bool(output_vit_layers_indices and len(output_vit_layers_indices) > 0)

        # Backbone -------------------------------------------------------
        self.processor = AutoImageProcessor.from_pretrained(vit_type, resume_download=True)
        self.vit_model = AutoModelForImageClassification.from_pretrained(
            vit_type,
            resume_download=True,
            num_labels=1000,
            ignore_mismatched_sizes=True,
            output_hidden_states=self._output_hidden_states,
        )
        for param in self.vit_model.parameters():
            param.requires_grad = False

        self.fc_classes = nn.Sequential(
            nn.Linear(1000, self.intermediate_embeddings),
            nn.GELU(),
            nn.Linear(self.intermediate_embeddings, self.intermediate_embeddings)
        )

        self.fc_embedding = nn.Sequential(
            nn.Linear(self.last_embedding_size, 384),
            nn.GELU(),
            nn.Linear(384, self.no_of_classes)
        )

        # Scene‑conditioning --------------------------------------------
        self.embeddings = nn.Embedding(num_embeddings=self.no_of_classes, embedding_dim=emb_dim)
        self.adapter = EmbeddingAdapter(emb_dim)
        self.gamma = nn.Parameter(torch.full((emb_dim,), 1e-4))

    # ------------------------------------------------------------------
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Pad, resize, and normalise to match backbone expectations."""
        x = pad_to_square(x)

        if x.shape[-1] != self.image_size:
            x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
        x = (x + 1.0) / 2.0
        x = x.clamp_(0, 1)
        with torch.no_grad():
            inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).pixel_values.to(x.device)

        return inputs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
        """
        Returns the CIDE scene embedding and optionally a list of selected ViT intermediate hidden states.
        """
        pixel_values = self._prep(x)

        intermediate_hidden_states_for_bifpn = None

        with torch.no_grad():
            vit_outputs = self.vit_model(pixel_values)
            final_logits = vit_outputs.logits  # These are the [B, 1000] logits from ViT

            if self._output_hidden_states:
                all_hidden_states = vit_outputs.hidden_states  # Tuple of hidden states
                intermediate_hidden_states_for_bifpn = []
                if self.output_vit_layers_indices:
                    for i in self.output_vit_layers_indices:
                        if 0 <= i < len(all_hidden_states):
                            h = all_hidden_states[i]  # (B, num_patches, C)
                            if len(h.shape) == 4:
                                intermediate_hidden_states_for_bifpn.append(h)
                            else:
                                # Let's reshape it!
                                batch_size, num_patches, dim = h.shape
                                # Suppose input image size = 256, patch size = 4 (for SwinV2-tiny-patch4)
                                h_w = int(num_patches ** 0.5)  # Should be 64 for 256/4=64

                                feature_map = h.permute(0, 2, 1).contiguous()  # (B, C, N)
                                feature_map = feature_map.view(batch_size, dim, h_w, h_w)  # (B, C, H, W)
                                intermediate_hidden_states_for_bifpn.append(feature_map)
                        else:
                            print(
                                f"Warning: Requested ViT layer index {i} is out of range (0-{len(all_hidden_states) - 1}).")

        # Original CIDE conditioning signal generation
        processed_logits = self.fc_classes(final_logits)
        activated_weights = self.softmax(processed_logits)
        bifpn_hidden_states = self.bifpn(intermediate_hidden_states_for_bifpn)
        pooled = []
        for px in bifpn_hidden_states:
            gap = F.adaptive_avg_pool2d(px, (1, 1))         # (B, C, 1, 1)
            flat = gap.view(gap.size(0), -1)                # (B, C)
            pooled.append(flat)
        pooled.append(activated_weights)
        bifpn_gap_concat = torch.cat(pooled, dim=1)

        # cide_scene_embedding = activated_weights @ self.embeddings.weight
        combined_features = self.fc_embedding(bifpn_gap_concat)
        activated_features_weights = self.activations(combined_features)
        cide_scene_embedding = activated_features_weights @ self.embeddings.weight
        cide_adapted_embedding = self.adapter(cide_scene_embedding, self.gamma)  # [B, 1, D]

        if self.testing:
            return cide_adapted_embedding, bifpn_hidden_states, bifpn_gap_concat, intermediate_hidden_states_for_bifpn
        return cide_adapted_embedding

class DummyArgs:
    no_of_classes = 100
    img_size = 224


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)  # Made image path required
    parser.add_argument("--vit_type", default="microsoft/swinv2-base-patch4-window16-256")
    parser.add_argument("--act", default="sigmoid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="outputs", type=Path)
    # Specify which layers to extract (0-indexed, including embedding layer)
    # For ViT-base (12 layers + embedding layer = 13 total hidden_states)
    # e.g., layers 3, 6, 9, 12 (last layer before classifier)
    parser.add_argument("--vit_layers", type=int, nargs='+', default=[0,1,2])
    opt = parser.parse_args()

    x = load_image(opt.image, image_size=DummyArgs.img_size).to(opt.device)  # Use consistent image size

    model = CIDE(
        DummyArgs(),
        emb_dim=768,
        vit_type=opt.vit_type,
        activations=opt.act,
        output_vit_layers_indices=opt.vit_layers  # Pass the layer indices
    ).to(opt.device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        cide_emb, bifpn_hidden, bifpn_gap, intermediate_layers = model(x)

    print("CIDE Scene embedding shape:", cide_emb.shape)
    print("BiFPN GAP shape:", bifpn_gap.shape)

    if intermediate_layers:
        print(f"Extracted {len(intermediate_layers)} ViT intermediate layers:")
        for i, layer_output in enumerate(intermediate_layers):
            print(f"  Layer {opt.vit_layers[i]} output shape: {layer_output.shape}")

    if bifpn_hidden:
        print(f"Extracted {len(bifpn_hidden)} BiFPN intermediate layers:")
        for i, layer_output in enumerate(bifpn_hidden):
            print(f"  BiFPN P{i} output shape: {layer_output.shape}")

    opt.out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{opt.vit_type.replace('/', '_')}_cide_scene_emb.npy"
    np.save(opt.out_dir / fn, cide_emb.squeeze(1).cpu().numpy())
    print(f"✅ CIDE embedding saved to {opt.out_dir / fn}")


if __name__ == "__main__":
    _cli()

