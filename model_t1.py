import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from timm.models.layers import trunc_normal_
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# 🔗  CIDE comes from the dedicated "test_cide.py" module so that the ViT backbone
#     can be swapped freely at run‑time via Args.
from test_cide import CIDE  # noqa: E402

from utils import pad, unpad, silog
from optimizer import get_optimizer
from metrics import compute_metrics
from utils import eigen_crop, garg_crop, custom_crop, no_crop

# ────────────────────────────────────────────────────────────────
#  Decoder hyper‑parameters (kept identical to the original file)
# ────────────────────────────────────────────────────────────────
NUM_DECONV = 3
NUM_FILTERS = [32, 32, 32]
DECONV_KERNELS = [2, 2, 2]


# ============================================================================
#  ENCODER – integrates CIDE with user‑selectable ViT backbone
# ============================================================================
class EcoDepthEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 1024,
        ldm_prior: list[int] = None,
        sd_path: str = None,
        emb_dim: int = 768,
        args=None,
        train_from_scratch: bool = False,
        bcb= "facebook/convnext-base-224"
    ) -> None:
        super().__init__()

        if ldm_prior is None:
            ldm_prior = [320, 640, 1280 + 1280]

        self.args = args

        # ── simple strided convs to down‑sample UNet feature maps ────────────
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        if train_from_scratch:
            self.apply(self._init_weights)

        # ── CIDE with **dynamic** ViT backbone selection ────────────────────
        self.cide_module = CIDE(
            args,
            emb_dim,
            train_from_scratch,
            vit_type=getattr(args, "vit_type", bcb),
            use_timm=getattr(args, "use_timm", False),
        )

        # ── Stable‑Diffusion UNet encoder (latent diffusion) ────────────────
        self.config = OmegaConf.load("./v1-inference.yaml")
        unet_config = self.config.model.params.unet_config
        first_stage_config = self.config.model.params.first_stage_config

        if train_from_scratch and sd_path is None:
            sd_path = "./checkpoints/v1-5-pruned-emaonly.ckpt"
            # unet_config.params.ckpt_path = sd_path  # uncomment if needed

        self.unet = instantiate_from_config(unet_config)
        self.encoder_vq = instantiate_from_config(first_stage_config)
        del self.encoder_vq.decoder
        del self.unet.out

        for p in self.encoder_vq.parameters():
            p.requires_grad = False

    # ---------------------------------------------------------------------
    #  Initialise freshly created layers when training from scratch
    # ---------------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ---------------------------------------------------------------------
    #  Forward pass
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB image to SD latent space (frozen VQ‑VAE)
        with torch.no_grad():
            latents = (
                self.encoder_vq.encode(x).mode().detach() * self.config.model.params.scale_factor
            )

        # Scene‑conditioning embedding from CIDE (frozen ViT backbone)
        conditioning_scene_embedding = self.cide_module(x)

        # SD UNet forward at timestep t=1 (or any placeholder)
        t = torch.ones((x.shape[0],), device=x.device, dtype=torch.long)
        outs = self.unet(latents, t, context=conditioning_scene_embedding)

        # Fast, lightweight feature aggregation
        feats = [
            outs[0],
            outs[1],
            torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1),
        ]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)


# ============================================================================
#  MAIN DEPTH‑ESTIMATION MODEL
# ============================================================================
class EcoDepth(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_depth = args.max_depth

        embed_dim = 192
        channels_in = embed_dim * 8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(
            out_dim=channels_in,
            args=args,
            train_from_scratch=args.train_from_scratch,
        )
        self.decoder = Decoder(channels_in, channels_out, args)

        # Evaluation crop selection
        self.eval_crop = {
            "eigen": eigen_crop,
            "garg": garg_crop,
            "custom": custom_crop,
            "none": no_crop,
        }.get(args.eval_crop, no_crop)

        # Only support fine‑tuning for now
        assert not args.train_from_scratch, "Training from scratch not yet supported."

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, padding=1),
        )
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x ∈ [0, 1]; convert to [-1, 1]
        x = x * 2.0 - 1.0
        x, padding = pad(x, 64)
        conv_feats = self.encoder(x)
        out = self.decoder([conv_feats])
        out = unpad(out, padding)
        depth_logits = self.last_layer_depth(out)
        return torch.sigmoid(depth_logits) * self.max_depth

    # ------------------------------------------------------------------
    #  Lightning boiler‑plate
    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        image, depth = batch["image"], batch["depth"]
        pred = self(image)
        return silog(pred, depth)

    def _shared_eval_step(self, batch, prefix):
        image, depth = batch["image"], batch["depth"]
        depth = self.eval_crop(depth)
        image_concat = torch.cat([image, image.flip(-1)])
        pred_concat = self(image_concat)
        pred = ((pred_concat[0] + pred_concat[1].flip(-1)) / 2).unsqueeze(0)
        loss = silog(pred, depth)
        metrics = compute_metrics(pred, depth, self.args)
        self.log(f"{prefix}_loss", loss)
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, _):
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        return self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        return get_optimizer(self, self.args)


# ============================================================================
#  U‑shaped DECODER (unchanged except for style tweaks)
# ============================================================================
class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, args):
        super().__init__()
        self.in_channels = in_channels
        self.args = args

        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV,
            NUM_FILTERS,
            DECONV_KERNELS,
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(NUM_FILTERS[-1], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)
        out = self.up(out)
        out = self.up(out)
        return out

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_planes,
                        planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            in_planes = planes
        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        if deconv_kernel == 4:
            return 4, 1, 0
        if deconv_kernel == 3:
            return 3, 1, 1
        if deconv_kernel == 2:
            return 2, 0, 0
        raise ValueError(f"Unsupported deconv kernel size: {deconv_kernel}")

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


