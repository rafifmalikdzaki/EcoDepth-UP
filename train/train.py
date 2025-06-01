import sys
sys.path.append("..")
from dataset import DepthDataset
import json
from torch.utils.data import DataLoader
from model_t2 import EcoDepth
from lightning.pytorch.loggers import WandbLogger
import wandb
from ldm.modules.diffusionmodules import util as ldm_util
# keep reference to the original backward
_orig_backward = ldm_util.CheckpointFunction.backward

def _safe_backward(ctx, *grad_outputs):
    # delegate to the real backward but tolerate unused tensors
    return _orig_backward(ctx, *grad_outputs, allow_unused=True)

ldm_util.CheckpointFunction.backward = _safe_backward

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from utils import download_model



class Args:
    def __init__(self):
        with open("train_config.json", "r") as f:
            config = json.load(f) 
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

wandb_logger = WandbLogger(
    project="EcoDepth",  # Change to your project name
    name=args.experiment_name if hasattr(args, "experiment_name") else None,
    config=vars(args),   # Logs your Args as hyperparameters
)

model = EcoDepth(args)

if args.ckpt_path == "":
    model_str = f"weights_{args.scene}.ckpt"
    download_model(model_str)
    args.ckpt_path = f"../checkpoints/{model_str}"

# model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["state_dict"])
# --- robust checkpoint loader ---------------------------------------------
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)

state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model_state = model.state_dict()
compatible = {}

for k, v in state.items():
    # 1) rename legacy embedding_adapter -> adapter
    if "embedding_adapter" in k:
        k = k.replace("embedding_adapter", "adapter")

    # 2) discard the old fc.* MLP — not used any more
    if ".fc." in k and "cide_module.adapter" not in k:
        continue

    # 3) keep only if the tensor shape matches the current model
    if k in model_state and v.shape == model_state[k].shape:
        compatible[k] = v
    else:
        print(f"↪︎  skipped {k}  {tuple(v.shape)}")

missing, unexpected = model.load_state_dict(compatible, strict=False)
print(f"✓ loaded {len(compatible)} tensors "
      f"(ignored {len(state)-len(compatible)})")
# ---------------------------------------------------------------------------

modules_to_not_freeze = ["decoder", "cide"]

# Freezing Layers

print("Freezing Layers")
for param in model.parameters():
    param.requires_grad = False


print("Unfreeze Layer Decoder and CIDE")
for param in model.decoder.parameters():
    param.requires_grad = True

for param in model.encoder.cide_module.parameters():
    param.requires_grad = True


train_dataset = DepthDataset(
    args=args, 
    is_train=True, 
    filenames_path=args.train_filenames_path, 
    data_path=args.train_data_path, 
    depth_factor=args.train_depth_factor
)

train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size)

val_dataset = DepthDataset(
    args=args, 
    is_train=False, 
    filenames_path=args.val_filenames_path, 
    data_path=args.val_data_path, 
    depth_factor=args.val_depth_factor
)

val_loader = DataLoader(val_dataset, num_workers=8)

checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    save_last=True,
    monitor="val_loss",
    save_weights_only=True,
)

trainer = L.Trainer(
    logger=wandb_logger,
    max_epochs=args.epochs,
    val_check_interval=args.val_check_interval,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=4
)

trainer.fit(
    model=model, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader,
)
