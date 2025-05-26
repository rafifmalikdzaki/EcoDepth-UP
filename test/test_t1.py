import sys
sys.path.append("..")
from dataset import DepthDataset
import json
from torch.utils.data import DataLoader
from model_t2 import EcoDepth
import lightning as L
import torch
from utils import download_model
from torchinfo import summary
class Args:
    def __init__(self):
        with open("test_config.json", "r") as f:
            config = json.load(f)
        for n, v in config.items():
           setattr(self, n, v)

args = Args()

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


test_dataset = DepthDataset(
    args=args,
    is_train=False,
    filenames_path=args.test_filenames_path,
    data_path=args.test_data_path,
    depth_factor=args.test_depth_factor
)

test_loader = DataLoader(test_dataset, num_workers=args.num_workers)

trainer = L.Trainer(logger=False)

trainer.test(model, dataloaders=test_loader)