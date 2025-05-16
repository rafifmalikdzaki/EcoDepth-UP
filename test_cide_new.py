import torch
import torch.nn as nn
from transformers import (
    ViTConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import timm
from torchvision import transforms as T
from einops import repeat
import numpy as np

def pad_to_make_square(x):
    y = 255*((x+1)/2)
    y = torch.permute(y, (0,2,3,1))
    bs, _, h, w = x.shape
    if w>h:
        patch = torch.zeros(bs, w-h, w, 3).to(x.device)
        y = torch.cat([y, patch], axis=1)
    else:
        patch = torch.zeros(bs, h, h-w, 3).to(x.device)
        y = torch.cat([y, patch], axis=2)
    return y.to(torch.int)


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts

class CIDE(nn.Module):
    def __init__(self, args, emb_dim, train_from_scratch, vit_type="google/vit-base-patch16-224", use_timm=False):
        super().__init__()
        self.args = args
        self.dim = emb_dim
        self.use_timm = use_timm
        self.vit_type = vit_type

        if self.use_timm:
            self.vit_model = timm.create_model(vit_type, pretrained=not train_from_scratch, num_classes=1000)
            self.vit_model.eval()
            for p in self.vit_model.parameters():
                p.requires_grad = False
            self.vit_processor = T.Compose([
                T.Resize((224, 224)),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        else:
            try:
                self.vit_processor = AutoImageProcessor.from_pretrained(vit_type, resume_download=True)
                self.vit_model = AutoModelForImageClassification.from_pretrained(vit_type, resume_download=True)
            except:
                vit_config = ViTConfig(num_labels=1000)
                self.vit_model = AutoModelForImageClassification.from_config(vit_config)
            for param in self.vit_model.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 400),
            nn.GELU(),
            nn.Linear(400, args.no_of_classes)
        )
        self.m = nn.Softmax(dim=1)
        self.embeddings = nn.Parameter(torch.randn(args.no_of_classes, emb_dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=emb_dim)
        self.gamma = nn.Parameter(torch.ones(emb_dim) * 1e-4)

    def forward(self, x):
        y = pad_to_make_square(x)

        with torch.no_grad():
            if self.use_timm:
                y = (y + 1.0) / 2.0  # convert from [-1, 1] to [0, 1]
                y = self.vit_processor(y)
                vit_logits = self.vit_model(y)
                if isinstance(vit_logits, tuple):
                    vit_logits = vit_logits[0]
            else:
                inputs = self.vit_processor(images=y, return_tensors="pt").to(x.device)
                vit_outputs = self.vit_model(**inputs)
                vit_logits = vit_outputs.logits

        class_probs = self.fc(vit_logits)
        class_probs = self.m(class_probs)

        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma)

        return conditioning_scene_embedding

from PIL import Image
import argparse
import os

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),           # Converts to [0, 1]
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),  # Normalize to [-1, 1] like training
    ])
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

class Args:
    no_of_classes = 100
    train_from_scratch = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--vit_type', type=str, default="google/vit-base-patch16-224", help='Backbone name')
    parser.add_argument('--use_timm', action='store_true', help='Use timm model instead of HuggingFace')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--out_dir', type=str, default="outputs", help='Directory to save embeddings')
    args_cli = parser.parse_args()

    # Load image
    img_tensor = load_image(args_cli.image).to(args_cli.device)

    # Initialize CIDE
    model = CIDE(
        args=Args(),
        emb_dim=768,
        train_from_scratch=Args.train_from_scratch,
        vit_type=args_cli.vit_type,
        use_timm=args_cli.use_timm
    ).to(args_cli.device)
    with torch.no_grad():
        out = model(img_tensor)

    print(f"Scene embedding shape: {out.shape}")

    # Save embedding
    os.makedirs(args_cli.out_dir, exist_ok=True)
    out_np = out.squeeze(1).cpu().numpy()  # shape [1, D]
    model_name = args_cli.vit_type.replace("/", "_")
    save_path = os.path.join(args_cli.out_dir, f"{model_name}_embedding.npy")
    np.save(save_path, out_np)
    print(f"✅ Saved embedding to: {save_path}")
