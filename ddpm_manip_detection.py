# contrastive_manip_detect.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

# ============================================================
# CONTRASTIVE ENCODER
# ============================================================

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, emb_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        z = F.normalize(z, dim=1)
        return z


# ============================================================
# CLASSIFIER
# ============================================================

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.fc(z).squeeze(1)
        return logits


# ============================================================
# IMAGE PREPROCESS (EĞİTİMDEKİYLE AYNI)
# ============================================================

def load_image_for_contrastive(path):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor


# ============================================================
# SAFE CKPT LOADER
# ============================================================

def safe_load(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model_dict = model.state_dict()
    new_state = {}

    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state[k] = v
        else:
            print(f"[SKIP] {k} (shape mismatch)")

    model_dict.update(new_state)
    model.load_state_dict(model_dict)
    print(f"[OK] Loaded {len(new_state)} weights")
    return model


# ============================================================
# ANA FONKSİYON — TEK SATIRDA KULLANIM
# ============================================================

@torch.no_grad()
def detect_manipulated(model_path, image_path):
    """
    Tek görüntüden REAL/FAKE tahmini döndürür.
    return:
        {
            "class_id": 0 veya 1,
            "class_name": "REAL" veya "FAKE",
            "prob_fake": float
        }
    """

    encoder = ConvEncoder(in_channels=3, base_channels=32, emb_dim=128)
    model = DeepFakeClassifier(encoder, emb_dim=128).to(DEVICE)

    model = safe_load(model, model_path)
    model.eval()

    img = load_image_for_contrastive(image_path).to(DEVICE)

    logits = model(img)[0]
    prob_fake = torch.sigmoid(logits).item()
    cls = 1 if prob_fake >= 0.5 else 0

    return {
        "class_id": cls,
        "class_name": "FAKE" if cls == 0 else "REAL",
        "prob_fake": prob_fake
    }


# ============================================================
# Opsiyonel CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    out = detect_manipulated(args.model, args.image)
    print(out)
