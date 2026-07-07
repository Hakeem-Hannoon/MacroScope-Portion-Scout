"""The scale-conditioned mass regressor (roadmap P3) — the model this project
trains from scratch, because nothing public does it (verified 2026-07,
docs/MODELS.md §4).

Architecture
============
Input:  RGB crop of one food region, metrically rectified (MATH.md §3.1),
        plus a conditioning vector of measured physics:
          [log(area_m2), height_m (or -1), has_height, scale-source one-hot(5)]
Output: log(mass_g) for the region (and an auxiliary kcal head).

    crop ──► CNN backbone (timm mobilenetv3_large_100, pretrained) ──► h ∈ R^960
    physics ──► MLP ──► (γ, β)          FiLM conditioning
    h' = γ ⊙ h + β ──► MLP head ──► [log_mass, log_kcal]

Why this shape and no recurrence: the input is a single image with scalar
side-information — a convolutional (or hybrid ViT) encoder plus feature-wise
conditioning is the right inductive bias. RNNs model sequences; there is no
sequence here. FiLM (Perez et al., 2018) lets the measured scale multiply
visual features, which mirrors the physics: doubling the metric area should
roughly double predicted mass for the same appearance, and the network learns
exactly that coupling instead of guessing scale from texture. If multi-frame
captures land later (video sweep), attention pooling over frame embeddings is
the upgrade path; recurrence stays unnecessary.

The backbone is exchangeable via --backbone (fastvit_t8 for ANE-friendly
inference, efficientnet-lite for LiteRT). Everything exports through
export/export_coreml.py.

Data: a manifest CSV produced by data/prepare_nutrition5k.py with columns
  image_path, area_m2, height_m, mass_g, kcal, split
Nutrition5k is CC BY 4.0 — commercially usable with attribution.

GPU job (Google Labs H100): ~1-2 h for 50 epochs at batch 128. The real cost
is the 181 GB dataset download and the manifest extraction, both one-time.

    python train/mass_regressor_nutrition5k.py --manifest out/n5k-manifest.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCALE_SOURCES = ["lidar", "ruler", "reference_object", "stated", "none"]
COND_DIM = 3 + len(SCALE_SOURCES)


class MealRegionDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 256, train: bool = True):
        self.rows = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image = Image.open(row.image_path).convert("RGB").resize(
            (self.image_size, self.image_size)
        )
        x = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        if self.train and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])

        height = float(row.get("height_m", -1) or -1)
        has_height = 1.0 if height > 0 else 0.0
        source = str(row.get("scale_source", "lidar"))
        one_hot = [1.0 if source == s else 0.0 for s in SCALE_SOURCES]
        cond = torch.tensor(
            [math.log(max(row.area_m2, 1e-6)), max(height, -1.0), has_height, *one_hot],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [math.log(max(row.mass_g, 1.0)), math.log(max(row.kcal, 1.0))],
            dtype=torch.float32,
        )
        return x, cond, target


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(), nn.Linear(128, 2 * feature_dim)
        )

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return features * (1 + gamma) + beta


class ScaleConditionedMassRegressor(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_large_100"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        self.film = FiLM(COND_DIM, feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 2)
        )

    def forward(self, image: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.backbone(image)
        return self.head(self.film(h, cond))


def mape(pred_log: torch.Tensor, true_log: torch.Tensor) -> float:
    pred = torch.exp(pred_log)
    true = torch.exp(true_log)
    return (torch.abs(pred - true) / true).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backbone", default="mobilenetv3_large_100")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="out/mass-regressor.pt")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest.split == "train"]
    test_df = manifest[manifest.split == "test"]
    train_loader = DataLoader(
        MealRegionDataset(train_df, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        MealRegionDataset(test_df, train=False),
        batch_size=args.batch_size,
        num_workers=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScaleConditionedMassRegressor(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    best_mape = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for image, cond, target in train_loader:
            image, cond, target = image.to(device), cond.to(device), target.to(device)
            loss = loss_fn(model(image, cond), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        mass_mapes, kcal_mapes = [], []
        with torch.no_grad():
            for image, cond, target in test_loader:
                pred = model(image.to(device), cond.to(device)).cpu()
                mass_mapes.append(mape(pred[:, 0], target[:, 0]))
                kcal_mapes.append(mape(pred[:, 1], target[:, 1]))
        mass_mape = float(np.mean(mass_mapes))
        kcal_mape = float(np.mean(kcal_mapes))
        print(f"epoch {epoch + 1}: mass MAPE {mass_mape:.3f}, kcal MAPE {kcal_mape:.3f}")
        if mass_mape < best_mape:
            best_mape = mass_mape
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"backbone": args.backbone, "state_dict": model.state_dict()},
                args.output,
            )
    print(f"best mass MAPE: {best_mape:.3f} → {args.output}")
    print("Benchmarks to beat (Nutrition5k, docs/MODELS.md): 26.1% RGB / 16.5% depth.")


if __name__ == "__main__":
    main()
