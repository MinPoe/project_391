"""Training script for the Behavioural Cloning model.

Usage:
    python bc/train.py --data processed/processed_simulator/data.csv \
                       --epochs 100 --batch-size 256 --lr 1e-3 \
                       --out bc/bc_model.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import BCNet


def load_data(csv_path: str):
    """Load CSV and split into LiDAR features and action labels.

    Uses chunked reading for large files to avoid memory issues.
    """
    # Read header to identify columns
    header = pd.read_csv(csv_path, nrows=0)
    lidar_cols = sorted(
        [c for c in header.columns if c.startswith("lidar_")],
        key=lambda c: int(c.split("_")[1]),
    )
    use_cols = lidar_cols + ["steering_angle", "speed"]

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=use_cols, chunksize=100_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    X = df[lidar_cols].values.astype(np.float32)
    y = df[["steering_angle", "speed"]].values.astype(np.float32)
    return X, y, len(lidar_cols)


def make_loaders(X, y, train_ratio=0.8, batch_size=256):
    """80/20 train/val split, returns DataLoaders."""
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(
        torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    )
    val_ds = TensorDataset(
        torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def train(args):
    X, y, num_lidar = load_data(args.data)
    print(f"Loaded {len(X)} samples, {num_lidar} LiDAR rays")

    train_loader, val_loader = make_loaders(X, y, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BCNet(num_lidar_rays=num_lidar).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved best model (val_loss={best_val_loss:.6f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC model")
    parser.add_argument("--data", required=True, help="Path to processed CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="bc/bc_model.pth", help="Output model path")
    args = parser.parse_args()
    train(args)
