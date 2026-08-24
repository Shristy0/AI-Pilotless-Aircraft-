from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _require_deps() -> tuple[object, object, object]:
    try:
        import cv2  # type: ignore
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception:
        print("Optional dependencies missing. Install with: pip install -r requirements-optional.txt")
        raise SystemExit(1)
    return cv2, torch, nn


def _make_sample(cv2, label: int, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    if label == 1:
        cv2.line(img, (size // 3, size - 4), (size // 2 - 4, 4), 200, 2)
        cv2.line(img, (2 * size // 3, size - 4), (size // 2 + 4, 4), 200, 2)
    return img


def build_dataset(cv2, n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n_samples, 1, 64, 64), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        label = 1 if i % 2 == 0 else 0
        img = _make_sample(cv2, label)
        X[i, 0] = img / 255.0
        y[i] = label
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def main() -> None:
    cv2, torch, nn = _require_deps()
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = build_dataset(cv2, n_samples=400)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(6):
        optimizer.zero_grad()
        preds = model(X_train_t).squeeze()
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).squeeze().numpy()
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == y_test).mean())

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "vision_demo_metrics.json"
    out_path.write_text(json.dumps({"vision_demo_accuracy": round(acc, 4)}, indent=2), encoding="utf-8")
    print(f"Saved vision demo metrics to {out_path}")


if __name__ == "__main__":
    main()
