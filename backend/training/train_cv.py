"""
training/train_cv.py — Fine-tune YOLOv8 on dark-pattern UI dataset
===================================================================
Trains a YOLOv8n model to detect visual dark-pattern UI elements.
Expects a Roboflow-exported YOLO-format dataset in data/cv_dataset/.

Usage:
    python training/train_cv.py [--epochs 50] [--imgsz 640]

Dataset sources:
  - Roboflow Universe: search "dark pattern" or "UI elements"
    https://universe.roboflow.com/
  - Export as "YOLOv8" format → extracts to data/cv_dataset/
"""

from __future__ import annotations
import argparse
import json
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for dark-pattern UI detection")
    parser.add_argument("--model",      default="yolov8n.pt",  help="Base YOLO model")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--data-dir",   default="data/cv_dataset")
    parser.add_argument("--output-dir", default="checkpoints/cv")
    parser.add_argument("--device",     default="",  help="'' for auto, '0' for GPU 0, 'cpu' for CPU")
    return parser.parse_args()


def create_dataset_yaml(data_dir: Path, yaml_path: Path):
    """
    Create a YOLO dataset YAML if one doesn't exist.
    Assumes Roboflow export structure:
      data/cv_dataset/
        train/images/, train/labels/
        valid/images/, valid/labels/
        test/images/,  test/labels/
    """
    if yaml_path.exists():
        print(f"[train_cv] Using existing dataset YAML: {yaml_path}")
        return

    # Auto-detect directories
    train_dir = data_dir / "train" / "images"
    valid_dir = data_dir / "valid" / "images"
    test_dir  = data_dir / "test"  / "images"

    config = {
        "path": str(data_dir.resolve()),
        "train": "train/images" if train_dir.exists() else "images",
        "val":   "valid/images" if valid_dir.exists() else "images",
        "test":  "test/images"  if test_dir.exists()  else None,
        "nc": 7,
        "names": [
            "countdown_timer",
            "prechecked_box",
            "shame_button",
            "hidden_link",
            "ad_container",
            "scarcity_badge",
            "consent_banner",
        ]
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[train_cv] Created dataset YAML: {yaml_path}")


def create_synthetic_dataset(data_dir: Path):
    """
    Create a minimal synthetic dataset for testing the training pipeline.
    In production, replace with a real annotated UI screenshot dataset.
    """
    import numpy as np
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[train_cv] Pillow not installed — cannot create synthetic dataset")
        return

    print("[train_cv] Creating synthetic demonstration dataset...")

    for split in ("train", "valid"):
        img_dir   = data_dir / split / "images"
        label_dir = data_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        n_images = 20 if split == "train" else 5

        for i in range(n_images):
            # Create a simple mock UI screenshot
            img = Image.new("RGB", (640, 480), color=(245, 245, 250))
            draw = ImageDraw.Draw(img)

            # Draw a fake countdown timer box (class 0 = countdown_timer)
            x1, y1, x2, y2 = 50, 50, 250, 100
            draw.rectangle([x1, y1, x2, y2], fill=(220, 50, 50), outline=(180, 0, 0))
            draw.text((60, 65), f"Offer ends in 02:3{i % 10}:00", fill=(255, 255, 255))

            # Draw a fake ad container (class 4 = ad_container)
            ax1, ay1, ax2, ay2 = 300, 200, 600, 350
            draw.rectangle([ax1, ay1, ax2, ay2], fill=(255, 250, 200), outline=(200, 180, 0))
            draw.text((310, 265), "Sponsored | Partner Content", fill=(100, 80, 0))

            img.save(img_dir / f"img_{i:04d}.jpg", quality=85)

            # Write YOLO annotation (normalized xywh)
            W, H = 640, 480
            with open(label_dir / f"img_{i:04d}.txt", "w") as f:
                # countdown_timer (class 0)
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                # ad_container (class 4)
                cx = ((ax1 + ax2) / 2) / W
                cy = ((ay1 + ay2) / 2) / H
                bw = (ax2 - ax1) / W
                bh = (ay2 - ay1) / H
                f.write(f"4 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[train_cv] Synthetic dataset created at {data_dir}")


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[train_cv] ultralytics not installed — pip install ultralytics")
        return

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check / create dataset ─────────────────────────────────────
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"[train_cv] No dataset found at {data_dir}")
        print("[train_cv] Creating synthetic demo dataset (replace with real annotated data)")
        create_synthetic_dataset(data_dir)

    yaml_path = data_dir / "dataset.yaml"
    create_dataset_yaml(data_dir, yaml_path)

    # ── Load model ─────────────────────────────────────────────────
    print(f"\n[train_cv] Loading base model: {args.model}")
    model = YOLO(args.model)

    # ── Train ──────────────────────────────────────────────────────
    print(f"[train_cv] Starting training ({args.epochs} epochs, imgsz={args.imgsz})...")
    results = model.train(
        data     = str(yaml_path),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device or None,
        project  = str(output_dir),
        name     = "dark_pattern_detector",
        patience = 10,         # early stopping
        save     = True,
        plots    = True,
        verbose  = True,
        augment  = True,      # mosaic, flips, color jitter
        degrees  = 0,         # no rotation (UI screenshots are upright)
        fliplr   = 0,         # no horizontal flip (UI is directional)
        mosaic   = 0.5,
        mixup    = 0.1,
    )

    # ── Copy best weights ──────────────────────────────────────────
    import shutil
    best_weights = output_dir / "dark_pattern_detector" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, output_dir / "best.pt")
        print(f"\n✅ Training complete! Best weights → {output_dir / 'best.pt'}")
    else:
        print(f"\n[train_cv] Training complete (weights at {output_dir})")

    # ── Validate ───────────────────────────────────────────────────
    print("\n[train_cv] Running validation...")
    val_results = model.val(data=str(yaml_path))
    metrics = {
        "mAP50":     float(val_results.box.map50),
        "mAP50-95":  float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall":    float(val_results.box.mr),
    }
    print(f"[train_cv] Validation: {metrics}")

    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
