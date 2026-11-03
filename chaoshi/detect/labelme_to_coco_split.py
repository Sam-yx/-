"""
Convert Labelme JSON annotations to COCO format and split dataset by ratio.

Default behavior:
- Read Labelme JSON files from detect_dataset
- Resolve images from my_photos/Baishi
- Create COCO files:
  - detect_dataset/annotations/instances_train.json
  - detect_dataset/annotations/instances_val.json
  - detect_dataset/annotations/instances_test.json
- Copy images into:
  - detect_dataset/images/train
  - detect_dataset/images/val
  - detect_dataset/images/test

Usage examples:
  python detect_dataset/labelme_to_coco_split.py
  python detect_dataset/labelme_to_coco_split.py --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Config:
    labelme_dir: Path
    image_dir: Path
    output_dir: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Compute polygon area with shoelace formula."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def points_to_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    y_min = min(ys)
    w = max(xs) - x_min
    h = max(ys) - y_min
    return x_min, y_min, w, h


def normalize_points(raw_points: List[List[float]]) -> List[Tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in raw_points if len(p) >= 2]


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Map image file names to absolute paths for fast lookup."""
    index: Dict[str, Path] = {}
    duplicates = set()

    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            name = p.name
            if name in index:
                duplicates.add(name)
            index[name] = p

    if duplicates:
        print(f"[WARN] Found duplicated image names ({len(duplicates)}). Last one wins.")
    return index


def discover_labelme_jsons(labelme_dir: Path) -> List[Path]:
    return sorted([p for p in labelme_dir.glob("*.json") if p.is_file()])


def build_full_coco(cfg: Config) -> Tuple[dict, Dict[int, Path]]:
    json_files = discover_labelme_jsons(cfg.labelme_dir)
    if not json_files:
        raise FileNotFoundError(f"No Labelme JSON files found in: {cfg.labelme_dir}")

    image_index = build_image_index(cfg.image_dir)

    categories: List[dict] = []
    category_name_to_id: Dict[str, int] = {}

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id_to_path: Dict[int, Path] = {}

    ann_id = 1
    image_id = 1

    missing_images = 0
    used_json = 0

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_path_raw = str(data.get("imagePath", ""))
        image_name = Path(image_path_raw).name if image_path_raw else ""
        if not image_name:
            print(f"[WARN] No imagePath in JSON, skipped: {jf.name}")
            continue

        abs_img_path = image_index.get(image_name)
        if abs_img_path is None:
            missing_images += 1
            print(f"[WARN] Image not found for JSON {jf.name}: {image_name}")
            continue

        width = int(data.get("imageWidth", 0) or 0)
        height = int(data.get("imageHeight", 0) or 0)
        if width <= 0 or height <= 0:
            # Fallback: lightweight read using OpenCV if dimensions missing.
            try:
                import cv2  # Local import to avoid hard dependency if not needed.

                img = cv2.imread(str(abs_img_path))
                if img is not None:
                    height, width = img.shape[:2]
            except Exception:
                pass

        if width <= 0 or height <= 0:
            print(f"[WARN] Invalid image size, skipped: {abs_img_path}")
            continue

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )
        image_id_to_path[image_id] = abs_img_path

        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if not label:
                continue

            if label not in category_name_to_id:
                cid = len(category_name_to_id) + 1
                category_name_to_id[label] = cid
                categories.append({"id": cid, "name": label, "supercategory": "none"})

            points = normalize_points(shape.get("points", []))
            if len(points) < 2:
                continue

            shape_type = str(shape.get("shape_type", "")).lower().strip()
            x, y, w, h = points_to_bbox(points)

            # COCO bbox must be non-negative. Skip degenerate boxes.
            if w <= 0 or h <= 0:
                continue

            segmentation: List[List[float]] = []
            area: float

            if shape_type == "polygon" and len(points) >= 3:
                flat = [v for pt in points for v in pt]
                segmentation = [flat]
                area = polygon_area(points)
                if area <= 0:
                    area = w * h
            elif shape_type == "rectangle" and len(points) >= 2:
                # Keep segmentation compatible with COCO consumers.
                x1, y1 = points[0]
                x2, y2 = points[1]
                rect = [x1, y1, x2, y1, x2, y2, x1, y2]
                segmentation = [rect]
                area = w * h
            else:
                # Unknown shape: still export bbox to maximize compatibility.
                area = w * h

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_name_to_id[label],
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            ann_id += 1

        used_json += 1
        image_id += 1

    print(f"[INFO] Labelme JSON found: {len(json_files)}")
    print(f"[INFO] JSON converted: {used_json}")
    if missing_images:
        print(f"[WARN] JSON skipped due to missing images: {missing_images}")
    print(f"[INFO] COCO images: {len(coco['images'])}")
    print(f"[INFO] COCO annotations: {len(coco['annotations'])}")
    print(f"[INFO] COCO categories: {len(coco['categories'])}")

    return coco, image_id_to_path


def split_images(images: List[dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[dict]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = images[:]
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def build_split_coco(full_coco: dict, split_images_list: List[dict]) -> dict:
    image_ids = {img["id"] for img in split_images_list}
    anns = [ann for ann in full_coco["annotations"] if ann["image_id"] in image_ids]
    return {
        "images": split_images_list,
        "annotations": anns,
        "categories": full_coco["categories"],
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def copy_split_images(split_name: str, split_images_list: List[dict], image_id_to_path: Dict[int, Path], output_images_root: Path) -> None:
    split_dir = output_images_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for image_info in split_images_list:
        src = image_id_to_path[image_info["id"]]
        dst = split_dir / image_info["file_name"]
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir.parent / "detect_dataset"
    default_images = script_dir.parent / "my_photos" / "Baishi"
    default_json_dir = script_dir.parent / "detect_dataset"

    parser = argparse.ArgumentParser(description="Labelme to COCO with train/val/test split")
    parser.add_argument("--labelme-dir", type=Path, default=script_dir, help="Directory containing Labelme JSON files (legacy)")
    parser.add_argument("--json-dir", type=Path, default=default_json_dir, help="Directory containing Labelme JSON files (preferred). Defaults to detect_dataset in workspace root")
    parser.add_argument("--image-dir", type=Path, default=default_images, help="Directory containing source images")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output root directory for COCO files")

    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Prefer explicit --json-dir when provided, otherwise fall back to --labelme-dir.
    selected_labelme_dir = args.json_dir if getattr(args, "json_dir", None) else args.labelme_dir

    cfg = Config(
        labelme_dir=selected_labelme_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("=" * 60)
    print("Labelme -> COCO conversion with dataset split")
    print("=" * 60)
    print(f"[INFO] labelme_dir: {cfg.labelme_dir}")
    print(f"[INFO] image_dir:   {cfg.image_dir}")
    print(f"[INFO] output_dir:  {cfg.output_dir}")
    print(
        f"[INFO] ratios: train={cfg.train_ratio} val={cfg.val_ratio} "
        f"test={cfg.test_ratio} seed={cfg.seed}"
    )

    full_coco, image_id_to_path = build_full_coco(cfg)
    split_map = split_images(
        full_coco["images"],
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        cfg.seed,
    )

    ann_root = cfg.output_dir / "annotations"
    img_root = cfg.output_dir / "images"

    for split_name in ["train", "val", "test"]:
        split_coco = build_split_coco(full_coco, split_map[split_name])
        ann_file = ann_root / f"instances_{split_name}.json"
        save_json(ann_file, split_coco)
        copy_split_images(split_name, split_map[split_name], image_id_to_path, img_root)

        print(
            f"[DONE] {split_name:<5} "
            f"images={len(split_coco['images']):<4} "
            f"annotations={len(split_coco['annotations']):<5} "
            f"file={ann_file}"
        )

    print("[SUCCESS] Conversion completed.")


if __name__ == "__main__":
    main()
