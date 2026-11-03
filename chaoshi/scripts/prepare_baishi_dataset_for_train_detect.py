from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml


# ------------------------
# Editable config
# ------------------------
SOURCE_ROOT = Path(r"C:/Users/OMENXCJ/Downloads/Baishi/dataset")
TARGET_ROOT = Path(r"C:/chaoshi/detect_dataset")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# If True, clean target split dirs before writing new files.
CLEAR_TARGET_SPLITS = True


def read_class_names(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")
    names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n]
    if not names:
        raise ValueError(f"classes.txt is empty: {classes_path}")
    return names


def ensure_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0, got {total}"
        )


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def gather_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    pairs: list[tuple[Path, Path]] = []
    missing_images: list[Path] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image_for_stem(images_dir, stem)
        if image_path is None:
            missing_images.append(label_path)
            continue
        pairs.append((image_path, label_path))

    return pairs, missing_images


def split_items(items: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    items = list(items)
    random.Random(RANDOM_SEED).shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val : n_train + n_val + n_test]

    return {"train": train_items, "val": val_items, "test": test_items}


def make_split_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_split_dirs(root: Path) -> None:
    for base in ("images", "labels"):
        p = root / base
        if p.exists():
            shutil.rmtree(p)


def move_to_splits(root: Path, split_map: dict[str, list[tuple[Path, Path]]]) -> dict[str, int]:
    stats = {"train": 0, "val": 0, "test": 0}
    for split, items in split_map.items():
        for image_path, label_path in items:
            dst_image = root / "images" / split / image_path.name
            dst_label = root / "labels" / split / label_path.name
            shutil.move(str(image_path), str(dst_image))
            shutil.move(str(label_path), str(dst_label))
            stats[split] += 1
    return stats


def write_data_yaml(root: Path, class_names: list[str]) -> Path:
    yaml_path = root / "data.yaml"
    content = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.write_text(
        yaml.dump(content, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    ensure_ratios()

    images_dir = SOURCE_ROOT / "images"
    labels_dir = SOURCE_ROOT / "labels"
    classes_path = SOURCE_ROOT / "classes.txt"

    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    class_names = read_class_names(classes_path)
    pairs, missing_images = gather_pairs(images_dir, labels_dir)
    split_map = split_items(pairs)

    if CLEAR_TARGET_SPLITS:
        clear_split_dirs(TARGET_ROOT)
    make_split_dirs(TARGET_ROOT)
    stats = move_to_splits(TARGET_ROOT, split_map)
    yaml_path = write_data_yaml(TARGET_ROOT, class_names)

    print("Done: dataset prepared for train_detect.py")
    print(f"source: {SOURCE_ROOT}")
    print(f"target: {TARGET_ROOT}")
    print(f"classes: {class_names}")
    print(f"paired items moved: {len(pairs)}")
    print(f"missing-image labels skipped: {len(missing_images)}")
    print(
        f"split counts -> train: {stats['train']}, val: {stats['val']}, test: {stats['test']}"
    )
    print(f"data yaml: {yaml_path}")

    if missing_images:
        log_path = TARGET_ROOT / "missing_images_labels.txt"
        log_path.write_text(
            "\n".join(str(p) for p in missing_images) + "\n",
            encoding="utf-8",
        )
        print(f"missing-image labels log: {log_path}")


if __name__ == "__main__":
    main()
