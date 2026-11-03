import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter 1920x1080 images and copy them to target directory with Baishi naming."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source folder to scan for images, for example: C:/chaoshi/my_photos/Baishi",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="C:/chaoshi/images",
        help="Target folder, default: C:/chaoshi/images",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Baishi",
        help="Output filename prefix, default: Baishi",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan source folder recursively",
    )
    return parser.parse_args()


def load_image_for_windows(path: Path):
    # Use imdecode to better handle non-ASCII paths on Windows.
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return data


def get_start_index(dst_dir: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\\d+)$", re.IGNORECASE)
    max_index = 0

    for file in dst_dir.iterdir():
        if not file.is_file():
            continue
        stem = file.stem
        match = pattern.match(stem)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def iter_image_files(src_dir: Path, recursive: bool):
    if recursive:
        candidates = src_dir.rglob("*")
    else:
        candidates = src_dir.glob("*")

    for p in candidates:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def main():
    args = parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    next_index = get_start_index(dst_dir, args.prefix)
    checked = 0
    copied = 0

    for img_path in iter_image_files(src_dir, args.recursive):
        checked += 1
        img = load_image_for_windows(img_path)
        if img is None:
            continue

        height, width = img.shape[:2]
        if width == 1920 and height == 1080:
            ext = img_path.suffix.lower() if img_path.suffix else ".jpg"
            new_name = f"{args.prefix}{next_index}{ext}"
            target_path = dst_dir / new_name
            shutil.copy2(img_path, target_path)
            next_index += 1
            copied += 1

    print(f"Done. Checked: {checked}, Copied: {copied}, Target: {dst_dir}")


if __name__ == "__main__":
    main()
