from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


DEFAULT_LABELS_DIR = Path(r"C:\Users\OMENXCJ\Downloads\bottle.yolov8-obb\vail/labels")
DEFAULT_IMAGES_DIR = Path(r"C:\Users\OMENXCJ\Downloads\bottle.yolov8-obb\vail/images")
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_UNLABELED_DIR = Path(r"C:/chaoshi/my_photos/unlabeled_images")


def is_zero_byte_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == 0


def build_image_candidates(images_dir: Path, stem: str, image_exts: tuple[str, ...]) -> list[Path]:
    return [images_dir / f"{stem}{ext}" for ext in image_exts]


def get_unique_target_path(target_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = target_dir / filename
    idx = 1
    while candidate.exists():
        candidate = target_dir / f"{base}_{idx}{ext}"
        idx += 1
    return candidate


def move_unlabeled_images(
    labels_dir: Path,
    images_dir: Path,
    image_exts: tuple[str, ...],
    unlabeled_dir: Path,
    dry_run: bool,
    log_lines: list[str],
) -> int:
    label_stems = {p.stem for p in labels_dir.glob("*.txt") if p.is_file()}
    image_files = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    moved_count = 0
    if not dry_run:
        unlabeled_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_files):
        if image_path.stem in label_stems:
            continue

        target_path = get_unique_target_path(unlabeled_dir, image_path.name)
        if not dry_run:
            image_path.replace(target_path)
        moved_count += 1
        log_lines.append(f"[MOVE_UNLABELED_IMAGE] {image_path} -> {target_path}")

    return moved_count


def cleanup_empty_labels(
    labels_dir: Path,
    images_dir: Path,
    image_exts: tuple[str, ...],
    unlabeled_dir: Path,
    dry_run: bool,
    log_path: Path,
) -> None:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels 目录不存在: {labels_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")

    txt_files = sorted(p for p in labels_dir.glob("*.txt") if p.is_file())

    deleted_labels = 0
    moved_images_from_empty_labels = 0
    skipped_missing_image = 0
    skipped_non_empty = 0
    moved_unlabeled_images = 0

    log_lines = []
    log_lines.append("=== cleanup_empty_labels_and_images ===")
    log_lines.append(f"time={datetime.now().isoformat(timespec='seconds')}")
    log_lines.append(f"labels_dir={labels_dir}")
    log_lines.append(f"images_dir={images_dir}")
    log_lines.append(f"unlabeled_dir={unlabeled_dir}")
    log_lines.append(f"dry_run={dry_run}")
    log_lines.append("")

    if not dry_run:
        unlabeled_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in txt_files:
        if not is_zero_byte_file(txt_path):
            skipped_non_empty += 1
            continue

        stem = txt_path.stem
        candidates = build_image_candidates(images_dir, stem, image_exts)
        existing_images = [p for p in candidates if p.exists()]

        if not dry_run:
            txt_path.unlink(missing_ok=True)
        deleted_labels += 1
        log_lines.append(f"[DELETE_LABEL] {txt_path}")

        if not existing_images:
            skipped_missing_image += 1
            log_lines.append(f"[SKIP_NO_IMAGE] {stem} -> no matching image found")
            continue

        for image_path in existing_images:
            target_path = get_unique_target_path(unlabeled_dir, image_path.name)
            if not dry_run:
                image_path.replace(target_path)
            moved_images_from_empty_labels += 1
            log_lines.append(
                f"[MOVE_IMAGE_FROM_EMPTY_LABEL] {image_path} -> {target_path}"
            )

    moved_unlabeled_images = move_unlabeled_images(
        labels_dir=labels_dir,
        images_dir=images_dir,
        image_exts=image_exts,
        unlabeled_dir=unlabeled_dir,
        dry_run=dry_run,
        log_lines=log_lines,
    )

    log_lines.append("")
    log_lines.append("=== summary ===")
    log_lines.append(f"txt_scanned={len(txt_files)}")
    log_lines.append(f"deleted_labels={deleted_labels}")
    log_lines.append(f"moved_images_from_empty_labels={moved_images_from_empty_labels}")
    log_lines.append(f"skipped_non_empty_labels={skipped_non_empty}")
    log_lines.append(f"skipped_missing_image={skipped_missing_image}")
    log_lines.append(f"moved_unlabeled_images={moved_unlabeled_images}")
    log_lines.append(f"unlabeled_dir={unlabeled_dir}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("处理完成")
    print(f"扫描 txt 数量: {len(txt_files)}")
    print(f"删除空 txt 数量: {deleted_labels}")
    print(f"空txt对应图片移动数量: {moved_images_from_empty_labels}")
    print(f"跳过(非空 txt): {skipped_non_empty}")
    print(f"跳过(未找到对应图片): {skipped_missing_image}")
    print(f"移动未标注图片数量: {moved_unlabeled_images}")
    print(f"未标注图片目录: {unlabeled_dir}")
    print(f"日志文件: {log_path}")


def parse_exts(raw: str) -> tuple[str, ...]:
    parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    normalized = []
    for ext in parts:
        normalized.append(ext if ext.startswith(".") else f".{ext}")
    return tuple(dict.fromkeys(normalized))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "删除 labels 当前层中 0 字节 txt，删除对应图片，并将未标注图片移动到单独目录。"
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="labels 目录（只扫描当前这一层）",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="images 目录（按同名匹配图片并删除）",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(DEFAULT_IMAGE_EXTS),
        help="要匹配删除的图片后缀，逗号分隔，如 .jpg,.jpeg,.png",
    )
    parser.add_argument(
        "--unlabeled-dir",
        type=Path,
        default=DEFAULT_UNLABELED_DIR,
        help="未标注图片移动到的目录",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("cleanup_empty_labels.log"),
        help="日志输出路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际删除",
    )
    args = parser.parse_args()

    cleanup_empty_labels(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        image_exts=parse_exts(args.image_exts),
        unlabeled_dir=args.unlabeled_dir,
        dry_run=args.dry_run,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
