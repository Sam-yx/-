import argparse
import os
import random
import shutil


DEFAULT_SOURCE_DIR = "images"  # 假设原始数据按类别目录组织在 images/ 下，如 images/cat/, images/dog/ 等
DEFAULT_NEW_DIR = "detect_dataset"
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_LABEL_EXTENSIONS = (".txt", ".xml", ".json")
SPLITS = ("train", "val", "test")


def _endswith_any(filename, exts):
    return filename.lower().endswith(exts)


def _expected_label_paths(category_dir, image_stem, label_exts):
    candidate_dirs = (
        category_dir,
        os.path.join(category_dir, "labels"),
        os.path.join(category_dir, "Annotations"),
    )
    candidates = []
    for base_dir in candidate_dirs:
        for ext in label_exts:
            candidates.append(os.path.join(base_dir, f"{image_stem}{ext}"))
    return candidates


def _find_label_path(category_dir, image_stem, label_exts):
    for path in _expected_label_paths(category_dir, image_stem, label_exts):
        if os.path.isfile(path):
            return path
    return None


def _write_missing_report(report_path, missing_records):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("missing_label_count=" + str(len(missing_records)) + "\n\n")
        for idx, rec in enumerate(missing_records, 1):
            f.write(f"[{idx}] category: {rec['category']}\n")
            f.write(f"image: {rec['image_path']}\n")
            f.write("expected_label_candidates:\n")
            for cand in rec["expected_label_candidates"]:
                f.write(f"  - {cand}\n")
            f.write("\n")


def _split_counts(total, val_ratio, test_ratio):
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    train_count = total - val_count - test_count

    # 尽量保证 train 至少有样本，避免训练集为空。
    if total > 0 and train_count == 0:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        train_count = total - val_count - test_count

    return train_count, val_count, test_count


def _split_samples(samples, val_ratio, test_ratio, rng):
    shuffled = list(samples)
    rng.shuffle(shuffled)
    train_count, val_count, test_count = _split_counts(
        len(shuffled), val_ratio, test_ratio
    )

    train_items = shuffled[:train_count]
    val_items = shuffled[train_count : train_count + val_count]
    test_items = shuffled[train_count + val_count : train_count + val_count + test_count]
    return train_items, val_items, test_items


def split_detection_dataset(
    source_dir,
    new_dir,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    overwrite=False,
    image_exts=DEFAULT_IMAGE_EXTENSIONS,
    label_exts=DEFAULT_LABEL_EXTENSIONS,
    missing_report_path="",
    print_missing=True,
):
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    if not 0 <= val_ratio <= 1 or not 0 <= test_ratio <= 1:
        raise ValueError("val_ratio 和 test_ratio 必须在 0 到 1 之间")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio 必须小于 1")

    for split in SPLITS:
        os.makedirs(os.path.join(new_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(new_dir, "labels", split), exist_ok=True)

    categories = sorted(
        name
        for name in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, name))
    )

    rng = random.Random(seed)
    stats = {
        "train": 0,
        "val": 0,
        "test": 0,
        "missing_label": 0,
        "skipped_exists": 0,
        "failed": 0,
    }
    missing_records = []

    for category in categories:
        category_dir = os.path.join(source_dir, category)
        category_samples = []

        for filename in sorted(os.listdir(category_dir)):
            image_path = os.path.join(category_dir, filename)
            if not os.path.isfile(image_path) or not _endswith_any(filename, image_exts):
                continue

            image_stem, _ = os.path.splitext(filename)
            label_path = _find_label_path(category_dir, image_stem, label_exts)
            if label_path is None:
                stats["missing_label"] += 1
                missing_records.append(
                    {
                        "category": category,
                        "image_path": image_path,
                        "expected_label_candidates": _expected_label_paths(
                            category_dir, image_stem, label_exts
                        ),
                    }
                )
                continue

            category_samples.append((image_path, label_path, filename))

        if not category_samples:
            print(f"[跳过] {category}: 未找到成对的图片和标注")
            continue

        train_items, val_items, test_items = _split_samples(
            category_samples, val_ratio, test_ratio, rng
        )
        split_map = {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        }

        for split_name, items in split_map.items():
            for image_path, label_path, image_filename in items:
                image_stem, _ = os.path.splitext(image_filename)
                label_ext = os.path.splitext(label_path)[1]

                dst_image_name = f"{category}_{image_filename}"
                dst_label_name = f"{category}_{image_stem}{label_ext}"
                dst_image = os.path.join(new_dir, "images", split_name, dst_image_name)
                dst_label = os.path.join(new_dir, "labels", split_name, dst_label_name)

                if (os.path.exists(dst_image) or os.path.exists(dst_label)) and not overwrite:
                    stats["skipped_exists"] += 1
                    continue

                try:
                    shutil.copy2(image_path, dst_image)
                    shutil.copy2(label_path, dst_label)
                    stats[split_name] += 1
                except OSError as exc:
                    stats["failed"] += 1
                    print(f"[失败] {image_path} / {label_path}: {exc}")

        print(
            f"[完成] {category}: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}"
        )

    print("\n划分完成")
    print(f"train: {stats['train']}")
    print(f"val: {stats['val']}")
    print(f"test: {stats['test']}")
    print(f"缺少标注: {stats['missing_label']}")
    print(f"跳过(目标已存在): {stats['skipped_exists']}")
    print(f"复制失败: {stats['failed']}")

    if missing_records:
        report_path = (
            missing_report_path
            if missing_report_path
            else os.path.join(new_dir, "missing_labels_report.txt")
        )
        _write_missing_report(report_path, missing_records)
        print(f"漏标注报告: {report_path}")

        if print_missing:
            print("\n漏标注明细:")
            for idx, rec in enumerate(missing_records, 1):
                print(f"[{idx}] 类别: {rec['category']}")
                print(f"    图片: {rec['image_path']}")
                print("    期望标注路径:")
                for cand in rec["expected_label_candidates"]:
                    print(f"      - {cand}")


def _parse_exts(text):
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    return tuple(p if p.startswith(".") else f".{p}" for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="将按类别目录组织的目标检测数据划分为 train/val/test"
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="源目录")
    parser.add_argument("--new-dir", default=DEFAULT_NEW_DIR, help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    parser.add_argument(
        "--missing-report-path",
        default="",
        help="漏标注报告输出路径，默认输出到 new_dir/missing_labels_report.txt",
    )
    parser.add_argument(
        "--no-print-missing",
        action="store_true",
        help="不在控制台打印漏标注明细，只输出报告文件",
    )
    parser.add_argument(
        "--image-exts",
        default=",".join(DEFAULT_IMAGE_EXTENSIONS),
        help="图片扩展名，逗号分隔，如 .jpg,.png",
    )
    parser.add_argument(
        "--label-exts",
        default=",".join(DEFAULT_LABEL_EXTENSIONS),
        help="标注扩展名，逗号分隔，如 .txt,.xml",
    )
    args = parser.parse_args()

    split_detection_dataset(
        source_dir=args.source_dir,
        new_dir=args.new_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        image_exts=_parse_exts(args.image_exts),
        label_exts=_parse_exts(args.label_exts),
        missing_report_path=args.missing_report_path,
        print_missing=not args.no_print_missing,
    )


if __name__ == "__main__":
    main()
