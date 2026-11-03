from __future__ import annotations

import argparse
import ast
import csv
import shutil
from pathlib import Path


TARGET_CLASS_NAME = "Bottle"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")


def parse_names_from_yaml(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            return list(ast.literal_eval(stripped.split(":", 1)[1].strip()))
    raise ValueError(f"names not found in {data_yaml}")


def build_index_mapping(original_names: list[str]) -> tuple[list[str], dict[int, int], int]:
    if TARGET_CLASS_NAME not in original_names:
        raise ValueError(f"{TARGET_CLASS_NAME} not found in data.yaml names")

    new_names: list[str] = []
    old_to_new: dict[int, int] = {}
    target_index = -1
    for old_index, name in enumerate(original_names):
        if name == TARGET_CLASS_NAME:
            target_index = old_index
            continue
        old_to_new[old_index] = len(new_names)
        new_names.append(name)

    return new_names, old_to_new, target_index


def resolve_image_path(images_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image for label not found: {stem} in {images_dir}")


def copy_to_backup(source_path: Path, dataset_root: Path, backup_root: Path, prefix: str) -> Path:
    relative_path = source_path.relative_to(dataset_root)
    backup_path = backup_root / prefix / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(backup_path))
    return backup_path


def move_to_backup(source_path: Path, dataset_root: Path, backup_root: Path, prefix: str) -> Path:
    relative_path = source_path.relative_to(dataset_root)
    backup_path = backup_root / prefix / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(backup_path))
    return backup_path


def update_data_yaml(data_yaml: Path, new_names: list[str]) -> None:
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    names_replaced = False
    nc_replaced = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("nc:"):
            indent = line[: len(line) - len(line.lstrip())]
            updated_lines.append(f"{indent}nc: {len(new_names)}")
            nc_replaced = True
        elif stripped.startswith("names:"):
            indent = line[: len(line) - len(line.lstrip())]
            updated_lines.append(f"{indent}names: {new_names!r}")
            names_replaced = True
        else:
            updated_lines.append(line)

    if not names_replaced or not nc_replaced:
        raise ValueError(f"failed to update nc/names in {data_yaml}")

    data_yaml.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the Bottle class from a YOLO dataset and back up affected files."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the YOLO dataset root")
    parser.add_argument(
        "--backup-root",
        type=Path,
        required=True,
        help="Directory where removed samples and original modified labels will be backed up",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned changes without touching files",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    backup_root = args.backup_root.resolve()
    if backup_root == dataset_root or dataset_root in backup_root.parents:
        raise ValueError("backup root must be outside the dataset root")

    data_yaml = dataset_root / "data.yaml"
    original_names = parse_names_from_yaml(data_yaml)
    new_names, old_to_new, target_index = build_index_mapping(original_names)

    modified_label_rows: list[list[str]] = []
    removed_pair_rows: list[list[str]] = []
    modified_label_files = 0
    removed_pairs = 0
    removed_annotations = 0
    copied_modified_dataset_yaml = False

    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"
        images_dir = dataset_root / split / "images"
        if not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = resolve_image_path(images_dir, label_path.stem)
            original_lines = label_path.read_text(encoding="utf-8").splitlines()
            new_lines: list[str] = []
            removed_in_file = 0

            for line in original_lines:
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                old_index = int(parts[0])
                if old_index == target_index:
                    removed_in_file += 1
                    continue

                new_index = old_to_new[old_index]
                new_lines.append(" ".join([str(new_index), *parts[1:]]))

            if removed_in_file == 0:
                continue

            removed_annotations += removed_in_file
            if not new_lines:
                removed_pairs += 1
                removed_pair_rows.append(
                    [
                        split,
                        image_path.relative_to(dataset_root).as_posix(),
                        label_path.relative_to(dataset_root).as_posix(),
                        str(removed_in_file),
                    ]
                )
                if not args.dry_run:
                    move_to_backup(image_path, dataset_root, backup_root, "removed")
                    move_to_backup(label_path, dataset_root, backup_root, "removed")
                continue

            modified_label_files += 1
            modified_label_rows.append(
                [
                    split,
                    image_path.relative_to(dataset_root).as_posix(),
                    label_path.relative_to(dataset_root).as_posix(),
                    str(removed_in_file),
                ]
            )
            if not args.dry_run:
                if not copied_modified_dataset_yaml:
                    copy_to_backup(data_yaml, dataset_root, backup_root, "modified_labels")
                    copied_modified_dataset_yaml = True
                copy_to_backup(image_path, dataset_root, backup_root, "modified_labels")
                copy_to_backup(label_path, dataset_root, backup_root, "modified_labels")
                label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"target_class={TARGET_CLASS_NAME}")
    print(f"original_names={original_names}")
    print(f"new_names={new_names}")
    print(f"removed_annotations={removed_annotations}")
    print(f"modified_label_files={modified_label_files}")
    print(f"removed_pairs={removed_pairs}")

    if args.dry_run:
        return

    backup_root.mkdir(parents=True, exist_ok=True)
    update_data_yaml(data_yaml, new_names)
    write_csv(
        backup_root / "modified_labels_manifest.csv",
        ["split", "image", "label", "removed_bottle_annotations"],
        modified_label_rows,
    )
    write_csv(
        backup_root / "removed_pairs_manifest.csv",
        ["split", "image", "label", "removed_bottle_annotations"],
        removed_pair_rows,
    )
    print(f"backup_root={backup_root}")


if __name__ == "__main__":
    main()