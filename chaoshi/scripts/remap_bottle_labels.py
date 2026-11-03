from __future__ import annotations

import argparse
import ast
from pathlib import Path


KEEP_NAMES = ["Baishi", "Wangzai", "Weitanai", "RIO_shuimitao"]
MERGED_NAME = "Bottle"
SPLITS = ("train", "valid", "test")


def parse_names_from_yaml(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            return list(ast.literal_eval(stripped.split(":", 1)[1].strip()))
    raise ValueError(f"names not found in {data_yaml}")


def build_name_mapping(original_names: list[str]) -> tuple[list[str], dict[int, int]]:
    missing = [name for name in KEEP_NAMES if name not in original_names]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"missing keep labels in data.yaml: {missing_text}")

    new_names = KEEP_NAMES + [MERGED_NAME]
    new_index_by_name = {name: index for index, name in enumerate(new_names)}
    merged_index = new_index_by_name[MERGED_NAME]

    old_to_new: dict[int, int] = {}
    for old_index, name in enumerate(original_names):
        old_to_new[old_index] = new_index_by_name.get(name, merged_index)
    return new_names, old_to_new


def remap_label_file(label_path: Path, old_to_new: dict[int, int]) -> bool:
    original_lines = label_path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    changed = False

    for line in original_lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append("")
            continue

        parts = stripped.split()
        old_index = int(parts[0])
        new_index = old_to_new[old_index]
        if new_index != old_index:
            changed = True
        new_lines.append(" ".join([str(new_index), *parts[1:]]))

    if changed:
        suffix = "\n" if original_lines else ""
        label_path.write_text("\n".join(new_lines) + suffix, encoding="utf-8")
    return changed


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep selected YOLO classes and merge all others into Bottle."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the YOLO dataset root")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    original_names = parse_names_from_yaml(data_yaml)
    new_names, old_to_new = build_name_mapping(original_names)

    changed_files = 0
    total_files = 0
    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            total_files += 1
            if remap_label_file(label_path, old_to_new):
                changed_files += 1

    update_data_yaml(data_yaml, new_names)

    print(f"dataset: {dataset_root}")
    print(f"updated label files: {changed_files}/{total_files}")
    print(f"new names: {new_names}")


if __name__ == "__main__":
    main()