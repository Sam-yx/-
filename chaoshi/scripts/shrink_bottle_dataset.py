from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


KEEP_NAMES = {"Baishi", "Wangzai", "Weitanai", "RIO_shuimitao"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class SamplePair:
    split: str
    label_path: Path
    image_path: Path
    pair_size: int

    @property
    def stable_key(self) -> str:
        digest = hashlib.sha1(self.label_path.as_posix().encode("utf-8")).hexdigest()
        return digest


def parse_names_from_yaml(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            return list(ast.literal_eval(stripped.split(":", 1)[1].strip()))
    raise ValueError(f"names not found in {data_yaml}")


def resolve_image_path(images_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image for label not found: {stem} in {images_dir}")


def total_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_sample_index(dataset_root: Path) -> tuple[list[SamplePair], list[SamplePair]]:
    names = parse_names_from_yaml(dataset_root / "data.yaml")
    keep_ids = {index for index, name in enumerate(names) if name in KEEP_NAMES}
    missing = KEEP_NAMES - {names[index] for index in keep_ids}
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"keep labels missing from data.yaml: {missing_text}")

    protected_pairs: list[SamplePair] = []
    candidate_pairs: list[SamplePair] = []

    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"
        images_dir = dataset_root / split / "images"
        if not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = resolve_image_path(images_dir, label_path.stem)
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            class_ids = {int(line.split()[0]) for line in lines}
            pair = SamplePair(
                split=split,
                label_path=label_path,
                image_path=image_path,
                pair_size=label_path.stat().st_size + image_path.stat().st_size,
            )
            if class_ids & keep_ids:
                protected_pairs.append(pair)
            else:
                candidate_pairs.append(pair)

    return protected_pairs, candidate_pairs


def plan_removals(candidate_pairs: list[SamplePair], bytes_to_remove: int) -> list[SamplePair]:
    if bytes_to_remove <= 0:
        return []

    split_totals: dict[str, int] = {split: 0 for split in SPLITS}
    split_pairs: dict[str, list[SamplePair]] = {split: [] for split in SPLITS}
    for pair in candidate_pairs:
        split_totals[pair.split] += pair.pair_size
        split_pairs[pair.split].append(pair)

    total_candidate_bytes = sum(split_totals.values())
    if total_candidate_bytes < bytes_to_remove:
        raise ValueError(
            f"not enough removable samples: need {bytes_to_remove} bytes, have {total_candidate_bytes} bytes"
        )

    selected: list[SamplePair] = []
    selected_paths: set[Path] = set()
    removed_bytes = 0
    allocated = 0
    active_splits = [split for split in SPLITS if split_totals[split] > 0]

    for index, split in enumerate(active_splits):
        if index == len(active_splits) - 1:
            quota = bytes_to_remove - allocated
        else:
            quota = math.floor(bytes_to_remove * split_totals[split] / total_candidate_bytes)
            allocated += quota

        split_removed = 0
        for pair in sorted(split_pairs[split], key=lambda item: item.stable_key):
            if split_removed >= quota:
                break
            selected.append(pair)
            selected_paths.add(pair.label_path)
            split_removed += pair.pair_size
            removed_bytes += pair.pair_size

    if removed_bytes >= bytes_to_remove:
        return selected

    for pair in sorted(candidate_pairs, key=lambda item: (item.stable_key, item.label_path.as_posix())):
        if pair.label_path in selected_paths:
            continue
        selected.append(pair)
        removed_bytes += pair.pair_size
        if removed_bytes >= bytes_to_remove:
            break

    return selected


def move_to_backup(dataset_root: Path, backup_root: Path, pair: SamplePair) -> None:
    for path in (pair.image_path, pair.label_path):
        relative_path = path.relative_to(dataset_root)
        backup_path = backup_root / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(backup_path))


def write_manifest(backup_root: Path, dataset_root: Path, removed_pairs: list[SamplePair]) -> Path:
    manifest_path = backup_root / "prune_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "label", "pair_size_bytes"])
        for pair in removed_pairs:
            writer.writerow(
                [
                    pair.split,
                    pair.image_path.relative_to(dataset_root).as_posix(),
                    pair.label_path.relative_to(dataset_root).as_posix(),
                    pair.pair_size,
                ]
            )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shrink a YOLO dataset to a target size while preserving protected classes."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the dataset root")
    parser.add_argument(
        "--backup-root",
        type=Path,
        required=True,
        help="Directory where removed image/label pairs will be moved",
    )
    parser.add_argument(
        "--target-gib",
        type=float,
        default=1.2,
        help="Target dataset size in GiB after removal (default: 1.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned removals without moving files",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    backup_root = args.backup_root.resolve()
    if backup_root == dataset_root or dataset_root in backup_root.parents:
        raise ValueError("backup root must be outside the dataset root")

    current_bytes = total_file_bytes(dataset_root)
    target_bytes = int(args.target_gib * (1024**3))
    bytes_to_remove = max(0, current_bytes - target_bytes)

    protected_pairs, candidate_pairs = build_sample_index(dataset_root)
    removable_bytes = sum(pair.pair_size for pair in candidate_pairs)

    if bytes_to_remove > removable_bytes:
        raise ValueError(
            "cannot reach target without touching protected samples: "
            f"need to remove {bytes_to_remove} bytes, but only {removable_bytes} bytes are removable"
        )

    removed_pairs = plan_removals(candidate_pairs, bytes_to_remove)
    removed_bytes = sum(pair.pair_size for pair in removed_pairs)
    projected_bytes = current_bytes - removed_bytes

    print(f"current_gib={current_bytes / (1024**3):.3f}")
    print(f"target_gib={target_bytes / (1024**3):.3f}")
    print(f"protected_pairs={len(protected_pairs)}")
    print(f"candidate_pairs={len(candidate_pairs)}")
    print(f"planned_remove_pairs={len(removed_pairs)}")
    print(f"planned_remove_gib={removed_bytes / (1024**3):.3f}")
    print(f"projected_gib={projected_bytes / (1024**3):.3f}")

    if args.dry_run:
        return

    backup_root.mkdir(parents=True, exist_ok=True)
    for pair in removed_pairs:
        move_to_backup(dataset_root, backup_root, pair)
    manifest_path = write_manifest(backup_root, dataset_root, removed_pairs)

    final_bytes = total_file_bytes(dataset_root)
    print(f"final_gib={final_bytes / (1024**3):.3f}")
    print(f"backup_manifest={manifest_path}")


if __name__ == "__main__":
    main()