from __future__ import annotations

import argparse
from pathlib import Path


def remap_line(line: str, mapping: dict[int, int]) -> tuple[str, str | None]:
    stripped = line.strip()
    if not stripped:
        return line, None

    parts = stripped.split()
    if len(parts) < 5:
        return line, "字段数不足(<5)"

    try:
        class_id = int(parts[0])
    except ValueError:
        return line, "class_id不是整数"

    new_class_id = mapping.get(class_id, class_id)
    if new_class_id == class_id:
        return line, None

    parts[0] = str(new_class_id)
    new_line = " ".join(parts)
    if line.endswith("\n"):
        new_line += "\n"
    return new_line, None


def process_file(file_path: Path, mapping: dict[int, int]) -> tuple[bool, list[str], int]:
    errors: list[str] = []
    changed = False
    changed_lines = 0

    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    output_lines: list[str] = []

    for i, line in enumerate(lines, start=1):
        new_line, err = remap_line(line, mapping)
        if err is not None:
            errors.append(f"{file_path} | line {i} | {err} | {line.rstrip()}")
            output_lines.append(line)
            continue

        if new_line != line:
            changed = True
            changed_lines += 1
        output_lines.append(new_line)

    if changed:
        file_path.write_text("".join(output_lines), encoding="utf-8", newline="")

    return changed, errors, changed_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="递归修改YOLO标签中的class_id")
    parser.add_argument(
        "--labels-dir",
        default=r"C:\Users\OMENXCJ\Downloads\bottle.yolov8-obb\train\labels",
        help="labels根目录（会递归处理所有txt）",
    )
    parser.add_argument(
        "--log-file",
        default=r"C:\chaoshi\detect_dataset\four\train\class_id_remap_skipped.log",
        help="跳过记录日志文件路径",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"labels目录不存在: {labels_dir}")

    mapping = {3: 1}  # 需要替换的class_id映射

    txt_files = sorted(labels_dir.rglob("*.txt"))
    total_files = len(txt_files)
    changed_files = 0
    changed_lines_total = 0
    skipped_records: list[str] = []

    for txt in txt_files:
        changed, errors, changed_lines = process_file(txt, mapping)
        if changed:
            changed_files += 1
        changed_lines_total += changed_lines
        skipped_records.extend(errors)

    log_path = Path(args.log_file)
    if skipped_records:
        log_path.write_text("\n".join(skipped_records) + "\n", encoding="utf-8")
    else:
        log_path.write_text("无跳过记录\n", encoding="utf-8")

    print(f"扫描txt文件: {total_files}")
    print(f"发生修改的文件: {changed_files}")
    print(f"替换的总行数: {changed_lines_total}")
    print(f"跳过记录数: {len(skipped_records)}")
    print(f"日志文件: {log_path}")


if __name__ == "__main__":
    main()
