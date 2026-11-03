import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SOURCE_DIR = Path("detect_dataset")
NAME_PATTERN = re.compile(r"baishi|百事", re.IGNORECASE)


def _safe_int(text, default=0):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def xml_to_labelme_dict(xml_path: Path) -> tuple[dict, Counter]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = (root.findtext("filename") or f"{xml_path.stem}.jpg").strip()
    width = _safe_int(root.findtext("size/width"), 0)
    height = _safe_int(root.findtext("size/height"), 0)

    shapes = []
    class_counter = Counter()

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "unknown").strip()
        xmin = _safe_int(obj.findtext("bndbox/xmin"), 0)
        ymin = _safe_int(obj.findtext("bndbox/ymin"), 0)
        xmax = _safe_int(obj.findtext("bndbox/xmax"), 0)
        ymax = _safe_int(obj.findtext("bndbox/ymax"), 0)

        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        shapes.append(
            {
                "label": name,
                "points": [[float(xmin), float(ymin)], [float(xmax), float(ymax)]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
        )
        class_counter[name] += 1

    data = {
        "version": "5.8.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    return data, class_counter


def main():
    xml_files = sorted(
        p for p in SOURCE_DIR.rglob("*.xml") if NAME_PATTERN.search(p.name)
    )

    total = len(xml_files)
    success = 0
    failed = 0
    failed_files = []
    class_counter = Counter()

    for xml_path in xml_files:
        json_path = xml_path.with_suffix(".json")  # 覆盖同名 JSON
        try:
            data, per_file_counter = xml_to_labelme_dict(xml_path)
            class_counter.update(per_file_counter)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            success += 1
        except Exception as exc:
            failed += 1
            failed_files.append((str(xml_path), str(exc)))

    print("=" * 60)
    print("XML -> LabelMe JSON 转换完成")
    print("=" * 60)
    print(f"源目录            : {SOURCE_DIR}")
    print("筛选条件          : 文件名包含 baishi 或 百事")
    print("输出规则          : 同目录同名 .json（覆盖）")
    print(f"匹配 XML 文件数    : {total}")
    print(f"成功转换          : {success}")
    print(f"失败              : {failed}")

    print("\n类别统计（object 数量）:")
    if class_counter:
        for cls_name, cnt in sorted(class_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {cls_name}: {cnt}")
    else:
        print("  无")

    if failed_files:
        print("\n失败明细:")
        for fp, err in failed_files:
            print(f"  - {fp} -> {err}")


if __name__ == "__main__":
    main()
