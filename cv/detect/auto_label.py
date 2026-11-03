"""
=================================================================
  自动标注工具 (Auto Labeling Tool)

  流程:
    1. 读取 SOURCE_DIR 下的子文件夹, 子文件夹名 = 标签名
    2. 加载 YOLO 预训练模型检测目标 (bottle / cup 等)
    3. 对检测到的目标生成紧贴边缘的 bbox, 用子文件夹名作为标签
    4. 按 VOC / COCO / YOLO 三种格式输出标注文件
    5. 按 train/val/test 比例划分并整理目录
    6. 生成带检测框的预览图供人工校验

  使用示例:
    python auto_label.py
    python auto_label.py --format yolo
    python auto_label.py --format coco --conf 0.7
=================================================================
"""

import os
import cv2
import json
import random
import shutil
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]


# ╔══════════════════════════════════════════════════════════════╗
# ║                    📋 可配置参数区域                          ║
# ╚══════════════════════════════════════════════════════════════╝

# ==================== 1. 模型配置 ====================
# 可更换为任意 YOLO 模型: yolov8n.pt / yolov8s.pt / yolov8m.pt / yolo11n.pt / best.pt ...
MODEL_PATH = str(BASE_DIR / "detect" / "yolov8n.pt")

# COCO 预训练模型中要检测的类别 ID (用于筛选检测结果)
# 饮料相关: 39=bottle, 40=wine glass, 41=cup, 75=vase(罐状物体常被误识为 vase)
# 设为 None 则不过滤, 接受所有检测结果 (用自训练模型时建议设 None)
COCO_TARGET_CLASS_IDS = [39, 40, 41, 75]   # bottle + wine glass + cup + vase

# ==================== 2. 源数据 ====================
SOURCE_DIR = str(BASE_DIR / "my_photos" / "Baishi")

# ==================== 3. 输出配置 ====================
# 标注格式: "voc" / "coco" / "yolo"
OUTPUT_FORMAT = "voc"

# 输出根目录
OUTPUT_DIR = str(BASE_DIR / "detect_dataset")

# ==================== 4. 检测参数 ====================
CONF_THRESHOLD = 0.5       # 最低置信度
IMGSZ          = 1280      # 推理图像尺寸 (长边)
DEVICE         = "0"       # "0"=GPU, "cpu"=CPU

# ==================== 5. 数据集划分比例 ====================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ==================== 6. 预览图 ====================
GENERATE_PREVIEW = True               # 是否生成预览图
PREVIEW_DIR = str(Path(OUTPUT_DIR) / "preview")

# ==================== 7. 随机种子 ====================
RANDOM_SEED = 42


# ╔══════════════════════════════════════════════════════════════╗
# ║                       核心功能实现                            ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(description="自动标注工具")
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖 MODEL_PATH")
    parser.add_argument("--source", type=str, default=None,
                        help="覆盖 SOURCE_DIR")
    parser.add_argument("--output", type=str, default=None,
                        help="覆盖 OUTPUT_DIR")
    parser.add_argument("--format", type=str, default=None,
                        choices=["voc", "coco", "yolo"],
                        help="覆盖 OUTPUT_FORMAT")
    parser.add_argument("--conf", type=float, default=None,
                        help="覆盖 CONF_THRESHOLD")
    parser.add_argument("--no-preview", action="store_true",
                        help="不生成预览图")
    return parser.parse_args()


def apply_args(opt):
    """命令行参数覆盖全局配置"""
    global MODEL_PATH, SOURCE_DIR, OUTPUT_DIR, OUTPUT_FORMAT, CONF_THRESHOLD, GENERATE_PREVIEW
    if opt.model:
        MODEL_PATH = opt.model
    if opt.source:
        SOURCE_DIR = opt.source
    if opt.output:
        OUTPUT_DIR = opt.output
    if opt.format:
        OUTPUT_FORMAT = opt.format
    if opt.conf is not None:
        CONF_THRESHOLD = opt.conf
    if opt.no_preview:
        GENERATE_PREVIEW = False


# ----------------------------------------------------------------
#  Step 1: 扫描源目录, 收集图片与类别
# ----------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_source_dir(source_dir):
    """
    扫描源目录:
      - 有子文件夹 → 每个子文件夹名 = 类别名, 提取其中的图片
      - 无子文件夹 → 用当前文件夹名作为类别名, 提取当前目录图片
    返回: [(image_path, class_name), ...]
    """
    source = Path(source_dir)
    subdirs = [d for d in source.iterdir() if d.is_dir()]

    samples = []
    if subdirs:
        for sub in sorted(subdirs):
            class_name = sub.name
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    samples.append((str(f), class_name))
    else:
        class_name = source.name
        for f in sorted(source.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((str(f), class_name))

    return samples


# ----------------------------------------------------------------
#  Step 2 & 3: 检测 + 生成标注数据
# ----------------------------------------------------------------

def is_truncated(box, img_w, img_h, margin=3):
    """判断 bbox 是否触及图像边缘 (truncated)"""
    x1, y1, x2, y2 = box
    return (x1 <= margin or y1 <= margin or
            x2 >= img_w - margin or y2 >= img_h - margin)


## 以下 IoU/NMS 函数已注释，便于回退。若需要恢复，请移除注释。
# def _compute_iou(box1, box2):
#     """计算两个 [x1,y1,x2,y2] bbox 的 IoU"""
#     xa = max(box1[0], box2[0])
#     ya = max(box1[1], box2[1])
#     xb = min(box1[2], box2[2])
#     yb = min(box1[3], box2[3])
#     inter = max(0, xb - xa) * max(0, yb - ya)
#     if inter == 0:
#         return 0.0
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     return inter / (area1 + area2 - inter)


## 跨类别 NMS 已注释，便于回退。若需要恢复，请移除注释并确保 _compute_iou 可用。
# def cross_class_nms(objects, iou_thresh=0.5):
#     """跨类别 NMS: 不同 COCO 类别检测同一物体时, 只保留置信度最高的框"""
#     if len(objects) <= 1:
#         return objects
#     # 按置信度降序排列
#     sorted_objs = sorted(objects, key=lambda o: o["conf"], reverse=True)
#     keep = []
#     suppressed = set()
#     for i, obj_i in enumerate(sorted_objs):
#         if i in suppressed:
#             continue
#         keep.append(obj_i)
#         for j in range(i + 1, len(sorted_objs)):
#             if j in suppressed:
#                 continue
#             if _compute_iou(obj_i["bbox"], sorted_objs[j]["bbox"]) >= iou_thresh:
#                 suppressed.add(j)
#     return keep


def detect_and_annotate(model, samples, class_names_list):
    """
    对所有样本运行检测, 返回标注数据列表:
    [
      {
        "image_path": str,
        "class_name": str,         # 子文件夹名 (标签)
        "img_w": int,
        "img_h": int,
        "objects": [
          {"label": str, "bbox": [x1,y1,x2,y2], "conf": float, "truncated": bool},
          ...
        ]
      },
      ...
    ]
    """
    results_list = []
    total = len(samples)

    for idx, (img_path, class_name) in enumerate(samples):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  检测进度: {idx + 1}/{total}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] 无法读取: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        # 推理
        detections = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        objects = []
        for r in detections:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])

                # 如果设定了目标类别过滤
                if COCO_TARGET_CLASS_IDS is not None:
                    if cls_id not in COCO_TARGET_CLASS_IDS:
                        continue

                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # 裁剪到图像边界 (确保坐标精确)
                x1 = max(0, int(round(x1)))
                y1 = max(0, int(round(y1)))
                x2 = min(img_w, int(round(x2)))
                y2 = min(img_h, int(round(y2)))

                # 跳过无效框
                if x2 <= x1 or y2 <= y1:
                    continue

                truncated = is_truncated([x1, y1, x2, y2], img_w, img_h)

                objects.append({
                    "label": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "truncated": truncated,
                })

        # 跨类别 NMS 已禁用（若需恢复，请取消注释）
        # objects = cross_class_nms(objects, iou_thresh=IOU_THRESHOLD)

        results_list.append({
            "image_path": img_path,
            "class_name": class_name,
            "img_w": img_w,
            "img_h": img_h,
            "objects": objects,
        })

    return results_list


# ----------------------------------------------------------------
#  Step 4a: VOC XML 生成
# ----------------------------------------------------------------

def _create_voc_xml(annotation):
    """生成 VOC 格式 XML 字符串"""
    img_path = annotation["image_path"]
    img_w = annotation["img_w"]
    img_h = annotation["img_h"]

    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = Path(img_path).parent.name
    ET.SubElement(root, "filename").text = Path(img_path).name
    ET.SubElement(root, "path").text = img_path

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "AutoLabel"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    for obj in annotation["objects"]:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj["label"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "1" if obj["truncated"] else "0"
        ET.SubElement(obj_elem, "difficult").text = "0"

        bbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj["bbox"][0])
        ET.SubElement(bbox, "ymin").text = str(obj["bbox"][1])
        ET.SubElement(bbox, "xmax").text = str(obj["bbox"][2])
        ET.SubElement(bbox, "ymax").text = str(obj["bbox"][3])

    # 美化输出
    rough = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding=None)


# ----------------------------------------------------------------
#  Step 4b: YOLO TXT 生成
# ----------------------------------------------------------------

def _create_yolo_txt(annotation, class_names_list):
    """生成 YOLO 格式 TXT 内容"""
    img_w = annotation["img_w"]
    img_h = annotation["img_h"]
    lines = []

    for obj in annotation["objects"]:
        label = obj["label"]
        if label not in class_names_list:
            continue
        cls_id = class_names_list.index(label)

        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


# ----------------------------------------------------------------
#  Step 4c: COCO JSON 生成
# ----------------------------------------------------------------

def _build_coco_json(annotations, class_names_list):
    """构建完整 COCO JSON 结构"""
    images = []
    coco_annotations = []
    categories = []

    for i, name in enumerate(class_names_list):
        categories.append({"id": i, "name": name, "supercategory": "beverage"})

    ann_id = 1
    for img_id, ann in enumerate(annotations, start=1):
        images.append({
            "id": img_id,
            "file_name": Path(ann["image_path"]).name,
            "width": ann["img_w"],
            "height": ann["img_h"],
        })

        for obj in ann["objects"]:
            label = obj["label"]
            if label not in class_names_list:
                continue
            cls_id = class_names_list.index(label)

            x1, y1, x2, y2 = obj["bbox"]
            w = x2 - x1
            h = y2 - y1

            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "truncated": 1 if obj["truncated"] else 0,
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": coco_annotations,
        "categories": categories,
    }


# ----------------------------------------------------------------
#  Step 5: 数据集划分 + 目录生成
# ----------------------------------------------------------------

def split_dataset(annotations):
    """按比例随机划分 train / val / test"""
    random.seed(RANDOM_SEED)
    indices = list(range(len(annotations)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [annotations[i] for i in train_idx],
        "val":   [annotations[i] for i in val_idx],
        "test":  [annotations[i] for i in test_idx],
    }


def save_voc(splits, class_names_list):
    """保存 VOC 格式数据集"""
    root = Path(OUTPUT_DIR)
    print(f"\n[VOC] 输出目录: {root}")

    for split, anns in splits.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for ann in anns:
            stem = Path(ann["image_path"]).stem
            # 为避免不同子文件夹同名文件冲突, 加上类别前缀
            new_name = f"{ann['class_name']}_{stem}"

            # 复制图片
            src = ann["image_path"]
            dst_img = img_dir / f"{new_name}{Path(src).suffix}"
            shutil.copy2(src, dst_img)

            # 生成 XML
            xml_content = _create_voc_xml(ann)
            dst_xml = lbl_dir / f"{new_name}.xml"
            with open(dst_xml, "w", encoding="utf-8") as f:
                f.write(xml_content)

        print(f"  [{split}] 图片: {len(anns)}, 标注: {len(anns)}")


def save_yolo(splits, class_names_list):
    """保存 YOLO 格式数据集"""
    root = Path(OUTPUT_DIR)
    print(f"\n[YOLO] 输出目录: {root}")

    for split, anns in splits.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for ann in anns:
            stem = Path(ann["image_path"]).stem
            new_name = f"{ann['class_name']}_{stem}"

            src = ann["image_path"]
            dst_img = img_dir / f"{new_name}{Path(src).suffix}"
            shutil.copy2(src, dst_img)

            txt_content = _create_yolo_txt(ann, class_names_list)
            dst_txt = lbl_dir / f"{new_name}.txt"
            with open(dst_txt, "w", encoding="utf-8") as f:
                f.write(txt_content)

        print(f"  [{split}] 图片: {len(anns)}, 标注: {len(anns)}")

    # 生成 data.yaml
    yaml_path = root / "data.yaml"
    yaml_content = (
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_names_list)}\n"
        f"names: {class_names_list}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  [YAML] {yaml_path}")


def save_coco(splits, class_names_list):
    """保存 COCO 格式数据集"""
    root = Path(OUTPUT_DIR)
    print(f"\n[COCO] 输出目录: {root}")

    for split, anns in splits.items():
        img_dir = root / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片 (并重命名避免冲突)
        renamed_anns = []
        for ann in anns:
            stem = Path(ann["image_path"]).stem
            new_name = f"{ann['class_name']}_{stem}"
            src = ann["image_path"]
            dst_img = img_dir / f"{new_name}{Path(src).suffix}"
            shutil.copy2(src, dst_img)

            # 更新 file_name 用于 JSON
            ann_copy = dict(ann)
            ann_copy["image_path"] = str(dst_img)
            renamed_anns.append(ann_copy)

        # 生成 JSON
        coco_data = _build_coco_json(renamed_anns, class_names_list)
        json_name = f"instances_{split}.json"
        json_path = root / json_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        print(f"  [{split}] 图片: {len(anns)}, 标注JSON: {json_path}")


# ----------------------------------------------------------------
#  Step 6: 预览图生成
# ----------------------------------------------------------------

COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


def generate_previews(annotations, class_names_list):
    """生成带检测框的预览图"""
    preview_dir = Path(PREVIEW_DIR)
    preview_dir.mkdir(parents=True, exist_ok=True)

    color_map = {}
    for i, name in enumerate(class_names_list):
        color_map[name] = COLORS[i % len(COLORS)]

    count = 0
    for ann in annotations:
        if not ann["objects"]:
            continue

        img = cv2.imread(ann["image_path"])
        if img is None:
            continue

        for obj in ann["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            conf = obj["conf"]
            color = color_map.get(label, (0, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            if obj["truncated"]:
                text += " [T]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        stem = Path(ann["image_path"]).stem
        out_name = f"{ann['class_name']}_{stem}_preview.jpg"
        cv2.imwrite(str(preview_dir / out_name), img)
        count += 1

    print(f"\n[预览] 已生成 {count} 张预览图 → {preview_dir}")


# ----------------------------------------------------------------
#  主流程
# ----------------------------------------------------------------

def main():
    opt = parse_args()
    apply_args(opt)

    print("=" * 60)
    print("  自动标注工具 (Auto Labeling Tool)")
    print("=" * 60)
    print(f"  模型       : {MODEL_PATH}")
    print(f"  源目录     : {SOURCE_DIR}")
    print(f"  输出格式   : {OUTPUT_FORMAT}")
    print(f"  输出目录   : {OUTPUT_DIR}")
    print(f"  置信度阈值 : {CONF_THRESHOLD}")
    print(f"  划分比例   : train={TRAIN_RATIO} / val={VAL_RATIO} / test={TEST_RATIO}")
    print(f"  预览图     : {'是' if GENERATE_PREVIEW else '否'}")
    print("=" * 60)

    # 1. 扫描源目录
    print("\n[Step 1] 扫描源目录...")
    samples = scan_source_dir(SOURCE_DIR)
    if not samples:
        print("  [ERROR] 未找到任何图片, 请检查 SOURCE_DIR")
        return

    class_names_list = sorted(set(cls for _, cls in samples))
    print(f"  图片总数: {len(samples)}")
    print(f"  类别: {class_names_list}")
    for cls in class_names_list:
        n = sum(1 for _, c in samples if c == cls)
        print(f"    {cls}: {n} 张")

    # 2. 加载模型
    print(f"\n[Step 2] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 3. 检测与标注
    print("\n[Step 3] 开始检测...")
    annotations = detect_and_annotate(model, samples, class_names_list)

    # 统计
    total_obj = sum(len(a["objects"]) for a in annotations)
    no_det = sum(1 for a in annotations if not a["objects"])
    trunc = sum(1 for a in annotations for o in a["objects"] if o["truncated"])
    print(f"\n  检测完成:")
    print(f"    总图片: {len(annotations)}")
    print(f"    检测到目标: {total_obj} 个")
    print(f"    截断目标 (truncated): {trunc} 个")
    print(f"    无检测结果的图片: {no_det} 张")

    # 过滤: 只保留有检测结果的图片
    annotations = [a for a in annotations if a["objects"]]
    print(f"    保留有标注的图片: {len(annotations)} 张")

    # 4. 生成预览图
    if GENERATE_PREVIEW:
        print("\n[Step 4] 生成预览图...")
        generate_previews(annotations, class_names_list)

    # 5. 划分数据集
    print("\n[Step 5] 划分数据集...")
    splits = split_dataset(annotations)
    for split, anns in splits.items():
        print(f"  {split}: {len(anns)} 张")

    # 6. 保存标注
    fmt = OUTPUT_FORMAT.lower()
    print(f"\n[Step 6] 保存 {fmt.upper()} 格式标注...")

    if fmt == "voc":
        save_voc(splits, class_names_list)
    elif fmt == "yolo":
        save_yolo(splits, class_names_list)
    elif fmt == "coco":
        save_coco(splits, class_names_list)
    else:
        print(f"  [ERROR] 不支持的格式: {fmt}")
        return

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
