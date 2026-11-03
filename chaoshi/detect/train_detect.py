"""
=================================================================
  目标检测训练脚本 (高度可配置版)

  ● 模型: YOLOv8 / YOLO11 / YOLO26 / RT-DETR …… (随时切换)
  ● 规格: n / s / m / l / x                      (随时切换)
  ● 数据: YOLO / VOC / COCO 三种格式              (随时切换)
  ● 功能: 训练 → 验证 → 推理测试 (一键完成)
  ● GPU : RTX 5080 (16 GB) 优化
  ● 输入: 720 × 1280
=================================================================
"""

import os
import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]


# ╔══════════════════════════════════════════════════════════════╗
# ║                   📋 可配置参数区域                          ║
# ║              (所有参数均可根据需求自由修改)                   ║
# ╚══════════════════════════════════════════════════════════════╝

# ==================== 1. 模型配置 ====================
# model_type  : 模型系列
#   可选 → "yolov8" / "yolov9" / "yolov10" / "yolo11" / "yolo26" / "rtdetr"
# model_scale : 模型规格
#   可选 → "n"(nano) / "s"(small) / "m"(medium) / "l"(large) / "x"(xlarge)
# pretrained  : 自定义预训练权重路径 (str)
#   设为 None 则自动使用/下载官方预训练权重
#   例如: str(BASE_DIR / "detect" / "yolov8n.pt")

MODEL_TYPE  = "yolov8"
MODEL_SCALE = "s"
PRETRAINED  = str(BASE_DIR / "detect" / "yolov8s.pt")  # 自定义预训练权重路径 (设为 None 则自动使用/下载官方权重)


# ==================== 2. 数据集配置 ====================
# dataset_format: 切换数据集格式
#   "yolo" → 需提供 DATA_YAML (标注为 txt)
#   "voc"  → 需提供 VOC_ROOT  (标注为 xml, 自动转换)
#   "coco" → 需提供 COCO 相关路径 (标注为 json, 自动转换)

DATASET_FORMAT = "yolo"  # "yolo" / "voc" / "coco"

# --- [YOLO格式] yaml 配置文件 ---
DATA_YAML = str(BASE_DIR / "detect_dataset" / "data.yaml")

# --- [VOC格式] 目录结构 ---
# 期望结构:
#   VOC_ROOT/images/train/   VOC_ROOT/images/val/   VOC_ROOT/images/test/
#   VOC_ROOT/labels/train/   VOC_ROOT/labels/val/   VOC_ROOT/labels/test/  (XML)
VOC_ROOT = str(BASE_DIR / "detect_dataset")

# --- [COCO格式] ---
COCO_IMG_DIR   = str(BASE_DIR / "detect_dataset" / "images")   # 图像根目录
COCO_TRAIN_JSON = str(BASE_DIR / "detect_dataset" / "annotations" / "instances_train.json")  # 训练集标注 instances_train.json
COCO_VAL_JSON   = str(BASE_DIR / "detect_dataset" / "annotations" / "instances_val.json")    # 验证集标注 instances_val.json
COCO_TEST_JSON  = str(BASE_DIR / "detect_dataset" / "annotations" / "instances_test.json")   # 测试集标注 (可留空)

# --- 类别 (VOC / COCO 格式转换时需要; YOLO 格式从 yaml 读取) ---
CLASS_NAMES = ["Baishi", "Weitanai", "RIO_shuimitao"]  # 类别名称列表 (示例: ["Baishi", "Wangzai"])
NC = len(CLASS_NAMES)

# ==================== 3. 训练超参数 ====================
# ┌──────────────────────────────────────────────────────┐
# │  batch size 推荐表 (RTX 5080 16GB, imgsz=3840x2160)   │
# │  nano  (n) → batch = 16                             │
# │  small (s) → batch = 12 ~ 16                        │
# │  medium(m) → batch = 8                              │
# │  large (l) → batch = 4 ~ 6                          │
# │  xlarge(x) → batch = 2 ~ 4                          │
# └──────────────────────────────────────────────────────┘

# 输入图像尺寸（宽 x 高）
# 你的数据集为 3840x2160 (宽 x 高)
IMG_W   = 1280        # 图像宽度 (pixels)
IMG_H   = 720       # 图像高度 (pixels)
IMGSZ   = max(IMG_H, IMG_W)
EPOCHS  = 100        # 训练总轮数
BATCH   = 16         # 批次大小
WORKERS = 8          # 数据加载线程数
DEVICE  = 0          # GPU 设备 (单卡: 0, 多卡: "0,1", CPU: "cpu")

# 优化器
OPTIMIZER      = "auto"    # "SGD" / "Adam" / "AdamW" / "auto"
LR0            = 0.01      # 初始学习率
LRF            = 0.01      # 最终学习率 = LR0 × LRF
MOMENTUM       = 0.937     # SGD 动量 / Adam beta1
WEIGHT_DECAY   = 0.0005
WARMUP_EPOCHS  = 3.0
WARMUP_MOMENTUM = 0.8

# 数据增强
HSV_H     = 0.015    # 色调
HSV_S     = 0.7      # 饱和度
HSV_V     = 0.4      # 明度
DEGREES   = 0.0      # 旋转角度
TRANSLATE = 0.1      # 平移
SCALE     = 0.5      # 缩放
FLIPUD    = 0.0      # 上下翻转概率
FLIPLR    = 0.5      # 左右翻转概率
MOSAIC    = 1.0      # Mosaic 增强
MIXUP     = 0.0      # MixUp 增强


# ==================== 4. 保存与输出 ====================
PROJECT     = str(BASE_DIR / "runs" / "detect")
NAME        = "bottle_train"
SAVE_PERIOD = -1     # 每 N epoch 存一次 (-1 = 仅 best + last)
PATIENCE    = 20     # 早停: 连续 N epoch 无改善则停止 (0 = 禁用)
EXIST_OK    = False  # True = 覆盖同名实验; False = 自动递增


# ==================== 4.1 TensorRT 导出 ====================
EXPORT_ENGINE        = True   # 训练完成后自动导出 best.pt -> best.engine
ENGINE_HALF          = True   # FP16 导出
ENGINE_DYNAMIC       = True   # 动态输入
ENGINE_BATCH         = 1      # 实时检测默认 batch=1


# ==================== 5. 推理测试 ====================
TEST_SOURCE = str(BASE_DIR / "detect_dataset" / "test" / "images")  # 推理测试图像路径 (文件或目录)
CONF_THRES  = 0.90    # 置信度阈值
IOU_THRES   = 0.45   # NMS IoU 阈值
MAX_DET     = 300    # 每张图最大检测数


# ╔══════════════════════════════════════════════════════════════╗
# ║                     核心功能实现                              ║
# ╚══════════════════════════════════════════════════════════════╝

def get_model_path():
    """根据 MODEL_TYPE + MODEL_SCALE 构建模型权重路径"""
    if PRETRAINED:
        return PRETRAINED

    name_map = {
        "yolov8":  f"yolov8{MODEL_SCALE}.pt",
        "yolov9":  f"yolov9{MODEL_SCALE}.pt",
        "yolov10": f"yolov10{MODEL_SCALE}.pt",
        "yolo11":  f"yolo11{MODEL_SCALE}.pt",
        "yolo26":  f"yolo26{MODEL_SCALE}.pt",
        "rtdetr":  f"rtdetr-{MODEL_SCALE}.pt",
    }
    model_name = name_map.get(MODEL_TYPE, f"{MODEL_TYPE}{MODEL_SCALE}.pt")

    # 优先使用本地已有权重
    local_path = Path(__file__).parent / model_name
    if local_path.exists():
        print(f"[INFO] 使用本地权重: {local_path}")
        return str(local_path)

    print(f"[INFO] 将自动下载官方预训练权重: {model_name}")
    return model_name


# ----------------------------------------------------------------
#  VOC XML → YOLO TXT 转换
# ----------------------------------------------------------------

def _voc_xml_to_yolo(xml_path, class_names):
    """将单个 VOC XML 标注转换为 YOLO 格式文本"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    if img_w == 0 or img_h == 0:
        return ""

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_names:
            print(f"  [WARN] 未知类别 '{name}'，跳过 → {xml_path}")
            continue
        cls_id = class_names.index(name)

        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def prepare_voc_dataset():
    """
    VOC 格式处理:
    1. 将 labels/{split}/ 下的 XML 转成 TXT (写入同目录, 与 XML 共存)
    2. 生成 YOLO 训练所需的 yaml 配置文件
    """
    root = Path(VOC_ROOT)
    print(f"[INFO] VOC 数据集根目录: {root}")
    print(f"[INFO] 类别: {CLASS_NAMES}  (共 {NC} 类)")
    print("[INFO] 正在将 VOC XML → YOLO TXT ...")

    for split in ("train", "val", "test"):
        xml_dir = root / "labels" / split
        if not xml_dir.exists():
            print(f"  [{split}] 目录不存在，跳过")
            continue

        xml_files = list(xml_dir.glob("*.xml"))
        for xf in xml_files:
            txt_content = _voc_xml_to_yolo(str(xf), CLASS_NAMES)
            txt_path = xf.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)
        print(f"  [{split}] 转换完成: {len(xml_files)} 个文件")

    # 生成 YOLO yaml
    yaml_path = root / "dataset_auto.yaml"
    data_cfg = {
        "path":  str(root),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    NC,
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, allow_unicode=True, default_flow_style=False)
  
  
    print(f"[INFO] YOLO yaml 已生成 → {yaml_path}")
    return str(yaml_path)


# ----------------------------------------------------------------
#  COCO JSON → YOLO TXT 转换
# ----------------------------------------------------------------

def _coco_json_to_yolo(json_path, output_dir, class_names):
    """将 COCO JSON 标注转换为 YOLO TXT 文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # 按图像分组
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for img_id, img_info in img_map.items():
        img_w, img_h = img_info["width"], img_info["height"]
        stem = Path(img_info["file_name"]).stem
        lines = []

        for ann in ann_by_img.get(img_id, []):
            cat_name = cat_map.get(ann["category_id"], "")
            if cat_name not in class_names:
                continue
            cls_id = class_names.index(cat_name)

            x, y, w, h = ann["bbox"]          # COCO: [x, y, w, h] 左上角
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(Path(output_dir) / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1

    return count


def prepare_coco_dataset():
    """
    COCO 格式处理: 将 JSON 标注转为 YOLO TXT, 生成 yaml
    Ultralytics 约定: images/{split} ↔ labels/{split} 必须在同一 path 下
    """
    root = Path(COCO_IMG_DIR)           # detect_dataset/images
    dataset_root = root.parent          # detect_dataset
    labels_root = dataset_root / "labels"
    print(f"[INFO] COCO 数据集图像目录: {root}")
    print(f"[INFO] YOLO 标签输出目录:   {labels_root}")
    print("[INFO] 正在将 COCO JSON → YOLO TXT ...")

    for split, json_path in [("train", COCO_TRAIN_JSON),
                              ("val",   COCO_VAL_JSON),
                              ("test",  COCO_TEST_JSON)]:
        if not json_path:
            continue
        jp = Path(json_path)
        if not jp.exists():
            print(f"  [{split}] JSON 不存在，跳过: {jp}")
            continue
        out_dir = labels_root / split
        n = _coco_json_to_yolo(json_path, str(out_dir), CLASS_NAMES)
        print(f"  [{split}] 转换完成: {n} 个文件")

    # 生成 yaml — 使用 path + 相对路径，让 Ultralytics 自动关联 images↔labels
    yaml_path = dataset_root / "coco_dataset_auto.yaml"
    data_cfg = {
        "path":  str(dataset_root),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NC,
        "names": CLASS_NAMES,
    }
    if COCO_TEST_JSON:
        data_cfg["test"] = "images/test"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"[INFO] YOLO yaml 已生成 → {yaml_path}")
    return str(yaml_path)


# ----------------------------------------------------------------
#  数据集统一入口
# ----------------------------------------------------------------

def prepare_dataset():
    """根据 DATASET_FORMAT 准备数据, 返回 yaml 路径"""
    fmt = DATASET_FORMAT.lower()

    if fmt == "yolo":
        print(f"[INFO] 使用 YOLO 格式: {DATA_YAML}")
        return DATA_YAML

    elif fmt == "voc":
        return prepare_voc_dataset()

    elif fmt == "coco":
        return prepare_coco_dataset()

    else:
        raise ValueError(f"不支持的数据集格式: {fmt}  (可选: yolo / voc / coco)")


# ----------------------------------------------------------------
#  训练
# ----------------------------------------------------------------

def train(model, data_yaml):
    """执行训练, 返回训练结果"""
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)

    results = model.train(
        data=data_yaml,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        # 优化器
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        # 数据增强
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        # 保存
        project=PROJECT,
        name=NAME,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        exist_ok=EXIST_OK,
        # 其他
        pretrained=True,
        verbose=True,
    )
    return results


# ----------------------------------------------------------------
#  验证
# ----------------------------------------------------------------

def validate(model, data_yaml):
    """在验证集上评估模型"""
    print("\n" + "=" * 60)
    print("  开始验证")
    print("=" * 60)

    metrics = model.val(
        data=data_yaml,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        verbose=True,
    )

    print("\n[验证结果]")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    return metrics


# ----------------------------------------------------------------
#  推理测试
# ----------------------------------------------------------------

def predict(model):
    """对测试集进行推理并保存可视化结果"""
    print("\n" + "=" * 60)
    print("  开始推理测试")
    print("=" * 60)

    if not TEST_SOURCE or not Path(TEST_SOURCE).exists():
        print(f"[WARN] 测试路径不存在: {TEST_SOURCE}")
        return None

    results = model.predict(
        source=TEST_SOURCE,
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        save=True,
        save_txt=True,
        save_conf=True,
        project=PROJECT,
        name=f"{NAME}_predict",
        exist_ok=EXIST_OK,
    )

    # 统计检测结果
    total_det = 0
    for r in results:
        total_det += len(r.boxes)

    print(f"\n[推理结果]")
    print(f"  测试图像数: {len(results)}")
    print(f"  检测到目标: {total_det} 个")
    print(f"  结果已保存至: {PROJECT}/{NAME}_predict")

    return results


def export_engine(model, weight_path):
    """将 .pt 权重导出为 TensorRT .engine，失败仅告警不中断流程。"""
    if not EXPORT_ENGINE:
        print("[INFO] 已关闭 TensorRT engine 自动导出")
        return None

    pt_path = Path(weight_path)
    if not pt_path.exists():
        print(f"[WARN] 无法导出 engine，权重不存在: {pt_path}")
        return None

    print("\n" + "=" * 60)
    print("  开始导出 TensorRT engine")
    print("=" * 60)
    print(f"  源权重   : {pt_path}")
    print(f"  精度模式 : {'FP16' if ENGINE_HALF else 'FP32'}")
    print(f"  动态输入 : {ENGINE_DYNAMIC}")
    print(f"  Batch    : {ENGINE_BATCH}")

    try:
        engine_path = model.export(
            format="engine",
            imgsz=IMGSZ,
            device=DEVICE,
            half=ENGINE_HALF,
            dynamic=ENGINE_DYNAMIC,
            batch=ENGINE_BATCH,
            verbose=True,
        )
    except Exception as e:
        print(f"[WARN] TensorRT engine 导出失败: {e}")
        return None

    print(f"[INFO] TensorRT engine 导出完成 → {engine_path}")
    return engine_path


# ----------------------------------------------------------------
#  主流程
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("  目标检测训练脚本")
    print("=" * 60)
    print(f"  模型     : {MODEL_TYPE}-{MODEL_SCALE}")
    print(f"  数据格式 : {DATASET_FORMAT}")
    print(f"  输入尺寸 : {IMG_W}x{IMG_H} (宽x高)")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Batch    : {BATCH}")
    print(f"  GPU      : {DEVICE}")
    print("=" * 60)

    # 1. 准备数据集
    data_yaml = prepare_dataset()

    # 2. 加载模型
    model_path = get_model_path()
    print(f"\n[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)

    # 3. 训练
    train_results = train(model, data_yaml)

    # 4. 加载本次训练输出目录中的最佳权重进行验证和推理
    #    避免使用固定 PROJECT/NAME 路径误加载到历史实验的 best.pt
    run_dir = Path(getattr(train_results, "save_dir", Path(PROJECT) / NAME))
    best_weight = run_dir / "weights" / "best.pt"
    if best_weight.exists():
        print(f"\n[INFO] 加载最佳权重: {best_weight}")
        best_model = YOLO(str(best_weight))
    else:
        print("[WARN] 未找到 best.pt，使用当前模型继续")
        best_model = model

    # 5. 验证
    validate(best_model, data_yaml)

    # 6. 推理测试
    predict(best_model)

    # 7. 自动导出 TensorRT engine
    if best_weight.exists():
        export_engine(best_model, best_weight)
    else:
        print("[WARN] 跳过 TensorRT engine 导出：未找到 best.pt")

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
