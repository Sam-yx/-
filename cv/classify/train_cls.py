"""
YOLOv8 图像分类模型训练脚本

数据集目录结构:
    cls_dataset/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
        val/
            class1/
            class2/
        test/  (可选)
            class1/
            class2/

使用方法:
    python train_cls.py                          # 使用默认配置训练
    python train_cls.py --epochs 50 --batch 32   # 自定义参数
    python train_cls.py --mode val               # 验证模型
    python train_cls.py --mode predict --source ./test_images  # 预测
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # 必须在 ultralytics/cv2 之前导入，否则 DLL 冲突
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 分类模型训练")

    # 模式选择
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "val", "predict", "export"],
                        help="运行模式: train/val/predict/export")

    # 模型配置
    parser.add_argument("--weights", type=str, default="yolov8n-cls.pt",
                        help="预训练权重路径 (yolov8n-cls.pt / yolov8s-cls.pt / yolov8m-cls.pt / yolov8x-cls.pt)")

    # 数据集配置
    parser.add_argument("--data", type=str, default=r"C:\cv\cls_dataset",
                        help="分类数据集根目录路径 (包含 train/val 子文件夹)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (0/1/2.../cpu)")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")

    # 优化器参数
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW", "RMSProp"],
                        help="优化器类型")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率 = lr0 * lrf")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减")

    # 训练策略
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 比率")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑")

    # 保存配置
    parser.add_argument("--project", type=str, default="runs/classify", help="结果保存目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")

    # 预测/导出参数
    parser.add_argument("--source", type=str, default="", help="预测图片路径")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "engine", "openvino"],
                        help="导出格式")

    # 导出选项
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本 (仅对 onnx 有效)")
    parser.add_argument("--simplify", action="store_true", help="导出后是否简化 ONNX 模型 (使用 onnx-simplifier)")
    parser.add_argument("--dynamic", action="store_true", help="是否导出动态轴 (batch size 可变)")

    return parser.parse_args()


def train(opt):
    """训练分类模型"""
    print("=" * 60)
    print("YOLOv8 分类模型训练")
    print("=" * 60)
    print(f"预训练权重: {opt.weights}")
    print(f"数据集路径: {opt.data}")
    print(f"训练轮次:   {opt.epochs}")
    print(f"批次大小:   {opt.batch}")
    print(f"图像尺寸:   {opt.imgsz}")
    print(f"设备:       {opt.device}")
    print("=" * 60)

    # 加载模型
    model = YOLO(opt.weights)

    # 开始训练
    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        patience=opt.patience,
        lr0=opt.lr0,
        lrf=opt.lrf,
        optimizer=opt.optimizer,
        weight_decay=opt.weight_decay,
        dropout=opt.dropout,
        label_smoothing=opt.label_smoothing,
        resume=opt.resume,
        pretrained=True,
        exist_ok=True,
        verbose=True,
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: {opt.project}/{opt.name}/weights/best.pt")
    return results


def validate(opt):
    """验证模型性能"""
    print("=" * 60)
    print("YOLOv8 分类模型验证")
    print("=" * 60)

    model = YOLO(opt.weights)
    metrics = model.val(
        data=opt.data,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        split="val",  # 使用 val 集验证，也可以改为 "test"
    )

    print("\n验证结果:")
    print(f"  Top-1 准确率: {metrics.top1:.4f}")
    print(f"  Top-5 准确率: {metrics.top5:.4f}")
    return metrics


def predict(opt):
    """使用模型进行预测"""
    if not opt.source:
        print("[错误] 预测模式需要指定 --source 参数")
        return

    print("=" * 60)
    print("YOLOv8 分类模型预测")
    print("=" * 60)
    print(f"模型: {opt.weights}")
    print(f"图片: {opt.source}")

    model = YOLO(opt.weights)
    results = model.predict(
        source=opt.source,
        imgsz=opt.imgsz,
        device=opt.device,
        save=True,
        project=opt.project,
        name="predict",
    )

    # 打印预测结果
    for r in results:
        probs = r.probs
        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()

        print(f"\n图片: {r.path}")
        print(f"预测类别: {r.names[probs.top1]} (置信度: {probs.top1conf:.4f})")
        print("Top-5 预测:")
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            print(f"  {i+1}. {r.names[idx]}: {conf:.4f}")

    return results


def export_model(opt):
    """导出模型"""
    print("=" * 60)
    print("YOLOv8 分类模型导出")
    print("=" * 60)
    print(f"模型: {opt.weights}")
    print(f"格式: {opt.format}")

    model = YOLO(opt.weights)
    model.export(
        format=opt.format,
        imgsz=opt.imgsz,
        device=opt.device,
    )

    print(f"\n导出完成!")


def main():
    opt = parse_args()

    # 检查 GPU
    if opt.device != "cpu":
        if torch.cuda.is_available():
            print(f"使用 GPU: {torch.cuda.get_device_name(int(opt.device))}")
        else:
            print("GPU 不可用，切换到 CPU")
            opt.device = "cpu"

    # 根据模式执行
    if opt.mode == "train":
        train(opt)
    elif opt.mode == "val":
        validate(opt)
    elif opt.mode == "predict":
        predict(opt)
    elif opt.mode == "export":
        export_model(opt)


if __name__ == "__main__":
    main()
