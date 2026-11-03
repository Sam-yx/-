"""
YOLOv8 模型训练脚本
- 支持目标检测 / 图像分类 / 实例分割 训练、验证、预测、导出
- 基于 ultralytics 库
- 自动根据模型类型（det/cls/seg）切换任务
"""

from ultralytics import YOLO
import argparse
import os
import torch


def get_task_type(weights):
    """根据权重文件名判断任务类型"""
    w = os.path.basename(weights).lower()
    if "-cls" in w:
        return "classify"
    elif "-seg" in w:
        return "segment"
    else:
        return "detect"


def train(opt):
    """训练 YOLOv8 模型"""
    # 加载模型：可以是预训练权重(.pt)或模型配置(.yaml)
    model = YOLO(opt.weights)

    # 开始训练
    results = model.train(
        data=opt.data,            # 数据集配置文件路径
        epochs=opt.epochs,        # 训练轮次
        batch=opt.batch,          # 批次大小
        imgsz=opt.imgsz,          # 输入图像尺寸
        device=opt.device,        # 训练设备 (0, 1, 2... 或 'cpu')
        workers=opt.workers,      # 数据加载线程数
        project=opt.project,      # 结果保存的项目目录
        name=opt.name,            # 本次训练的实验名称
        patience=opt.patience,    # 早停耐心值（多少轮无提升则停止）
        lr0=opt.lr0,              # 初始学习率
        lrf=opt.lrf,              # 最终学习率 = lr0 * lrf
        optimizer=opt.optimizer,  # 优化器: SGD, Adam, AdamW 等
        resume=opt.resume,        # 是否从上次中断处继续训练
        augment=True,             # 启用数据增强
        cache=opt.cache,          # 缓存图像到内存/磁盘以加速训练
        exist_ok=True,            # 允许覆盖已有实验目录
    )
    return results


def val(opt):
    """验证模型性能"""
    task = get_task_type(opt.weights)
    model = YOLO(opt.weights)
    metrics = model.val(
        data=opt.data,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        project=opt.project,
        name=opt.name + "_val",
        exist_ok=True,
    )
    # 打印主要指标
    if task == "classify":
        print(f"\nTop-1 准确率: {metrics.top1:.4f}")
        print(f"Top-5 准确率: {metrics.top5:.4f}")
    else:
        print(f"\nmAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


def predict(opt):
    """使用训练好的模型进行预测"""
    model = YOLO(opt.weights)
    results = model.predict(
        source=opt.source,        # 图片/视频/目录路径
        imgsz=opt.imgsz,
        device=opt.device,
        conf=opt.conf,            # 置信度阈值
        iou=opt.iou,              # NMS IoU 阈值
        save=True,                # 保存预测结果图片
        save_txt=True,            # 保存预测结果标签
        project=opt.project,
        name=opt.name + "_predict",
        exist_ok=True,
    )
    return results


def export_model(opt):
    """导出模型为其他格式 (ONNX, TorchScript 等)"""
    model = YOLO(opt.weights)
    model.export(
        format=opt.export_format,  # 导出格式
        imgsz=opt.imgsz,
        device=opt.device,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 训练/验证/预测/导出")

    # 基本参数
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "val", "predict", "export"],
                        help="运行模式: train | val | predict | export")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="预训练权重路径 (默认: yolov8n.pt)")
    parser.add_argument("--data", type=str, default="data/bottle.yaml",
                        help="数据集路径: 检测用 .yaml, 分类用目录路径 (如 cls_dataset)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="输入图像尺寸 (分类默认224, 检测建议640)")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备: 'auto' | '0'(GPU) | 'cpu' (默认: auto)")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数 (默认: 8)")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值 (默认: 50)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率 (默认: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="最终学习率比例 (默认: 0.01)")
    parser.add_argument("--optimizer", type=str, default="auto",
                        help="优化器: SGD | Adam | AdamW | auto (默认: auto)")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续训练")
    parser.add_argument("--cache", type=str, default=False,
                        help="缓存模式: True | ram | disk (默认: False)")

    # 输出参数
    parser.add_argument("--project", type=str, default="runs",
                        help="结果保存目录 (默认: runs)")
    parser.add_argument("--name", type=str, default="bottle",
                        help="实验名称 (默认: bottle)")

    # 预测参数
    parser.add_argument("--source", type=str, default="VOCdevkit/test/images",
                        help="预测图片/视频路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="预测置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值 (默认: 0.7)")

    # 导出参数
    parser.add_argument("--export-format", type=str, default="onnx",
                        help="导出格式: onnx | torchscript | engine 等 (默认: onnx)")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 自动检测设备
    if opt.device == "auto":
        opt.device = "0" if torch.cuda.is_available() else "cpu"

    # 自动判断任务类型
    task = get_task_type(opt.weights)

    print(f"模式: {opt.mode}")
    print(f"任务: {task}")
    print(f"权重: {opt.weights}")
    print(f"数据集: {opt.data}")
    print(f"设备: {opt.device}")
    print("-" * 50)

    if opt.mode == "train":
        train(opt)
    elif opt.mode == "val":
        val(opt)
    elif opt.mode == "predict":
        predict(opt)
    elif opt.mode == "export":
        export_model(opt)
