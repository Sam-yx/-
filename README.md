# Computer Vision Projects

此仓库归档两个基于 Python 的计算机视觉项目：

| 目录 | 说明 |
| --- | --- |
| `chaoshi/` | 超市机器人视觉控制项目：商品 OCR、YOLO 商品检测、多帧追踪、人脸识别与 MCU 串口通信。核心入口为 `detect/unified_detect.py`。 |
| `cv/` | 通用计算机视觉实验与工具：YOLO 目标检测和分类、自动标注、数据格式转换、分割、DINOv2 和本地 Tesseract OCR 集成。 |

## 使用说明

每个子项目都有独立的模型、数据集与运行环境需求。请先按所使用脚本的 import 安装相应 Python 依赖，再在对应目录执行脚本。

常用入口：

```powershell
# 超市机器人统一视觉控制端
python chaoshi/detect/unified_detect.py --no-serial

# YOLO 图像分类训练
python cv/classify/train_cls.py --mode train
```

## 未纳入版本控制的内容

为避免仓库过大及提交机器专用文件，模型权重、训练数据、摄像头采集照片、推理结果、本地 Tesseract 安装包、缓存和虚拟环境均由 `.gitignore` 排除。将这些资源放回脚本所需路径即可运行。

## 许可证

本项目采用 [MIT License](LICENSE)。
