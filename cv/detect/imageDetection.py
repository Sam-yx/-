import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def yolo_detection(image_path, model_path='yolov8x-cls.pt', save_dir='detection_results', target_class='bottle'):
    """
    使用YOLOv8进行图像目标检测并提取指定类别物体

    参数:
        image_path: 输入图像路径
        model_path: 模型路径，默认为预训练的yolov8x模型
        save_dir: 结果保存目录
        target_class: 目标类别名称（默认 'bottle'）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载YOLOv8检测模型
    model = YOLO(model_path)  # 可替换为 'yolov8s.pt'、'yolov8m.pt' 等更大模型

  # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为RGB格式（OpenCV默认BGR，matplotlib需要RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行目标检测
    results = model(image_rgb)  # 返回检测结果列表

    # 记录检测到的目标
    detected_bottles = []

    # 处理检测结果
    for result in results:
        # 绘制检测结果（边界框、类别标签、置信度）
        annotated_image = result.plot(
            boxes=True,  # 显示边界框
            labels=True,  # 显示类别标签
            conf=True,  # 显示置信度
            line_width=2,  # 边界框线宽
            font_size=12  # 字体大小
        )

        # 展示检测结果
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.axis('off')  # 关闭坐标轴
        plt.title('YOLOv8 目标检测结果')
        plt.tight_layout()
        plt.show()

        # 保存检测结果（转换回BGR格式用于OpenCV保存）
        img_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"detected_{img_name}")
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, annotated_image_bgr)
        print(f"检测结果已保存至: {save_path}")

        # 打印检测到的目标信息
        print("\n检测到的目标:")
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])  # 类别ID
            confidence = float(box.conf[0])  # 置信度
            cls_name = model.names[cls_id]  # 类别名称
            # 边界框坐标 (x1, y1, x2, y2)
            bbox = [int(coord) for coord in box.xyxy[0].tolist()]
            print(f"- 类别: {cls_name} (置信度: {confidence:.2f}), 位置: {bbox}")

            # 如果检测到的物品是 'bottle'
            if cls_name == target_class:
                # 提取物体区域
                x1, y1, x2, y2 = bbox
                object_region = image[y1:y2, x1:x2]

                # 保存提取出来的每个 'bottle' 图像，确保每个瓶子有唯一的文件名
                bottle_save_path = os.path.join(save_dir, f"{target_class}_{i+1}_{img_name}")
                cv2.imwrite(bottle_save_path, object_region)
                print(f"'{target_class}' 物体已保存至: {bottle_save_path}")
                detected_bottles.append(bottle_save_path)

    if not detected_bottles:
        print(f"未检测到目标物品: {target_class}")

if __name__ == "__main__":
    # 示例：检测单张图像
    input_image = "C:\\cv\\dinov2\\lv.jpg"  # 替换为你的图像路径
    yolo_detection(
        image_path=input_image,
        model_path='yolov8x-seg.pt',  # 可替换为其他模型
        save_dir='detection_results',
        target_class='bottle'  # 指定目标物体类别为 'bottle'
    )

