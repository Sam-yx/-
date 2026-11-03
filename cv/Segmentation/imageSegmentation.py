import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


def yolo_segmentation(image_path, model_path='yolov8x-seg.pt', save_path='segmented_result.jpg'):
    """
    使用YOLOv8进行图像分割

    参数:
        image_path: 输入图像路径
        model_path: 模型路径，默认为预训练的yolov8n-seg模型
        save_path: 结果保存路径
    """
    # 加载YOLOv8分割模型
    model = YOLO(model_path)  # 可以使用 'yolov8s-seg.pt' 等更大的模型

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为RGB格式格式（OpenCV默认BGR，matplotlib需要RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行分割预测
    results = model(image_rgb)

    # 处理结果
    for result in results:
        # 绘制分割结果
        annotated_image = result.plot(
            masks=True,  # 显示分割掩码
            boxes=True,  # 显示边界框
            labels=True,  # 显示类别标签
            conf=True  # 显示置信度
        )

        # 展示结果
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title('YOLOv8 图像分割结果')
        plt.show()

        # 保存结果（转换回BGR格式用于OpenCV保存）
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, annotated_image_bgr)
        print(f"分割结果已保存至: {save_path}")

        # 打印检测到的目标信息
        print("\n检测到的目标:")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls]
            print(f"- 类别: {cls_name}, 置信度: {conf:.2f}")


if __name__ == "__main__":
    # 示例：处理一张图片
    input_image = "test1.jpg"  # 替换为你的图片路径
    yolo_segmentation(input_image)
