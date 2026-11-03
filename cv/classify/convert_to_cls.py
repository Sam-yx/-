"""
将检测格式数据集转换为 YOLOv8 分类格式
检测格式: VOCdevkit/{train,val,test}/images/  （所有图片混在一起）
分类格式: cls_dataset/{train,val,test}/{类名}/  （按类名分文件夹）

类别通过文件名前缀自动判断：
  - RIOPuTao*.jpg -> RIOPuTao
  - WangZai*.jpg  -> WangZai
"""

import os
import shutil
import re

# === 配置 ===
SRC_ROOT = r"C:\cv\VOCdevkit"          # 原始检测数据集根目录
DST_ROOT = r"C:\cv\cls_dataset"        # 输出的分类数据集根目录
SPLITS = ["train", "val", "test"]       # 数据划分

# 类名列表（按文件名前缀匹配）
CLASS_NAMES = ["RIOPuTao", "WangZai"]


def get_class_from_filename(filename):
    """根据文件名前缀判断类别"""
    for cls_name in CLASS_NAMES:
        if filename.startswith(cls_name):
            return cls_name
    return None


def convert():
    total = 0
    for split in SPLITS:
        src_img_dir = os.path.join(SRC_ROOT, split, "images")
        if not os.path.exists(src_img_dir):
            print(f"[跳过] 目录不存在: {src_img_dir}")
            continue

        for img_file in os.listdir(src_img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            cls_name = get_class_from_filename(img_file)
            if cls_name is None:
                print(f"[警告] 无法识别类别: {img_file}，已跳过")
                continue

            # 创建目标目录: cls_dataset/{split}/{class_name}/
            dst_dir = os.path.join(DST_ROOT, split, cls_name)
            os.makedirs(dst_dir, exist_ok=True)

            src_path = os.path.join(src_img_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)
            shutil.copy2(src_path, dst_path)
            total += 1

        # 统计每个 split 的分布
        split_dir = os.path.join(DST_ROOT, split)
        if os.path.exists(split_dir):
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len(os.listdir(cls_dir))
                    print(f"  {split}/{cls}: {count} 张图片")

    print(f"\n转换完成！共处理 {total} 张图片")
    print(f"分类数据集目录: {DST_ROOT}")
    print(f"\n目录结构:")
    print(f"  {DST_ROOT}/")
    for split in SPLITS:
        print(f"    {split}/")
        for cls in CLASS_NAMES:
            print(f"      {cls}/")


if __name__ == "__main__":
    convert()
