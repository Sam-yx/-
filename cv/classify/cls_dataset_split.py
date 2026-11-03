"""
分类数据集划分工具

将按类别文件夹组织的图片数据集按比例划分为 train / val / test 三个子集。

源目录结构 (每个子文件夹代表一个类别):
  source_dir/
      ClassA/
          img1.jpg
          img2.jpg
      ClassB/
          img1.jpg

输出目录结构:
  output_dir/
      train/
          ClassA/
          ClassB/
      val/
          ClassA/
          ClassB/
      test/
          ClassA/
          ClassB/
"""

import os
import shutil
import random
from collections import defaultdict

# ======================== 配置区 ========================
SOURCE_DIR = r"C:\MaixCAM_Pro\data\images\sd\sd"       # 源数据目录（每个子文件夹为一个类别）
OUTPUT_DIR = r"C:\MaixCAM_Pro\data\dataset"      # 输出目录

TRAIN_RATIO = 0.7                      # 训练集比例
VAL_RATIO   = 0.2                      # 验证集比例
TEST_RATIO  = 0.1                      # 测试集比例

COPY_MODE = "copy"                     # "copy" 复制文件 / "move" 移动文件
RANDOM_SEED = 42                       # 随机种子，保证可复现
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
# ========================================================


def collect_images(source_dir):
    """收集所有类别的图片路径，返回 {类名: [图片路径列表]}"""
    class_images = defaultdict(list)
    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in sorted(os.listdir(class_dir)):
            if os.path.splitext(img_file)[1].lower() in IMG_EXTENSIONS:
                class_images[class_name].append(img_file)
    return class_images


def split_list(items, train_ratio, val_ratio, test_ratio):
    """将列表按比例划分为三部分"""
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def split_dataset(source_dir, output_dir, train_ratio, val_ratio, test_ratio,
                  copy_mode="copy", seed=42):
    """执行分类数据集划分"""
    # 参数校验
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        print(f"[警告] 比例之和为 {ratio_sum:.2f}，不等于 1.0，已自动归一化")
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

    random.seed(seed)

    # 收集图片
    class_images = collect_images(source_dir)
    if not class_images:
        print(f"[错误] 在 {source_dir} 中未找到任何类别文件夹或图片")
        return

    print(f"找到 {len(class_images)} 个类别，共 {sum(len(v) for v in class_images.values())} 张图片")
    print(f"划分比例 -> train: {train_ratio:.0%}  val: {val_ratio:.0%}  test: {test_ratio:.0%}")
    print(f"文件操作模式: {copy_mode}")
    print("-" * 60)

    stats = defaultdict(lambda: defaultdict(int))  # stats[split][class] = count
    op_func = shutil.copy2 if copy_mode == "copy" else shutil.move

    for class_name, images in class_images.items():
        # 随机打乱
        random.shuffle(images)

        # 按比例划分
        train_imgs, val_imgs, test_imgs = split_list(images, train_ratio, val_ratio, test_ratio)

        split_map = {
            "train": train_imgs,
            "val":   val_imgs,
            "test":  test_imgs,
        }

        for split_name, img_list in split_map.items():
            if not img_list:
                continue
            dst_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dst_dir, exist_ok=True)

            for img_file in img_list:
                src_path = os.path.join(source_dir, class_name, img_file)
                dst_path = os.path.join(dst_dir, img_file)
                op_func(src_path, dst_path)

            stats[split_name][class_name] = len(img_list)

    # 打印统计信息
    print(f"\n{'类别':<20} {'train':>8} {'val':>8} {'test':>8} {'总计':>8}")
    print("=" * 56)
    total_train = total_val = total_test = 0

    for class_name in sorted(class_images.keys()):
        t = stats["train"].get(class_name, 0)
        v = stats["val"].get(class_name, 0)
        te = stats["test"].get(class_name, 0)
        total = t + v + te
        total_train += t
        total_val += v
        total_test += te
        print(f"{class_name:<20} {t:>8} {v:>8} {te:>8} {total:>8}")

    print("-" * 56)
    grand_total = total_train + total_val + total_test
    print(f"{'合计':<20} {total_train:>8} {total_val:>8} {total_test:>8} {grand_total:>8}")
    print(f"\n数据集划分完成！输出目录: {output_dir}")


if __name__ == "__main__":
    split_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        copy_mode=COPY_MODE,
        seed=RANDOM_SEED,
    )
