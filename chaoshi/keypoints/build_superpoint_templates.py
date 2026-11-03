"""
SuperPoint 模板批量生成工具

将模板图片批量转换为 superpoint_lightglue 格式的 pkl 文件，
供 detect/realtime_detect.py 中的 KeypointVerifier 直接加载。

默认输出目录: C:/chaoshi/keypoints

示例:
    python build_superpoint_templates.py --input-dir C:/chaoshi/keypoints/templates
    python build_superpoint_templates.py --input-dir C:/chaoshi/keypoints --recursive
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import torch
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description="批量生成 SuperPoint 模板 pkl")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="模板图片目录，文件名(不含后缀)会作为类别名",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\chaoshi\keypoints",
        help="输出 pkl 目录，默认 C:/chaoshi/keypoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="特征提取设备: cuda 或 cpu",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=512,
        help="SuperPoint 最大关键点数",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归扫描子目录",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    use_cuda = device_str.lower() == "cuda" and torch.cuda.is_available()
    if device_str.lower() == "cuda" and not use_cuda:
        print("[WARN] CUDA 不可用，自动回退 CPU")
    return torch.device("cuda" if use_cuda else "cpu")


def load_lightglue_components():
    try:
        lg = importlib.import_module("lightglue")
        super_point_cls = getattr(lg, "SuperPoint")
        return super_point_cls
    except Exception as e:
        raise RuntimeError(
            "未安装 lightglue 依赖。请先安装: pip install lightglue"
        ) from e


def iter_images(root: Path, recursive: bool):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    if recursive:
        for pattern in patterns:
            files.extend(root.rglob(pattern))
            files.extend(root.rglob(pattern.upper()))
    else:
        for pattern in patterns:
            files.extend(root.glob(pattern))
            files.extend(root.glob(pattern.upper()))
    files = sorted(set(files))
    return files


def image_to_tensor_gray(image_bgr, device):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def features_to_cpu_serializable(feats: dict):
    out = {}
    for k, v in feats.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        elif isinstance(v, list):
            out[k] = [x.detach().cpu() if torch.is_tensor(x) else x for x in v]
        else:
            out[k] = v
    return out


def main():
    opt = parse_args()
    in_dir = Path(opt.input_dir)
    out_dir = Path(opt.output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(opt.device)
    print("=" * 60)
    print("SuperPoint 模板批量生成")
    print("=" * 60)
    print(f"输入目录: {in_dir}")
    print(f"输出目录: {out_dir}")
    print(f"设备: {device}")
    print(f"最大关键点: {opt.max_keypoints}")
    print("=" * 60)

    super_point_cls = load_lightglue_components()
    extractor = super_point_cls(max_num_keypoints=opt.max_keypoints).eval().to(device)

    image_files = iter_images(in_dir, opt.recursive)
    if not image_files:
        print("[ERROR] 未找到模板图片")
        return

    ok_cnt = 0
    skip_cnt = 0

    with torch.no_grad():
        for img_path in image_files:
            class_name = img_path.stem
            out_path = out_dir / f"{class_name}.pkl"

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[SKIP] 无法读取图片: {img_path}")
                skip_cnt += 1
                continue

            try:
                t = image_to_tensor_gray(image, device)
                feats = extractor.extract(t)
                cpu_feats = features_to_cpu_serializable(feats)

                data = {
                    "method": "superpoint_lightglue",
                    "class_name": class_name,
                    "source_image": str(img_path),
                    "max_keypoints": opt.max_keypoints,
                    "features": cpu_feats,
                }

                with open(out_path, "wb") as f:
                    pickle.dump(data, f)

                kp_num = int(cpu_feats["keypoints"].shape[1]) if "keypoints" in cpu_feats else -1
                print(f"[OK] {class_name} -> {out_path} | keypoints={kp_num}")
                ok_cnt += 1
            except Exception as e:
                print(f"[SKIP] 处理失败: {img_path} | 错误: {e}")
                skip_cnt += 1

    print("=" * 60)
    print(f"完成: 成功 {ok_cnt}，跳过 {skip_cnt}")
    print("输出已写入:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
