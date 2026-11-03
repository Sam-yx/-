"""
DINOv2 模板批量生成工具

将模板图片批量转换为 DINOv2 embedding 格式的 pkl 文件，
供 detect/realtime_detect.py 中的 DINOv2Verifier 加载。

默认模板目录结构:
    keypoints/templates/
        类名1/
            img1.jpg
            img2.jpg
        类名2/
            ...

输出:
    keypoints/
        类名1_dinov2.pkl
        类名2_dinov2.pkl

示例:
    python build_dinov2_templates.py --input-dir C:/chaoshi/keypoints/templates
    python build_dinov2_templates.py --input-dir C:/chaoshi/keypoints/templates --model dinov2_vitb14
"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import cv2
import torch


def _load_dinov2_hub(model_name, device):
    """加载 DINOv2，优先用本地缓存，避免 GitHub API 限流。"""
    hub_dir = torch.hub.get_dir()
    cached = sorted(glob.glob(os.path.join(hub_dir, "facebookresearch_dinov2_*")))
    if cached:
        repo_dir = cached[-1]
        print(f"[INFO] 使用本地缓存: {repo_dir}")
        model = torch.hub.load(repo_dir, model_name, source="local", trust_repo=True)
    else:
        print("[INFO] 本地无缓存，从 GitHub 下载...")
        model = torch.hub.load(
            "facebookresearch/dinov2", model_name, trust_repo=True
        )
    return model.eval().to(device)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="批量生成 DINOv2 模板 pkl")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="模板图片根目录，子目录名为类别名",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\chaoshi\keypoints",
        help="输出 pkl 目录，默认 C:/chaoshi/keypoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vits14",
        help="DINOv2 模型: dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备: cuda 或 cpu",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    use_cuda = device_str.lower() == "cuda" and torch.cuda.is_available()
    if device_str.lower() == "cuda" and not use_cuda:
        print("[WARN] CUDA 不可用，回退 CPU")
    return torch.device("cuda" if use_cuda else "cpu")


def preprocess(image_bgr, device, mean_t, std_t):
    """BGR image → [1, 3, 224, 224] normalized tensor"""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = t.to(device)
    t = (t - mean_t) / std_t
    return t


@torch.no_grad()
def extract_embedding(model, image_bgr, device, mean_t, std_t):
    """Extract L2-normalized DINOv2 [CLS] token embedding"""
    t = preprocess(image_bgr, device, mean_t, std_t)
    emb = model(t)  # [1, embed_dim]
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu()  # [embed_dim]


def iter_images(folder: Path):
    """List image files in a directory"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(f)
    return files


def main():
    opt = parse_args()
    device = resolve_device(opt.device)

    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    print(f"[INFO] 加载 {opt.model} ...")
    model = _load_dinov2_hub(opt.model, device)
    print(f"[INFO] 模型已加载到 {device}")

    input_dir = Path(opt.input_dir)
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描子目录，每个子目录作为一个类别
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[ERROR] 在 {input_dir} 下未找到子目录（每个子目录代表一个类别）")
        return

    print(f"[INFO] 找到 {len(class_dirs)} 个类别目录")
    print()

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images = iter_images(cls_dir)

        if not images:
            print(f"  [跳过] {cls_name}: 目录下无图片")
            continue

        print(f"  [{cls_name}] 处理 {len(images)} 张图片...")

        embeddings = []
        filenames = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [WARN] 无法读取: {img_path}")
                continue
            emb = extract_embedding(model, img, device, mean_t, std_t)
            embeddings.append(emb)
            filenames.append(img_path.name)

        if not embeddings:
            print(f"    [WARN] 无有效图片，跳过")
            continue

        emb_tensor = torch.stack(embeddings)  # [N, embed_dim]

        pkl_data = {
            "method": "dinov2",
            "model": opt.model,
            "embeddings": emb_tensor,
            "image_files": filenames,
            "num_templates": len(filenames),
        }

        pkl_path = output_dir / f"{cls_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        print(
            f"    → 保存 {pkl_path} "
            f"({len(filenames)} 张, embed_dim={emb_tensor.shape[1]})"
        )

    print()
    print("[完成] DINOv2 模板构建完毕。")


if __name__ == "__main__":
    main()
