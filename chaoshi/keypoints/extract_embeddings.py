"""
特征嵌入提取工具

从训练集提取各类别的特征向量，计算类别中心并保存。
用于后续实时检测时判断是否为已知类别。

使用方法:
    python extract_embeddings.py
    python extract_embeddings.py --data C:\cv\cls_dataset --weights best.pt
"""

import os
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="提取分类模型特征嵌入")
    parser.add_argument("--data", type=str, default=r"C:\cv\cls_dataset",
                        help="分类数据集路径")
    parser.add_argument("--weights", type=str, 
                        default=r"C:\cv\runs\classify\runs\classify\exp\weights\best.pt",
                        help="分类模型权重路径")
    parser.add_argument("--output", type=str, default="class_embeddings.pkl",
                        help="输出文件路径")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--device", type=str, default="0",
                        help="设备")
    parser.add_argument("--split", type=str, default="train",
                        help="使用哪个数据集分割 (train/val)")
    return parser.parse_args()


def get_embedding(model, img_path, imgsz, device):
    """提取单张图片的特征向量"""
    # 使用模型进行预测，获取特征
    results = model.predict(
        source=img_path,
        imgsz=imgsz,
        device=device,
        verbose=False,
        embed=[model.model.model[-1].i - 1]  # 获取分类头前一层的特征
    )
    
    # 从结果中提取特征向量
    if hasattr(results[0], 'probs') and results[0].probs is not None:
        # YOLOv8 分类模型，使用 probs 的原始数据
        # 实际上我们需要获取倒数第二层的特征
        pass
    
    return results


def extract_features_hook(model, img_path, imgsz, device):
    """使用 hook 提取特征向量"""
    features = []
    
    def hook_fn(module, input, output):
        # 全局平均池化后的特征
        if len(output.shape) == 4:
            # 如果是 4D tensor (B, C, H, W)，做全局平均池化
            feat = torch.nn.functional.adaptive_avg_pool2d(output, 1)
            feat = feat.flatten(1)
        else:
            feat = output
        features.append(feat.detach().cpu().numpy())
    
    # 注册 hook 到分类头之前的层
    # 对于 YOLOv8-cls，特征提取层是 model.model.model[-2]
    try:
        # 尝试获取分类头前的特征
        target_layer = model.model.model[-2]
        handle = target_layer.register_forward_hook(hook_fn)
        
        # 前向传播
        model.predict(source=img_path, imgsz=imgsz, device=device, verbose=False)
        
        # 移除 hook
        handle.remove()
        
        if features:
            return features[0]
    except Exception as e:
        print(f"Hook 方式失败: {e}")
    
    return None


def extract_embeddings_simple(model, img_path, imgsz, device):
    """简化的特征提取方式 - 使用模型输出的概率分布作为特征"""
    results = model.predict(
        source=img_path,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    
    if results and results[0].probs is not None:
        # 使用 softmax 前的 logits 或 softmax 后的概率作为特征
        probs = results[0].probs.data.cpu().numpy()
        return probs
    return None


def extract_all_embeddings(model, data_dir, split, imgsz, device):
    """提取所有类别的特征向量"""
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"[错误] 目录不存在: {split_dir}")
        return None
    
    class_embeddings = {}
    class_names = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d))])
    
    print(f"找到 {len(class_names)} 个类别: {class_names}")
    print("-" * 60)
    
    # 首先，我们需要用 hook 的方式提取真正的特征
    # 获取模型最后一层前的特征维度
    features_dict = {}
    
    for cls_name in class_names:
        cls_dir = os.path.join(split_dir, cls_name)
        img_files = [f for f in os.listdir(cls_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"处理类别 [{cls_name}]: {len(img_files)} 张图片")
        
        cls_features = []
        
        for img_file in tqdm(img_files, desc=f"  {cls_name}", leave=False):
            img_path = os.path.join(cls_dir, img_file)
            
            # 提取特征（这里使用概率分布，也可以改用 hook 方式）
            feat = extract_embeddings_simple(model, img_path, imgsz, device)
            
            if feat is not None:
                cls_features.append(feat)
        
        if cls_features:
            cls_features = np.array(cls_features)
            
            # 计算类别中心和统计信息
            center = np.mean(cls_features, axis=0)
            std = np.std(cls_features, axis=0)
            
            # 计算类内最大距离（用于确定阈值）
            distances = [np.linalg.norm(f - center) for f in cls_features]
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)
            
            class_embeddings[cls_name] = {
                'center': center,
                'std': std,
                'max_dist': max_dist,
                'mean_dist': mean_dist,
                'n_samples': len(cls_features),
            }
            
            print(f"  -> 中心计算完成，样本数: {len(cls_features)}, "
                  f"平均距离: {mean_dist:.4f}, 最大距离: {max_dist:.4f}")
    
    return class_embeddings


def main():
    opt = parse_args()
    
    print("=" * 60)
    print("特征嵌入提取工具")
    print("=" * 60)
    print(f"数据集: {opt.data}")
    print(f"模型: {opt.weights}")
    print(f"数据分割: {opt.split}")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(opt.weights):
        print(f"[错误] 模型文件不存在: {opt.weights}")
        return
    
    if not os.path.exists(opt.data):
        print(f"[错误] 数据集目录不存在: {opt.data}")
        return
    
    # 加载模型
    print("加载模型...")
    model = YOLO(opt.weights)
    print(f"类别: {model.names}")
    
    # 提取特征
    print("\n开始提取特征...")
    embeddings = extract_all_embeddings(
        model, opt.data, opt.split, opt.imgsz, opt.device
    )
    
    if embeddings is None:
        print("[错误] 特征提取失败")
        return
    
    # 计算推荐的未知类别阈值
    all_max_dists = [v['max_dist'] for v in embeddings.values()]
    recommended_threshold = max(all_max_dists) * 1.5  # 使用最大距离的1.5倍作为阈值
    
    # 保存结果
    output_data = {
        'embeddings': embeddings,
        'model_path': opt.weights,
        'class_names': list(embeddings.keys()),
        'recommended_threshold': recommended_threshold,
    }
    
    output_path = opt.output
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("\n" + "=" * 60)
    print("特征提取完成!")
    print(f"保存到: {output_path}")
    print(f"推荐的未知类别阈值: {recommended_threshold:.4f}")
    print("\n各类别统计:")
    for cls_name, data in embeddings.items():
        print(f"  {cls_name}: {data['n_samples']} 样本, "
              f"平均距离 {data['mean_dist']:.4f}, 最大距离 {data['max_dist']:.4f}")
    print("=" * 60)
    print("\n使用方法:")
    print(f"  python camera_classify.py --embedding {output_path} --roi")


if __name__ == "__main__":
    main()
