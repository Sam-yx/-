"""
全图特征模板提取工具

从训练集中提取各类别的全图特征模板，用于后续实时检测时的特征匹配。
使用 YOLOv8 分类模型的特征提取层获取深度特征。

使用方法:
    python extract_templates.py
    python extract_templates.py --data C:\cv\cls_dataset --weights best.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import pickle
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="提取全图特征模板")
    parser.add_argument("--data", type=str, default=r"C:\cv\cls_dataset",
                        help="分类数据集路径")
    parser.add_argument("--weights", type=str, 
                        default=r"C:\cv\runs\classify\runs\classify\exp\weights\best.pt",
                        help="分类模型权重路径")
    parser.add_argument("--output", type=str, default="feature_templates.pkl",
                        help="输出文件路径")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--device", type=str, default="0",
                        help="设备")
    parser.add_argument("--split", type=str, default="train",
                        help="使用哪个数据集分割 (train/val)")
    parser.add_argument("--max-templates", type=int, default=20,
                        help="每个类别最多保存的模板数量")
    return parser.parse_args()


class FeatureExtractor:
    """特征提取器：使用 YOLOv8 分类模型提取全图特征"""
    
    def __init__(self, model_path, device='0'):
        self.model = YOLO(model_path)
        self.device = device
        self.features = []
        self.hook_handle = None
        
        # 注册 hook 到分类头之前的层
        self._register_hook()
    
    def _register_hook(self):
        """注册前向钩子以提取特征"""
        def hook_fn(module, input, output):
            # 对特征进行全局平均池化
            if len(output.shape) == 4:
                feat = F.adaptive_avg_pool2d(output, 1)
                feat = feat.flatten(1)
            else:
                feat = output
            self.features.append(feat.detach().cpu())
        
        # YOLOv8-cls 的特征层是倒数第二层
        try:
            target_layer = self.model.model.model[-2]
            self.hook_handle = target_layer.register_forward_hook(hook_fn)
        except Exception as e:
            print(f"[警告] 无法注册 hook: {e}")
    
    def extract(self, img):
        """提取单张图片的特征向量"""
        self.features = []
        
        # 运行推理（触发 hook）
        self.model.predict(
            source=img,
            imgsz=224,
            device=self.device,
            verbose=False
        )
        
        if self.features:
            return self.features[0].numpy().flatten()
        return None
    
    def extract_from_path(self, img_path):
        """从图片路径提取特征"""
        return self.extract(img_path)
    
    def __del__(self):
        if self.hook_handle:
            self.hook_handle.remove()


def extract_all_templates(extractor, data_dir, split, max_templates):
    """提取所有类别的特征模板"""
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"[错误] 目录不存在: {split_dir}")
        return None
    
    class_templates = {}
    class_names = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d))])
    
    print(f"找到 {len(class_names)} 个类别: {class_names}")
    print("-" * 60)
    
    for cls_name in class_names:
        cls_dir = os.path.join(split_dir, cls_name)
        img_files = [f for f in os.listdir(cls_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 随机选择部分图片作为模板
        if len(img_files) > max_templates:
            import random
            random.seed(42)
            img_files = random.sample(img_files, max_templates)
        
        print(f"处理类别 [{cls_name}]: {len(img_files)} 张图片")
        
        cls_features = []
        
        for img_file in tqdm(img_files, desc=f"  {cls_name}", leave=False):
            img_path = os.path.join(cls_dir, img_file)
            
            # 提取特征
            feat = extractor.extract_from_path(img_path)
            
            if feat is not None:
                # L2 归一化
                feat = feat / (np.linalg.norm(feat) + 1e-8)
                cls_features.append(feat)
        
        if cls_features:
            cls_features = np.array(cls_features)
            
            # 计算类别中心特征
            center = np.mean(cls_features, axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            
            # 计算类内相似度统计
            similarities = [np.dot(f, center) for f in cls_features]
            min_sim = np.min(similarities)
            mean_sim = np.mean(similarities)
            
            class_templates[cls_name] = {
                'templates': cls_features,   # 所有模板特征
                'center': center,            # 类别中心
                'min_similarity': min_sim,   # 类内最小相似度
                'mean_similarity': mean_sim, # 类内平均相似度
                'n_templates': len(cls_features),
            }
            
            print(f"  -> 模板数: {len(cls_features)}, "
                  f"平均相似度: {mean_sim:.4f}, 最小相似度: {min_sim:.4f}")
    
    return class_templates


def main():
    opt = parse_args()
    
    print("=" * 60)
    print("全图特征模板提取工具")
    print("=" * 60)
    print(f"数据集: {opt.data}")
    print(f"模型: {opt.weights}")
    print(f"数据分割: {opt.split}")
    print(f"每类最大模板数: {opt.max_templates}")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(opt.weights):
        print(f"[错误] 模型文件不存在: {opt.weights}")
        return
    
    if not os.path.exists(opt.data):
        print(f"[错误] 数据集目录不存在: {opt.data}")
        return
    
    # 创建特征提取器
    print("加载模型...")
    extractor = FeatureExtractor(opt.weights, opt.device)
    
    # 提取特征模板
    print("\n开始提取特征模板...")
    templates = extract_all_templates(
        extractor, opt.data, opt.split, opt.max_templates
    )
    
    if templates is None:
        print("[错误] 特征提取失败")
        return
    
    # 计算推荐的相似度阈值
    all_min_sims = [v['min_similarity'] for v in templates.values()]
    # 使用最小相似度的 0.8 倍作为阈值（留一些余量）
    recommended_threshold = min(all_min_sims) * 0.8
    
    # 保存结果
    output_data = {
        'templates': templates,
        'model_path': opt.weights,
        'class_names': list(templates.keys()),
        'recommended_threshold': recommended_threshold,
        'feature_dim': templates[list(templates.keys())[0]]['center'].shape[0],
    }
    
    output_path = opt.output
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("\n" + "=" * 60)
    print("特征模板提取完成!")
    print(f"保存到: {output_path}")
    print(f"特征维度: {output_data['feature_dim']}")
    print(f"推荐的相似度阈值: {recommended_threshold:.4f}")
    print("\n各类别统计:")
    for cls_name, data in templates.items():
        print(f"  {cls_name}: {data['n_templates']} 模板, "
              f"平均相似度 {data['mean_similarity']:.4f}, "
              f"最小相似度 {data['min_similarity']:.4f}")
    print("=" * 60)
    print("\n使用方法:")
    print(f"  python camera_classify.py --template {output_path} --roi")


if __name__ == "__main__":
    main()
