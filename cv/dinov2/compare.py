import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

# 参数设置
patch_h = 300
patch_w = 300
feat_dim = 384

# 图像预处理
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 加载DINOv2模型
dinov2_vits14 = torch.hub.load('', 'dinov2_vits14', source='local').cuda()
dinov2_vits14.eval()  # 设置为评估模式

def extract_image_features(img_path):
    """提取单张图片的DINOv2特征"""
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()  # 添加批次维度并移至GPU

    with torch.no_grad():
        # 提取DINOv2特征
        features_dict = dinov2_vits14.forward_features(img_tensor)
        patch_features = features_dict['x_norm_patchtokens']  # patch级特征
        cls_feature = features_dict['x_norm_clstoken']  # 整体图像特征
    
    # 返回两种特征
    return patch_features, cls_feature


def calculate_similarity(feat1, feat2, method='cosine'):
    """计算两种特征的相似度"""
    if method == 'cosine':
        # 对于patch特征，计算平均相似度
        if feat1.dim() == 3:  # 如果是patch特征
            return F.cosine_similarity(feat1.mean(dim=1), feat2.mean(dim=1)).item()
        # 对于cls特征，直接计算相似度
        return F.cosine_similarity(feat1, feat2).item()
    elif method == 'euclidean':
        # 欧几里得距离
        return F.pairwise_distance(feat1, feat2, p=2).item()
    return 0.0


# 两张图片的路径
img_path1 = 'bottle_1_jiance.jpg'
img_path2 = 'bottle_2_jiance.jpg'  # 第二张图片的路径

# 提取两张图片的特征
patch_feats1, cls_feats1 = extract_image_features(img_path1)
patch_feats2, cls_feats2 = extract_image_features(img_path2)

# 将patch特征通过平均化转换为二维张量
patch_feats1_avg = patch_feats1.mean(dim=1)  # [batch_size, feat_dim]
patch_feats2_avg = patch_feats2.mean(dim=1)  # [batch_size, feat_dim]

# 计算两种特征的余弦相似度
patch_similarity = calculate_similarity(patch_feats1_avg, patch_feats2_avg, method='cosine')
cls_similarity = calculate_similarity(cls_feats1, cls_feats2, method='cosine')

# 输出结果
print(f"基于Patch特征的余弦相似度: {patch_similarity:.4f}")
print(f"基于CLS特征的余弦相似度: {cls_similarity:.4f}")

# 使用 patch_feats1 来进行 PCA 可视化
features = patch_feats1.view(-1, feat_dim).cpu()  # 展平 patch_feats1

# 获取patch数量
num_patches = patch_feats1.shape[1]

# 设置 PCA 的 n_components 为3，以便可视化RGB颜色
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(features)
pca_features = pca.transform(features)

# 归一化处理
for i in range(n_components):
    pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (
            pca_features[:, i].max() - pca_features[:, i].min())

# 将 PCA 特征调整为适用于可视化的形状
side_length = int(np.sqrt(num_patches))
if side_length * side_length == num_patches:
    pca_features_rgb = pca_features.reshape(side_length, side_length, 3)
else:
    pca_features_rgb = pca_features.reshape(-1, 1, 3)
    print(f"警告: 无法将patch重塑为正方形，使用一维可视化")

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False 

plt.figure(figsize=(10, 10))
plt.imshow(pca_features_rgb)
plt.colorbar()
plt.title(f"PCA特征可视化 (Patch相似度: {patch_similarity:.4f})")  # 现在中文可以正常显示
plt.savefig('features_pca.png')
plt.show()
plt.close()
