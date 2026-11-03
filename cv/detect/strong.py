import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import numpy as np
import os
import random


class ImageAugmentation:
    def __init__(self, seed=0):
        super(ImageAugmentation, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def resize_keep_ratio(self, image, target_size):
        """将图像缩放到目标尺寸，保持长宽比（最长边为target_size）"""
        old_w, old_h = image.size
        ratio = min(float(target_size) / old_w, float(target_size) / old_h)
        new_w, new_h = int(old_w * ratio), int(old_h * ratio)
        return image.resize((new_w, new_h), Image.BILINEAR)

    def resizeDown_keep_ratio(self, image, target_size):
        """缩小图像尺寸，保持长宽比，但不放大图像"""
        old_w, old_h = image.size
        ratio = min(float(target_size) / old_w, float(target_size) / old_h)
        ratio = min(ratio, 1.0)  # 确保不放大
        new_w, new_h = int(old_w * ratio), int(old_h * ratio)
        return image.resize((new_w, new_h), Image.BILINEAR)

    def resize(self, img, size):
        """将图像缩放到指定尺寸"""
        return img.resize((size, size), Image.BILINEAR)

    def random_flip_horizon(self, img, h_rate=0.5):
        """随机水平翻转"""
        if np.random.random() < h_rate:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    # def random_flip_vertical(self, img, v_rate=0.5):
    #     """随机垂直翻转"""
    #     if np.random.random() < v_rate:
    #         return img.transpose(Image.FLIP_TOP_BOTTOM)
    #     return img

    def center_crop(self, img, target_size=None):
        """中心裁剪"""
        w, h = img.size
        crop_size = min(w, h)
        
        # 计算裁剪区域
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # 裁剪图像
        cropped_img = img.crop((left, top, right, bottom))
        
        # 如果需要，调整到目标尺寸
        if target_size and target_size != crop_size:
            cropped_img = cropped_img.resize((target_size, target_size), Image.BILINEAR)
        
        return cropped_img

    def random_bright(self, img_tensor, p=0.5):
        """随机亮度变换"""
        if np.random.random() < p:
            alpha = np.random.uniform(-0.2, 0.2)  # 亮度变化范围
            img_tensor = img_tensor + alpha
            img_tensor = torch.clamp(img_tensor, min=0.0, max=1.0)
        return img_tensor

    def random_contrast(self, img_tensor, p=0.5):
        """随机对比度变换"""
        if np.random.random() < p:
            alpha = np.random.uniform(0.5, 1.5)  # 对比度变化范围
            img_tensor = img_tensor * alpha
            img_tensor = torch.clamp(img_tensor, min=0.0, max=1.0)
        return img_tensor

    def random_saturation(self, img_tensor, p=0.5):
        """随机饱和度变换"""
        if np.random.random() < p:
            # 仅对RGB图像的第二个通道（绿色通道）进行调整
            if img_tensor.shape[0] >= 3:  # 确保是RGB图像
                alpha = np.random.uniform(0.5, 1.5)  # 饱和度变化范围
                img_tensor[1] = img_tensor[1] * alpha
                img_tensor[1] = torch.clamp(img_tensor[1], min=0.0, max=1.0)
        return img_tensor

    def add_gaussian_noise(self, img_tensor, p=0.5):
        """添加高斯噪声"""
        if np.random.random() < p:
            mean = 0
            std = np.random.uniform(0.01, 0.1)  # 噪声强度范围
            noise = torch.normal(mean, std, img_tensor.shape)
            img_tensor = img_tensor + noise
            img_tensor = torch.clamp(img_tensor, min=0.0, max=1.0)
        return img_tensor

    def add_salt_noise(self, img_tensor, p=0.5):
        """添加盐噪声"""
        if np.random.random() < p:
            # 盐噪声比例
            noise_ratio = np.random.uniform(0.01, 0.05)
            mask = torch.rand(img_tensor.shape) < noise_ratio
            img_tensor = img_tensor.clone()  # 避免原地修改
            img_tensor[mask] = 1.0
        return img_tensor

    def add_pepper_noise(self, img_tensor, p=0.5):
        """添加胡椒噪声"""
        if np.random.random() < p:
            # 胡椒噪声比例
            noise_ratio = np.random.uniform(0.01, 0.05)
            mask = torch.rand(img_tensor.shape) < noise_ratio
            img_tensor = img_tensor.clone()  # 避免原地修改
            img_tensor[mask] = 0.0
        return img_tensor


def get_image_list(image_path):
    """获取图像文件列表"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    files_list = []
    for root, _, files in os.walk(image_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                files_list.append(os.path.join(root, file))
    return files_list


def save_image(img, save_path, prefix, image_path):
    """保存增强后的图像"""
    # 创建保存目录
    img_save_dir = os.path.join(save_path, "images")
    os.makedirs(img_save_dir, exist_ok=True)
    
    # 生成保存文件名
    image_name = os.path.basename(image_path)
    save_img_name = f"{prefix}{image_name}"
    save_img_path = os.path.join(img_save_dir, save_img_name)
    
    # 保存图像
    try:
        img.save(save_img_path)
    except Exception as e:
        print(f"Error saving image {save_img_path}: {e}")


def runAugumentation(image_path, save_path):
    """运行数据增强"""
    image_list = get_image_list(image_path)
    if not image_list:
        print("No images found in the specified directory.")
        return
        
    aug = ImageAugmentation()
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()
    
    for img_path in image_list:
        try:
            print(f"Processing: {os.path.basename(img_path)}")
            img = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
            
            # 尺寸变换 - 缩小尺寸
            t_img = aug.resizeDown_keep_ratio(img.copy(), 1024)
            save_image(t_img, save_path, "rs_", img_path)
            
            # 水平翻转
            t_img = aug.random_flip_horizon(img.copy())
            save_image(t_img, save_path, "fh_", img_path)
            
            # # 竖直翻转
            # t_img = aug.random_flip_vertical(img.copy())
            # save_image(t_img, save_path, "fv_", img_path)
            
            # 中心裁剪
            t_img = aug.center_crop(img.copy(), 1024)
            save_image(t_img, save_path, "cc_", img_path)
            
            # 转换为tensor进行像素级变换
            img_tensor = to_tensor(img)
            
            # 随机亮度调整
            t_img_tensor = aug.random_bright(img_tensor.clone())
            save_image(to_image(t_img_tensor), save_path, "rb_", img_path)
            
            # 随机对比度调整
            t_img_tensor = aug.random_contrast(img_tensor.clone())
            save_image(to_image(t_img_tensor), save_path, "rc_", img_path)
            
            # 随机饱和度调整
            t_img_tensor = aug.random_saturation(img_tensor.clone())
            save_image(to_image(t_img_tensor), save_path, "rsat_", img_path)
            
            # 添加高斯噪声
            t_img_tensor = aug.add_gaussian_noise(img_tensor.clone())
            save_image(to_image(t_img_tensor), save_path, "gn_", img_path)
            
            # 添加盐噪声
            t_img_tensor = aug.add_salt_noise(img_tensor.clone())
            save_image(to_image(t_img_tensor), save_path, "sn_", img_path)
            
            # 添加胡椒噪声
            t_img_tensor = aug.add_pepper_noise(img_tensor.clone())
            save_image(to_image(t_img_tensor), save_path, "pn_", img_path)
            
            print(f"Completed: {os.path.basename(img_path)}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
            continue


if __name__ == '__main__':
    # 图像文件夹和保存路径
    image_path = r"../captures"  # 图片位置路径
    save_path = r"Augumentation" # 结果保存位置路径
    
    # 运行数据增强
    runAugumentation(image_path, save_path)
    print("Image augmentation completed!")
    