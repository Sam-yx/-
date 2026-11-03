"""
饮料识别脚本 —— 摄像头实时检测
方法: YOLOv8 目标检测 + 分类模型 + ORB 关键点匹配 (双重验证)

流程:
  1. 用 yolov8n.pt 检测画面中的瓶子 (COCO class 39=bottle)
  2. 裁剪瓶子区域, 用训练好的分类模型 (best.pt) 识别饮料类别
  3. 用 ORB 关键点匹配参考图做二次验证
  4. 在终端打印识别结果, 画面显示标注框

用法:
  python camera_detect.py
  python camera_detect.py --cls-weights runs/classify/runs/bottle/weights/best.pt
  python camera_detect.py --camera 1          # 切换摄像头
  python camera_detect.py --no-show           # 不显示窗口, 仅终端输出
  python camera_detect.py --det-conf 0.4      # 调整检测置信度
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ─── 确保 numpy 在 torch 之前加载 (torch 2.9+cu130 兼容性) ───
import numpy
import torch
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════
#  ORB 关键点匹配器
# ═══════════════════════════════════════════════════════════════
class ORBMatcher:
    """用 ORB 特征 + BFMatcher 对参考图做关键点匹配验证"""

    def __init__(self, ref_dirs: dict, max_refs_per_class: int = 10):
        """
        ref_dirs: {"RIOPuTao": "cls_dataset/train/RIOPuTao", ...}
        """
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ref_features = {}  # {class_name: [(kp, des), ...]}

        for cls_name, dir_path in ref_dirs.items():
            self.ref_features[cls_name] = []
            p = Path(dir_path)
            if not p.exists():
                print(f"[ORB] 警告: 参考目录不存在 {dir_path}")
                continue
            imgs = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
            imgs = imgs[:max_refs_per_class]
            for img_path in imgs:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None and len(des) > 10:
                    self.ref_features[cls_name].append(des)
            print(f"[ORB] 加载 {cls_name}: {len(self.ref_features[cls_name])} 张参考图")

    def match(self, crop_bgr, top_n: int = 2) -> dict:
        """
        对裁剪区域做 ORB 匹配, 返回各类别的平均好匹配数
        crop_bgr: BGR 图像 (numpy array)
        返回: {"RIOPuTao": 35.2, "WangZai": 12.1, ...}
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(des) < 5:
            return {cls: 0.0 for cls in self.ref_features}

        scores = {}
        for cls_name, ref_list in self.ref_features.items():
            if not ref_list:
                scores[cls_name] = 0.0
                continue
            match_counts = []
            for ref_des in ref_list:
                try:
                    matches = self.bf.knnMatch(des, ref_des, k=2)
                    # Lowe's ratio test
                    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                    match_counts.append(len(good))
                except Exception:
                    match_counts.append(0)
            scores[cls_name] = np.mean(sorted(match_counts, reverse=True)[:top_n])
        return scores


# ═══════════════════════════════════════════════════════════════
#  主检测流程
# ═══════════════════════════════════════════════════════════════
def run(opt):
    # ---- 加载模型 ----
    print(f"[模型] 检测模型: {opt.det_weights}")
    print(f"[模型] 分类模型: {opt.cls_weights}")
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device} ({'GPU' if device == '0' else 'CPU'})")

    det_model = YOLO(opt.det_weights)  # 目标检测 (yolov8n)
    cls_model = YOLO(opt.cls_weights)  # 分类 (训练好的 best.pt)

    # ---- 获取分类模型的类别名 ----
    # 从模型中读取类别映射
    cls_names = cls_model.names  # {0: 'RIOPuTao', 1: 'WangZai'}
    print(f"[分类] 类别: {cls_names}")

    # ---- 初始化 ORB 匹配器 ----
    ref_dirs = {}
    for idx, name in cls_names.items():
        ref_dir = os.path.join(opt.ref_dir, name)
        if os.path.isdir(ref_dir):
            ref_dirs[name] = ref_dir
    orb_matcher = ORBMatcher(ref_dirs, max_refs_per_class=opt.max_refs)

    # ---- 打开摄像头 ----
    cap = cv2.VideoCapture(opt.camera)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {opt.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[摄像头] 已打开 (索引={opt.camera})")
    print(f"[提示] 按 'q' 退出")
    print("=" * 60)

    frame_count = 0
    last_print_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[摄像头] 读取失败, 退出")
            break

        frame_count += 1
        annotated = frame.copy()

        # ---- Step 1: 目标检测 —— 找瓶子 ----
        det_results = det_model.predict(
            frame, imgsz=640, device=device,
            conf=opt.det_conf, verbose=False,
            classes=[39, 40, 41, 42, 43, 44, 45, 46, 47],
            # COCO: 39=bottle, 40=wine glass, 41=cup, 42=fork...
            # 主要检测 bottle(39), 也包含杯子等容器
        )

        detections = det_results[0].boxes
        now = time.time()

        if len(detections) == 0:
            # 没有检测到瓶子时, 尝试整帧分类
            if now - last_print_time > 2.0:  # 每2秒提示一次
                cls_results = cls_model.predict(
                    frame, imgsz=224, device=device, verbose=False
                )
                top1_idx = cls_results[0].probs.top1
                top1_conf = cls_results[0].probs.top1conf.item()
                cls_name = cls_names[top1_idx]
                if top1_conf > 0.7:
                    print(f"[整帧分类] {cls_name} (置信度: {top1_conf:.2f})")
                else:
                    print(f"[等待] 未检测到饮料瓶...")
                last_print_time = now

            if not opt.no_show:
                cv2.putText(annotated, "No bottle detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det_conf = box.conf[0].item()
                det_cls = int(box.cls[0].item())

                # 保证裁剪区域合法
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                crop = frame[y1:y2, x1:x2]

                # ---- Step 2: 分类模型识别 ----
                cls_results = cls_model.predict(
                    crop, imgsz=224, device=device, verbose=False
                )
                top1_idx = cls_results[0].probs.top1
                top1_conf = cls_results[0].probs.top1conf.item()
                cls_name = cls_names[top1_idx]

                # ---- Step 3: ORB 关键点匹配验证 ----
                orb_scores = orb_matcher.match(crop)
                orb_best = max(orb_scores, key=orb_scores.get) if orb_scores else ""
                orb_best_score = orb_scores.get(orb_best, 0)

                # ---- 双重验证逻辑 ----
                if cls_name == orb_best or top1_conf > 0.85:
                    # 分类和关键点匹配一致, 或分类置信度极高
                    final_label = cls_name
                    verify = "双重确认 ✓" if cls_name == orb_best else "高置信分类"
                    color = (0, 255, 0)  # 绿色
                elif top1_conf > 0.6:
                    # 分类置信度中等, 以分类结果为准
                    final_label = cls_name
                    verify = "分类结果"
                    color = (0, 255, 255)  # 黄色
                else:
                    # 置信度低且不一致
                    final_label = "未知饮料"
                    verify = "待确认"
                    color = (0, 0, 255)  # 红色

                # ---- 终端输出 ----
                if now - last_print_time > 1.0:  # 每秒输出一次
                    print(f"[识别] 饮料类别: {final_label:10s} | "
                          f"分类: {cls_name}({top1_conf:.2f}) | "
                          f"ORB匹配: {orb_best}({orb_best_score:.0f}) | "
                          f"检测置信度: {det_conf:.2f} | "
                          f"{verify}")
                    last_print_time = now

                # ---- 画面标注 ----
                if not opt.no_show:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{final_label} {top1_conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # ---- 显示画面 ----
        if not opt.no_show:
            cv2.imshow("Beverage Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[退出] 用户按下 q")
                break
        else:
            # 无窗口模式, 按 Ctrl+C 退出
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("[完成] 摄像头已关闭")


def parse_args():
    parser = argparse.ArgumentParser(description="饮料识别 - 摄像头实时检测")

    # 模型路径
    parser.add_argument("--det-weights", type=str, default="yolov8n.pt",
                        help="目标检测模型 (默认: yolov8n.pt)")
    parser.add_argument("--cls-weights", type=str,
                        default="runs/classify/runs/bottle/weights/best.pt",
                        help="分类模型 (默认: 训练好的 best.pt)")

    # 摄像头
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头索引 (默认: 0)")

    # 参考图目录 (用于 ORB 匹配)
    parser.add_argument("--ref-dir", type=str, default="cls_dataset/train",
                        help="ORB 参考图目录 (默认: cls_dataset/train)")
    parser.add_argument("--max-refs", type=int, default=10,
                        help="每类最多加载的参考图数 (默认: 10)")

    # 阈值
    parser.add_argument("--det-conf", type=float, default=0.3,
                        help="检测置信度阈值 (默认: 0.3)")

    # 显示
    parser.add_argument("--no-show", action="store_true",
                        help="不显示窗口, 仅终端输出")

    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    opt = parse_args()
    run(opt)
