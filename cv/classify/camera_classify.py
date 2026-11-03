"""
YOLOv8 分类模型 - 摄像头实时识别

使用训练好的分类模型，通过摄像头进行实时分类检测。
支持：本地摄像头、USB 外接摄像头、手机摄像头（通过 USB/WiFi）
支持：未知类别检测（通过关键点匹配方案）

=== 关键点匹配方案（未知类别检测）===
1. 先运行 capture_keypoints.py 采集各类别饮料瓶的关键点
2. 关键点文件会保存到 keypoints/ 目录
3. 运行时自动加载关键点文件进行验证
4. 流程: 模型预测类别 -> 用该类别的关键点验证 -> 匹配失败则为未知

=== 手机摄像头使用方法 ===
1. 在手机上安装 IP Webcam (Android) 或 DroidCam 等 App
2. 手机通过 USB 连接电脑，并开启 USB 网络共享
   或者手机与电脑连接同一 WiFi
3. 在手机 App 中启动服务器，获取视频流地址
4. 使用 --source 参数指定视频流 URL

=== ROI 区域检测 ===
使用 --roi 参数启用 ROI 区域检测，只对画面中心的指定区域进行分类。
适合固定位置放置饮料瓶进行识别的场景。

=== 手机竖屏拍摄（看两层货架） ===
手机 App 推流通常是横屏，加 --rotate 90 可将画面转为竖屏。
示例：python camera_classify.py --dual-roi --rotate 90 --source http://手机IP:8080/video

使用示例:
    # 1. 采集关键点（每个类别运行一次）
    python capture_keypoints.py --class-name BaiShi --output keypoints
    python capture_keypoints.py --class-name WangZai --output keypoints
    
    # 2. 运行识别（启用 ROI + 关键点验证）
    python camera_classify.py --roi --keypoints-dir keypoints
    
    # 调整匹配参数
    python camera_classify.py --roi --min-matches 15 --match-ratio 0.7

快捷键:
    Q / ESC : 退出程序
    S       : 截图保存
    Space   : 暂停/继续
"""

import cv2
import argparse
import time
import os
import socket
import urllib.request
import pickle
import numpy as np
import threading
import queue
from datetime import datetime
from ultralytics import YOLO

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# =================== 商品编码映射 ===================
PRODUCT_CODE = {
    # --- 牛奶类 ---
    "旺仔牛奶": 0x01,
    "金典牛奶": 0x02,
    "特仑苏牛奶": 0x03,
    "光明牛奶": 0x04,
    "蒙牛牛奶": 0x05,
    "伊利牛奶": 0x06,
    "牛奶": 0x01,
    # --- 薯片类 ---
    "乐事薯片": 0x10,
    "品客薯片": 0x11,
    "好丽友薯片": 0x12,
    "薯片": 0x10,
    # --- 可乐类 ---
    "可口可乐": 0x20,
    "百事可乐": 0x21,
    "可乐": 0x20,
    # --- 功能饮料 ---
    "红牛": 0x30,
    "佳得乐": 0x31,
    "功能饮料": 0x30,
    # --- 水 ---
    "农夫山泉": 0x40,
    "怡宝": 0x41,
    "矿泉水": 0x40,
    # --- 咖啡/茶 ---
    "咖啡": 0x50,
    "茶": 0x51,
    # --- 饼干 ---
    "奥利奥饼干": 0x60,
    "饼干": 0x60,
    # --- 巧克力 ---
    "德芙巧克力": 0x70,
    "巧克力": 0x70,
    # --- 坚果 ---
    "三只松鼠坚果": 0x80,
    "坚果": 0x80,
    # --- 模型英文类名映射（根据你的训练类别添加） ---
    "BaiShi": 0x21,      # 百事可乐
    "WangZai": 0x01,     # 旺仔牛奶
    "RIO_mitao": 0x30,   # RIO蜜桃（归入功能饮料，可自行修改）
    # --- 未知 ---
    "未知": 0x00,
    "Unknown": 0x00,
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 分类模型摄像头实时识别")
    
    # 模型配置
    parser.add_argument("--weights", type=str, 
                        default="runs/classify/runs/classify/exp/weights/best.pt",
                        help="分类模型权重路径")
    
    # 摄像头配置
    parser.add_argument("--camera", type=int, default=2,
                        help="摄像头索引 (0=内置, 1=外接, 以此类推)")
    parser.add_argument("--source", type=str, default="",
                        help="视频源 URL (手机摄像头流地址，如 http://192.168.1.100:8080/video)")
    parser.add_argument("--phone", action="store_true",
                        help="自动扫描并连接手机摄像头 (扫描常见端口)")
    parser.add_argument("--width", type=int, default=1280,
                        help="摄像头画面宽度")
    parser.add_argument("--height", type=int, default=720,
                        help="摄像头画面高度")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="画面旋转角度 (0/90/180/270)。手机竖拍时用 90 或 270 将横屏流转为竖屏")
    
    # 推理配置
    parser.add_argument("--imgsz", type=int, default=224,
                        help="模型输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备 (0/1/cpu)")
    
    # 显示配置
    parser.add_argument("--show-top5", action="store_true",
                        help="显示Top-5预测结果")
    parser.add_argument("--no-show", action="store_true",
                        help="不显示画面（仅终端输出）")
    
    # 保存配置
    parser.add_argument("--save-dir", type=str, default="screenshots",
                        help="截图保存目录")
    
    # ROI 配置（单区域）
    parser.add_argument("--roi", action="store_true",
                        help="启用单 ROI 区域检测（只对 ROI 区域进行分类）")
    parser.add_argument("--roi-width", type=int, default=200,
                        help="ROI 区域宽度（像素，略大于饮料瓶宽度）")
    parser.add_argument("--roi-height", type=int, default=500,
                        help="ROI 区域高度（像素，约为饮料瓶高度）")
    parser.add_argument("--roi-x", type=int, default=-1,
                        help="ROI 左上角 X 坐标（-1 表示居中）")
    parser.add_argument("--roi-y", type=int, default=-1,
                        help="ROI 左上角 Y 坐标（-1 表示居中）")
    
    # 双 ROI 配置（A/B 两个区域）
    parser.add_argument("--dual-roi", action="store_true",
                        help="启用双 ROI 区域检测（A/B 两个区域同时检测）")
    parser.add_argument("--roi-a-x", type=int, default=-1,
                        help="ROI A 左上角 X 坐标（-1 表示自动计算：左侧 1/4 位置）")
    parser.add_argument("--roi-a-y", type=int, default=-1,
                        help="ROI A 左上角 Y 坐标（-1 表示垂直居中）")
    parser.add_argument("--roi-b-x", type=int, default=-1,
                        help="ROI B 左上角 X 坐标（-1 表示居中）")
    parser.add_argument("--roi-b-y", type=int, default=-1,
                        help="ROI B 左上角 Y 坐标（-1 表示自动计算：下方位置）")
    parser.add_argument("--dual-gap", type=int, default=100,
                        help="双 ROI（上下）之间的垂直间距（像素, 默认100）")
    
    # 关键点匹配配置（用于未知类别检测）
    parser.add_argument("--keypoints-dir", type=str, default="keypoints",
                        help="关键点文件目录（包含各类别的 .pkl 文件）")
    parser.add_argument("--keypoint-method", type=str, default="orb",
                        choices=["orb", "sift"],
                        help="关键点检测算法: orb(快速) 或 sift(精确)")
    parser.add_argument("--min-matches", type=int, default=10,
                        help="最少匹配点数量，低于此值判定为未知 (默认10)")
    parser.add_argument("--match-ratio", type=float, default=0.75,
                        help="Lowe's ratio test 阈值 (默认0.75)")
    parser.add_argument("--unknown-label", type=str, default="Unknown",
                        help="未知类别显示标签")
    parser.add_argument("--min-model-conf", type=float, default=0.6,
                        help="模型最低置信度，低于此值直接判定为未知 (默认0.6)")
    
    # 串口通信配置
    parser.add_argument("--serial-port", type=str, default="",
                        help="串口端口号（如 COM3、/dev/ttyUSB0）。为空则不启用串口")
    parser.add_argument("--baud-rate", type=int, default=9600,
                        help="串口波特率 (默认9600)")
    parser.add_argument("--ack-timeout", type=float, default=2.0,
                        help="等待单片机确认(ACK)的超时秒数 (默认2.0)")
    parser.add_argument("--consecutive-frames", type=int, default=3,
                        help="连续识别相同结果的帧数要求 (默认3)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="超时后最大重发次数 (默认3)")
    
    return parser.parse_args()


class KeypointMatcher:
    """关键点匹配器"""
    
    def __init__(self, method="orb", ratio=0.75):
        self.method = method
        self.ratio = ratio
        
        # 初始化检测器
        if method == "orb":
            self.detector = cv2.ORB_create(nfeatures=500)
            # ORB 使用 Hamming 距离
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # sift
            self.detector = cv2.SIFT_create(nfeatures=500)
            # SIFT 使用 L2 距离
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def extract(self, image):
        """提取图像的关键点和描述子"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match(self, descriptors1, descriptors2):
        """
        匹配两组描述子，使用 Lowe's ratio test
        
        Returns:
            good_matches: 通过比率测试的匹配点列表
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) < 2 or len(descriptors2) < 2:
            return []
        
        try:
            # KNN 匹配，k=2
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.ratio * n.distance:
                        good_matches.append(m)
            
            return good_matches
        except Exception as e:
            return []


def load_keypoints(keypoints_dir, class_names):
    """
    加载关键点文件
    
    Args:
        keypoints_dir: 关键点文件目录
        class_names: 模型的类别名称列表
    
    Returns:
        dict: {类别名: {'descriptors': ..., 'method': ...}}
    """
    keypoints_data = {}
    
    if not os.path.exists(keypoints_dir):
        print(f"[警告] 关键点目录不存在: {keypoints_dir}")
        return keypoints_data
    
    for cls_name in class_names:
        pkl_path = os.path.join(keypoints_dir, f"{cls_name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            keypoints_data[cls_name] = data
            print(f"  已加载: {cls_name} ({data['total_keypoints']} 个关键点)")
        else:
            print(f"  [跳过] 未找到: {pkl_path}")
    
    return keypoints_data


def match_keypoints(current_descriptors, keypoints_data, class_name, matcher, min_matches=10):
    """
    关键点匹配验证
    
    Args:
        current_descriptors: 当前图像的描述子
        keypoints_data: 所有类别的关键点数据
        class_name: 模型预测的类别名
        matcher: KeypointMatcher 实例
        min_matches: 最少匹配点数量
    
    Returns:
        dict: 匹配结果
    """
    # 如果该类别没有关键点数据，不进行验证（视为通过）
    if class_name not in keypoints_data:
        return {
            'verified': True,
            'num_matches': -1,
            'message': 'no_keypoints_file'
        }
    
    if current_descriptors is None or len(current_descriptors) == 0:
        return {
            'verified': False,
            'num_matches': 0,
            'message': 'no_descriptors'
        }
    
    # 获取该类别的模板描述子
    template_data = keypoints_data[class_name]
    template_descriptors = template_data['descriptors']
    
    # 检查描述子类型是否匹配
    if template_data['method'] != matcher.method:
        print(f"[警告] 关键点方法不匹配: 模板={template_data['method']}, 当前={matcher.method}")
        return {
            'verified': False,
            'num_matches': 0,
            'message': 'method_mismatch'
        }
    
    # 进行匹配
    good_matches = matcher.match(current_descriptors, template_descriptors)
    num_matches = len(good_matches)
    
    # 判断是否通过验证
    verified = num_matches >= min_matches
    
    return {
        'verified': verified,
        'num_matches': num_matches,
        'min_required': min_matches,
        'message': 'ok' if verified else 'insufficient_matches'
    }


def scan_phone_camera():
    """扫描局域网内的手机摄像头"""
    print("正在扫描手机摄像头...")
    
    # 获取本机IP地址，推断局域网网段
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "192.168.1.1"
    
    # 提取网段前缀
    ip_prefix = ".".join(local_ip.split(".")[:3]) + "."
    
    # 常见手机摄像头 App 端口和路径
    phone_apps = [
        {"name": "IP Webcam", "port": 8080, "paths": ["/video", "/videofeed", ""]},
        {"name": "DroidCam", "port": 4747, "paths": ["/video", "/mjpegfeed", ""]},
        {"name": "Iriun", "port": 4747, "paths": ["/video", ""]},
        {"name": "EpocCam", "port": 8080, "paths": ["/video", ""]},
    ]
    
    found_sources = []
    
    # 扫描常见 IP 范围 (快速扫描)
    print(f"本机 IP: {local_ip}, 扫描网段: {ip_prefix}x")
    
    # 先尝试常见的 USB 网络共享 IP
    common_ips = [
        "192.168.42.129",  # Android USB 网络共享常用 IP
        "192.168.137.1",   # Windows 热点
        local_ip,          # 本机
    ]
    
    # 添加局域网常见 IP
    for i in [1, 2, 100, 101, 102, 103, 104, 105, 50, 51]:
        common_ips.append(f"{ip_prefix}{i}")
    
    for ip in common_ips:
        for app in phone_apps:
            for path in app["paths"]:
                url = f"http://{ip}:{app['port']}{path}"
                try:
                    # 快速测试连接
                    req = urllib.request.Request(url, method='HEAD')
                    urllib.request.urlopen(req, timeout=0.5)
                    print(f"  [发现] {app['name']}: {url}")
                    found_sources.append({"name": app["name"], "url": url})
                except Exception:
                    pass
    
    return found_sources


def get_video_source(opt):
    """获取视频源（摄像头索引或URL）"""
    
    # 优先使用指定的 source URL
    if opt.source:
        print(f"使用指定视频源: {opt.source}")
        return opt.source
    
    # 自动扫描手机摄像头
    if opt.phone:
        sources = scan_phone_camera()
        if sources:
            print(f"\n找到 {len(sources)} 个手机摄像头:")
            for i, src in enumerate(sources):
                print(f"  {i+1}. [{src['name']}] {src['url']}")
            
            if len(sources) == 1:
                selected = sources[0]["url"]
            else:
                try:
                    choice = input("\n请选择 (输入序号，默认1): ").strip()
                    idx = int(choice) - 1 if choice else 0
                    selected = sources[idx]["url"]
                except (ValueError, IndexError):
                    selected = sources[0]["url"]
            
            print(f"\n已选择: {selected}")
            return selected
        else:
            print("\n未找到手机摄像头，请检查:")
            print("  1. 手机和电脑是否在同一网络")
            print("  2. 手机 App 是否已启动服务器")
            print("  3. 手动指定: --source http://手机IP:端口/video")
            return None
    
    # 默认使用本地摄像头
    print(f"使用本地摄像头: 索引 {opt.camera}")
    return opt.camera


def get_roi_coords(frame_width, frame_height, roi_width, roi_height, roi_x, roi_y):
    """计算 ROI 区域坐标"""
    # 如果 x 或 y 为 -1，则居中
    if roi_x < 0:
        x1 = (frame_width - roi_width) // 2
    else:
        x1 = roi_x
    
    if roi_y < 0:
        y1 = (frame_height - roi_height) // 2
    else:
        y1 = roi_y
    
    # 确保不超出边界
    x1 = max(0, min(x1, frame_width - roi_width))
    y1 = max(0, min(y1, frame_height - roi_height))
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    
    return x1, y1, x2, y2


def get_dual_roi_coords(frame_width, frame_height, roi_width, roi_height, opt):
    """
    计算双 ROI 区域坐标（A 在上方，B 在下方）

    布局规则（默认）：
    - 水平居中（若用户未指定 roi_a_x/roi_b_x）
    - 两个 ROI 垂直排列，中间间距由 opt.dual_gap 控制
    - 若用户通过命令行指定了任意坐标则以用户坐标为准（并做边界裁剪）
    """
    gap = getattr(opt, 'dual_gap', 100)

    # 计算默认垂直起始位置（使两块 ROI 与间距整体居中）
    total_h = roi_height * 2 + gap
    start_y = (frame_height - total_h) // 2

    # A（上）坐标（默认水平居中）
    if opt.roi_a_x < 0:
        a_x1 = (frame_width - roi_width) // 2
    else:
        a_x1 = opt.roi_a_x

    if opt.roi_a_y < 0:
        a_y1 = start_y
    else:
        a_y1 = opt.roi_a_y

    # B（下）坐标（默认水平居中）
    if opt.roi_b_x < 0:
        b_x1 = (frame_width - roi_width) // 2
    else:
        b_x1 = opt.roi_b_x

    if opt.roi_b_y < 0:
        b_y1 = start_y + roi_height + gap
    else:
        b_y1 = opt.roi_b_y

    # 边界裁剪
    a_x1 = max(0, min(a_x1, frame_width - roi_width))
    a_y1 = max(0, min(a_y1, frame_height - roi_height))
    b_x1 = max(0, min(b_x1, frame_width - roi_width))
    b_y1 = max(0, min(b_y1, frame_height - roi_height))

    roi_a = (a_x1, a_y1, a_x1 + roi_width, a_y1 + roi_height)
    roi_b = (b_x1, b_y1, b_x1 + roi_width, b_y1 + roi_height)

    return roi_a, roi_b


def draw_roi(frame, x1, y1, x2, y2, color=(0, 255, 255), thickness=3, label="ROI"):
    """在画面上绘制 ROI 框"""
    # 绘制矩形框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制角标（让 ROI 框更明显）
    corner_len = 20
    # 左上角
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 2)
    # 右上角
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 2)
    # 左下角
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 2)
    # 右下角
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 2)
    
    # 绘制标签
    cv2.putText(frame, label, (x1 + 5, y1 - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def draw_predictions(frame, results, show_top5=False, conf_threshold=0.5,
                     keypoint_result=None, unknown_label="Unknown"):
    """在画面上绘制预测结果（支持关键点匹配的未知类别检测）"""
    probs = results[0].probs
    names = results[0].names
    
    # 获取Top-1预测
    top1_idx = probs.top1
    top1_conf = float(probs.top1conf)
    top1_name = names[top1_idx]
    
    # 未知类别检测（基于关键点匹配）
    is_unknown = False
    match_info = ""
    
    if keypoint_result is not None:
        is_unknown = keypoint_result['is_unknown']
        num_matches = keypoint_result.get('num_matches', 0)
        reason = keypoint_result.get('reason', '')
        
        if num_matches >= 0:
            match_info = f" (kp={num_matches})"
        
        if is_unknown and reason:
            match_info += f" [{reason}]"
    
    # 设置颜色和文本
    if is_unknown:
        color = (0, 0, 255)  # 红色 - 未知类别
        display_name = unknown_label
        text = f"{display_name}: ???{match_info}"
    elif top1_conf >= conf_threshold:
        color = (0, 255, 0)  # 绿色 - 高置信度
        display_name = top1_name
        text = f"{display_name}: {top1_conf:.2%}{match_info}"
    else:
        color = (0, 165, 255)  # 橙色 - 低置信度
        display_name = top1_name
        text = f"{display_name}: {top1_conf:.2%}{match_info}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # 绘制背景框
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 15 + text_h), font, font_scale, color, thickness)
    
    # 绘制Top-5预测（可选）
    if show_top5:
        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()
        
        y_offset = 80
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            cls_name = names[idx]
            text = f"{i+1}. {cls_name}: {conf:.2%}"
            
            # 颜色渐变
            alpha = 1.0 - (i * 0.15)
            text_color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
            
            cv2.putText(frame, text, (15, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # 返回结果
    if is_unknown:
        return frame, unknown_label, 0.0
    return frame, top1_name, top1_conf


def draw_dual_predictions(frame, result_a, result_b, conf_threshold=0.5, unknown_label="Unknown"):
    """
    绘制双 ROI 区域的预测结果
    
    Args:
        frame: 图像帧
        result_a: ROI A 的检测结果 dict
        result_b: ROI B 的检测结果 dict
        conf_threshold: 置信度阈值
        unknown_label: 未知类别标签
    
    Returns:
        frame: 绘制后的帧
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # 绘制 ROI A 结果（左上角）
    text_a = format_result_text("A", result_a, conf_threshold, unknown_label)
    color_a = get_result_color(result_a, conf_threshold)
    
    (text_w, text_h), baseline = cv2.getTextSize(text_a, font, font_scale, thickness)
    cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text_a, (15, 15 + text_h), font, font_scale, color_a, thickness)
    
    # 绘制 ROI B 结果（左上角第二行）
    text_b = format_result_text("B", result_b, conf_threshold, unknown_label)
    color_b = get_result_color(result_b, conf_threshold)
    
    y_offset = 25 + text_h + baseline
    (text_w, text_h), baseline = cv2.getTextSize(text_b, font, font_scale, thickness)
    cv2.rectangle(frame, (10, y_offset), (20 + text_w, y_offset + 10 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text_b, (15, y_offset + 5 + text_h), font, font_scale, color_b, thickness)
    
    return frame


def format_result_text(region_label, result, conf_threshold, unknown_label):
    """格式化单个区域的结果文本"""
    if result is None:
        return f"[{region_label}] ---"
    
    pred_name = result.get('name', '---')
    pred_conf = result.get('conf', 0.0)
    is_unknown = result.get('is_unknown', False)
    num_matches = result.get('num_matches', -1)
    
    match_info = ""
    if num_matches >= 0:
        match_info = f" (kp={num_matches})"
    
    if is_unknown:
        return f"[{region_label}] {unknown_label}: ???{match_info}"
    else:
        return f"[{region_label}] {pred_name}: {pred_conf:.2%}{match_info}"


def get_result_color(result, conf_threshold):
    """根据结果获取显示颜色"""
    if result is None:
        return (128, 128, 128)  # 灰色
    
    is_unknown = result.get('is_unknown', False)
    pred_conf = result.get('conf', 0.0)
    
    if is_unknown:
        return (0, 0, 255)  # 红色 - 未知
    elif pred_conf >= conf_threshold:
        return (0, 255, 0)  # 绿色 - 高置信度
    else:
        return (0, 165, 255)  # 橙色 - 低置信度


def detect_single_roi(model, roi_frame, opt, keypoints_data, keypoint_matcher):
    """
    对单个 ROI 区域进行检测
    
    Args:
        model: YOLO 模型
        roi_frame: ROI 区域图像
        opt: 命令行参数
        keypoints_data: 关键点数据
        keypoint_matcher: 关键点匹配器
    
    Returns:
        dict: 检测结果 {'name': str, 'conf': float, 'is_unknown': bool, 'num_matches': int}
    """
    # 模型推理
    results = model.predict(
        source=roi_frame,
        imgsz=opt.imgsz,
        device=opt.device,
        verbose=False,
    )
    
    # 获取预测结果
    probs = results[0].probs
    top1_name = model.names[probs.top1]
    top1_conf = float(probs.top1conf)
    
    result = {
        'name': top1_name,
        'conf': top1_conf,
        'is_unknown': False,
        'num_matches': -1
    }
    
    # 关键点匹配检测未知类别
    # 条件1: 模型置信度必须超过阈值
    if top1_conf < opt.min_model_conf:
        result['is_unknown'] = True
        result['reason'] = 'low_conf'
    # 条件2: 如果有关键点数据，进行关键点验证
    elif keypoints_data and keypoint_matcher:
        if top1_name in keypoints_data:
            _, current_descriptors = keypoint_matcher.extract(roi_frame)
            match_result = match_keypoints(
                current_descriptors, keypoints_data, 
                top1_name, keypoint_matcher, 
                min_matches=opt.min_matches
            )
            result['is_unknown'] = not match_result['verified']
            result['num_matches'] = match_result['num_matches']
            if not match_result['verified']:
                result['reason'] = match_result['message']
    
    return result


def open_serial(port, baud_rate):
    """
    打开串口连接
    
    Returns:
        serial.Serial 实例，失败返回 None
    """
    if not HAS_SERIAL:
        print("[错误] 未安装 pyserial，请运行: pip install pyserial")
        return None
    try:
        ser = serial.Serial(port, baud_rate, timeout=1, write_timeout=1)
        print(f"[串口] 已打开: {port} @ {baud_rate}bps")
        return ser
    except Exception as e:
        print(f"[错误] 无法打开串口 {port}: {e}")
        return None


def get_product_code(class_name):
    """
    根据类别名称查找商品编码
    
    Args:
        class_name: 模型输出的类名（英文或中文）
    
    Returns:
        int: 16 进制编码，未找到返回 0x00
    """
    return PRODUCT_CODE.get(class_name, 0x00)


def send_product_code(ser, code_a, code_b=None):
    """
    通过串口发送商品编码给单片机
    
    协议格式: [0xAA] [code_a] [code_b] [checksum] [0x55]
      - 0xAA  : 帧头
      - code_a: ROI A 商品编码 (1 字节)
      - code_b: ROI B 商品编码 (1 字节, 单 ROI 模式为 0x00)
      - checksum: code_a ^ code_b (异或校验)
      - 0x55  : 帧尾
    """
    if ser is None:
        return
    if code_b is None:
        code_b = 0x00
    checksum = code_a ^ code_b
    packet = bytes([0xAA, code_a, code_b, checksum, 0xBB])
    try:
        ser.write(packet)
    except Exception as e:
        print(f"[串口] 发送失败: {e}")


class SerialComm:
    """
    串口通信管理器 — 请求/应答协议

    单片机 → 电脑:
      请求识别: 0x55 0x01 0xBB   (3 字节)
      确认收到: 0x55 0x02 0xBB   (3 字节)

    电脑 → 单片机:
      商品信息: 0xAA [code_a] [code_b] [XOR] 0xBB  (5 字节)

    流程:
      1. 单片机发请求 → 电脑进入检测状态
      2. 连续 N 帧识别结果相同 → 电脑发送商品信息
      3. 电脑等待确认（超时则重发）
      4. 收到确认 → 回到空闲状态
    """

    STATE_IDLE      = 0  # 等待单片机请求
    STATE_DETECTING = 1  # 识别中（等待连续 N 帧相同）
    STATE_WAIT_ACK  = 2  # 等待单片机确认

    def __init__(self, ser, consecutive=3, ack_timeout=2.0, max_retries=3):
        self.ser = ser
        self.consecutive = consecutive
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries

        self.state = self.STATE_IDLE
        self._consec_count = 0
        self._last_codes = None
        self._last_packet = None
        self._send_time = 0
        self._retry_count = 0

        # 后台接收
        self._rx_buf = bytearray()
        self._rx_queue = queue.Queue()
        self._stopped = False
        self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._rx_thread.start()

    # ---------- 后台接收线程 ----------
    def _rx_worker(self):
        """持续读取串口并解析 [0x55 cmd 0xBB] 格式的单片机命令"""
        while not self._stopped:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self._rx_buf.extend(data)
                    self._parse_rx()
                else:
                    time.sleep(0.01)
            except Exception:
                if not self._stopped:
                    time.sleep(0.1)

    def _parse_rx(self):
        while len(self._rx_buf) >= 3:
            try:
                idx = self._rx_buf.index(0x55)
            except ValueError:
                self._rx_buf.clear()
                break
            if idx > 0:
                del self._rx_buf[:idx]
            if len(self._rx_buf) < 3:
                break
            if self._rx_buf[2] == 0xBB:
                cmd = self._rx_buf[1]
                del self._rx_buf[:3]
                self._rx_queue.put(cmd)
            else:
                del self._rx_buf[:1]

    # ---------- 主循环每帧调用 ----------
    def update(self, code_a, code_b=0x00):
        """
        驱动状态机，每帧调用。

        Returns:
            str: 状态描述（用于 OSD / 终端显示）
        """
        # 1. 处理接收队列
        while not self._rx_queue.empty():
            try:
                cmd = self._rx_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == 0x01:          # 单片机请求识别
                if self.state == self.STATE_IDLE:
                    self.state = self.STATE_DETECTING
                    self._consec_count = 0
                    self._last_codes = None
                    print("[串口] <<< 收到请求，开始检测...")
            elif cmd == 0x02:        # 单片机确认收到
                if self.state == self.STATE_WAIT_ACK:
                    self.state = self.STATE_IDLE
                    print("[串口] <<< 单片机已确认，回到空闲")

        # 2. 状态机
        if self.state == self.STATE_IDLE:
            return "IDLE"

        if self.state == self.STATE_DETECTING:
            codes = (code_a, code_b)
            if codes == self._last_codes:
                self._consec_count += 1
            else:
                self._last_codes = codes
                self._consec_count = 1
            if self._consec_count >= self.consecutive:
                self._do_send(code_a, code_b)
                self.state = self.STATE_WAIT_ACK
                self._send_time = time.time()
                self._retry_count = 0
                return "SENT"
            return f"DET {self._consec_count}/{self.consecutive}"

        if self.state == self.STATE_WAIT_ACK:
            if time.time() - self._send_time > self.ack_timeout:
                self._retry_count += 1
                if self._retry_count <= self.max_retries:
                    self._resend()
                    self._send_time = time.time()
                    return f"RETRY {self._retry_count}/{self.max_retries}"
                else:
                    self.state = self.STATE_IDLE
                    print("[串口] 超过最大重试次数，回到空闲")
                    return "TIMEOUT"
            return "WAIT_ACK"

        return ""

    # ---------- 发送 ----------
    def _do_send(self, code_a, code_b):
        checksum = code_a ^ code_b
        self._last_packet = bytes([0xAA, code_a, code_b, checksum, 0xBB])
        try:
            self.ser.write(self._last_packet)
            print(f"[串口] >>> 已发送: A=0x{code_a:02X} B=0x{code_b:02X}")
        except Exception as e:
            print(f"[串口] 发送失败: {e}")

    def _resend(self):
        if self._last_packet:
            try:
                self.ser.write(self._last_packet)
                print(f"[串口] >>> 重发 ({self._retry_count}/{self.max_retries})")
            except Exception as e:
                print(f"[串口] 重发失败: {e}")

    def stop(self):
        self._stopped = True
        self._rx_thread.join(timeout=2)


def draw_info(frame, fps, paused=False, comm_status=""):
    """绘制FPS和状态信息"""
    h, w = frame.shape[:2]
    
    # FPS（右上角）
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 暂停状态
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # 串口通信状态（底部上方）
    if comm_status:
        if "IDLE" in comm_status:
            sc_color = (180, 180, 180)
        elif "DET" in comm_status:
            sc_color = (0, 255, 255)     # 黄色
        elif "SENT" in comm_status or "WAIT_ACK" in comm_status:
            sc_color = (255, 255, 0)     # 青色
        elif "RETRY" in comm_status:
            sc_color = (0, 165, 255)     # 橙色
        elif "TIMEOUT" in comm_status:
            sc_color = (0, 0, 255)       # 红色
        else:
            sc_color = (200, 200, 200)
        cv2.putText(frame, f"COMM: {comm_status}", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc_color, 2)
    
    # 操作提示（底部）
    tips = "Q:Quit | S:Screenshot | Space:Pause"
    cv2.putText(frame, tips, (10, h - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame


def main():
    opt = parse_args()
    
    # 检查模型文件
    if not os.path.exists(opt.weights):
        print(f"[错误] 模型文件不存在: {opt.weights}")
        print("请指定正确的模型路径，例如:")
        print("  python camera_classify.py --weights runs/classify/exp/weights/best.pt")
        return
    
    # 创建截图目录
    os.makedirs(opt.save_dir, exist_ok=True)
    
    # 加载模型
    print("=" * 60)
    print("YOLOv8 分类模型 - 摄像头实时识别")
    print("=" * 60)
    print(f"模型: {opt.weights}")
    print(f"设备: {opt.device}")
    print("=" * 60)
    
    model = YOLO(opt.weights)
    
    # 获取类别信息
    class_names = list(model.names.values())
    print(f"类别列表: {model.names}")
    print("=" * 60)
    
    # 加载关键点数据（用于未知类别检测）
    keypoints_data = {}
    keypoint_matcher = None
    if os.path.exists(opt.keypoints_dir):
        print(f"加载关键点数据: {opt.keypoints_dir}")
        keypoints_data = load_keypoints(opt.keypoints_dir, class_names)
        if keypoints_data:
            keypoint_matcher = KeypointMatcher(
                method=opt.keypoint_method, 
                ratio=opt.match_ratio
            )
            print(f"关键点算法: {opt.keypoint_method.upper()}")
            print(f"最少匹配点: {opt.min_matches}")
            print(f"Ratio阈值: {opt.match_ratio}")
            print(f"最低置信度: {opt.min_model_conf}")
            print("=" * 60)
        else:
            print("[警告] 未找到任何关键点文件")
            print("=" * 60)
    else:
        print(f"[提示] 关键点目录不存在: {opt.keypoints_dir}")
        print("  如需启用未知类别检测，请先采集关键点:")
        print(f"  python capture_keypoints.py --class-name BaiShi --output {opt.keypoints_dir}")
        print("=" * 60)
    
    # 确定视频源
    video_source = get_video_source(opt)
    if video_source is None:
        return
    
    # 显示视频源信息
    if isinstance(video_source, str):
        print(f"视频源: {video_source} (网络流)")
    else:
        print(f"视频源: 摄像头索引 {video_source}")
    print(f"分辨率设置: {opt.width}x{opt.height}")
    print("=" * 60)
    
    # 打开摄像头/视频流
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {video_source}")
        print("\n可能的解决方案:")
        print("  1. 本地摄像头: python camera_classify.py --camera 0")
        print("  2. 外接摄像头: python camera_classify.py --camera 1")
        print("  3. 手机摄像头: python camera_classify.py --source http://手机IP:端口/video")
        print("  4. 自动扫描:   python camera_classify.py --phone")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少延迟
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"实际分辨率: {actual_width}x{actual_height}")
    if opt.rotate:
        print(f"画面旋转: {opt.rotate}°")
        if opt.rotate in (90, 270):
            print(f"旋转后分辨率: {actual_height}x{actual_width}")
    
    # 显示 ROI 模式
    if opt.dual_roi:
        print("ROI 模式: 双区域 (A/B)")
    elif opt.roi:
        print("ROI 模式: 单区域")
    else:
        print("ROI 模式: 全画面")
    
    # 初始化串口（请求/应答协议）
    ser = None
    serial_comm = None
    if opt.serial_port:
        ser = open_serial(opt.serial_port, opt.baud_rate)
        if ser:
            serial_comm = SerialComm(
                ser,
                consecutive=opt.consecutive_frames,
                ack_timeout=opt.ack_timeout,
                max_retries=opt.max_retries,
            )
            print(f"[串口] 请求/应答模式 (连续{opt.consecutive_frames}帧, 超时{opt.ack_timeout}s)")
    else:
        print("[串口] 未指定 --serial-port，串口通信已禁用")
    print("按 Q 或 ESC 退出，按 S 截图，按 空格 暂停")
    print("-" * 60)
    
    # 主循环
    paused = False
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    last_frame = None
    pred_conf = 0.0  # 初始化预测置信度
    pred_name = ""   # 初始化预测类名
    comm_status = "" # 串口通信状态
    
    # 双 ROI 模式的结果
    result_a = None
    result_b = None
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[警告] 无法读取摄像头画面")
                    break
                # 旋转画面（手机竖拍时将横屏流转为竖屏）
                if opt.rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif opt.rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif opt.rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                last_frame = frame.copy()
            else:
                frame = last_frame.copy() if last_frame is not None else None
                if frame is None:
                    continue
            
            # 计算FPS
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()
            
            # 获取帧尺寸
            h, w = frame.shape[:2]
            
            # ========== 双 ROI 模式 ==========
            if opt.dual_roi:
                # 计算双 ROI 区域坐标
                roi_a, roi_b = get_dual_roi_coords(
                    w, h, opt.roi_width, opt.roi_height, opt
                )
                a_x1, a_y1, a_x2, a_y2 = roi_a
                b_x1, b_y1, b_x2, b_y2 = roi_b
                
                # 提取 ROI 区域图像
                roi_frame_a = frame[a_y1:a_y2, a_x1:a_x2].copy()
                roi_frame_b = frame[b_y1:b_y2, b_x1:b_x2].copy()
                
                # 推理
                if not paused:
                    # ===== 检测 ROI A =====
                    result_a = detect_single_roi(
                        model, roi_frame_a, opt, 
                        keypoints_data, keypoint_matcher
                    )
                    
                    # ===== 检测 ROI B =====
                    result_b = detect_single_roi(
                        model, roi_frame_b, opt, 
                        keypoints_data, keypoint_matcher
                    )
                    
                    # 绘制双区域预测结果
                    frame = draw_dual_predictions(
                        frame, result_a, result_b,
                        conf_threshold=opt.conf,
                        unknown_label=opt.unknown_label
                    )
                    
                    # 获取商品编码
                    code_a = get_product_code(result_a.get('name', '')) if result_a and not result_a.get('is_unknown') else 0x00
                    code_b = get_product_code(result_b.get('name', '')) if result_b and not result_b.get('is_unknown') else 0x00
                    if serial_comm:
                        comm_status = serial_comm.update(code_a, code_b)
                    
                    # 终端输出
                    if frame_count == 1:
                        name_a = result_a.get('name', '---') if result_a else '---'
                        name_b = result_b.get('name', '---') if result_b else '---'
                        hex_a = f"0x{code_a:02X}"
                        hex_b = f"0x{code_b:02X}"
                        cs = f" [{comm_status}]" if comm_status else ""
                        print(f"[A] {name_a}({hex_a}) | [B] {name_b}({hex_b}){cs} | FPS: {fps:.1f}")
                
                # 绘制双 ROI 框
                color_a = get_result_color(result_a, opt.conf)
                color_b = get_result_color(result_b, opt.conf)
                frame = draw_roi(frame, a_x1, a_y1, a_x2, a_y2, color_a, label="A")
                frame = draw_roi(frame, b_x1, b_y1, b_x2, b_y2, color_b, label="B")
            
            # ========== 单 ROI 模式 ==========
            elif opt.roi:
                roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_coords(
                    w, h, opt.roi_width, opt.roi_height, opt.roi_x, opt.roi_y
                )
                # 提取 ROI 区域图像用于推理
                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                
                # 推理
                if not paused:
                    results = model.predict(
                        source=roi_frame,
                        imgsz=opt.imgsz,
                        device=opt.device,
                        verbose=False,
                    )
                    
                    # 获取模型预测结果
                    probs = results[0].probs
                    top1_name = model.names[probs.top1]
                    top1_conf = float(probs.top1conf)
                    
                    # 关键点匹配检测未知类别
                    keypoint_result = None
                    
                    # 条件1: 模型置信度必须超过阈值
                    if top1_conf < opt.min_model_conf:
                        keypoint_result = {
                            'is_unknown': True,
                            'num_matches': -1,
                            'reason': 'low_conf'
                        }
                    # 条件2: 如果有关键点数据，进行关键点验证
                    elif keypoints_data and keypoint_matcher:
                        if top1_name in keypoints_data:
                            _, current_descriptors = keypoint_matcher.extract(roi_frame)
                            match_result = match_keypoints(
                                current_descriptors, keypoints_data, 
                                top1_name, keypoint_matcher, 
                                min_matches=opt.min_matches
                            )
                            keypoint_result = {
                                'is_unknown': not match_result['verified'],
                                'num_matches': match_result['num_matches'],
                                'reason': match_result['message'] if not match_result['verified'] else ''
                            }
                    
                    # 绘制预测结果
                    frame, pred_name, pred_conf = draw_predictions(
                        frame, results, 
                        show_top5=opt.show_top5,
                        conf_threshold=opt.conf,
                        keypoint_result=keypoint_result,
                        unknown_label=opt.unknown_label
                    )
                    
                    # 获取商品编码
                    code = get_product_code(pred_name) if pred_name != opt.unknown_label else 0x00
                    if serial_comm:
                        comm_status = serial_comm.update(code)
                    
                    # 终端输出
                    if frame_count == 1:
                        hex_code = f"0x{code:02X}"
                        cs = f" [{comm_status}]" if comm_status else ""
                        print(f"预测: {pred_name}({hex_code}) ({pred_conf:.2%}){cs} | FPS: {fps:.1f}")
                
                # 绘制 ROI 框
                if not paused and pred_conf >= opt.conf:
                    roi_color = (0, 255, 0)
                else:
                    roi_color = (0, 255, 255)
                frame = draw_roi(frame, roi_x1, roi_y1, roi_x2, roi_y2, roi_color, label="ROI")
            
            # ========== 全画面模式 ==========
            else:
                if not paused:
                    results = model.predict(
                        source=frame,
                        imgsz=opt.imgsz,
                        device=opt.device,
                        verbose=False,
                    )
                    
                    probs = results[0].probs
                    top1_name = model.names[probs.top1]
                    top1_conf = float(probs.top1conf)
                    
                    keypoint_result = None
                    if top1_conf < opt.min_model_conf:
                        keypoint_result = {
                            'is_unknown': True,
                            'num_matches': -1,
                            'reason': 'low_conf'
                        }
                    elif keypoints_data and keypoint_matcher:
                        if top1_name in keypoints_data:
                            _, current_descriptors = keypoint_matcher.extract(frame)
                            match_result = match_keypoints(
                                current_descriptors, keypoints_data, 
                                top1_name, keypoint_matcher, 
                                min_matches=opt.min_matches
                            )
                            keypoint_result = {
                                'is_unknown': not match_result['verified'],
                                'num_matches': match_result['num_matches'],
                                'reason': match_result['message'] if not match_result['verified'] else ''
                            }
                    
                    frame, pred_name, pred_conf = draw_predictions(
                        frame, results, 
                        show_top5=opt.show_top5,
                        conf_threshold=opt.conf,
                        keypoint_result=keypoint_result,
                        unknown_label=opt.unknown_label
                    )
                    
                    # 获取商品编码
                    code = get_product_code(pred_name) if pred_name != opt.unknown_label else 0x00
                    if serial_comm:
                        comm_status = serial_comm.update(code)
                    
                    if frame_count == 1:
                        hex_code = f"0x{code:02X}"
                        cs = f" [{comm_status}]" if comm_status else ""
                        print(f"预测: {pred_name}({hex_code}) ({pred_conf:.2%}){cs} | FPS: {fps:.1f}")
            
            # 绘制信息
            frame = draw_info(frame, fps, paused, comm_status)
            
            # 显示画面
            if not opt.no_show:
                cv2.imshow("YOLOv8 Classification", frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC 退出
                print("\n退出程序")
                break
            elif key == ord('s'):  # S 截图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                filepath = os.path.join(opt.save_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"[截图] 已保存: {filepath}")
            elif key == ord(' '):  # 空格 暂停/继续
                paused = not paused
                print(f"[状态] {'暂停' if paused else '继续'}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if serial_comm:
            serial_comm.stop()
        if ser:
            ser.close()
            print("[串口] 已关闭")
        print("摄像头已释放")


if __name__ == "__main__":
    main()
