"""
YOLOv8 检测模型 - 摄像头实时识别

使用训练好的检测模型，通过摄像头进行实时目标检测。
支持：本地摄像头、USB 外接摄像头、手机摄像头（通过 USB/WiFi）

=== 手机摄像头使用方法 ===
1. 在手机上安装 IP Webcam (Android) 或 DroidCam 等 App
2. 手机通过 USB 连接电脑，并开启 USB 网络共享
   或者手机与电脑连接同一 WiFi
3. 在手机 App 中启动服务器，获取视频流地址
4. 使用 --source 参数指定视频流 URL

=== ROI 区域检测 ===
使用 --roi 参数启用 ROI 区域检测，只对画面中心的指定区域进行检测。
适合固定位置放置饮料瓶进行识别的场景。

=== 手机竖屏拍摄（看两层货架） ===
手机 App 推流通常是横屏，加 --rotate 90 可将画面转为竖屏。
示例：python camera_detect.py --dual-roi --rotate 90 --source http://手机IP:8080/video

使用示例:
    # 本地摄像头 + 全画面检测
    python camera_detect.py --camera 0

    # 手机摄像头 + 单 ROI
    python camera_detect.py --roi --source http://192.168.1.100:8080/video

    # 双 ROI + 串口通信
    python camera_detect.py --dual-roi --serial-port COM3

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
    "RIO_mitao": 0x30,   # RIO蜜桃
    "RIOPuTao": 0x31,    # RIO葡萄
    "koko": 0x50,        # koko
    "xuebi": 0x22,       # 雪碧
    # --- 未知 ---
    "未知": 0x00,
    "Unknown": 0x00,
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 检测模型摄像头实时识别")

    # 模型配置
    parser.add_argument("--weights", type=str,
                        default="runs/detect/bottle_train/weights/best.pt",
                        help="检测模型权重路径")

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
    parser.add_argument("--imgsz", type=int, default=640,
                        help="模型输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备 (0/1/cpu)")

    # 显示配置
    parser.add_argument("--no-show", action="store_true",
                        help="不显示画面（仅终端输出）")

    # 保存配置
    parser.add_argument("--save-dir", type=str, default="screenshots",
                        help="截图保存目录")

    # ROI 配置（单区域）
    parser.add_argument("--roi", action="store_true",
                        help="启用单 ROI 区域检测（只对 ROI 区域进行检测）")
    parser.add_argument("--roi-width", type=int, default=200,
                        help="ROI 区域宽度（像素）")
    parser.add_argument("--roi-height", type=int, default=500,
                        help="ROI 区域高度（像素）")
    parser.add_argument("--roi-x", type=int, default=-1,
                        help="ROI 左上角 X 坐标（-1 表示居中）")
    parser.add_argument("--roi-y", type=int, default=-1,
                        help="ROI 左上角 Y 坐标（-1 表示居中）")

    # 双 ROI 配置（A/B 两个区域）
    parser.add_argument("--dual-roi", action="store_true",
                        help="启用双 ROI 区域检测（A/B 两个区域同时检测）")
    parser.add_argument("--roi-a-x", type=int, default=-1,
                        help="ROI A 左上角 X 坐标（-1 表示自动计算）")
    parser.add_argument("--roi-a-y", type=int, default=-1,
                        help="ROI A 左上角 Y 坐标（-1 表示垂直居中）")
    parser.add_argument("--roi-b-x", type=int, default=-1,
                        help="ROI B 左上角 X 坐标（-1 表示居中）")
    parser.add_argument("--roi-b-y", type=int, default=-1,
                        help="ROI B 左上角 Y 坐标（-1 表示自动计算）")
    parser.add_argument("--dual-gap", type=int, default=100,
                        help="双 ROI（上下）之间的垂直间距（像素, 默认100）")

    # 多帧确认配置
    parser.add_argument("--confirm-frames", type=int, default=5,
                        help="连续多少帧检测到相同结果才确认 (默认5)")
    parser.add_argument("--iou-match", type=float, default=0.5,
                        help="多帧匹配时判定同一目标的 IoU 阈值")

    # 串口通信配置
    parser.add_argument("--serial-port", type=str, default="",
                        help="串口端口号（如 COM3、/dev/ttyUSB0）。为空则不启用串口")
    parser.add_argument("--baud-rate", type=int, default=9600,
                        help="串口波特率 (默认9600)")
    parser.add_argument("--ack-timeout", type=float, default=2.0,
                        help="等待单片机确认(ACK)的超时秒数 (默认2.0)")
    parser.add_argument("--consecutive-frames", type=int, default=3,
                        help="串口发送前连续识别相同结果的帧数要求 (默认3)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="超时后最大重发次数 (默认3)")

    return parser.parse_args()


# ================================================================
#  多帧确认追踪器 (从 realtime_detect.py 复用)
# ================================================================

def _box_iou(b1, b2):
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


class MultiFrameTracker:
    """
    多帧确认机制：
    - 每帧的检测结果与已有追踪目标按 IoU + 类别 进行匹配
    - 连续 confirm_frames 帧匹配成功 → 标记为"已确认"
    - 只有已确认的目标才会被显示和发送串口
    """

    def __init__(self, confirm_frames=5, iou_thresh=0.5):
        self.confirm_frames = confirm_frames
        self.iou_thresh = iou_thresh
        self.tracks = []
        self._next_id = 0
        self.printed_ids = set()

    def update(self, detections):
        """
        输入: detections = [(class_name, conf, [x1,y1,x2,y2]), ...]
        返回: confirmed = [(class_name, conf, [x1,y1,x2,y2], track_id), ...]
        """
        used_det = set()

        # 1. 匹配现有 track
        for ti, trk in enumerate(self.tracks):
            best_iou = 0
            best_di = -1
            for di, (cls, conf, box) in enumerate(detections):
                if di in used_det:
                    continue
                if cls != trk["cls"]:
                    continue
                iou = _box_iou(box, trk["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_iou >= self.iou_thresh and best_di >= 0:
                trk["hits"] += 1
                trk["miss"] = 0
                trk["box"] = detections[best_di][2]
                trk["conf"] = detections[best_di][1]
                used_det.add(best_di)
            else:
                trk["miss"] += 1

        # 2. 新建 track（未匹配的检测）
        for di, (cls, conf, box) in enumerate(detections):
            if di not in used_det:
                self.tracks.append({
                    "id": self._next_id,
                    "cls": cls,
                    "conf": conf,
                    "box": box,
                    "hits": 1,
                    "miss": 0,
                })
                self._next_id += 1

        # 3. 清除连续丢失的 track
        self.tracks = [t for t in self.tracks if t["miss"] <= self.confirm_frames]

        # 4. 筛选已确认的
        confirmed = []
        for trk in self.tracks:
            if trk["hits"] >= self.confirm_frames:
                confirmed.append((trk["cls"], trk["conf"], trk["box"], trk["id"]))
        return confirmed


# ================================================================
#  手机摄像头扫描
# ================================================================

def scan_phone_camera():
    """扫描局域网内的手机摄像头"""
    print("正在扫描手机摄像头...")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "192.168.1.1"

    ip_prefix = ".".join(local_ip.split(".")[:3]) + "."

    phone_apps = [
        {"name": "IP Webcam", "port": 8080, "paths": ["/video", "/videofeed", ""]},
        {"name": "DroidCam", "port": 4747, "paths": ["/video", "/mjpegfeed", ""]},
        {"name": "Iriun", "port": 4747, "paths": ["/video", ""]},
        {"name": "EpocCam", "port": 8080, "paths": ["/video", ""]},
    ]

    found_sources = []
    print(f"本机 IP: {local_ip}, 扫描网段: {ip_prefix}x")

    common_ips = [
        "192.168.42.129",
        "192.168.137.1",
        local_ip,
    ]
    for i in [1, 2, 100, 101, 102, 103, 104, 105, 50, 51]:
        common_ips.append(f"{ip_prefix}{i}")

    for ip in common_ips:
        for app in phone_apps:
            for path in app["paths"]:
                url = f"http://{ip}:{app['port']}{path}"
                try:
                    req = urllib.request.Request(url, method='HEAD')
                    urllib.request.urlopen(req, timeout=0.5)
                    print(f"  [发现] {app['name']}: {url}")
                    found_sources.append({"name": app["name"], "url": url})
                except Exception:
                    pass

    return found_sources


def get_video_source(opt):
    """获取视频源（摄像头索引或URL）"""
    if opt.source:
        print(f"使用指定视频源: {opt.source}")
        return opt.source

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

    print(f"使用本地摄像头: 索引 {opt.camera}")
    return opt.camera


# ================================================================
#  ROI 区域坐标计算
# ================================================================

def get_roi_coords(frame_width, frame_height, roi_width, roi_height, roi_x, roi_y):
    """计算 ROI 区域坐标"""
    if roi_x < 0:
        x1 = (frame_width - roi_width) // 2
    else:
        x1 = roi_x

    if roi_y < 0:
        y1 = (frame_height - roi_height) // 2
    else:
        y1 = roi_y

    x1 = max(0, min(x1, frame_width - roi_width))
    y1 = max(0, min(y1, frame_height - roi_height))
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    return x1, y1, x2, y2


def get_dual_roi_coords(frame_width, frame_height, roi_width, roi_height, opt):
    """计算双 ROI 区域坐标（A 在上方，B 在下方）"""
    gap = getattr(opt, 'dual_gap', 100)
    total_h = roi_height * 2 + gap
    start_y = (frame_height - total_h) // 2

    if opt.roi_a_x < 0:
        a_x1 = (frame_width - roi_width) // 2
    else:
        a_x1 = opt.roi_a_x

    if opt.roi_a_y < 0:
        a_y1 = start_y
    else:
        a_y1 = opt.roi_a_y

    if opt.roi_b_x < 0:
        b_x1 = (frame_width - roi_width) // 2
    else:
        b_x1 = opt.roi_b_x

    if opt.roi_b_y < 0:
        b_y1 = start_y + roi_height + gap
    else:
        b_y1 = opt.roi_b_y

    a_x1 = max(0, min(a_x1, frame_width - roi_width))
    a_y1 = max(0, min(a_y1, frame_height - roi_height))
    b_x1 = max(0, min(b_x1, frame_width - roi_width))
    b_y1 = max(0, min(b_y1, frame_height - roi_height))

    roi_a = (a_x1, a_y1, a_x1 + roi_width, a_y1 + roi_height)
    roi_b = (b_x1, b_y1, b_x1 + roi_width, b_y1 + roi_height)

    return roi_a, roi_b


# ================================================================
#  绘制辅助函数
# ================================================================

# 颜色映射（最多支持 20 个类别，循环使用）
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
]
_class_color_map = {}


def get_class_color(cls_name):
    """为类别分配一个固定颜色"""
    if cls_name not in _class_color_map:
        _class_color_map[cls_name] = COLORS[len(_class_color_map) % len(COLORS)]
    return _class_color_map[cls_name]


def draw_roi(frame, x1, y1, x2, y2, color=(0, 255, 255), thickness=3, label="ROI"):
    """在画面上绘制 ROI 框"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    corner_len = 20
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 2)

    cv2.putText(frame, label, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def draw_detections(frame, confirmed, offset_x=0, offset_y=0):
    """
    在画面上绘制已确认的检测框

    Args:
        frame: 画面
        confirmed: MultiFrameTracker 返回的已确认列表
        offset_x, offset_y: ROI 模式下的坐标偏移量（将 ROI 内坐标映射回原图）

    Returns:
        frame: 绘制后的画面
    """
    for cls_name, conf, box, track_id in confirmed:
        color = get_class_color(cls_name)

        x1 = int(box[0]) + offset_x
        y1 = int(box[1]) + offset_y
        x2 = int(box[2]) + offset_x
        y2 = int(box[3]) + offset_y

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_info(frame, fps, paused=False, comm_status="", num_confirmed=0):
    """绘制FPS和状态信息"""
    h, w = frame.shape[:2]

    # FPS + 确认数（右上角）
    info_text = f"FPS: {fps:.1f} | Confirmed: {num_confirmed}"
    cv2.putText(frame, info_text, (w - 320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if comm_status:
        if "IDLE" in comm_status:
            sc_color = (180, 180, 180)
        elif "DET" in comm_status:
            sc_color = (0, 255, 255)
        elif "SENT" in comm_status or "WAIT_ACK" in comm_status:
            sc_color = (255, 255, 0)
        elif "RETRY" in comm_status:
            sc_color = (0, 165, 255)
        elif "TIMEOUT" in comm_status:
            sc_color = (0, 0, 255)
        else:
            sc_color = (200, 200, 200)
        cv2.putText(frame, f"COMM: {comm_status}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc_color, 2)

    tips = "Q:Quit | S:Screenshot | Space:Pause"
    cv2.putText(frame, tips, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


# ================================================================
#  检测推理辅助
# ================================================================

def run_detect(model, source_frame, opt):
    """
    对一帧图像进行目标检测

    Args:
        model: YOLO 检测模型
        source_frame: 输入图像
        opt: 命令行参数

    Returns:
        detections: [(class_name, conf, [x1,y1,x2,y2]), ...]
    """
    results = model.predict(
        source=source_frame,
        imgsz=opt.imgsz,
        conf=opt.conf,
        iou=opt.iou,
        device=opt.device,
        verbose=False,
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = r.names[cls_id]
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append((cls_name, conf, [x1, y1, x2, y2]))

    return detections


def get_best_detection(confirmed):
    """
    从已确认的检测结果中取置信度最高的目标

    Returns:
        (class_name, conf) 或 (None, 0.0)
    """
    if not confirmed:
        return None, 0.0
    best = max(confirmed, key=lambda x: x[1])
    return best[0], best[1]


# ================================================================
#  商品编码 & 串口通信
# ================================================================

def get_product_code(class_name):
    """根据类别名称查找商品编码"""
    return PRODUCT_CODE.get(class_name, 0x00)


def open_serial(port, baud_rate):
    """打开串口连接"""
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


class SerialComm:
    """
    串口通信管理器 — 请求/应答协议

    单片机 → 电脑:
      请求识别: 0x55 0x01 0xBB
      确认收到: 0x55 0x02 0xBB

    电脑 → 单片机:
      商品信息: 0xAA [code_a] [code_b] [XOR] 0xBB
    """

    STATE_IDLE      = 0
    STATE_DETECTING = 1
    STATE_WAIT_ACK  = 2

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

        self._rx_buf = bytearray()
        self._rx_queue = queue.Queue()
        self._stopped = False
        self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._rx_thread.start()

    def _rx_worker(self):
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

    def update(self, code_a, code_b=0x00):
        """驱动状态机，每帧调用"""
        while not self._rx_queue.empty():
            try:
                cmd = self._rx_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == 0x01:
                if self.state == self.STATE_IDLE:
                    self.state = self.STATE_DETECTING
                    self._consec_count = 0
                    self._last_codes = None
                    print("[串口] <<< 收到请求，开始检测...")
            elif cmd == 0x02:
                if self.state == self.STATE_WAIT_ACK:
                    self.state = self.STATE_IDLE
                    print("[串口] <<< 单片机已确认，回到空闲")

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


# ================================================================
#  主流程
# ================================================================

def main():
    opt = parse_args()

    # 检查模型文件
    if not os.path.exists(opt.weights):
        print(f"[错误] 模型文件不存在: {opt.weights}")
        print("请指定正确的模型路径，例如:")
        print("  python camera_detect.py --weights runs/detect/bottle_train/weights/best.pt")
        return

    # 创建截图目录
    os.makedirs(opt.save_dir, exist_ok=True)

    # 加载模型
    print("=" * 60)
    print("YOLOv8 检测模型 - 摄像头实时识别")
    print("=" * 60)
    print(f"模型: {opt.weights}")
    print(f"设备: {opt.device}")
    print("=" * 60)

    model = YOLO(opt.weights)

    # 获取类别信息
    print(f"类别列表: {model.names}")
    print("=" * 60)

    # 确定视频源
    video_source = get_video_source(opt)
    if video_source is None:
        return

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
        print("  1. 本地摄像头: python camera_detect.py --camera 0")
        print("  2. 外接摄像头: python camera_detect.py --camera 1")
        print("  3. 手机摄像头: python camera_detect.py --source http://手机IP:端口/video")
        print("  4. 自动扫描:   python camera_detect.py --phone")
        return

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"实际分辨率: {actual_width}x{actual_height}")
    if opt.rotate:
        print(f"画面旋转: {opt.rotate}°")
        if opt.rotate in (90, 270):
            print(f"旋转后分辨率: {actual_height}x{actual_width}")

    if opt.dual_roi:
        print("ROI 模式: 双区域 (A/B)")
    elif opt.roi:
        print("ROI 模式: 单区域")
    else:
        print("ROI 模式: 全画面")

    print(f"多帧确认: 连续 {opt.confirm_frames} 帧")

    # 初始化串口
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

    # 初始化多帧追踪器
    # 全画面 / 单ROI 各一个 tracker；双ROI 模式各一个
    if opt.dual_roi:
        tracker_a = MultiFrameTracker(opt.confirm_frames, opt.iou_match)
        tracker_b = MultiFrameTracker(opt.confirm_frames, opt.iou_match)
    else:
        tracker = MultiFrameTracker(opt.confirm_frames, opt.iou_match)

    # 主循环
    paused = False
    frame_count = 0
    fps = 0.0
    fps_start_time = time.time()
    last_frame = None
    comm_status = ""

    # 缓存上一帧已确认结果（暂停时复用）
    last_confirmed = []
    last_confirmed_a = []
    last_confirmed_b = []

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[警告] 无法读取摄像头画面")
                    break
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

            # 计算 FPS
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()

            h, w = frame.shape[:2]

            # ========== 双 ROI 模式 ==========
            if opt.dual_roi:
                roi_a, roi_b = get_dual_roi_coords(
                    w, h, opt.roi_width, opt.roi_height, opt
                )
                a_x1, a_y1, a_x2, a_y2 = roi_a
                b_x1, b_y1, b_x2, b_y2 = roi_b

                if not paused:
                    # 检测 ROI A
                    roi_frame_a = frame[a_y1:a_y2, a_x1:a_x2].copy()
                    dets_a = run_detect(model, roi_frame_a, opt)
                    last_confirmed_a = tracker_a.update(dets_a)

                    # 检测 ROI B
                    roi_frame_b = frame[b_y1:b_y2, b_x1:b_x2].copy()
                    dets_b = run_detect(model, roi_frame_b, opt)
                    last_confirmed_b = tracker_b.update(dets_b)

                    # 串口：各区域取置信度最高的目标
                    best_a_name, _ = get_best_detection(last_confirmed_a)
                    best_b_name, _ = get_best_detection(last_confirmed_b)
                    code_a = get_product_code(best_a_name) if best_a_name else 0x00
                    code_b = get_product_code(best_b_name) if best_b_name else 0x00
                    if serial_comm:
                        comm_status = serial_comm.update(code_a, code_b)

                    # 终端输出
                    if frame_count == 1:
                        na = best_a_name or "---"
                        nb = best_b_name or "---"
                        cs = f" [{comm_status}]" if comm_status else ""
                        print(f"[A] {na}(0x{code_a:02X}) | [B] {nb}(0x{code_b:02X}){cs} | FPS: {fps:.1f}")

                # 绘制检测框（坐标偏移到原图）
                frame = draw_detections(frame, last_confirmed_a, offset_x=a_x1, offset_y=a_y1)
                frame = draw_detections(frame, last_confirmed_b, offset_x=b_x1, offset_y=b_y1)

                # 绘制 ROI 框
                frame = draw_roi(frame, a_x1, a_y1, a_x2, a_y2, (0, 255, 255), label="A")
                frame = draw_roi(frame, b_x1, b_y1, b_x2, b_y2, (255, 255, 0), label="B")

                total_confirmed = len(last_confirmed_a) + len(last_confirmed_b)

            # ========== 单 ROI 模式 ==========
            elif opt.roi:
                roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_coords(
                    w, h, opt.roi_width, opt.roi_height, opt.roi_x, opt.roi_y
                )

                if not paused:
                    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                    dets = run_detect(model, roi_frame, opt)
                    last_confirmed = tracker.update(dets)

                    best_name, best_conf = get_best_detection(last_confirmed)
                    code = get_product_code(best_name) if best_name else 0x00
                    if serial_comm:
                        comm_status = serial_comm.update(code)

                    if frame_count == 1:
                        n = best_name or "---"
                        cs = f" [{comm_status}]" if comm_status else ""
                        print(f"预测: {n}(0x{code:02X}) ({best_conf:.2%}){cs} | FPS: {fps:.1f}")

                # 绘制检测框（坐标偏移到原图）
                frame = draw_detections(frame, last_confirmed, offset_x=roi_x1, offset_y=roi_y1)

                # 绘制 ROI 框
                has_confirmed = len(last_confirmed) > 0
                roi_color = (0, 255, 0) if has_confirmed else (0, 255, 255)
                frame = draw_roi(frame, roi_x1, roi_y1, roi_x2, roi_y2, roi_color, label="ROI")

                total_confirmed = len(last_confirmed)

            # ========== 全画面模式 ==========
            else:
                if not paused:
                    dets = run_detect(model, frame, opt)
                    last_confirmed = tracker.update(dets)

                    best_name, best_conf = get_best_detection(last_confirmed)
                    code = get_product_code(best_name) if best_name else 0x00
                    if serial_comm:
                        comm_status = serial_comm.update(code)

                    if frame_count == 1:
                        n = best_name or "---"
                        cs = f" [{comm_status}]" if comm_status else ""
                        print(f"预测: {n}(0x{code:02X}) ({best_conf:.2%}){cs} | FPS: {fps:.1f}")

                frame = draw_detections(frame, last_confirmed)
                total_confirmed = len(last_confirmed)

            # 终端打印新确认的目标（每个 track_id 只打印一次）
            all_confirmed = last_confirmed if not opt.dual_roi else (last_confirmed_a + last_confirmed_b)
            active_tracker_list = [tracker] if not opt.dual_roi else [tracker_a, tracker_b]
            for cls_name, conf, box, track_id in all_confirmed:
                for trk in active_tracker_list:
                    if track_id not in trk.printed_ids:
                        trk.printed_ids.add(track_id)
                        bx = [int(v) for v in box]
                        print(f"  [确认] {cls_name}  置信度: {conf:.2f}  "
                              f"位置: ({bx[0]},{bx[1]})-({bx[2]},{bx[3]})")

            # 绘制信息
            frame = draw_info(frame, fps, paused, comm_status,
                              num_confirmed=total_confirmed)

            # 显示画面
            if not opt.no_show:
                cv2.imshow("YOLOv8 Detection", frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\n退出程序")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                filepath = os.path.join(opt.save_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"[截图] 已保存: {filepath}")
            elif key == ord(' '):
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