"""
=================================================================
  实时饮料瓶检测 (手机摄像头 / 本地摄像头)

  - 使用 YOLO 模型进行实时推理
  - 支持: 本地摄像头 / USB 摄像头 / 手机摄像头 (IP Webcam / DroidCam)
  - 多帧确认机制: 连续 N 帧检测到同一目标后才显示框 + 终端输出

使用示例:
    python realtime_detect.py --model best.pt --phone
    python realtime_detect.py --model best.pt --source http://192.168.1.100:8080/video
    python realtime_detect.py --model best.pt --camera 0

快捷键:
    ESC / Q : 退出
=================================================================
"""

import argparse
import socket
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="实时饮料瓶检测")

    # --- 模型 ---
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型权重路径, 例如 best.pt")

    # --- 视频源 ---
    parser.add_argument("--camera", type=int, default=2,
                        help="本地摄像头索引 (0=内置, 1=外接)")
    parser.add_argument("--source", type=str, default="",
                        help="视频源 URL (手机摄像头流地址)")
    parser.add_argument("--phone", action="store_true",
                        help="自动扫描并连接手机摄像头")

    # --- 画面 ---
    parser.add_argument("--width", type=int, default=1920,
                        help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=1080,
                        help="摄像头采集高度")
    parser.add_argument("--rotate", type=int, default=0,
                        choices=[0, 90, 180, 270],
                        help="画面旋转角度 (90=顺时针90°, 270=逆时针90°, 180=翻转)")
    parser.add_argument("--resize-w", type=int, default=0,
                        help="检测前 resize 目标宽度，0 表示不启用")
    parser.add_argument("--resize-h", type=int, default=0,
                        help="检测前 resize 目标高度，0 表示不启用")
    parser.add_argument("--resize-scale", type=float, default=0.0,
                        help="检测前按比例缩放，例如 0.5，0 表示不启用")

    # --- 检测参数 ---
    parser.add_argument("--conf", type=float, default=0.9,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.9,
                        help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1920,
                        help="推理图像尺寸 (长边)")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备 (0=GPU, cpu=CPU)")

    # --- 多帧确认 ---
    parser.add_argument("--confirm-frames", type=int, default=5,
                        help="需要连续多少帧检测到相同结果才确认 (默认 5)")
    parser.add_argument("--iou-match", type=float, default=0.5,
                        help="多帧匹配时, 判定同一目标的 IoU 阈值")

    return parser.parse_args()


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
    ]

    found_sources = []
    common_ips = ["192.168.42.129", "192.168.137.1", local_ip]
    for i in [1, 2, 100, 101, 102, 103, 104, 105]:
        common_ips.append(f"{ip_prefix}{i}")

    print(f"本机 IP: {local_ip}")

    for ip in common_ips:
        for app in phone_apps:
            for path in app["paths"]:
                url = f"http://{ip}:{app['port']}{path}"
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    urllib.request.urlopen(req, timeout=0.5)
                    print(f"  [发现] {app['name']}: {url}")
                    found_sources.append({"name": app["name"], "url": url})
                except Exception:
                    pass

    return found_sources


def get_video_source(opt):
    """根据参数获取视频源"""
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

        print("\n未找到手机摄像头，使用本地摄像头")
        return opt.camera

    return opt.camera


def rotate_frame(frame, angle):
    """按指定角度旋转画面"""
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def resolve_resize_target(opt, frame_w, frame_h):
    """根据参数解析检测前 resize 的目标尺寸；未启用时返回 None。"""
    if opt.resize_w > 0 and opt.resize_h > 0:
        return opt.resize_w, opt.resize_h
    if opt.resize_scale > 0:
        tw = max(1, int(round(frame_w * opt.resize_scale)))
        th = max(1, int(round(frame_h * opt.resize_scale)))
        return tw, th
    return None


def letterbox_for_detect(frame, target_w, target_h):
    """保持比例缩放并补边，返回处理后的图像和映射信息。"""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_w": w,
        "orig_h": h,
    }
    return canvas, meta


def map_box_back(box, meta):
    """将检测图坐标映射回原图坐标。"""
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    x1 = (box[0] - pad_x) / scale
    y1 = (box[1] - pad_y) / scale
    x2 = (box[2] - pad_x) / scale
    y2 = (box[3] - pad_y) / scale

    x1 = max(0.0, min(float(orig_w - 1), x1))
    y1 = max(0.0, min(float(orig_h - 1), y1))
    x2 = max(0.0, min(float(orig_w - 1), x2))
    y2 = max(0.0, min(float(orig_h - 1), y2))
    return [x1, y1, x2, y2]


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
    - 每帧检测结果与已有追踪目标按 IoU + 类别 进行匹配
    - 连续 confirm_frames 帧匹配成功后标记为已确认
    - 只有已确认的目标才会显示和打印
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
        for trk in self.tracks:
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

        # 4. 筛选已确认
        confirmed = []
        for trk in self.tracks:
            if trk["hits"] >= self.confirm_frames:
                confirmed.append((trk["cls"], trk["conf"], trk["box"], trk["id"]))
        return confirmed


def main():
    opt = parse_args()

    # ---------- 加载模型 ----------
    print(f"[INFO] 加载模型: {opt.model}")
    model = YOLO(opt.model)

    # ---------- 视频源 ----------
    video_source = get_video_source(opt)

    # ---------- 打开视频 ----------
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("错误：无法打开视频源，请检查设备连接。")
        print("  --source http://手机IP:端口/video")
        print("  --phone  自动扫描")
        print("  --camera 0  本地摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)

    print("=" * 55)
    src_desc = (f"{video_source} (网络流)"
                if isinstance(video_source, str)
                else f"摄像头索引 {video_source}")
    print(f"  视频源     : {src_desc}")
    print(f"  画面旋转   : {opt.rotate}°")
    print(f"  推理尺寸   : {opt.imgsz}")
    if (opt.resize_w > 0 and opt.resize_h > 0) or (opt.resize_scale > 0):
        if opt.resize_w > 0 and opt.resize_h > 0:
            print(f"  检测前缩放 : 开启 ({opt.resize_w}x{opt.resize_h}, 保持比例+补边)")
        else:
            print(f"  检测前缩放 : 开启 (scale={opt.resize_scale}, 保持比例+补边)")
    else:
        print("  检测前缩放 : 关闭")
    print(f"  置信度阈值 : {opt.conf}")
    print(f"  多帧确认   : 连续 {opt.confirm_frames} 帧")
    print(f"  设备       : {opt.device}")
    print("=" * 55)
    print("  [ESC / Q] 退出")
    print("=" * 55)

    tracker = MultiFrameTracker(
        confirm_frames=opt.confirm_frames,
        iou_thresh=opt.iou_match,
    )

    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 255, 0), (255, 0, 255), (128, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
    ]
    class_color_map = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 无法读取画面，尝试重连...")
            cap.release()
            cap = cv2.VideoCapture(video_source)
            continue

        frame = rotate_frame(frame, opt.rotate)
        h, w = frame.shape[:2]
        display = frame.copy()

        def run_detect(img):
            results = model.predict(
                source=img,
                imgsz=opt.imgsz,
                conf=opt.conf,
                iou=opt.iou,
                device=opt.device,
                verbose=False,
            )
            dets = []
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = r.names[cls_id]
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    dets.append((cls_name, conf, [x1, y1, x2, y2]))
            return dets

        resize_target = resolve_resize_target(opt, w, h)
        if resize_target is not None:
            detect_frame, detect_meta = letterbox_for_detect(
                frame, resize_target[0], resize_target[1]
            )
            dets = run_detect(detect_frame)
            dets = [(cls, conf, map_box_back(box, detect_meta))
                    for cls, conf, box in dets]
        else:
            dets = run_detect(frame)

        confirmed = tracker.update(dets)

        for cls_name, conf, box, track_id in confirmed:
            if cls_name not in class_color_map:
                class_color_map[cls_name] = colors[len(class_color_map) % len(colors)]
            color = class_color_map[cls_name]

            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if track_id not in tracker.printed_ids:
                tracker.printed_ids.add(track_id)
                print(f"  [确认] {cls_name}  置信度: {conf:.2f}  "
                      f"位置: ({x1},{y1})-({x2},{y2})")

        cv2.putText(display, f"Confirmed: {len(confirmed)} | ESC: Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Realtime Detect", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q") or key == ord("Q"):
            print("\n退出检测。")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
