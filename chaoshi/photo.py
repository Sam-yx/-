"""
摄像头拍照工具

支持：本地摄像头、USB 外接摄像头、手机摄像头（通过 USB/WiFi）

=== 手机摄像头使用方法 ===
1. 在手机上安装 IP Webcam (Android) 或 DroidCam 等 App
2. 手机通过 USB 连接电脑，并开启 USB 网络共享
   或者手机与电脑连接同一 WiFi
3. 在手机 App 中启动服务器，获取视频流地址
4. 使用 --source 参数指定视频流 URL

使用示例:
    python photo.py                                    # 使用默认摄像头
    1                         # 使用外接摄像头
    python photo.py --source http://192.168.1.100:8080/video  # 手机摄像头
    python photo.py --phone                            # 自动扫描手机摄像头
    python photo.py --phone --rotate 90                # 手机画面旋转为横屏

快捷键:
    空格键 : 拍照保存
    ESC    : 退出
"""

import cv2
import os
import argparse
import socket
import urllib.request
import time
from datetime import datetime


def parse_args()  :
    parser = argparse.ArgumentParser(description="摄像头拍照工具")
    parser.add_argument("--camera", type=int, default=2,
                        help="摄像头索引 (0=内置, 1=外接)")
    parser.add_argument("--source", type=str, default="",
                        help="视频源 URL (手机摄像头流地址)")
    parser.add_argument("--phone", action="store_true",
                        help="自动扫描并连接手机摄像头")
    parser.add_argument("--save-dir", type=str, default="my_photos/14/",  # 保存目录",
                        help="照片保存目录")
    parser.add_argument("--prefix", type=str, default="14",   
                        help="照片文件名前缀")
    parser.add_argument("--width", type=int, default=1280,
                        help="摄像头画面宽度")
    parser.add_argument("--height", type=int, default=720,
                        help="摄像头画面高度")
    parser.add_argument("--rotate", type=int, default=-1,
                        choices=[-1, 0, 90, 180, 270],
                        help="画面旋转角度，-1=自动(手机摄像头默认90度)")
    parser.add_argument("--interval", type=float, default=0,
                        help="定时拍照间隔（秒），<=0 表示关闭定时拍照")
    parser.add_argument("--max-photos", type=int, default=0,
                        help="定时拍照最大张数，<=0 表示不限制")
    return parser.parse_args()


def rotate_frame(frame, angle):
    """按指定角度旋转画面。"""
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


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
    
    # USB 网络共享常用 IP
    common_ips = ["192.168.42.129", "192.168.137.1", local_ip]
    for i in [1, 2, 100, 101, 102, 103, 104, 105]:
        common_ips.append(f"{ip_prefix}{i}")
    
    print(f"本机 IP: {local_ip}")
    
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
    """获取视频源"""
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
            print("\n未找到手机摄像头，使用本地摄像头")
            return opt.camera
    
    return opt.camera


def capture_photo(save_dir='C:\\cv\\my_photos', camera_index=0):
    """
    使用 OpenCV 打开摄像头，按空格键拍照保存，按 ESC 键退出。
    """
    opt = parse_args()
    save_dir = opt.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取视频源
    video_source = get_video_source(opt)
    
    print("=" * 50)
    if isinstance(video_source, str):
        print(f"视频源: {video_source} (网络流)")
    else:
        print(f"视频源: 摄像头索引 {video_source}")

    if opt.rotate == -1:
        rotate_angle = 90 if isinstance(video_source, str) else 0
    else:
        rotate_angle = opt.rotate

    print(f"画面旋转: {rotate_angle} 度")
    print(f"保存目录: {save_dir}")
    if opt.interval > 0:
        print(f"定时拍照: 开启 (每 {opt.interval} 秒一张)")
        if opt.max_photos > 0:
            print(f"最大张数: {opt.max_photos}")
        else:
            print("最大张数: 不限制")
    else:
        print("定时拍照: 关闭")
    # ROI 功能已移除，始终保存全画面
    print("=" * 50)

    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查设备连接。")
        print("\n可能的解决方案:")
        print("  1. 本地摄像头: python photo.py --camera 0")
        print("  2. 外接摄像头: python photo.py --camera 1")
        print("  3. 手机摄像头: python photo.py --source http://手机IP:端口/video")
        print("  4. 自动扫描:   python photo.py --phone")
        return

    if isinstance(video_source, int):
        # 对 USB 摄像头优先使用 MJPG，可减少驱动回退到低分辨率的情况
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)

    # 打印请求分辨率、驱动上报分辨率、真实读取帧分辨率
    print("请求分辨率:", opt.width, opt.height)
    print("驱动上报分辨率:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 预热几帧后再读取，拿到更稳定的真实分辨率
    for _ in range(3):
        cap.read()
    ret, first_frame = cap.read()
    if not ret:
        print("错误：无法读取摄像头画面。")
        cap.release()
        return
    real_h, real_w = first_frame.shape[:2]
    print("真实帧分辨率:", real_w, real_h)

    print("摄像头已打开：")
    print("  [空格键] 拍照保存")
    print("  [ESC]    退出")
    if opt.interval > 0:
        print("  [自动]   按设定时间间隔自动拍照")

    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    photo_count = 0
    last_capture_time = time.time()

    def save_current_frame(current_frame):
        nonlocal photo_count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_count += 1
        filename = f"{opt.prefix}{photo_count}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        # 保存整张图片（已移除 ROI 裁剪）
        cv2.imwrite(filepath, current_frame)
        print(f"已保存照片: {filepath}")

    frame = first_frame
    while True:
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面。")
                break

        frame = rotate_frame(frame, rotate_angle)
        display = frame.copy()
        # ROI 功能已移除；显示原图预览

        cv2.putText(display, "Space: Capture | ESC: Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Photos: {photo_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if opt.interval > 0:
            elapsed = time.time() - last_capture_time
            remaining = max(0.0, opt.interval - elapsed)
            cv2.putText(display, f"Auto every {opt.interval:.1f}s | Next: {remaining:.1f}s",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera Preview", display)

        if opt.interval > 0:
            now = time.time()
            if (now - last_capture_time) >= opt.interval:
                save_current_frame(frame)
                last_capture_time = now
                if opt.max_photos > 0 and photo_count >= opt.max_photos:
                    print("已达到最大拍照张数，自动结束。")
                    break

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("退出拍照。")
            break
        elif key == 32:  # 空格键
            save_current_frame(frame)
            last_capture_time = time.time()
            if opt.max_photos > 0 and photo_count >= opt.max_photos:
                print("已达到最大拍照张数，自动结束。")
                break

        # 下一轮开始前清空，继续读取新帧
        frame = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_photo()
