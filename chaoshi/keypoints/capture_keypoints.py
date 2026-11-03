"""
饮料瓶关键点采集工具

使用摄像头采集饮料瓶的关键点信息，并保存到文件中。
支持：本地摄像头、USB 外接摄像头、手机摄像头（通过 USB/WiFi）
采用 ORB 或 SIFT 算法提取关键点和描述子。

使用方法:
    1. 运行脚本，将饮料瓶放入 ROI 区域
    2. 按 C 键采集当前帧的关键点
    3. 可以从多个角度采集多次
    4. 按 S 键保存关键点文件
    5. 按 Q 键退出

示例:
    # 采集百事可乐的关键点
    python capture_keypoints.py --class-name BaiShi --output keypoints/
    
    # 使用 SIFT 算法（更精确但较慢）
    python capture_keypoints.py --class-name BaiShi --method sift
    
    # 指定摄像头
    python capture_keypoints.py --class-name WangZai --camera 1
    
    # 使用手机摄像头（自动扫描）
    python capture_keypoints.py --class-name BaiShi --phone
    
    # 使用手机摄像头（指定 URL）
    python capture_keypoints.py --class-name BaiShi --source http://192.168.42.129:8080/video
"""

import cv2
import argparse
import pickle
import os
import numpy as np
import socket
import urllib.request
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="饮料瓶关键点采集工具")
    
    parser.add_argument("--class-name", type=str, required=True,
                        help="类别名称（必须与训练时一致，如 BaiShi, WangZai）")
    parser.add_argument("--output", type=str, default="keypoints",
                        help="关键点文件保存目录")
    
    # 摄像头配置
    parser.add_argument("--camera", type=int, default=2,
                        help="摄像头索引")
    parser.add_argument("--source", type=str, default="",
                        help="视频源 URL (手机摄像头流地址，如 http://192.168.42.129:8080/video)")
    parser.add_argument("--phone", action="store_true",
                        help="自动扫描并连接手机摄像头")
    parser.add_argument("--width", type=int, default=1920,
                        help="画面宽度")
    parser.add_argument("--height", type=int, default=1080,
                        help="画面高度")
    
    # ROI 配置
    parser.add_argument("--roi-width", type=int, default=400,
                        help="ROI 区域宽度")
    parser.add_argument("--roi-height", type=int, default=600,
                        help="ROI 区域高度")
    
    # 关键点算法配置
    parser.add_argument("--method", type=str, default="orb",
                        choices=["orb", "sift"],
                        help="关键点检测算法: orb(快速) 或 sift(精确)")
    parser.add_argument("--max-keypoints", type=int, default=500,
                        help="每帧最大关键点数量")
    
    return parser.parse_args()


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
            print("  1. 手机和电脑是否在同一网络 (或 USB 网络共享已开启)")
            print("  2. 手机 App 是否已启动服务器")
            print("  3. 手动指定: --source http://手机IP:端口/video")
            return None
    
    # 默认使用本地摄像头
    print(f"使用本地摄像头: 索引 {opt.camera}")
    return opt.camera


class KeypointExtractor:
    """关键点提取器"""
    
    def __init__(self, method="orb", max_keypoints=500):
        self.method = method
        self.max_keypoints = max_keypoints
        
        if method == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_keypoints)
        else:  # sift
            self.detector = cv2.SIFT_create(nfeatures=max_keypoints)
    
    def extract(self, image):
        """
        提取图像的关键点和描述子
        
        Returns:
            keypoints: 关键点列表
            descriptors: 描述子数组
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 提取关键点和描述子
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def serialize_keypoints(self, keypoints):
        """将关键点序列化为可保存的格式"""
        kp_list = []
        for kp in keypoints:
            kp_list.append({
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            })
        return kp_list
    
    def deserialize_keypoints(self, kp_list):
        """从序列化格式恢复关键点"""
        keypoints = []
        for kp_dict in kp_list:
            kp = cv2.KeyPoint(
                x=kp_dict['pt'][0],
                y=kp_dict['pt'][1],
                size=kp_dict['size'],
                angle=kp_dict['angle'],
                response=kp_dict['response'],
                octave=kp_dict['octave'],
                class_id=kp_dict['class_id']
            )
            keypoints.append(kp)
        return keypoints


def get_roi_coords(frame_width, frame_height, roi_width, roi_height):
    """计算居中的 ROI 坐标"""
    roi_x1 = (frame_width - roi_width) // 2
    roi_y1 = (frame_height - roi_height) // 2
    roi_x2 = roi_x1 + roi_width
    roi_y2 = roi_y1 + roi_height
    return roi_x1, roi_y1, roi_x2, roi_y2


def draw_keypoints_info(frame, num_keypoints, num_samples, class_name):
    """绘制关键点信息"""
    h, w = frame.shape[:2]
    
    # 信息文本
    info_lines = [
        f"Class: {class_name}",
        f"Keypoints: {num_keypoints}",
        f"Samples: {num_samples}",
        "",
        "C: Capture | S: Save | Q: Quit"
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(frame, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    return frame


def main():
    opt = parse_args()
    
    # 创建输出目录
    os.makedirs(opt.output, exist_ok=True)
    
    # 初始化关键点提取器
    extractor = KeypointExtractor(method=opt.method, max_keypoints=opt.max_keypoints)
    
    print("=" * 60)
    print("饮料瓶关键点采集工具")
    print("=" * 60)
    print(f"类别名称: {opt.class_name}")
    print(f"检测算法: {opt.method.upper()}")
    print(f"最大关键点: {opt.max_keypoints}")
    print(f"输出目录: {opt.output}")
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
    print("=" * 60)
    
    # 打开摄像头/视频流
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {video_source}")
        print("\n可能的解决方案:")
        print("  1. 本地摄像头: python capture_keypoints.py --class-name XXX --camera 0")
        print("  2. 外接摄像头: python capture_keypoints.py --class-name XXX --camera 1")
        print("  3. 手机摄像头: python capture_keypoints.py --class-name XXX --source http://手机IP:端口/video")
        print("  4. 自动扫描:   python capture_keypoints.py --class-name XXX --phone")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)
    
    # 采集的关键点数据
    collected_samples = []
    
    print("\n操作说明:")
    print("  C - 采集当前帧的关键点")
    print("  S - 保存关键点文件")
    print("  Q - 退出")
    print("-" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 无法读取画面")
                break
            
            h, w = frame.shape[:2]
            
            # 计算 ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_coords(
                w, h, opt.roi_width, opt.roi_height
            )
            
            # 提取 ROI 区域
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            # 提取关键点（用于显示）
            keypoints, descriptors = extractor.extract(roi_frame)
            num_keypoints = len(keypoints) if keypoints else 0
            
            # 在 ROI 上绘制关键点
            roi_with_kp = cv2.drawKeypoints(
                roi_frame, keypoints, None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # 将带关键点的 ROI 放回原图
            display_frame = frame.copy()
            display_frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi_with_kp
            
            # 绘制 ROI 框
            cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), 
                         (0, 255, 255), 2)
            
            # 绘制信息
            display_frame = draw_keypoints_info(
                display_frame, num_keypoints, len(collected_samples), opt.class_name
            )
            
            cv2.imshow("Keypoint Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                print("\n退出采集")
                break
            
            elif key == ord('c'):
                # 采集当前帧的关键点
                if descriptors is not None and len(descriptors) > 0:
                    sample = {
                        'keypoints': extractor.serialize_keypoints(keypoints),
                        'descriptors': descriptors,
                        'num_keypoints': num_keypoints,
                        'timestamp': datetime.now().isoformat()
                    }
                    collected_samples.append(sample)
                    print(f"[采集] 样本 #{len(collected_samples)}: {num_keypoints} 个关键点")
                else:
                    print("[警告] 未检测到关键点，请调整饮料瓶位置")
            
            elif key == ord('s'):
                # 保存关键点文件
                if len(collected_samples) == 0:
                    print("[警告] 没有采集任何样本")
                    continue
                
                # 合并所有样本的描述子
                all_descriptors = []
                all_keypoints = []
                
                for sample in collected_samples:
                    all_descriptors.append(sample['descriptors'])
                    all_keypoints.extend(sample['keypoints'])
                
                merged_descriptors = np.vstack(all_descriptors)
                
                # 保存数据
                save_data = {
                    'class_name': opt.class_name,
                    'method': opt.method,
                    'num_samples': len(collected_samples),
                    'total_keypoints': len(all_keypoints),
                    'descriptors': merged_descriptors,
                    'keypoints': all_keypoints,
                    'samples': collected_samples,
                    'created_at': datetime.now().isoformat()
                }
                
                output_path = os.path.join(opt.output, f"{opt.class_name}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(save_data, f)
                
                print(f"\n[保存成功]")
                print(f"  文件: {output_path}")
                print(f"  样本数: {len(collected_samples)}")
                print(f"  总关键点数: {len(all_keypoints)}")
                print(f"  描述子形状: {merged_descriptors.shape}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 如果有未保存的数据，提示保存
        if len(collected_samples) > 0:
            print(f"\n注意: 有 {len(collected_samples)} 个样本未保存")
            save_anyway = input("是否保存? (y/n): ").strip().lower()
            if save_anyway == 'y':
                all_descriptors = [s['descriptors'] for s in collected_samples]
                all_keypoints = []
                for s in collected_samples:
                    all_keypoints.extend(s['keypoints'])
                
                merged_descriptors = np.vstack(all_descriptors)
                
                save_data = {
                    'class_name': opt.class_name,
                    'method': opt.method,
                    'num_samples': len(collected_samples),
                    'total_keypoints': len(all_keypoints),
                    'descriptors': merged_descriptors,
                    'keypoints': all_keypoints,
                    'samples': collected_samples,
                    'created_at': datetime.now().isoformat()
                }
                
                output_path = os.path.join(opt.output, f"{opt.class_name}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(save_data, f)
                print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()
