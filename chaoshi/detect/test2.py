"""
==============================================================
 统一视觉识别系统 —— 小车三合一控制端（检测版）
==============================================================
功能：
  1. OCR 模式    —— 拍照 → PaddleOCR → 大模型提取商品列表 → 串口返回
  2. 饮料检测模式 —— YOLOv8 目标检测 + 双ROI + MultiFrameTracker → 串口返回
  3. 人脸识别模式 —— 人脸+全图特征匹配 → 实时串口推送坐标

与 unified_vision.py 的区别：
  - 饮料模式从 **分类(ClassifyMode)** 改为 **检测(DetectMode)**
  - 使用 MultiFrameTracker (IoU追踪) 替代连续N帧计数
  - 绘制方式从单行文字 → bbox 矩形框 + 类名标签
  - 删除关键点匹配（检测模型自带类别）
  - 默认 imgsz 720x1280，新增 --iou 参数

================================================================
 ★★★ 串口通信协议（MCU ↔ PC）★★★
================================================================
  
【格式】  帧头 0xAA + 命令/类型 + 数据 + XOR校验 + 帧尾 0xBB
【校验】  XOR = 命令字节 XOR 每个数据字节（不含帧头帧尾）

------- MCU → PC -----------------------------------------
  3字节帧: 0xAA [CMD] 0xBB
    CMD=0x01  切换到 OCR 文字识别模式
    CMD=0x02  切换到 饮料瓶检测模式
    CMD=0x03  切换到 人脸识别模式
    CMD=0x04  OCR模式：触发拍照
                CMD=0x05  停止帧应答（饮料/人脸模式收到0x51后）
        CMD=0x06  饮料检测模式：抓取帧应答（继续识别）
    CMD=0x07  饮料检测模式：目标丢失帧应答（收到0x55后恢复识别）

  4字节帧: 0xAA [CMD] [XOR] 0xBB
    CMD=0xA2  微调结束（人脸模式）

------- PC → MCU ----------------------------------------

  模式确认（5字节）：
    0xAA  0xF0  [mode]  0xF1  0xBB
    mode: 0x01=OCR 0x02=饮料 0x03=人脸 0x04=OCR触发拍照ACK
    校验位固定为 0xF1

  OCR 结果（5字节）：
    0xAA  0x11  [flag]  [XOR]  0xBB
    flag: 0x01=识别到商品, 0x00=未识别到

  饮料检测-位置（6字节）：
    0xAA  0x52  [dxH][dxL]  [XOR]  0xBB
    dxH/dxL: 有符号16位，目标中心X与图片中心X的差值

    饮料检测-停止（4字节）：
        0xAA  0x51  [XOR]  0xBB

  饮料检测-抓取（4字节）：
        0xAA  0x53  [code]  0xBB
        code: 0x03=RIO_shuimitao, 0x04=Baishi, 0x05=Wangzai, 0x06=Weitanai
                    其他目标保持旧协议 0x53

  饮料检测-目标丢失（4字节）：
      0xAA  0x55  0x55  0xBB
      已开始发送位置帧后，若连续多帧未识别到目标则发送
      20ms重发，直到MCU回复0x07

    人脸停止（4字节）：
        0xAA  0x51  0x51  0xBB
        匹配到目标且 |Δx| < 停止阈值后发送，20ms重发直到MCU回复0x05

  人脸位置（6字节）：
    0xAA  0x52  [dxH][dxL]  0x63  0xBB
    dxH/dxL: 有符号16位，人脸中心X与图片中心X的差值
              正值=人脸在图片中心右侧，负值=左侧
    校验位固定为 0x63
        MCU回复0x05后开始发送；若人脸暂时丢失则暂停发送，重识别后继续
        20ms重发，直到 |Δx| < face-align-threshold

  人脸放货（4字节）：
    0xAA  0x53  0x53  0xBB
    |Δx| < 阈值后发送，20ms重发直到MCU发模式切换

  OCR模式-识别到饮料瓶（4字节）：
    0xAA  0x52  0x52  0xBB
    固定帧，20ms重发直到MCU发模式切换帧

  PC收到确认（4字节）：
    0xAA  0xB1  [XOR]  0xBB
    PC回复MCU的退出识别(0xA4)命令确认

================================================================
运行：
  python unified_detect.py --port COM3
  python unified_detect.py --port COM3 --camera 0 --baud 115200
  python unified_detect.py --no-serial              # 键盘模拟
================================================================
"""

import argparse
import importlib
import os
import sys

# 解决 Windows 上 libomp / libiomp5md 双 OpenMP 运行时冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_WINDOWS_DLL_HANDLES = []
_WINDOWS_DLL_DIRS = set()


def _add_windows_dll_dir(path):
    if sys.platform != "win32" or not path or not hasattr(os, "add_dll_directory"):
        return
    abs_path = os.path.abspath(path)
    norm_path = os.path.normcase(abs_path)
    if norm_path in _WINDOWS_DLL_DIRS or not os.path.isdir(abs_path):
        return
    try:
        handle = os.add_dll_directory(abs_path)
    except OSError:
        return
    _WINDOWS_DLL_HANDLES.append(handle)
    _WINDOWS_DLL_DIRS.add(norm_path)
    os.environ["PATH"] = abs_path + os.pathsep + os.environ.get("PATH", "")


def _setup_windows_inference_runtime():
    if sys.platform != "win32":
        return

    lib_dir = os.path.dirname(os.__file__)
    site_candidates = []
    for site_dir in (
        os.path.join(lib_dir, "site-packages"),
        os.path.join(sys.prefix, "Lib", "site-packages"),
    ):
        norm_site = os.path.normcase(os.path.abspath(site_dir))
        if norm_site not in site_candidates:
            site_candidates.append(norm_site)

    dll_candidates = []
    for site_dir in site_candidates:
        dll_candidates.append(os.path.join(site_dir, "torch", "lib"))
        dll_candidates.append(os.path.join(site_dir, "tensorrt_libs"))

    dll_candidates.append(os.path.join(sys.prefix, "Library", "bin"))
    cuda_path = os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    )
    dll_candidates.append(os.path.join(cuda_path, "bin"))

    for dll_dir in dll_candidates:
        _add_windows_dll_dir(dll_dir)


def _is_tensorrt_engine_path(path):
    return str(path).lower().endswith(".engine")


# 解决 Windows 上 torch / TensorRT 运行时 DLL 查找问题
if sys.platform == "win32":
    _setup_windows_inference_runtime()
    import torch as _torch_preload              # noqa: F401

import re
import json
import time
import threading
import queue
import cv2
import numpy as np
from datetime import datetime
from collections import deque

# =================== 可选依赖（缺失只禁用对应模式） ===================
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import face_recognition as face_rec
    HAS_FACE = True
except ImportError:
    HAS_FACE = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import pickle
    HAS_PICKLE = True
except ImportError:
    HAS_PICKLE = False

try:
    import torch
    LIGHTGLUE_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "LightGlue-0.2")
    )
    if LIGHTGLUE_ROOT not in sys.path:
        sys.path.insert(0, LIGHTGLUE_ROOT)
    _lightglue = importlib.import_module("lightglue")
    _lightglue_utils = importlib.import_module("lightglue.utils")
    LightGlue = getattr(_lightglue, "LightGlue")
    SuperPoint = getattr(_lightglue, "SuperPoint")
    rbd = getattr(_lightglue_utils, "rbd")
    HAS_LIGHTGLUE = True
except Exception:
    HAS_LIGHTGLUE = False

# =================== 商品名称映射（OCR 文本 → 标签名）===================
# 标签名与 YOLO 模型类名一致，用户可自行增删改
PRODUCT_LABEL = {
    # --- 牛奶类 ---
    "旺仔牛奶": "Wangzai", "旺仔": "Wangzai",
    "金典牛奶": "Jindianniunai",
    "特仑苏牛奶": "Telunsu",
    "光明牛奶": "Guangming",
    "蒙牛牛奶": "Mengniu",
    "伊利牛奶": "Yili",
    "牛奶": "Wangzai",
    # --- 薯片类 ---
    "乐事薯片": "Leshi", "乐事": "Leshi",
    "品客薯片": "Pringles",
    "好丽友薯片": "Haoliyou",
    "薯片": "Leshi",
    # --- 可乐/饮料 ---
    "可口可乐": "Coco",
    "百事可乐": "Baishi", "百事": "Baishi",
    "百事草莓奶昔": "Baishi_caomei_wutang",
    "RIO水蜜桃": "RIO_shuimitao", "RIO": "RIO_shuimitao",
    "维他奶": "Weitanai", "维他": "Weitanai",
    "可乐": "Coco",
    # --- 功能饮料 ---
    "红牛": "Hongniu",
    "佳得乐": "Jiadele",
    "功能饮料": "Hongniu",
    # --- 水 ---
    "农夫山泉": "Nongfushanquan",
    "怡宝": "Yibao",
    "矿泉水": "Nongfushanquan",
    # --- 咖啡/茶 ---
    "咖啡": "Kafei",
    "茶": "Cha",
    # --- 饼干 ---
    "奥利奥饼干": "Aoliao", "奥利奥": "Aoliao",
    "饼干": "Binggan",
    # --- 巧克力 ---
    "德芙巧克力": "Dove",
    "巧克力": "Qiaokeli",
    # --- 坚果 ---
    "三只松鼠坚果": "Sanzhisongshu", "三只松鼠": "Sanzhisongshu",
    "坚果": "Jianguo",
    # --- YOLO 模型类名（直接映射自身）---
    "Baishi": "Baishi",
    "Baishi_caomei_wutang": "Baishi_caomei_wutang",
    "Wangzai": "Wangzai",
    "RIO_shuimitao": "RIO_shuimitao",
    "Weitanai": "Weitanai",
    # --- 未知 ---
    "未知": "Unknown",
    "Unknown": "Unknown",
}

# =================== 目标清单（要抓取的商品标签及数量）===================
# 手动指定的目标（用户可自行增删，标签名须与 YOLO 类名一致，值为数量）
MANUAL_TARGETS = {
    "Wangzai": 1,   
    "Baishi": 1,    
    "Weitanai": 1,   
    "RIO_shuimitao": 1, 
    # 在此添加更多标签及数量...
}


DET_GRAB_CODES = {
    "RIO_shuimitao": 0x03,
    "Baishi": 0x04,
    "Wangzai": 0x05,
    "Weitanai": 0x06,
}


def get_target_dict(shared_state):
    """获取合并后的目标清单（手动 + OCR）{label: total_qty}"""
    merged = dict(MANUAL_TARGETS)
    for label, qty in shared_state.get("ocr_targets", {}).items():
        merged[label] = merged.get(label, 0) + qty
    return merged


def get_remaining_targets(shared_state):
    """获取每个目标的剩余数量 {label: remaining_qty}"""
    merged = get_target_dict(shared_state)
    grabbed = shared_state.get("grabbed_counts", {})
    return {label: total - grabbed.get(label, 0)
            for label, total in merged.items()}


def _fmt_remaining(shared_state):
    """格式化剩余数量字符串，用于终端/屏幕显示"""
    remaining = get_remaining_targets(shared_state)
    if not remaining:
        return "无"
    return ", ".join(f"{k} x{v}" for k, v in sorted(remaining.items()))


def get_target_set(shared_state):
    """获取当前要抓取的目标标签集合（剩余数量 > 0 的）"""
    remaining = get_remaining_targets(shared_state)
    return {label for label, qty in remaining.items() if qty > 0}


CHINESE_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
               "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


# =====================================================================
# 1. 串口通信层
# =====================================================================

def _xor_check(cmd_byte, data_bytes):
    """计算 XOR 校验：CMD ⊕ 每个数据字节"""
    xor = cmd_byte
    for b in data_bytes:
        xor ^= b
    return xor & 0xFF


def build_frame(cmd, data_bytes):
    """构建 PC→MCU 数据帧：0xAA [CMD] [DATA...] [XOR] 0xBB"""
    xor = _xor_check(cmd, data_bytes)
    return bytes([0xAA, cmd] + list(data_bytes) + [xor, 0xBB])


def build_mode_ack(mode):
    """模式确认帧 (5B)：0xAA 0xF0 [mode] 0xF1 0xBB（校验固定0xF1）"""
    return bytes([0xAA, 0xF0, mode & 0xFF, 0xF1, 0xBB])


def build_ocr_result(recognized):
    """OCR 结果帧 (5B)：0xAA 0x11 [flag] [XOR] 0xBB
    recognized: True/False 或 1/0
    """
    flag = 0x01 if recognized else 0x00
    return build_frame(0x11, [flag])


def build_detect_result(code_a, code_b):
    """饮料检测结果帧 (6B)：[codeA][codeB]"""
    return build_frame(0x21, [code_a & 0xFF, code_b & 0xFF])


def build_det_position(delta_x):
    """饮料检测-位置帧 (6B): 0xAA 0x52 [dxH][dxL] [XOR] 0xBB
    delta_x: 有符号整数 (-32768~32767)，负=左侧，正=右侧
    XOR: 固定为 0x63
    """
    dx = max(-32768, min(32767, int(delta_x)))
    dx_u16 = dx + 65536 if dx < 0 else dx
    dx_h = (dx_u16 >> 8) & 0xFF
    dx_l = dx_u16 & 0xFF
    return bytes([0xAA, 0x52, dx_h, dx_l, 0x63, 0xBB])


def build_det_grab(target_label=None):
    """饮料检测-抓取帧 (4B): 0xAA 0x53 [code] 0xBB
    指定 4 种目标时发送对应商品码；其他目标保持旧协议 0x53。
    """
    grab_code = DET_GRAB_CODES.get(target_label, 0x53)
    return bytes([0xAA, 0x53, grab_code & 0xFF, 0xBB])


def build_det_stop():
    """饮料检测-停止帧 (4B): 0xAA 0x51 [XOR] 0xBB"""
    return build_frame(0x51, [])


def build_det_lost():
    """饮料检测-目标丢失帧 (4B): 0xAA 0x55 0x55 0xBB"""
    return bytes([0xAA, 0x55, 0x55, 0xBB])


def build_face_stop():
    """人脸识别-停止帧 (4B): 0xAA 0x51 0x51 0xBB"""
    return bytes([0xAA, 0x51, 0x51, 0xBB])


def build_face_pos(delta_x):
    """人脸位置帧 (6B)：0xAA 0x31 [dxH][dxL] [XOR] 0xBB
    delta_x: 人脸中心X与图片中心X的有符号差值，正=右侧，负=左侧
    """
    dx = max(-32768, min(32767, int(delta_x)))
    dx_u16 = dx + 65536 if dx < 0 else dx
    return build_frame(0x31, [(dx_u16 >> 8) & 0xFF, dx_u16 & 0xFF])


def build_no_target():
    """未找到目标帧 (5B)：0xAA 0x3F 0x00 [XOR] 0xBB"""
    return build_frame(0x3F, [0x00])


def build_pc_ack():
    """PC→MCU 收到确认帧 (4B): 0xAA 0xB1 [XOR] 0xBB"""
    return build_frame(0xB1, [])


def build_face_deliver():
    """人脸识别-放货帧 (4B): 0xAA 0x33 [XOR] 0xBB"""
    return build_frame(0x33, [])


def build_ocr_drink_frame():
    """OCR模式-识别到饮料瓶帧 (4B): 0xAA 0x52 0x52 0xBB
    固定帧，20ms重发直到MCU发模式切换"""
    return bytes([0xAA, 0x52, 0x52, 0xBB])


def build_face_position(delta_x):
    """人脸位置帧 (6B): 0xAA 0x52 [dxH][dxL] 0x63 0xBB
    delta_x: 有符号整数，目标人脸中心X与图片中心X的差值
    校验位固定为 0x63
    """
    dx = max(-32768, min(32767, int(delta_x)))
    dx_u16 = dx + 65536 if dx < 0 else dx
    dx_h = (dx_u16 >> 8) & 0xFF
    dx_l = dx_u16 & 0xFF
    return bytes([0xAA, 0x52, dx_h, dx_l, 0x63, 0xBB])


def build_face_deliver_new():
    """人脸放货帧 (4B): 0xAA 0x53 0x53 0xBB
    |Δx|<阈值后发送，20ms重发直到MCU发模式切换"""
    return bytes([0xAA, 0x53, 0x53, 0xBB])


def build_face_no_target_new():
    """人脸未匹配目标帧 (4B): 0xAA 0x54 0x54 0xBB
    未匹配到目标人脸时发送，20ms重发直到识别到人脸"""
    return bytes([0xAA, 0x54, 0x54, 0xBB])


def _log_comm_frame(enabled, direction, frame_bytes):
    """生成通信日志行，并按需打印到终端。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    hex_str = " ".join(f"{b:02X}" for b in bytes(frame_bytes))
    line = f"[COMM {ts}] {direction} {hex_str}"
    if enabled:
        print(line)
    return line


class CommManager:
    """
    串口通信管理器
    - 后台线程持续接收 MCU 指令
    - 主线程通过 get_cmd() 获取最新指令
    - 通过 send() 发送帧
    """

    CMD_OCR       = 0x01
    CMD_CLASSIFY  = 0x02
    CMD_FACE      = 0x03
    CMD_TRIGGER   = 0x04
    CMD_STOP_ACK   = 0x05   # MCU→PC: 停止帧应答（饮料/人脸收到0x51后）
    CMD_DET_ACK    = 0x06   # MCU→PC: 抓取帧应答（继续识别）
    CMD_DET_RESUME = 0x07   # MCU→PC: 目标丢失帧应答（收到0x55后恢复识别）
    CMD_ADJUST_DONE = 0xA2   # MCU→PC: 微调结束
    CMD_EXIT_DET    = 0xA4   # MCU→PC: 退出识别

    def __init__(self, port, baud, comm_log=False, comm_log_file=True):
        self.ser = serial.Serial(port, baud, timeout=0.1, write_timeout=0.5)
        self.cmd_queue = queue.Queue()
        self._rx_buf = bytearray()
        self._rx_enabled_evt = threading.Event()
        self._comm_log = bool(comm_log)
        self._comm_log_file = bool(comm_log_file)
        self._comm_log_fp = None
        self._comm_log_file_lock = threading.Lock()
        self._comm_log_file_error = False

        if self._comm_log_file:
            try:
                self._comm_log_fp = open(
                    os.path.abspath("comm_log.txt"),
                    "w",
                    encoding="utf-8",
                    buffering=1,
                )
                print(f"[串口] 通信日志文件: {self._comm_log_fp.name}")
            except Exception as e:
                self._comm_log_file = False
                print(f"[串口] 打开通信日志文件失败: {e}")

        self._stop_evt = threading.Event()
        self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._rx_thread.start()
        print(f"[串口] 已打开 {port} @ {baud}")
        print("[串口] 接收门控已启用：请在 OpenCV 窗口按一次 Enter 后开始接收 MCU->PC 消息")

    def enable_rx_once(self):
        """按一次 Enter 后打开 MCU->PC 接收门控。"""
        if self._rx_enabled_evt.is_set():
            return
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                break
        self._rx_enabled_evt.set()
        print("[串口] 已解锁接收 MCU->PC 消息（本次运行只需按一次 Enter）")

    def is_rx_enabled(self):
        return self._rx_enabled_evt.is_set()

    def _write_comm_line(self, line):
        """写入通信日志文件（当前目录 comm_log.txt）。"""
        if not self._comm_log_file or self._comm_log_fp is None:
            return
        try:
            with self._comm_log_file_lock:
                self._comm_log_fp.write(line + "\n")
                self._comm_log_fp.flush()
        except Exception as e:
            if not self._comm_log_file_error:
                print(f"[串口] 写通信日志文件失败: {e}")
                self._comm_log_file_error = True

    def _rx_worker(self):
        """后台接收线程：解析 0xAA [CMD] 0xBB 三字节帧"""
        while not self._stop_evt.is_set():
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self._rx_buf.extend(data)
                    self._parse()
                else:
                    time.sleep(0.01)
            except Exception:
                if not self._stop_evt.is_set():
                    time.sleep(0.1)

    def _parse(self):
        while len(self._rx_buf) >= 3:
            try:
                idx = self._rx_buf.index(0xAA)
            except ValueError:
                self._rx_buf.clear()
                return
            if idx > 0:
                del self._rx_buf[:idx]
            if len(self._rx_buf) < 3:
                return
            # 3-byte frame: 0xAA CMD 0xBB
            if self._rx_buf[2] == 0xBB:
                frame = bytes(self._rx_buf[:3])
                cmd = frame[1]
                del self._rx_buf[:3]
                if not self._rx_enabled_evt.is_set():
                    continue
                line = _log_comm_frame(self._comm_log, "MCU->PC", frame)
                self._write_comm_line(line)
                self.cmd_queue.put(cmd)
                continue
            # 4-byte frame: 0xAA CMD XOR 0xBB
            if len(self._rx_buf) >= 4 and self._rx_buf[3] == 0xBB:
                frame = bytes(self._rx_buf[:4])
                cmd = frame[1]
                xor_byte = frame[2]
                if xor_byte == cmd:  # XOR = CMD (no data)
                    del self._rx_buf[:4]
                    if not self._rx_enabled_evt.is_set():
                        continue
                    line = _log_comm_frame(self._comm_log, "MCU->PC", frame)
                    self._write_comm_line(line)
                    self.cmd_queue.put(cmd)
                    continue
            # Neither matched
            del self._rx_buf[:1]

    def get_cmd(self):
        """非阻塞获取最新 MCU 命令，无则返回 None"""
        cmd = None
        while not self.cmd_queue.empty():
            try:
                cmd = self.cmd_queue.get_nowait()
            except queue.Empty:
                break
        return cmd

    def send(self, frame):
        """发送帧"""
        try:
            frame_bytes = bytes(frame)
            self.ser.write(frame_bytes)
            line = _log_comm_frame(self._comm_log, "PC->MCU", frame_bytes)
            self._write_comm_line(line)
        except Exception as e:
            print(f"[串口] 发送失败: {e}")

    def close(self):
        self._stop_evt.set()
        self._rx_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        if self._comm_log_fp is not None:
            try:
                self._comm_log_fp.close()
            except Exception:
                pass


class _DummyComm:
    """调试模式：键盘模拟 MCU 指令（--no-serial）"""

    CMD_OCR      = 0x01
    CMD_CLASSIFY = 0x02
    CMD_FACE     = 0x03
    CMD_TRIGGER  = 0x04
    CMD_STOP_ACK  = 0x05
    CMD_DET_ACK   = 0x06
    CMD_DET_RESUME = 0x07
    CMD_ADJUST_DONE = 0xA2
    CMD_EXIT_DET    = 0xA4

    _key_map = {
        ord('1'): 0x01, ord('2'): 0x02, ord('3'): 0x03,
        ord('4'): 0x04, ord('t'): 0x05,
        ord('7'): 0x07, ord('8'): 0x06, ord('9'): 0xA2,
        ord('-'): 0xA4,
        # 兼容 Numpad '-' 在部分系统/驱动返回的键值（常见 VK_SUBTRACT=109）
        109: 0xA4,
    }

    def __init__(self):
        self._pending = None
        print("[调试] 键盘模拟模式：1=OCR 2=饮料检测 3=人脸 4=触发 "
              "t=0x05停止应答 7=0x07丢失恢复 8=0x06抓取应答 9=微调结束 -=退出识别")

    def feed_key(self, key):
        # 有时 OpenCV/底层驱动会返回不同的 key 值（例如 numpad '-' 可能为 109）
        # 我们先直接按 key 查表，若未命中再尝试低字节备用匹配。
        # 当使用 `key = cv2.waitKey(1) & 0xFF` 时，未按键会返回 255（0xFF）
        # 忽略该值以避免持续打印未映射按键日志。
        if key is None:
            return
        if key == 255 or key == -1:
            return

        cmd = None
        cmd = self._key_map.get(key)
        if cmd is None and isinstance(key, int):
            cmd = self._key_map.get(key & 0xFF)
        if cmd is not None:
            self._pending = cmd
            return

        # 未映射按键时打印调试信息，方便定位按键实际返回的值
        try:
            print(f"[调试] 未映射按键 key={key} (hex={key:#x} low_byte={key & 0xFF if isinstance(key,int) else 'N/A'})")
        except Exception:
            pass

    def get_cmd(self):
        cmd = self._pending
        self._pending = None
        return cmd

    def send(self, frame):
        hex_str = " ".join(f"{b:02X}" for b in frame)
        #print(f"[调试TX] {hex_str}")

    def close(self):
        pass


def _handle_enter_to_enable_rx(comm, key):
    """在 OpenCV 窗口按 Enter 后，开启真实串口接收。"""
    if key not in (10, 13):
        return
    if isinstance(comm, CommManager):
        comm.enable_rx_once()


MODE_SWITCH_CMDS = (
    CommManager.CMD_OCR,
    CommManager.CMD_CLASSIFY,
    CommManager.CMD_FACE,
)


def _is_mode_switch_cmd(cmd):
    return cmd in MODE_SWITCH_CMDS


def _log_ignore_same_mode_cmd(prefix, cmd):
    print(f"{prefix} <<< 收到当前模式重复切换命令 0x{cmd:02X}，忽略")


class SharedCamera:
    """共享摄像头：全局只打开一次，统一分辨率并只预热一次。"""

    def __init__(self, camera_idx, cam_w, cam_h, warmup_frames=20):
        self.camera_idx = int(camera_idx)
        self.cam_w = int(cam_w)
        self.cam_h = int(cam_h)
        self.warmup_frames = max(1, int(warmup_frames))
        self.cap = None

    def open(self):
        if self.cap is not None and self.cap.isOpened():
            return self.cap

        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开共享摄像头 {self.camera_idx}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap

    def warmup(self):
        cap = self.open()
        ok_count = 0
        for _ in range(self.warmup_frames):
            ret, _ = cap.read()
            if ret:
                ok_count += 1
            time.sleep(0.01)

        if ok_count == 0:
            raise RuntimeError("共享摄像头预热失败，未读取到有效画面")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[CAM] 共享摄像头已就绪: index={self.camera_idx} "
            f"actual={actual_w}x{actual_h} warmup={self.warmup_frames}帧"
        )

    def get(self):
        return self.open()

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# =====================================================================
# 2. OCR 模式（PaddleOCR + Ollama 大模型）—— 完全保留
# =====================================================================

class OCRMode:
    """
    OCR 模式流程:
        MCU 发 0x01 → 切换到本模式 → 摄像头预览
        MCU 发 0x04 → 拍照 → OCR → 大模型提取 → 串口返回商品编码列表
    """

    def __init__(self, camera_idx, cam_w, cam_h, save_dir,
                 ollama_url, llm_model, shared_state=None,
                 send_interval=0.02):
        self.camera_idx = camera_idx
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.save_dir = save_dir
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.shared_state = shared_state if shared_state is not None else {}
        self.send_interval = max(0.001, float(send_interval))
        self.ocr_engine = None
        if HAS_PADDLE:
            self.ocr_engine = PaddleOCR(
                lang="ch",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _poll_cmd_nonblocking(comm):
        """非阻塞取一条命令，不清空整队列。"""
        if isinstance(comm, CommManager):
            try:
                return comm.cmd_queue.get_nowait()
            except queue.Empty:
                return None
        if isinstance(comm, _DummyComm):
            return comm.get_cmd()
        return None

    @staticmethod
    def _push_back_cmd(comm, cmd):
        """把不需要当前处理的命令放回通信对象。"""
        if cmd is None:
            return
        if isinstance(comm, CommManager):
            comm.cmd_queue.put(cmd)
        elif isinstance(comm, _DummyComm) and comm._pending is None:
            comm._pending = cmd

    def _trigger_ack_worker(self, comm, stop_evt, state):
        """OCR 触发后只回复一次 ACK，期间持续监听中断命令。"""
        ack_frame = build_mode_ack(0x04)
        comm.send(ack_frame)  # 先立即回 ACK

        while not stop_evt.is_set():
            pending_cmd = self._poll_cmd_nonblocking(comm)
            if pending_cmd is not None:
                if pending_cmd == CommManager.CMD_OCR:
                    _log_ignore_same_mode_cmd("[OCR]", pending_cmd)
                elif (
                    _is_mode_switch_cmd(pending_cmd)
                    or pending_cmd == CommManager.CMD_EXIT_DET
                ):
                    state["interrupt_cmd"] = pending_cmd
                    stop_evt.set()
                    break
                else:
                    self._push_back_cmd(comm, pending_cmd)

            if stop_evt.wait(0.01):
                break

    def _send_ocr_result_until_mode_switch(self, comm, frame_data, current_mode_cmd):
        """循环发送 OCR 结果帧，直到收到模式切换命令。"""
        print(
            f"[OCR] >>> OCR结果帧循环发送已启动，每"
            f"{int(self.send_interval * 1000)}ms发送一次，等待模式切换命令"
        )

        next_send_ts = 0.0
        while True:
            now = time.time()
            if now >= next_send_ts:
                comm.send(frame_data)
                next_send_ts = now + self.send_interval

            mode_cmd = None
            while True:
                cmd = self._poll_cmd_nonblocking(comm)
                if cmd is None:
                    break
                if not _is_mode_switch_cmd(cmd):
                    continue
                if cmd == current_mode_cmd:
                    _log_ignore_same_mode_cmd("[OCR]", cmd)
                    continue
                if _is_mode_switch_cmd(cmd):
                    mode_cmd = cmd

            if mode_cmd is not None:
                print(
                    f"[OCR] <<< 收到模式切换命令 0x{mode_cmd:02X}，"
                    f"停止OCR结果帧循环发送"
                )
                return mode_cmd

            time.sleep(0.001)

    def run(self, comm, shared_camera=None):
        """进入OCR模式主循环，直到收到新模式/停止命令才退出"""
        print("\n[OCR] ===== 进入 OCR 文字识别模式 =====")
        if not HAS_PADDLE:
            print("[OCR] 错误: 缺少 paddleocr，pip install paddleocr")
            return None

        owns_cap = False
        if shared_camera is not None:
            try:
                cap = shared_camera.get()
            except Exception as e:
                print(f"[OCR] 无法获取共享摄像头: {e}")
                return None
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[OCR] 复用共享摄像头: {actual_w}x{actual_h}")
        else:
            cap = cv2.VideoCapture(self.camera_idx)
            if not cap.isOpened():
                print(f"[OCR] 无法打开摄像头 {self.camera_idx}")
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
            owns_cap = True
            print("[OCR] 摄像头已打开，等待触发 (CMD=0x04)...")

        print("[OCR] 摄像头已就绪，等待触发 (CMD=0x04)...")

        next_cmd = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display, "OCR Mode | Waiting TRIGGER(4)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Unified Vision", display)
            key = cv2.waitKey(1) & 0xFF

            if isinstance(comm, _DummyComm):
                comm.feed_key(key)
            _handle_enter_to_enable_rx(comm, key)

            if key == ord('q') or key == 27:
                next_cmd = None
                break

            cmd = comm.get_cmd()
            if cmd is None:
                continue

            if _is_mode_switch_cmd(cmd):
                if cmd == CommManager.CMD_OCR:
                    _log_ignore_same_mode_cmd("[OCR]", cmd)
                    continue
                next_cmd = cmd
                break

            if cmd == CommManager.CMD_EXIT_DET:   # 0xA4
                comm.send(build_pc_ack())
                print("[OCR] <<< MCU退出识别，已回复ACK")
                next_cmd = None
                break

            if cmd == CommManager.CMD_TRIGGER:
                ack_stop_evt = threading.Event()
                ack_state = {"interrupt_cmd": None}
                recognized_frame = None
                ack_thread = threading.Thread(
                    target=self._trigger_ack_worker,
                    args=(comm, ack_stop_evt, ack_state),
                    daemon=True,
                )
                ack_thread.start()
                print(
                    "[OCR] >>> 触发拍照识别（ACK已发送一次: F0 04 F1）"
                )

                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(self.save_dir, f"OCR_{ts}.png")
                    cv2.imwrite(img_path, frame)
                    cv2.destroyWindow("Unified Vision")
                    cv2.waitKey(1)

                    print(
                        f"[OCR] 已保存原图: {img_path} "
                        f"size={frame.shape[1]}x{frame.shape[0]}"
                    )

                    self._capture_face(frame, img_path)

                    ocr_input = self._resize_ocr_input(frame)
                    print("[OCR] OCR推理输入已压缩到 1280x720（仅内存）")

                    all_text = self._ocr(ocr_input)
                    if not all_text:
                        print("[OCR] 未识别到文本，不发送OCR结果帧")
                    else:
                        structured = self._llm_extract(all_text)
                        items = self._to_labels(structured, all_text)

                        recognized = len(items) > 0
                        if recognized:
                            recognized_frame = build_ocr_drink_frame()
                            hex_str = " ".join(f"{b:02X}" for b in recognized_frame)
                            print(f"[OCR] 识别到 {len(items)} 件商品，结果帧: {hex_str}")
                            for it in items:
                                print(f"       {it['name']} -> {it['label']}")
                            ocr_targets = {}
                            for it in items:
                                if it["label"] != "Unknown":
                                    ocr_targets[it["label"]] = (
                                        ocr_targets.get(it["label"], 0) + 1)
                            self.shared_state["ocr_targets"] = ocr_targets
                            print(f"[OCR] 目标清单已更新(OCR部分): {ocr_targets}")
                            print(f"[OCR] 合并后清单: {_fmt_remaining(self.shared_state)}")
                        else:
                            print("[OCR] 未识别到商品，不发送OCR结果帧")
                finally:
                    ack_stop_evt.set()
                    ack_thread.join(timeout=1.0)
                    print("[OCR] <<< OCR触发ACK监听已停止")

                interrupt_cmd = ack_state.get("interrupt_cmd")
                if interrupt_cmd is not None:
                    if interrupt_cmd == CommManager.CMD_EXIT_DET:
                        comm.send(build_pc_ack())
                        print("[OCR] <<< OCR处理中收到退出命令，已回复ACK")
                        next_cmd = None
                    else:
                        print(f"[OCR] <<< OCR处理中收到模式切换: 0x{interrupt_cmd:02X}")
                        next_cmd = interrupt_cmd
                    break

                if recognized_frame is not None:
                    next_cmd = self._send_ocr_result_until_mode_switch(
                        comm, recognized_frame, CommManager.CMD_OCR
                    )
                    if next_cmd is not None:
                        break

        if owns_cap:
            cap.release()
        return next_cmd

    def _capture_face(self, frame, img_path):
        """从当前帧中检测人脸并保存目标图路径，供人脸模板模式使用。"""
        if not HAS_FACE:
            print("  [OCR-FACE] face_recognition 未安装，跳过人脸采集")
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = preprocess_image(rgb)
            model = "cnn"
            try:
                locs = face_rec.face_locations(rgb, model=model)
            except Exception:
                model = "hog"
                locs = face_rec.face_locations(rgb, model=model)
            if not locs:
                print("  [OCR-FACE] 未检测到人脸")
                return
            best_idx = 0
            if len(locs) > 1:
                areas = [(b - t) * (r - l) for t, r, b, l in locs]
                best_idx = int(np.argmax(areas))
            self.shared_state["face_img_path"] = img_path
            t, r, b, l = locs[best_idx]
            print(f"  [OCR-FACE] ✓ 人脸已采集 model={model} template=local "
                  f"loc=({t},{r},{b},{l})  共{len(locs)}张人脸")
        except Exception as e:
            print(f"  [OCR-FACE] 采集异常: {e}")

    @staticmethod
    def _resize_ocr_input(frame, size=(1280, 720)):
        """生成 OCR 专用缩放图，避免高分辨率整图推理过慢。"""
        target_w, target_h = size
        interpolation = cv2.INTER_AREA
        if frame.shape[1] < target_w or frame.shape[0] < target_h:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

    def _ocr(self, img_input):
        result = list(self.ocr_engine.predict(img_input))
        all_text = ""
        if result and len(result) > 0:
            ocr_result = result[0]
            for text, score in zip(ocr_result.get('rec_texts', []),
                                   ocr_result.get('rec_scores', [])):
                print(f"  OCR: {text} ({score:.0%})")
                all_text += text
        return all_text

    def _llm_extract(self, text):
        if not HAS_REQUESTS:
            print("[OCR] 缺少 requests 库")
            return None
        prompt = (
            f"任务：从文本中提取所有饮料和零食，输出JSON数组。\n"
            f"文本：{text}\n"
            f"提取规则：\n"
            f"1. 提取所有饮料和零食\n"
            f'2. 品牌信息可能在后文说明："牛奶要光明的" → 牛奶品牌是"光明"\n'
            f'3. 同类商品出现品牌偏好时，必须把品牌补到该商品上："薯片我只吃好丽友" → 品牌是"好丽友"，商品是"薯片"\n'
            f'4. 没有明确品牌的，品牌字段填""\n'
            f'只输出JSON数组: [{{"数量":"数字","单位":"瓶/罐/盒","品牌":"","商品":"商品名"}}]\n'
            f"JSON："
        )
        try:
            resp = _requests.post(self.ollama_url, json={
                "model": self.llm_model, "prompt": prompt,
                "stream": False, "options": {"temperature": 0}
            }, timeout=60)
            if resp.status_code == 200:
                raw = resp.json().get("response", "").strip()
                m = re.search(r'\[.*\]', raw, re.DOTALL)
                if m:
                    return json.loads(m.group())
        except _requests.exceptions.ConnectionError:
            print("[OCR] 无法连接 Ollama，请运行 'ollama serve'")
        except Exception as e:
            print(f"[OCR] LLM 错误: {e}")
        return None

    @staticmethod
    def _canonical_product_name(product):
        product = str(product or "").strip()
        if not product:
            return ""
        if "牛奶" in product:
            return "牛奶"
        if "薯片" in product:
            return "薯片"
        if "可乐" in product:
            return "可乐"
        if "红牛" in product:
            return "红牛"
        if "矿泉水" in product or "纯净水" in product:
            return "矿泉水"
        return product

    @staticmethod
    def _extract_brand_preferences(text):
        text = str(text or "")
        if not text:
            return {}

        hint_map = {
            "牛奶": ["蒙牛", "伊利", "光明", "旺仔", "金典", "特仑苏"],
            "薯片": ["好丽友", "乐事", "品客"],
            "可乐": ["可口可乐", "百事", "可口"],
        }
        preferences = {}
        for product, brands in hint_map.items():
            for brand in brands:
                pattern = rf"{re.escape(product)}[^。；，,]*{re.escape(brand)}|{re.escape(brand)}[^。；，,]*{re.escape(product)}"
                if re.search(pattern, text):
                    preferences[product] = brand
                    break
        return preferences

    @staticmethod
    def _backfill_missing_structured_items(structured, raw_text):
        if not structured or not isinstance(structured, list):
            return []

        text = str(raw_text or "")
        if not text:
            return list(structured)

        specs = (
            {
                "product": "牛奶",
                "quantity_keywords": ["纯牛奶", "牛奶"],
                "presence_tokens": [
                    "纯牛奶", "牛奶", "光明牛奶", "蒙牛牛奶",
                    "伊利牛奶", "旺仔牛奶", "金典牛奶", "特仑苏牛奶",
                ],
            },
            {
                "product": "薯片",
                "quantity_keywords": ["薯片"],
                "presence_tokens": [
                    "薯片", "乐事", "品客", "好丽友",
                    "乐事薯片", "品客薯片", "好丽友薯片",
                ],
            },
            {
                "product": "可乐",
                "quantity_keywords": ["可口可乐", "百事可乐", "可乐"],
                "presence_tokens": ["可口可乐", "百事可乐", "可乐"],
            },
            {
                "product": "红牛",
                "quantity_keywords": ["红牛"],
                "presence_tokens": ["红牛"],
            },
        )
        qty_pattern = r"([0-9零一二两三四五六七八九十百]+)\s*(瓶|罐|盒|桶|包|袋|听|个|支|杯)?\s*({keywords})"

        filled = list(structured)
        present_products = set()
        for entry in filled:
            product = str(entry.get("商品", "") or "").strip()
            brand = str(entry.get("品牌", "") or "").strip()
            text_blob = brand + product

            canonical_product = OCRMode._canonical_product_name(product)
            if canonical_product:
                present_products.add(canonical_product)

            for spec in specs:
                if any(token and token in text_blob for token in spec["presence_tokens"]):
                    present_products.add(spec["product"])

        for spec in specs:
            if spec["product"] in present_products:
                continue

            keywords = "|".join(
                sorted((re.escape(keyword) for keyword in spec["quantity_keywords"]),
                       key=len, reverse=True)
            )
            match = re.search(qty_pattern.format(keywords=keywords), text)
            if not match:
                continue

            filled.append({
                "数量": match.group(1),
                "单位": match.group(2) or "",
                "品牌": "",
                "商品": match.group(3),
            })
            present_products.add(spec["product"])

        return filled

    def _to_labels(self, structured, raw_text="", max_items=6):
        items = []
        if not structured or not isinstance(structured, list):
            return items
        structured = self._backfill_missing_structured_items(structured, raw_text)
        brand_preferences = self._extract_brand_preferences(raw_text)
        for entry in structured:
            product = str(entry.get("商品", "") or "").strip()
            brand = str(entry.get("品牌", "") or "").strip()
            qty = self._parse_qty(entry.get("数量", "1"))
            canonical_product = self._canonical_product_name(product)
            if not brand and canonical_product:
                brand = brand_preferences.get(canonical_product, "")
            label = self._lookup(product, brand)
            name = (brand + product) if brand else product
            for _ in range(qty):
                if len(items) < max_items:
                    items.append({"label": label, "name": name})
        return items

    @staticmethod
    def _parse_qty(s):
        if not s:
            return 1
        try:
            return int(s)
        except Exception:
            for cn, n in CHINESE_NUM.items():
                if cn in str(s):
                    return n
            m = re.search(r'\d+', str(s))
            return int(m.group()) if m else 1

    @staticmethod
    def _lookup(product, brand=""):
        product = str(product or "").strip()
        brand = str(brand or "").strip()
        canonical_product = OCRMode._canonical_product_name(product)

        candidates = []
        if brand and product:
            candidates.append(brand + product)
        if brand and canonical_product and canonical_product != product:
            candidates.append(brand + canonical_product)
        if product:
            candidates.append(product)
        if canonical_product and canonical_product != product:
            candidates.append(canonical_product)

        for candidate in candidates:
            if candidate in PRODUCT_LABEL:
                return PRODUCT_LABEL[candidate]

        specific_items = sorted(
            PRODUCT_LABEL.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for candidate in candidates:
            for key, label in specific_items:
                if key in candidate or candidate in key:
                    return label
        return "Unknown"


# =====================================================================
# 3. MultiFrameTracker（IoU 追踪，从 camera_detect.py 复用）
# =====================================================================

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

        # 4. 筛选已确认的
        confirmed = []
        for trk in self.tracks:
            if trk["hits"] >= self.confirm_frames:
                confirmed.append((trk["cls"], trk["conf"], trk["box"], trk["id"]))
        return confirmed

    def reset(self):
        """重置所有追踪"""
        self.tracks.clear()
        self._next_id = 0


# =====================================================================
# 4. 饮料检测模式（YOLOv8 目标检测 + 双ROI + MultiFrameTracker）
# =====================================================================

# 颜色映射
_COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
]
_class_color_map = {}


def _get_class_color(cls_name):
    """为类别分配一个固定颜色"""
    if cls_name not in _class_color_map:
        _class_color_map[cls_name] = _COLORS[len(_class_color_map) % len(_COLORS)]
    return _class_color_map[cls_name]


class DetectMode:
    """
    饮料检测模式流程（全画面检测 + 目标抓取协议）:
        MCU发送 0x02 切换模式后，PC回复模式确认并立即进入检测
        检测到目标且 |Δx| < det-stop-threshold → 发送停止帧(0x51)，20ms重发
        收到 MCU 0x05 应答后 → 发送位置帧(0x52)，20ms重发
        若位置阶段连续 det-lost-frames 帧未识别到目标 → 发送目标丢失帧(0x55)，20ms重发
        收到 MCU 0x07 应答后 → 停止0x55并恢复检测
        |Δx| < 抓取阈值(默认10) → 按目标发送抓取帧(0x53+商品码)，20ms重发
        抓取帧重发阶段持续识别，但直到收到 0x06 前仅发送 0x53
        收到 MCU 0x06 应答后 → 回到检测阶段继续识别

    状态: DETECTING → STOP_SEND → POS_SEND → LOST_SEND/GRAB_SEND → DETECTING
    """

    _ST_DETECTING = 0   # 检测中，等待触发停止帧
    _ST_STOP_SEND = 1   # 发送停止帧(0x51)，等待 MCU 0x05
    _ST_POS_SEND  = 2   # 发送位置帧(0x52)
    _ST_GRAB_SEND = 3   # 发送抓取帧(0x53)，等待 MCU 0x06
    _ST_LOST_SEND = 4   # 发送目标丢失帧(0x55)，等待 MCU 0x07

    def __init__(self, weights, device, conf, iou_nms, imgsz,
                 camera_idx, cam_w, cam_h, rotate,
                 source="", phone=False,
                 no_show=False, save_dir="screenshots",
                 confirm_frames=5, iou_match=0.5,
                 stop_threshold=50,
                 align_threshold=10,
                 det_lost_frames=5,
                 shared_state=None,
                 send_interval=0.02,
                 notfound_delay=1.0,
                 notfound_min_frames=15):
        self.weights = weights
        self.device = device
        self.conf = conf
        self.iou_nms = iou_nms
        self.imgsz = imgsz
        self.camera_idx = camera_idx
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.rotate = rotate
        self.source = source
        self.phone = phone
        self.no_show = no_show
        self.save_dir = save_dir
        self.confirm_frames = confirm_frames
        self.iou_match = iou_match
        self.stop_threshold = int(stop_threshold)
        self.align_threshold = align_threshold   # 对齐像素阈值，可自定义
        self.det_lost_frames = max(1, int(det_lost_frames))
        self.shared_state = shared_state if shared_state is not None else {}
        self.send_interval = send_interval
        self.notfound_delay = max(0.0, float(notfound_delay))
        self.notfound_min_frames = max(1, int(notfound_min_frames))
        os.makedirs(save_dir, exist_ok=True)

    # ==================== run ====================
    def run(self, comm, shared_camera=None):
        print("\n[DET] ===== 进入饮料检测模式 =====")
        if not HAS_YOLO:
            print("[DET] 错误: 缺少 ultralytics，pip install ultralytics")
            return None
        if not os.path.exists(self.weights):
            print(f"[DET] 模型文件不存在: {self.weights}")
            return None

        try:
            model = YOLO(self.weights)
        except Exception as e:
            if _is_tensorrt_engine_path(self.weights):
                print(f"[DET] TensorRT engine 加载失败: {e}")
                print("[DET] 请确认当前环境已安装 tensorrt，且 TensorRT / torch / conda DLL 目录已加入进程")
            else:
                print(f"[DET] 模型加载失败: {e}")
            return None

        if _is_tensorrt_engine_path(self.weights):
            print(f"[DET] 手动 TensorRT 切换已启用: {self.weights}")

        print(f"[DET] 类别: {model.names}")

        targets = get_target_set(self.shared_state)
        print(f"[DET] 目标清单: {_fmt_remaining(self.shared_state)}")

        # 视频源
        video_src = self._get_video_source()
        if video_src is None:
            return None
        owns_cap = False
        using_shared_camera = False
        if isinstance(video_src, str):
            print(f"[DET] 视频源: {video_src} (网络流)")
            cap = cv2.VideoCapture(video_src)
            if not cap.isOpened():
                print(f"[DET] 无法打开视频源: {video_src}")
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            owns_cap = True
        else:
            if shared_camera is not None and int(video_src) == shared_camera.camera_idx:
                try:
                    cap = shared_camera.get()
                except Exception as e:
                    print(f"[DET] 无法获取共享摄像头: {e}")
                    return None
                using_shared_camera = True
                print(f"[DET] 视频源: 共享摄像头索引 {video_src}")
            else:
                print(f"[DET] 视频源: 摄像头索引 {video_src}")
                cap = cv2.VideoCapture(video_src)
                if not cap.isOpened():
                    print(f"[DET] 无法打开视频源: {video_src}")
                    return None
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                owns_cap = True

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if using_shared_camera:
            print(f"[DET] 复用共享摄像头，实际分辨率: {actual_w}x{actual_h}")
        else:
            print(f"[DET] 实际分辨率: {actual_w}x{actual_h}")
        if self.rotate:
            print(f"[DET] 画面旋转: {self.rotate}°")
            if self.rotate in (90, 270):
                print(f"[DET] 旋转后分辨率: {actual_h}x{actual_w}")
        print("[DET] 全画面检测模式")
        print(f"[DET] MultiFrameTracker: confirm={self.confirm_frames} "
              f"iou_match={self.iou_match}")
        print(f"[DET] 停止阈值: {self.stop_threshold}px")
        print(f"[DET] 对齐阈值: {self.align_threshold}px")
        print(f"[DET] 丢失阈值: 连续{self.det_lost_frames}帧未识别到目标则发送0x55")
        print("[DET] 快捷键: Q/ESC=退出 | S=截图 | 空格=暂停")
        print("[DET] 已进入检测流程，等待目标触发停止帧(0x51)...")

        tracker = MultiFrameTracker(self.confirm_frames, self.iou_match)

        # 状态变量
        state = self._ST_DETECTING
        fps, fps_count, fps_start = 0, 0, time.time()
        paused = False
        last_frame = None
        frame_count = 0
        next_cmd = None
        confirmed = []
        last_send_time = 0.0
        current_grab_target = None   # 当前正在抓取的目标标签名
        last_pos_dx = 0
        SEND_INTERVAL = self.send_interval
        last_raw_dets = []
        lost_target_frames = 0

        def _poll_det_cmd():
            """LOST_SEND 阶段优先取 0x07 和模式切换，避免被无关命令覆盖。"""
            if state != self._ST_LOST_SEND or not isinstance(comm, CommManager):
                return comm.get_cmd()

            drained = []
            while True:
                try:
                    drained.append(comm.cmd_queue.get_nowait())
                except queue.Empty:
                    break
            if not drained:
                return None

            for pending_cmd in drained:
                if pending_cmd == CommManager.CMD_DET_RESUME:
                    return pending_cmd
            for pending_cmd in drained:
                if _is_mode_switch_cmd(pending_cmd):
                    return pending_cmd
            return drained[-1]

        def _enter_lost_send(now_ts):
            nonlocal state, current_grab_target, last_pos_dx
            nonlocal last_send_time, lost_target_frames, confirmed, last_raw_dets

            tracker.reset()
            current_grab_target = None
            last_pos_dx = 0
            lost_target_frames = 0
            confirmed = []
            last_raw_dets = []
            comm.send(build_det_lost())
            last_send_time = now_ts
            state = self._ST_LOST_SEND
            print("[DET] >>> 位置阶段连续丢失目标，发送目标丢失帧(0x55)，等待MCU回复0x07")

        def _resume_from_lost_send():
            nonlocal state, current_grab_target, last_pos_dx
            nonlocal last_send_time, lost_target_frames, confirmed, last_raw_dets

            tracker.reset()
            current_grab_target = None
            last_pos_dx = 0
            lost_target_frames = 0
            confirmed = []
            last_raw_dets = []
            last_send_time = time.time()
            state = self._ST_DETECTING
            print("[DET] <<< 收到0x07，停止发送0x55，恢复检测模式重新识别")

        while True:
            # ---- 暂停支持 ----
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[DET] 无法读取画面")
                    break
                frame = self._rotate_frame(frame)
                last_frame = frame.copy()
            else:
                frame = last_frame.copy() if last_frame is not None else None
                if frame is None:
                    continue

            frame_count += 1
            fps_count += 1
            if fps_count >= 10:
                fps = fps_count / (time.time() - fps_start)
                fps_count, fps_start = 0, time.time()

            h, w = frame.shape[:2]
            display = frame.copy()
            img_center_x = w // 2

            # ---- YOLO 检测（LOST_SEND 阶段暂停识别）----
            if not paused and state != self._ST_LOST_SEND:
                last_raw_dets = self._run_detect(model, frame)
                confirmed = tracker.update(last_raw_dets)
            else:
                last_raw_dets = []
                if state == self._ST_LOST_SEND:
                    confirmed = []

            # ---- 刷新目标清单 ----
            targets = get_target_set(self.shared_state)

            # ---- 找距离图片中心最近的目标 ----
            target_info = self._find_nearest_target(
                confirmed, img_center_x, targets)
            if target_info is not None:
                last_pos_dx = int(target_info[1])

            # ---- 绘制检测框（目标清单内=绿，不在=灰）----
            self._draw_confirmed_with_targets(display, confirmed, targets)

            # ---- 绘制中心线 + 目标高亮 ----
            cv2.line(display, (img_center_x, 0), (img_center_x, h),
                     (128, 128, 128), 1)
            if target_info:
                t_name, t_delta, t_box = target_info
                tcx = int((t_box[0] + t_box[2]) / 2)
                tcy = int((t_box[1] + t_box[3]) / 2)
                cv2.circle(display, (tcx, tcy), 8, (0, 0, 255), -1)
                cv2.line(display, (img_center_x, tcy), (tcx, tcy),
                         (0, 0, 255), 2)
                cv2.putText(display, f"dx={t_delta}",
                            (min(tcx, img_center_x) + 5, tcy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ---- 状态栏 ----
            _sn = {
                self._ST_DETECTING: "DETECTING",
                self._ST_STOP_SEND: "STOP_SEND",
                self._ST_POS_SEND: "POS_SEND",
                self._ST_GRAB_SEND: "GRAB_SEND",
                self._ST_LOST_SEND: "LOST_SEND",
            }
            st_text = _sn.get(state, "???")
            if state == self._ST_LOST_SEND:
                tgt_text = "Waiting MCU 0x07"
            else:
                tgt_text = (f"Target: {target_info[0]} dx={target_info[1]}"
                            if target_info else "No target")
            cv2.putText(display, f"[{st_text}] {tgt_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            remaining = get_remaining_targets(self.shared_state)
            tgt_str = ", ".join(
                f"{k} x{v}" for k, v in sorted(remaining.items())
            ) if remaining else "NONE"
            cv2.putText(display, f"Targets: {tgt_str}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            self._draw_info(display, fps, paused, st_text)

            if not self.no_show:
                cv2.imshow("Unified Vision", display)

            key = cv2.waitKey(1) & 0xFF
            if isinstance(comm, _DummyComm):
                comm.feed_key(key)
            _handle_enter_to_enable_rx(comm, key)

            # ---- 键盘控制 ----
            if key == ord('q') or key == 27:
                next_cmd = None
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fpath = os.path.join(self.save_dir, f"det_{ts}.jpg")
                cv2.imwrite(fpath, display)
                print(f"[DET] 截图已保存: {fpath}")
            elif key == ord(' '):
                paused = not paused
                print(f"[DET] {'暂停' if paused else '继续'}")

            # ---- 命令处理 ----
            cmd = _poll_det_cmd()
            if cmd is not None:
                if _is_mode_switch_cmd(cmd):
                    if cmd == CommManager.CMD_CLASSIFY:
                        _log_ignore_same_mode_cmd("[DET]", cmd)
                    else:
                        next_cmd = cmd
                        break

                elif cmd == CommManager.CMD_DET_RESUME:   # 0x07
                    if state == self._ST_LOST_SEND:
                        _resume_from_lost_send()

                elif cmd == CommManager.CMD_STOP_ACK:   # 0x05
                    if state == self._ST_STOP_SEND:
                        if target_info is not None:
                            last_pos_dx = int(target_info[1])
                            current_grab_target = target_info[0]
                        lost_target_frames = 0
                        pkt = build_det_position(last_pos_dx)
                        comm.send(pkt)
                        last_send_time = time.time()
                        state = self._ST_POS_SEND
                        hx = " ".join(f"{b:02X}" for b in pkt)
                        print("[DET] <<< 收到停止应答(0x05)，"
                              f"立即发送位置帧 dx={last_pos_dx} 帧:{hx}")

                elif cmd == CommManager.CMD_DET_ACK:   # 0x06
                    if state == self._ST_GRAB_SEND:
                        # --- 抓取计数 -1 ---
                        if current_grab_target:
                            grabbed = self.shared_state.setdefault(
                                "grabbed_counts", {})
                            grabbed[current_grab_target] = (
                                grabbed.get(current_grab_target, 0) + 1)
                            rem = get_remaining_targets(self.shared_state)
                            rem_qty = rem.get(current_grab_target, 0)
                            print(f"[DET] <<< MCU收到抓取(ACK)，"
                                  f"{current_grab_target} 剩余: {rem_qty}")
                            print(f"[DET] 当前清单: "
                                  f"{_fmt_remaining(self.shared_state)}")
                        else:
                            print("[DET] <<< MCU收到抓取(ACK)")
                        state = self._ST_DETECTING
                        current_grab_target = None
                        print("[DET] <<< 收到0x06，继续识别")

            # ---- 状态机驱动（自动发送 / 重发）----
            now = time.time()

            if state == self._ST_DETECTING:
                if not paused and now - last_send_time >= 0.001:
                    if target_info:
                        delta_x = target_info[1]
                        if abs(delta_x) < self.stop_threshold:
                            # 进入停止帧阶段
                            current_grab_target = target_info[0]
                            comm.send(build_det_stop())
                            last_send_time = now
                            state = self._ST_STOP_SEND
                            print(f"[DET] >>> 目标进入停止阈值(|dx|={abs(delta_x)}"
                                  f"<{self.stop_threshold})，发送停止帧(0x51)")

            elif state == self._ST_STOP_SEND:
                if target_info is not None:
                    current_grab_target = target_info[0]
                    last_pos_dx = int(target_info[1])
                lost_target_frames = 0

                if now - last_send_time >= SEND_INTERVAL:
                    comm.send(build_det_stop())
                    last_send_time = now

            elif state == self._ST_POS_SEND:
                if target_info is not None:
                    current_grab_target = target_info[0]
                    last_pos_dx = int(target_info[1])
                    lost_target_frames = 0
                else:
                    lost_target_frames += 1
                    if lost_target_frames == 1:
                        print("[DET] >>> 位置阶段目标暂时丢失，停止重发上一帧位置，等待目标恢复")
                    if lost_target_frames >= self.det_lost_frames:
                        _enter_lost_send(now)
                        continue

                if target_info is not None and now - last_send_time >= SEND_INTERVAL:
                    if abs(last_pos_dx) < self.align_threshold:
                        grab_frame = build_det_grab(current_grab_target)
                        comm.send(grab_frame)
                        last_send_time = now
                        state = self._ST_GRAB_SEND
                        hx = " ".join(f"{b:02X}" for b in grab_frame)
                        print(f"[DET] >>> 已对齐(|dx|={abs(last_pos_dx)}"
                              f"<{self.align_threshold})，发送抓取帧:{hx}")
                    else:
                        pkt = build_det_position(last_pos_dx)
                        comm.send(pkt)
                        last_send_time = now
                        hx = " ".join(f"{b:02X}" for b in pkt)
                        print(f"[DET] >>> 发送位置 dx={last_pos_dx} 帧:{hx}")

            elif state == self._ST_GRAB_SEND:
                lost_target_frames = 0
                if now - last_send_time >= SEND_INTERVAL:
                    # 收到0x06前，无论中途识别到什么都只重发抓取帧
                    comm.send(build_det_grab(current_grab_target))
                    last_send_time = now

            elif state == self._ST_LOST_SEND:
                if now - last_send_time >= SEND_INTERVAL:
                    comm.send(build_det_lost())
                    last_send_time = now

            # ---- 终端日志 ----
            if not paused and frame_count % 30 == 1:
                rem_str = _fmt_remaining(self.shared_state)
                if target_info:
                    print(f"[DET] [{st_text}] target={target_info[0]} "
                          f"dx={target_info[1]} "
                          f"confirmed={len(confirmed)} "
                          f"FPS: {fps:.1f}  清单: {rem_str}")
                else:
                    if state == self._ST_DETECTING and last_raw_dets:
                        top_raw = sorted(last_raw_dets,
                                         key=lambda x: x[1],
                                         reverse=True)[:3]
                        top_desc = ", ".join(
                            f"{cls}:{conf:.2f}" for cls, conf, _ in top_raw)
                        print(f"[DET] [{st_text}] no target | "
                              f"raw={len(last_raw_dets)} [{top_desc}] "
                              f"confirmed={len(confirmed)} "
                              f"FPS: {fps:.1f}  清单: {rem_str}")
                    else:
                        print(f"[DET] [{st_text}] no target | "
                              f"confirmed={len(confirmed)} "
                              f"FPS: {fps:.1f}  清单: {rem_str}")

        if owns_cap:
            cap.release()
        return next_cmd

    # ==================== 检测推理 ====================

    def _run_detect(self, model, img):
        """对一帧图像做目标检测，返回 [(cls_name, conf, [x1,y1,x2,y2]), ...]"""
        predict_kwargs = dict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou_nms,
            verbose=False,
        )
        if not _is_tensorrt_engine_path(self.weights):
            predict_kwargs["device"] = self.device

        results = model.predict(
            **predict_kwargs,
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

    @staticmethod
    def _find_nearest_target(confirmed, img_center_x, targets):
        """从已确认目标中找到在目标清单中且离图片中心最近的
        返回 (class_name, delta_x, box) 或 None
        delta_x: 正=目标在中心右侧，负=左侧
        """
        best = None
        best_dist = float('inf')
        for cls_name, conf, box, track_id in confirmed:
            if cls_name not in targets:
                continue
            cx = (box[0] + box[2]) / 2
            delta = int(cx - img_center_x)
            dist = abs(delta)
            if dist < best_dist:
                best_dist = dist
                best = (cls_name, delta, box)
        return best

    @staticmethod
    def _get_best(confirmed):
        """从已确认目标中取置信度最高的，返回 (cls_name, conf) 或 (None, 0.0)"""
        if not confirmed:
            return (None, 0.0)
        best = max(confirmed, key=lambda x: x[1])
        return (best[0], best[1])

    # ==================== 辅助方法 ====================

    def _get_video_source(self):
        if self.source:
            return self.source
        if self.phone:
            return self._scan_phone()
        return self.camera_idx

    def _scan_phone(self):
        import socket as _sock
        import urllib.request as _url_req

        print("[DET] 正在扫描手机摄像头...")
        try:
            s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "192.168.1.1"

        ip_prefix = ".".join(local_ip.split(".")[:3]) + "."
        apps = [
            {"name": "IP Webcam", "port": 8080,
             "paths": ["/video", "/videofeed", ""]},
            {"name": "DroidCam",  "port": 4747,
             "paths": ["/video", "/mjpegfeed", ""]},
            {"name": "Iriun",     "port": 4747,
             "paths": ["/video", ""]},
            {"name": "EpocCam",   "port": 8080,
             "paths": ["/video", ""]},
        ]
        common_ips = ["192.168.42.129", "192.168.137.1", local_ip]
        for i in [1, 2, 100, 101, 102, 103, 104, 105, 50, 51]:
            common_ips.append(f"{ip_prefix}{i}")

        print(f"  本机 IP: {local_ip}, 扫描网段: {ip_prefix}x")
        for ip in common_ips:
            for app in apps:
                for path in app["paths"]:
                    url = f"http://{ip}:{app['port']}{path}"
                    try:
                        req = _url_req.Request(url, method='HEAD')
                        _url_req.urlopen(req, timeout=0.5)
                        print(f"  [发现] {app['name']}: {url}")
                        return url
                    except Exception:
                        pass
        print("  未找到手机摄像头")
        return None

    def _rotate_frame(self, frame):
        if self.rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _single_roi_coords(self, w, h):
        x1 = (w - self.roi_w) // 2 if self.roi_x < 0 else self.roi_x
        y1 = (h - self.roi_h) // 2 if self.roi_y < 0 else self.roi_y
        x1 = max(0, min(x1, w - self.roi_w))
        y1 = max(0, min(y1, h - self.roi_h))
        return x1, y1, x1 + self.roi_w, y1 + self.roi_h

    def _dual_roi_coords(self, w, h):
        gap = self.dual_gap
        total_h = self.roi_h * 2 + gap
        start_y = (h - total_h) // 2

        a_x1 = ((w - self.roi_w) // 2 if self.roi_a_x < 0
                 else self.roi_a_x)
        a_y1 = start_y if self.roi_a_y < 0 else self.roi_a_y
        b_x1 = ((w - self.roi_w) // 2 if self.roi_b_x < 0
                 else self.roi_b_x)
        b_y1 = (start_y + self.roi_h + gap if self.roi_b_y < 0
                 else self.roi_b_y)

        a_x1 = max(0, min(a_x1, w - self.roi_w))
        a_y1 = max(0, min(a_y1, h - self.roi_h))
        b_x1 = max(0, min(b_x1, w - self.roi_w))
        b_y1 = max(0, min(b_y1, h - self.roi_h))
        return ((a_x1, a_y1, a_x1 + self.roi_w, a_y1 + self.roi_h),
                (b_x1, b_y1, b_x1 + self.roi_w, b_y1 + self.roi_h))

    # ==================== 绘制方法 ====================

    @staticmethod
    def _draw_roi_box(frame, x1, y1, x2, y2,
                      color=(0, 255, 255), label="ROI"):
        """带角标的 ROI 框"""
        th = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
        cl, ct = 20, th + 2
        cv2.line(frame, (x1, y1), (x1 + cl, y1), color, ct)
        cv2.line(frame, (x1, y1), (x1, y1 + cl), color, ct)
        cv2.line(frame, (x2, y1), (x2 - cl, y1), color, ct)
        cv2.line(frame, (x2, y1), (x2, y1 + cl), color, ct)
        cv2.line(frame, (x1, y2), (x1 + cl, y2), color, ct)
        cv2.line(frame, (x1, y2), (x1, y2 - cl), color, ct)
        cv2.line(frame, (x2, y2), (x2 - cl, y2), color, ct)
        cv2.line(frame, (x2, y2), (x2, y2 - cl), color, ct)
        cv2.putText(frame, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    @staticmethod
    def _draw_confirmed(frame, confirmed, offset_x=0, offset_y=0):
        """绘制已确认的检测框 + 类名标签"""
        for cls_name, conf, box, track_id in confirmed:
            color = _get_class_color(cls_name)
            x1 = int(box[0]) + offset_x
            y1 = int(box[1]) + offset_y
            x2 = int(box[2]) + offset_x
            y2 = int(box[3]) + offset_y

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6),
                          (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_confirmed_with_targets(frame, confirmed, targets):
        """绘制已确认检测框：目标清单内=绿色粗框，不在=灰色细框"""
        for cls_name, conf, box, track_id in confirmed:
            x1, y1, x2, y2 = [int(v) for v in box]
            in_target = cls_name in targets
            color = (0, 255, 0) if in_target else (128, 128, 128)
            thickness = 2 if in_target else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{cls_name} {conf:.2f}"
            if in_target:
                label += " [TARGET]"
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
            cv2.rectangle(frame, (x1, y1 - tht - 4),
                          (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), font, fs,
                        (255, 255, 255), th)

    @staticmethod
    def _draw_single_label(frame, name, code, n_confirmed):
        """左上角单行结果标签"""
        text = f"{name} (0x{code:02X}) | confirmed: {n_confirmed}"
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        (tw, tht), bl = cv2.getTextSize(text, font, fs, th)
        cv2.rectangle(frame, (10, 10),
                      (20 + tw, 20 + tht + bl), (0, 0, 0), -1)
        color = (0, 255, 0) if code != 0x00 else (0, 165, 255)
        cv2.putText(frame, text, (15, 15 + tht), font, fs, color, th)

    @staticmethod
    def _draw_dual_labels(frame, name_a, code_a, name_b, code_b):
        """左上角双行结果标签"""
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2

        text_a = f"[A] {name_a} (0x{code_a:02X})"
        ca = (0, 255, 0) if code_a != 0x00 else (0, 165, 255)
        (tw, tht), bl = cv2.getTextSize(text_a, font, fs, th)
        cv2.rectangle(frame, (10, 10),
                      (20 + tw, 20 + tht + bl), (0, 0, 0), -1)
        cv2.putText(frame, text_a, (15, 15 + tht), font, fs, ca, th)

        text_b = f"[B] {name_b} (0x{code_b:02X})"
        cb = (0, 255, 0) if code_b != 0x00 else (0, 165, 255)
        y_off = 25 + tht + bl
        (tw2, tht2), bl2 = cv2.getTextSize(text_b, font, fs, th)
        cv2.rectangle(frame, (10, y_off),
                      (20 + tw2, y_off + 10 + tht2 + bl2),
                      (0, 0, 0), -1)
        cv2.putText(frame, text_b, (15, y_off + 5 + tht2),
                    font, fs, cb, th)

    @staticmethod
    def _draw_info(frame, fps, paused=False, comm_status=""):
        fh, fw = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.1f}", (fw - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if paused:
            cv2.putText(frame, "PAUSED", (fw // 2 - 60, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if comm_status:
            sc_map = {"IDLE": (180, 180, 180), "DET": (0, 255, 255),
                      "SENT": (255, 255, 0), "WAIT_ACK": (255, 255, 0),
                      "RETRY": (0, 165, 255), "TIMEOUT": (0, 0, 255)}
            sc = (200, 200, 200)
            for k, v in sc_map.items():
                if k in comm_status:
                    sc = v
                    break
            cv2.putText(frame, f"COMM: {comm_status}", (10, fh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
        cv2.putText(frame, "Q:Quit | S:Screenshot | Space:Pause",
                    (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)


# =====================================================================
# 5. 人脸识别模式 —— 完全保留
# =====================================================================

def preprocess_image(img):
    """图像预处理：CLAHE自适应直方图均衡化 + 双边滤波去噪"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    return img

# ====== SuperPoint + LightGlue 匹配参数 ======
FACE_SUPERPOINT_MAX_KEYPOINTS = 1024 # SuperPoint 最大关键点数
FACE_SUPERPOINT_RESIZE = 1024        # SuperPoint 在线缩放长边
FACE_MIN_MATCH_COUNT = 12           # LightGlue 最小匹配数
FACE_RANSAC_REPROJ_THRESHOLD = 5.0
FACE_LOCAL_TEMPLATE_THRESHOLD = 0.85
FACE_LOCAL_TEMPLATE_MIN_MATCHES = 40
FACE_LOCAL_CONTEXT_THRESHOLD = 0.8
FACE_LOCAL_CONTEXT_MIN_MATCHES = 18
FACE_LOCAL_CONTEXT_WEIGHT = 0.8
FACE_LOCAL_TEMPLATE_REGION_EXPAND = 2.0
FACE_LOCAL_CONTEXT_REGION_EXPAND = 3.6
FACE_X_FILTER_WINDOW = 5
FACE_TRACK_RECHECK_FRAMES = 15


class _FaceTemplateMatcher:
    _shared_device = None
    _shared_extractor = None
    _shared_matcher = None

    def __init__(self, max_keypoints=FACE_SUPERPOINT_MAX_KEYPOINTS,
                 resize=FACE_SUPERPOINT_RESIZE, target_expand_up=0.8,
                 target_expand_down=1.2, target_expand_side=0.6):
        if not HAS_LIGHTGLUE:
            raise RuntimeError("LightGlue/SuperPoint 不可用")

        if _FaceTemplateMatcher._shared_device is None:
            _FaceTemplateMatcher._shared_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if _FaceTemplateMatcher._shared_extractor is None:
            _FaceTemplateMatcher._shared_extractor = SuperPoint(
                max_num_keypoints=max_keypoints
            ).eval().to(_FaceTemplateMatcher._shared_device)
        if _FaceTemplateMatcher._shared_matcher is None:
            _FaceTemplateMatcher._shared_matcher = LightGlue(
                features="superpoint"
            ).eval().to(_FaceTemplateMatcher._shared_device)

        self.device = _FaceTemplateMatcher._shared_device
        self.resize = resize
        self.extractor = _FaceTemplateMatcher._shared_extractor
        self.matcher = _FaceTemplateMatcher._shared_matcher
        self.method_name = "SuperPoint+LightGlue"
        self.target_img = None
        self.target_feats = None
        self.target_expand_up = target_expand_up
        self.target_expand_down = target_expand_down
        self.target_expand_side = target_expand_side

    @staticmethod
    def _to_numpy_keypoints(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _to_tensor_gray(self, image):
        if image is None or image.size == 0:
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        tensor = torch.from_numpy(gray).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def _extract_features(self, image):
        tensor = self._to_tensor_gray(image)
        if tensor is None:
            return None
        with torch.no_grad():
            return self.extractor.extract(tensor, resize=self.resize)

    @staticmethod
    def _count_keypoints(feats):
        if feats is None or "keypoints" not in feats:
            return 0
        return int(feats["keypoints"].shape[1])

    @staticmethod
    def _expand_face_box(img_shape, face_location, expand_up=0.8,
                         expand_down=1.2, expand_side=0.6):
        top, right, bottom, left = face_location
        h, w = img_shape[:2]
        face_h = bottom - top
        face_w = right - left

        new_top = max(0, int(top - face_h * expand_up))
        new_bottom = min(h, int(bottom + face_h * expand_down))
        new_left = max(0, int(left - face_w * expand_side))
        new_right = min(w, int(right + face_w * expand_side))
        return new_top, new_right, new_bottom, new_left

    def set_target(self, img_path, face_location=None):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return False

        if face_location is not None:
            top, right, bottom, left = self._expand_face_box(
                img_bgr.shape,
                face_location,
                expand_up=self.target_expand_up,
                expand_down=self.target_expand_down,
                expand_side=self.target_expand_side,
            )
            img_bgr = img_bgr[top:bottom, left:right]
            print(
                f"  [FACE] 模板使用目标脸扩展区域: "
                f"({left},{top})-({right},{bottom})"
            )

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self.target_img = gray
        self.target_feats = self._extract_features(gray)

        kp_num = self._count_keypoints(self.target_feats)
        if kp_num < 10:
            print("  [FACE] 警告: 模板特征点过少")
            return False

        print(f"  [FACE] OK 提取{self.method_name}模板特征点: {kp_num} 个")
        return True

    def _estimate_homography(self, good_matches, frame_kp):
        if good_matches is None or len(good_matches) < FACE_MIN_MATCH_COUNT:
            return None, 0, None

        match_array = np.asarray(good_matches, dtype=np.int32)
        if match_array.ndim != 2 or match_array.shape[1] != 2:
            return None, 0, None

        target_keypoints = frame_kp["target_keypoints"]
        frame_keypoints = frame_kp["frame_keypoints"]
        src_pts = target_keypoints[match_array[:, 0]].reshape(-1, 1, 2)
        dst_pts = frame_keypoints[match_array[:, 1]].reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            FACE_RANSAC_REPROJ_THRESHOLD,
        )
        if H is None or mask is None:
            return None, 0, None

        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(np.sum(inlier_mask))
        return H, inlier_count, inlier_mask

    def match(self, frame_gray):
        if self.target_feats is None:
            return 0, 0, None, None

        query_feats = self._extract_features(frame_gray)
        if self._count_keypoints(query_feats) < 4:
            return 0, 0, None, None

        with torch.no_grad():
            try:
                match_out = self.matcher({"image0": self.target_feats, "image1": query_feats})
            except Exception:
                return 0, 0, None, None

        target_feats, query_feats, match_out = [
            rbd(x) for x in [self.target_feats, query_feats, match_out]
        ]
        matches = match_out.get("matches", None)
        if matches is None or len(matches) == 0:
            return 0, 0, None, None

        match_array = matches.detach().cpu().numpy().astype(np.int32)
        frame_meta = {
            "target_keypoints": self._to_numpy_keypoints(target_feats["keypoints"]),
            "frame_keypoints": self._to_numpy_keypoints(query_feats["keypoints"]),
        }
        good_matches = match_array.tolist()
        num_matches = int(match_array.shape[0])

        if num_matches < FACE_MIN_MATCH_COUNT:
            similarity = num_matches / FACE_MIN_MATCH_COUNT * 0.35
        else:
            _, inlier_count, _ = self._estimate_homography(good_matches, frame_meta)
            if inlier_count == 0:
                similarity = min(0.45, num_matches / (FACE_MIN_MATCH_COUNT * 2))
            else:
                inlier_ratio = inlier_count / max(1, num_matches)
                quantity_score = min(1.0, inlier_count / (FACE_MIN_MATCH_COUNT * 2))
                similarity = 0.65 * inlier_ratio + 0.35 * quantity_score

            num_matches = inlier_count

        return float(similarity), int(num_matches), good_matches, frame_meta

    def find_target_location(self, frame_gray, good_matches, frame_kp):
        H, inlier_count, _ = self._estimate_homography(good_matches, frame_kp)
        if H is None or inlier_count < max(8, FACE_MIN_MATCH_COUNT // 2):
            return None

        h_tgt, w_tgt = self.target_img.shape[:2]
        corners = np.float32([
            [0, 0],
            [w_tgt - 1, 0],
            [w_tgt - 1, h_tgt - 1],
            [0, h_tgt - 1],
        ]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        x_min = int(np.min(projected[:, 0]))
        x_max = int(np.max(projected[:, 0]))
        y_min = int(np.min(projected[:, 1]))
        y_max = int(np.max(projected[:, 1]))

        frame_h, frame_w = frame_gray.shape[:2]
        x_min = max(0, min(x_min, frame_w - 1))
        y_min = max(0, min(y_min, frame_h - 1))
        x_max = max(0, min(x_max, frame_w - 1))
        y_max = max(0, min(y_max, frame_h - 1))

        width = x_max - x_min
        height = y_max - y_min
        if width < 30 or height < 30:
            return None

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        return x_min, y_min, width, height, center_x, center_y

    def extract_region(self, frame, face_location, expand_ratio=2.0,
                       return_bounds=False):
        t, r, b, l = face_location
        h, w = frame.shape[:2]

        face_h = b - t
        face_w = r - l
        scale = expand_ratio / 2.0
        top_expand = int(face_h * 0.5 * scale)
        bottom_expand = int(face_h * 1.0 * scale)
        side_expand = int(face_w * 0.3 * scale)

        new_t = max(0, t - top_expand)
        new_b = min(h, b + bottom_expand)
        new_l = max(0, l - side_expand)
        new_r = min(w, r + side_expand)

        region = frame[new_t:new_b, new_l:new_r]
        if return_bounds:
            return region, (new_t, new_r, new_b, new_l)
        return region


class FaceMode:
    """
    人脸识别模式流程:
        MCU 发 0x03 → 切换 → 实时检测目标人脸
        匹配到目标且 |Δx| < face-stop-threshold → 发送停止帧(0x51)
        收到 MCU 0x05 后 → 发送位置帧(0x52)
        |Δx| < face-align-threshold → 发送放货帧(0x53)
        未匹配到目标人脸时不发送任何消息
    """

    def __init__(self, target_img, camera_idx, cam_w, cam_h,
                 push_interval, face_threshold,
                 face_weight=1.0, global_weight=0.0,
                 detect_every_n=2, num_jitters=10,
                 orb_features=1000, min_match_count=15,
                 voting_frames=3, shared_state=None,
                 use_cnn=True, preprocess=True, target_jitters=100,
                 face_stop_threshold=200,
                 face_align_threshold=10):
        self.target_img = target_img
        self.camera_idx = camera_idx
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.push_interval = push_interval
        self.face_threshold = face_threshold
        self.fw = face_weight
        self.gw = global_weight
        self.detect_every = detect_every_n
        self.num_jitters = num_jitters
        self.orb_features = orb_features
        self.min_match = min_match_count
        self.voting_n = voting_frames
        self.shared_state = shared_state if shared_state is not None else {}
        self.use_cnn = use_cnn
        self.preprocess = preprocess
        self.target_jitters = target_jitters
        self.cnn_available = use_cnn
        self.face_stop_threshold = int(face_stop_threshold)
        self.face_align_threshold = face_align_threshold

    @staticmethod
    def _clip_track_box(box, frame_shape):
        if box is None:
            return None

        try:
            x, y, w, h = [int(round(v)) for v in box]
        except Exception:
            return None

        frame_h, frame_w = frame_shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        if w < 10 or h < 10:
            return None
        return (x, y, w, h)

    @staticmethod
    def _create_light_tracker():
        if hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerMIL_create"):
            return legacy.TrackerMIL_create()
        raise RuntimeError("OpenCV 不支持 TrackerMIL_create")

    def _start_light_tracker(self, frame, box):
        clipped_box = self._clip_track_box(box, frame.shape)
        if clipped_box is None:
            return None, None

        try:
            tracker = self._create_light_tracker()
            init_result = tracker.init(frame, clipped_box)
        except Exception as e:
            print(f"  [FACE] 轻量跟踪初始化失败: {e}")
            return None, None

        if init_result is False:
            print("  [FACE] 轻量跟踪初始化失败: tracker.init 返回 False")
            return None, None
        return tracker, clipped_box

    def _run_full_face_recognition(self, frame, detection_model,
                                   local_matcher, context_matcher,
                                   votes, x_history,
                                   refresh_mode=False):
        scale = 0.75 if self.cnn_available else 0.25
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.preprocess:
            rgb = preprocess_image(rgb)
        try:
            face_locs = face_rec.face_locations(rgb, model=detection_model)
        except Exception:
            self.cnn_available = False
            detection_model = "hog"
            print("  [FACE] CNN 不可用，已降级为 HOG")
            face_locs = face_rec.face_locations(rgb, model="hog")
        scale_factor = 1.0 / scale

        best_result = None
        for (t, r, b, l) in face_locs:
            t = int(t * scale_factor)
            r = int(r * scale_factor)
            b = int(b * scale_factor)
            l = int(l * scale_factor)

            region, region_bounds = local_matcher.extract_region(
                frame,
                (t, r, b, l),
                expand_ratio=FACE_LOCAL_TEMPLATE_REGION_EXPAND,
                return_bounds=True,
            )
            if region.size == 0:
                continue

            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            local_similarity, num_matches, good_matches, region_kp = (
                local_matcher.match(gray_region)
            )

            context_similarity = 0.0
            context_matches = 0
            if context_matcher is not None:
                context_region = context_matcher.extract_region(
                    frame,
                    (t, r, b, l),
                    expand_ratio=FACE_LOCAL_CONTEXT_REGION_EXPAND,
                )
                if context_region is not None and context_region.size > 0:
                    gray_context = cv2.cvtColor(
                        context_region, cv2.COLOR_BGR2GRAY)
                    context_similarity, context_matches, _, _ = (
                        context_matcher.match(gray_context)
                    )

            target_loc = None
            if good_matches is not None and len(good_matches) >= FACE_MIN_MATCH_COUNT:
                local_loc = local_matcher.find_target_location(
                    gray_region, good_matches, region_kp)
                if local_loc:
                    x, y, w, h, center_x, center_y = local_loc
                    region_top, region_right, region_bottom, region_left = region_bounds
                    target_loc = (
                        x + region_left,
                        y + region_top,
                        w,
                        h,
                        center_x + region_left,
                        center_y + region_top,
                    )

            region_top, region_right, region_bottom, region_left = region_bounds
            region_w = region_right - region_left
            region_h = region_bottom - region_top
            region_center_x = (region_left + region_right) // 2
            region_center_y = (region_top + region_bottom) // 2

            combined_similarity = (
                local_similarity * (1.0 - FACE_LOCAL_CONTEXT_WEIGHT)
                + context_similarity * FACE_LOCAL_CONTEXT_WEIGHT
            )
            candidate = {
                "similarity": combined_similarity,
                "local_similarity": local_similarity,
                "context_similarity": context_similarity,
                "num_matches": num_matches,
                "context_matches": context_matches,
                "target_loc": target_loc,
                "region_box": (
                    region_left,
                    region_top,
                    region_w,
                    region_h,
                    region_center_x,
                    region_center_y,
                ),
            }
            if best_result is None or candidate["similarity"] > best_result["similarity"]:
                best_result = candidate

        result = {
            "detection_model": detection_model,
            "face_count": len(face_locs),
            "box": None,
            "box_center_y": None,
            "best_cx": None,
            "status_text": "NO FACE ROI",
            "color": (0, 0, 255),
            "thickness": 2,
            "confirmed": False,
            "track_box": None,
        }

        if best_result is None:
            if not refresh_mode:
                votes.clear()
            return result

        local_similarity = best_result["local_similarity"]
        context_similarity = best_result["context_similarity"]
        combined_similarity = best_result["similarity"]
        num_matches = best_result["num_matches"]
        context_matches = best_result["context_matches"]
        matched = (
            local_similarity >= FACE_LOCAL_TEMPLATE_THRESHOLD
            and num_matches >= FACE_LOCAL_TEMPLATE_MIN_MATCHES
            and (
                context_matcher is None
                or (
                    context_similarity >= FACE_LOCAL_CONTEXT_THRESHOLD
                    and context_matches >= FACE_LOCAL_CONTEXT_MIN_MATCHES
                )
            )
        )

        box = best_result["target_loc"] or best_result["region_box"]
        x, y, w, h, center_x, center_y = box

        if refresh_mode:
            confirmed = matched
            avg_sim = combined_similarity if matched else 0.0
            filtered_center_x = center_x
        else:
            votes.append((matched, combined_similarity))
            confirmed, avg_sim = self._vote_check(votes)
            x_history.append(center_x)
            filtered_center_x = int(np.median(x_history))

        if confirmed:
            if refresh_mode:
                color, thickness = (0, 255, 255), 2
                label = f"REFRESH {avg_sim:.0%}"
            else:
                color, thickness = (0, 255, 0), 3
                label = f"CONFIRMED! {avg_sim:.0%}"
            best_cx = filtered_center_x
        elif matched:
            color, thickness = (0, 200, 255), 2
            label = f"WAIT {combined_similarity:.0%}"
            best_cx = None
        else:
            color, thickness = (0, 0, 255), 2
            label = f"NO {combined_similarity:.0%}"
            best_cx = None

        result.update({
            "box": (x, y, w, h),
            "box_center_y": center_y,
            "best_cx": best_cx,
            "status_text": f"{label} (L:{num_matches} C:{context_matches})",
            "color": color,
            "thickness": thickness,
            "confirmed": confirmed,
            "track_box": (x, y, w, h),
        })
        return result

    def run(self, comm, shared_camera=None):
        print("\n[FACE] ===== 进入人脸识别模式 =====")
        if not HAS_FACE:
            print("[FACE] 错误: 缺少 face_recognition, "
                  "pip install face_recognition")
            return None
        if not HAS_LIGHTGLUE:
            print("[FACE] 错误: 缺少 LightGlue/SuperPoint 依赖")
            return None

        detection_model = "cnn" if self.cnn_available else "hog"
        print(f"  [FACE] 检测模型: {detection_model.upper()}")
        if self.preprocess:
            print("  [FACE] 图像预处理: CLAHE + 双边滤波")

        # 优先使用 OCR 阶段保存的目标图
        ocr_img = self.shared_state.get("face_img_path")
        if ocr_img and os.path.isfile(ocr_img):
            self.target_img = ocr_img
            print("  [FACE] ✓ 使用 OCR 阶段保存的目标图")
        else:
            print("  [FACE] OCR 阶段未保存目标图，回退到静态目标图片")

        resolved_target_img = self._resolve_target_image(self.target_img)
        if resolved_target_img is None:
            print(f"[FACE] 目标图片不存在或目录中无可用图片: {self.target_img}")
            return None
        if resolved_target_img != self.target_img:
            print(f"  [FACE] 目标图片已解析为: {resolved_target_img}")
        self.target_img = resolved_target_img

        target_location = self._load_face_location()
        if target_location is None:
            print("[FACE] 目标图片未检测到人脸，无法初始化局部模板模式")
            return None

        local_matcher = _FaceTemplateMatcher(
            max_keypoints=FACE_SUPERPOINT_MAX_KEYPOINTS,
            target_expand_up=0.8,
            target_expand_down=1.2,
            target_expand_side=0.6,
        )
        if not local_matcher.set_target(self.target_img, face_location=target_location):
            print("[FACE] 局部模板初始化失败")
            return None

        context_matcher = _FaceTemplateMatcher(
            max_keypoints=FACE_SUPERPOINT_MAX_KEYPOINTS,
            target_expand_up=1.4,
            target_expand_down=2.2,
            target_expand_side=1.8,
        )
        if not context_matcher.set_target(self.target_img, face_location=target_location):
            print("  [FACE] 警告: 大上下文模板初始化失败，仍继续")
            context_matcher = None

        tgt_show = cv2.imread(self.target_img)
        if tgt_show is not None:
            top, right, bottom, left = target_location
            cv2.rectangle(tgt_show, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Target", cv2.resize(tgt_show, None, fx=0.5, fy=0.5))

        owns_cap = False
        if shared_camera is not None:
            try:
                cap = shared_camera.get()
            except Exception as e:
                print(f"[FACE] 无法获取共享摄像头: {e}")
                return None
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[FACE] 复用共享摄像头: {actual_w}x{actual_h}")
        else:
            cap = cv2.VideoCapture(self.camera_idx)
            if not cap.isOpened():
                print(f"[FACE] 无法打开摄像头 {self.camera_idx}")
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
            owns_cap = True
        print("[FACE] 实时推送人脸坐标中...")
        print(f"  [FACE] 停止阈值: {self.face_stop_threshold}px")
        print(f"  [FACE] 放货阈值: {self.face_align_threshold}px")

        frame_count = 0
        votes = deque(maxlen=10)
        x_history = deque(maxlen=FACE_X_FILTER_WINDOW)
        tracker = None
        tracker_bbox = None
        tracker_active = False
        tracker_refresh_age = 0
        force_full_recognition = True
        last_face_count = 0
        # 人脸模式状态机
        _FACE_SEARCHING    = 0   # 搜索目标人脸；未匹配时静默
        _FACE_STOP_SEND    = 1   # 发送停止帧(0x51)，等待 MCU 0x05
        _FACE_POS_SEND     = 2   # 收到 0x05 后发送位置帧(0x52)
        _FACE_DELIVER_SEND = 3   # 发送放货帧(0x53)直到模式切换/退出
        face_state = _FACE_SEARCHING
        face_send_time = 0
        FACE_SEND_INTERVAL = self.push_interval
        face_pos_sent = False

        next_cmd = None
        img_center_x = self.cam_w // 2
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            h_frame, w_frame = frame.shape[:2]
            img_center_x = w_frame // 2
            display = frame.copy()
            best_cx = None
            current_face_dx = None
            visual_face_dx = None
            overlay_box = None
            overlay_color = (0, 0, 255)
            overlay_thickness = 2
            overlay_text = "NO FACE ROI"
            vision_state = "RECOGNIZING"
            tracker_updated = False

            cv2.line(display, (img_center_x, 0), (img_center_x, h_frame),
                     (128, 128, 128), 1)

            if tracker_active:
                ok, tracked_box = tracker.update(frame)
                if ok:
                    tracked_box = self._clip_track_box(tracked_box, frame.shape)
                if tracked_box is None:
                    tracker_active = False
                    tracker = None
                    tracker_bbox = None
                    tracker_refresh_age = 0
                    force_full_recognition = True
                    votes.clear()
                    x_history.clear()
                    print("  [FACE] 轻量跟踪丢失，切回完整识别")
                else:
                    tracker_updated = True
                    tracker_bbox = tracked_box
                    tracker_refresh_age += 1
                    tx, ty, tw, th = tracked_box
                    best_cx = tx + tw // 2
                    current_face_dx = best_cx - img_center_x
                    overlay_box = tracked_box
                    overlay_color = (255, 255, 0)
                    overlay_thickness = 2
                    overlay_text = "TRACKING"
                    vision_state = "TRACKING"

            refresh_mode = tracker_active and tracker_updated and (
                tracker_refresh_age >= FACE_TRACK_RECHECK_FRAMES
            )
            run_full_recognition = False
            if refresh_mode:
                run_full_recognition = True
            elif not tracker_active and (
                force_full_recognition or frame_count % self.detect_every == 0
            ):
                run_full_recognition = True

            if run_full_recognition:
                recognition = self._run_full_face_recognition(
                    frame,
                    detection_model,
                    local_matcher,
                    context_matcher,
                    votes,
                    x_history,
                    refresh_mode=refresh_mode,
                )
                detection_model = recognition["detection_model"]
                last_face_count = recognition["face_count"]

                if recognition["box"] is not None:
                    overlay_box = recognition["box"]
                    overlay_color = recognition["color"]
                    overlay_thickness = recognition["thickness"]
                    overlay_text = recognition["status_text"]
                elif not tracker_active:
                    overlay_box = None
                    overlay_color = recognition["color"]
                    overlay_thickness = recognition["thickness"]
                    overlay_text = recognition["status_text"]

                if recognition["best_cx"] is not None:
                    best_cx = recognition["best_cx"]
                    current_face_dx = best_cx - img_center_x

                if recognition["confirmed"] and recognition["track_box"] is not None:
                    tracker, tracker_bbox = self._start_light_tracker(
                        frame, recognition["track_box"])
                    if tracker is not None:
                        was_tracking = tracker_active
                        tracker_active = True
                        tracker_refresh_age = 0
                        force_full_recognition = False
                        x_history.clear()
                        votes.clear()
                        if was_tracking:
                            print("  [FACE] 定期重识别校正成功，更新轻量跟踪框")
                        else:
                            print("  [FACE] 已确认目标，切换到轻量跟踪")
                        vision_state = "TRACKING"
                        if tracker_bbox is not None:
                            overlay_box = tracker_bbox
                    else:
                        tracker_active = False
                        tracker_bbox = None
                        force_full_recognition = True
                elif refresh_mode and tracker_updated:
                    tracker_refresh_age = 0
                    print("  [FACE] 定期重识别未确认目标，继续使用轻量跟踪")
                    vision_state = "TRACKING"
                else:
                    force_full_recognition = False

            if overlay_box is not None:
                x, y, w, h = overlay_box
                center_y = y + h // 2
                draw_cx = best_cx if best_cx is not None else (x + w // 2)
                visual_face_dx = int(draw_cx - img_center_x)
                cv2.rectangle(display, (x, y), (x + w, y + h),
                              overlay_color, overlay_thickness)
                cv2.circle(display, (draw_cx, center_y), 5, overlay_color, -1)
                cv2.putText(display, f"X:{draw_cx}", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 2)
                cv2.circle(display, (draw_cx, center_y), 8, (0, 0, 255), -1)
                cv2.line(display, (img_center_x, center_y), (draw_cx, center_y),
                         (0, 0, 255), 2)
                cv2.putText(display, f"dx={visual_face_dx}",
                            (min(draw_cx, img_center_x) + 5, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display, overlay_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        overlay_color, 2)

            state_name = {
                _FACE_SEARCHING: "SEARCHING",
                _FACE_STOP_SEND: "STOP_SEND",
                _FACE_POS_SEND: "POS_SEND",
                _FACE_DELIVER_SEND: "DELIVER_SEND",
            }.get(face_state, "UNKNOWN")
            cv2.putText(display, f"COMM: {state_name}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"VISION: {vision_state}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            det_lbl = "CNN" if self.cnn_available else "HOG"
            cv2.putText(
                display,
                f"Mode: Local Template+Track [{det_lbl}] Faces:{last_face_count} | 'q':quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

            cv2.imshow("Unified Vision", display)
            key = cv2.waitKey(1) & 0xFF
            if isinstance(comm, _DummyComm):
                comm.feed_key(key)
            _handle_enter_to_enable_rx(comm, key)
            if key == ord('q') or key == 27:
                next_cmd = None
                break

            cmd = comm.get_cmd()
            if cmd is not None:
                if _is_mode_switch_cmd(cmd):
                    if cmd == CommManager.CMD_FACE:
                        _log_ignore_same_mode_cmd("  [FACE]", cmd)
                    else:
                        next_cmd = cmd
                        break
                elif cmd == CommManager.CMD_STOP_ACK:
                    if face_state == _FACE_STOP_SEND:
                        face_state = _FACE_POS_SEND
                        face_pos_sent = False
                        face_send_time = time.time()
                        if current_face_dx is not None:
                            pkt = build_face_position(current_face_dx)
                            comm.send(pkt)
                            face_send_time = time.time()
                            face_pos_sent = True
                            hx = " ".join(f"{b:02X}" for b in pkt)
                            print("  [FACE] <<< 收到停止应答(0x05)，"
                                  f"立即发送位置帧 dx={current_face_dx} 帧:{hx}")
                        else:
                            print("  [FACE] <<< 收到停止应答(0x05)，"
                                  "等待重新检测到目标人脸后发送位置帧")
                elif cmd == CommManager.CMD_EXIT_DET:   # 0xA4
                    comm.send(build_pc_ack())
                    print("  [FACE] <<< MCU退出识别，已回复ACK")
                    next_cmd = None
                    break

            # ---- 20ms 定时发送逻辑 ----
            now = time.time()
            if now - face_send_time >= FACE_SEND_INTERVAL:
                if face_state == _FACE_SEARCHING:
                    if (current_face_dx is not None
                            and abs(current_face_dx) < self.face_stop_threshold):
                        comm.send(build_face_stop())
                        face_send_time = now
                        face_state = _FACE_STOP_SEND
                        print(f"  [FACE] >>> 目标进入停止阈值(|dx|={abs(current_face_dx)}"
                              f"<{self.face_stop_threshold})，发送停止帧(0x51)")
                elif face_state == _FACE_STOP_SEND:
                    comm.send(build_face_stop())
                    face_send_time = now
                elif face_state == _FACE_POS_SEND:
                    if current_face_dx is None:
                        continue
                    if not face_pos_sent:
                        pkt = build_face_position(current_face_dx)
                        comm.send(pkt)
                        face_send_time = now
                        face_pos_sent = True
                    elif abs(current_face_dx) < self.face_align_threshold:
                        comm.send(build_face_deliver_new())
                        face_send_time = now
                        face_state = _FACE_DELIVER_SEND
                        print(f"  [FACE] >>> 已对齐(|dx|={abs(current_face_dx)}"
                              f"<{self.face_align_threshold})，发送放货帧(0x53)")
                    else:
                        pkt = build_face_position(current_face_dx)
                        comm.send(pkt)
                        face_send_time = now
                elif face_state == _FACE_DELIVER_SEND:
                    comm.send(build_face_deliver_new())
                    face_send_time = now

        if owns_cap:
            cap.release()
        cv2.destroyWindow("Target")
        return next_cmd

    # ---------- helpers ----------
    @staticmethod
    def _resolve_target_image(target_path):
        if not target_path:
            return None

        normalized = os.path.abspath(target_path)
        if os.path.isfile(normalized):
            return normalized
        if not os.path.isdir(normalized):
            return None

        preferred_names = ["OCR.png", "ocr.png"]
        for name in preferred_names:
            candidate = os.path.join(normalized, name)
            if os.path.isfile(candidate):
                return candidate

        image_files = []
        for name in os.listdir(normalized):
            candidate = os.path.join(normalized, name)
            lower_name = name.lower()
            if os.path.isfile(candidate) and lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                image_files.append(candidate)

        if not image_files:
            return None
        image_files.sort(key=os.path.getmtime, reverse=True)
        return image_files[0]

    def _load_face_location(self):
        img = face_rec.load_image_file(self.target_img)
        if self.preprocess:
            img = preprocess_image(img)
        model = "cnn" if self.cnn_available else "hog"
        try:
            locs = face_rec.face_locations(img, model=model)
        except Exception:
            self.cnn_available = False
            locs = face_rec.face_locations(img, model="hog")
            print("  [FACE] CNN 不可用，已降级为 HOG")
        if not locs:
            return None
        print(f"  目标人脸已加载 loc={locs[0]}  model={model}  "
              f"template=local")
        return locs[0]

    def _vote_check(self, votes):
        recent = list(votes)[-self.voting_n:]
        if len(recent) < self.voting_n:
            return False, 0
        match_count = sum(1 for m, s in recent if m)
        avg_sim = (np.mean([s for m, s in recent if m])
                   if match_count > 0 else 0)
        confirmed = match_count >= self.voting_n
        return confirmed, avg_sim

# =====================================================================
# 6. 主程序
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="统一视觉识别系统（检测版）—— 小车三合一控制端",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- 串口 ---
    p.add_argument("--port", default="COM3", help="串口号 (默认COM3)")
    p.add_argument("--baud", type=int, default=115200, help="波特率")

    # --- 摄像头 ---
    p.add_argument("--camera", type=int, default=1, help="摄像头索引")
    p.add_argument("--cam-w", type=int, default=1280, help="摄像头宽度")
    p.add_argument("--cam-h", type=int, default=720, help="摄像头高度")

    # --- 饮料检测 ---
    p.add_argument(
        "--weights",
        default=r"C:\chaoshi\runs\detect\bottle_train12\weights\best.pt",
        help="YOLOv8 检测模型权重路径（支持 .pt / .engine；传 .engine 时为手动 TensorRT 切换）")
    p.add_argument("--device", default="0", help="推理设备 cpu/0/1")
    p.add_argument("--conf", type=float, default=0.9,
                   help="检测置信度阈值")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="模型输入尺寸")
    p.add_argument("--rotate", type=int, default=0,
                   choices=[0, 90, 180, 270],
                   help="画面旋转角度 (手机竖拍用90)")

    # ---- 检测模式视频源 ----
    p.add_argument("--det-source", type=str, default="",
                   help="检测模式视频源URL")
    p.add_argument("--det-phone", action="store_true",
                   help="检测模式自动扫描手机摄像头")
    p.add_argument("--no-show", action="store_true",
                   help="不显示画面(仅终端)")
    p.add_argument("--det-save-dir", default="screenshots",
                   help="截图保存目录")

    # ---- MultiFrameTracker 参数 ----
    p.add_argument("--confirm-frames", type=int, default=1,
                   help="MultiFrameTracker 确认帧数 (默认1)")
    p.add_argument("--iou-match", type=float, default=0.5,
                   help="追踪匹配 IoU 阈值 (默认0.5)")
    p.add_argument("--notfound-delay", type=float, default=1.0,
                   help="兼容保留参数（当前检测流程未使用）")
    p.add_argument("--notfound-min-frames", type=int, default=15,
                   help="兼容保留参数（当前检测流程未使用）")

    # ---- 抓取对齐参数 ----
    p.add_argument("--det-stop-threshold", type=int, default=80,
                   help="饮料瓶停止阈值(px)，|Δx|<此值则发停止帧")
    p.add_argument("--align-threshold", type=int, default=20,
                   help="饮料瓶对齐阈值(px)，|Δx|<此值则抓取")
    p.add_argument("--det-lost-frames", type=int, default=5,
                   help="位置阶段连续丢失目标多少帧后发送0x55，默认5")

    # --- OCR ---
    p.add_argument("--save-dir", default=r"C:\chaoshi\images",
                   help="OCR 截图保存目录")
    p.add_argument("--ollama-url",
                   default="http://localhost:11434/api/generate",
                   help="Ollama 地址")
    p.add_argument("--llm-model", default="qwen2.5:1.5b",
                   help="大模型名称") 

    # --- 人脸 ---
    p.add_argument("--target-img", default=r"C:\chaoshi\images",
                   help="目标人脸图片路径，传目录时会自动选择 OCR.png 或目录中最新图片")
    p.add_argument("--face-cam-w", type=int, default=1280,
                   help="兼容保留参数；共享摄像头下统一使用 --cam-w")
    p.add_argument("--face-cam-h", type=int, default=720,
                   help="兼容保留参数；共享摄像头下统一使用 --cam-h")
    p.add_argument("--send-interval", type=float, default=0.02,
                   help="统一帧发送/重发间隔(秒)，默认 0.02 (20ms)，含OCR触发ACK循环")
    p.add_argument("--face-threshold", type=float, default=0.55,
                   help="人脸混合相似度阈值")
    p.add_argument("--use-cnn", action="store_true", default=True,
                   help="使用CNN模型检测人脸")
    p.add_argument("--no-cnn", dest="use_cnn", action="store_false",
                   help="强制使用HOG模型（CPU友好）")
    p.add_argument("--no-preprocess", dest="preprocess",
                   action="store_false", default=True,
                   help="禁用图像预处理(CLAHE+双边滤波)")
    p.add_argument("--target-jitters", type=int, default=100,
                   help="目标人脸编码抖动次数")
    p.add_argument("--num-jitters", type=int, default=10,
                   help="实时检测人脸编码抖动次数")
    p.add_argument("--face-weight", type=float, default=0.6,
                   help="人脸特征权重(混合模式)")
    p.add_argument("--global-weight", type=float, default=0.4,
                   help="全图特征权重(混合模式)")
    p.add_argument("--voting-frames", type=int, default=3,
                   help="多帧投票确认帧数")
    p.add_argument("--face-stop-threshold", type=int, default=200,
                   help="人脸停止阈值(px)，|Δx|<此值则发停止帧")
    p.add_argument("--face-align-threshold", type=int, default=25,
                   help="人脸对齐阈值(px)，|Δx|<此值则发放货帧")

    # --- 调试 ---
    p.add_argument("--no-serial", action="store_true",
                   help="不使用串口（键盘模拟命令）")
    p.add_argument("--comm-log", action="store_true",
                   help="打印串口通信日志（时间戳 + 方向 + 十六进制），默认关闭")
    p.add_argument("--comm-log-file", dest="comm_log_file",
                   action="store_true", default=True,
                   help="写入通信日志到当前目录 comm_log.txt（默认开启，启动覆盖）")
    p.add_argument("--no-comm-log-file", dest="comm_log_file",
                   action="store_false",
                   help="关闭通信日志txt输出")
    return p


def main():
    args = build_parser().parse_args()

    print("=" * 60)
    print(" 统一视觉识别系统（检测版）—— 小车三合一控制端")
    print("=" * 60)

    # 初始化串口
    comm = None
    if args.no_serial:
        comm = _DummyComm()
    else:
        if not HAS_SERIAL:
            print("[错误] pyserial 未安装，pip install pyserial")
            print("       或使用 --no-serial 进行键盘调试")
            return
        try:
            comm = CommManager(
                args.port,
                args.baud,
                comm_log=args.comm_log,
                comm_log_file=args.comm_log_file,
            )
        except Exception as e:
            print(f"[错误] 无法打开串口 {args.port}: {e}")
            print("       请检查端口号，或使用 --no-serial 调试")
            return

    shared_state = {"face_img_path": None,
                     "ocr_targets": {}, "grabbed_counts": {}}

    if args.face_cam_w != args.cam_w or args.face_cam_h != args.cam_h:
        print(
            "[主循环] 已启用共享摄像头，人脸分辨率参数将被忽略，"
            f"统一使用 {args.cam_w}x{args.cam_h}"
        )

    shared_camera = None
    try:
        shared_camera = SharedCamera(
            camera_idx=args.camera,
            cam_w=args.cam_w,
            cam_h=args.cam_h,
            warmup_frames=20,
        )
        print(
            f"[主循环] 初始化共享摄像头: index={args.camera} "
            f"target={args.cam_w}x{args.cam_h}"
        )
        shared_camera.open()
        shared_camera.warmup()
    except Exception as e:
        print(f"[错误] 共享摄像头初始化失败: {e}")
        comm.close()
        return

    # 构建三个模式
    ocr_mode = OCRMode(
        camera_idx=args.camera, cam_w=args.cam_w, cam_h=args.cam_h,
        save_dir=args.save_dir, ollama_url=args.ollama_url,
        llm_model=args.llm_model, shared_state=shared_state,
        send_interval=args.send_interval,
    )
    det_mode = DetectMode(
        weights=args.weights, device=args.device,
        conf=args.conf, iou_nms=args.iou, imgsz=args.imgsz,
        camera_idx=args.camera, cam_w=args.cam_w, cam_h=args.cam_h,
        rotate=args.rotate,
        source=args.det_source, phone=args.det_phone,
        no_show=args.no_show,
        save_dir=args.det_save_dir,
        confirm_frames=args.confirm_frames,
        iou_match=args.iou_match,
        stop_threshold=args.det_stop_threshold,
        notfound_delay=args.notfound_delay,
        notfound_min_frames=args.notfound_min_frames,
        align_threshold=args.align_threshold,
        det_lost_frames=args.det_lost_frames,
        shared_state=shared_state,
        send_interval=args.send_interval,
    )
    face_mode = FaceMode(
        target_img=args.target_img, camera_idx=args.camera,
        cam_w=args.cam_w, cam_h=args.cam_h,
        push_interval=args.send_interval,
        face_threshold=args.face_threshold,
        face_weight=args.face_weight,
        global_weight=args.global_weight,
        detect_every_n=2,
        num_jitters=args.num_jitters,
        voting_frames=args.voting_frames,
        shared_state=shared_state,
        use_cnn=args.use_cnn,
        preprocess=args.preprocess,
        target_jitters=args.target_jitters,
        face_stop_threshold=args.face_stop_threshold,
        face_align_threshold=args.face_align_threshold,
    )

    mode_map = {
        CommManager.CMD_OCR:      ocr_mode,
        CommManager.CMD_CLASSIFY: det_mode,    # 0x02 → 检测模式
        CommManager.CMD_FACE:     face_mode,
    }

    print("\n[主循环] 等待 MCU 指令...")
    print("  0x01(1)=OCR  0x02(2)=饮料检测  0x03(3)=人脸")
    print("-" * 60)

    try:
        while True:
            idle_frame = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(idle_frame, "IDLE - Waiting MCU Command",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(idle_frame,
                        "1:OCR  2:Detect  3:Face  Q:Quit",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)
            if isinstance(comm, CommManager) and not comm.is_rx_enabled():
                cv2.putText(idle_frame,
                            "Press ENTER once to enable MCU RX",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 200, 255), 1)
            cv2.imshow("Unified Vision", idle_frame)
            key = cv2.waitKey(100) & 0xFF

            if isinstance(comm, _DummyComm):
                comm.feed_key(key)
            _handle_enter_to_enable_rx(comm, key)

            if key == ord('q') or key == 27:
                break

            cmd = comm.get_cmd()
            if cmd is None:
                continue

            mode_obj = mode_map.get(cmd)
            if mode_obj is None:
                print(f"[主循环] 未知命令: 0x{cmd:02X}")
                continue

            mode_id = cmd
            comm.send(build_mode_ack(mode_id))
            print(f"[主循环] >>> 进入模式 0x{mode_id:02X}，已发送确认")

            next_cmd = mode_obj.run(comm, shared_camera)
            cv2.destroyAllWindows()

            if next_cmd is None:
                print("[主循环] 回到空闲")
                continue
            elif next_cmd in mode_map:
                new_id = next_cmd
                comm.send(build_mode_ack(new_id))
                print(f"[主循环] >>> 直接切换模式 0x{new_id:02X}")
                next_cmd2 = mode_map[new_id].run(comm, shared_camera)
                cv2.destroyAllWindows()
                if next_cmd2 and next_cmd2 in mode_map:
                    if isinstance(comm, _DummyComm):
                        comm._pending = next_cmd2
                    else:
                        comm.cmd_queue.put(next_cmd2)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        if shared_camera is not None:
            shared_camera.release()
        comm.close()
        print("[系统] 已退出")


if __name__ == "__main__":
    main()
