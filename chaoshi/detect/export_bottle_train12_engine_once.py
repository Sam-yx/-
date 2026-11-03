import os
import shutil
import sys
from pathlib import Path


SOURCE_PT = Path("C:/chaoshi/runs/detect/bottle_train12/weights/best.pt")
TARGET_ENGINE = Path("C:/chaoshi/runs/detect/bottle_train12/best.engine")
IMGSZ = 1280
DEVICE = 0
ENGINE_HALF = True
ENGINE_DYNAMIC = True
ENGINE_BATCH = 1

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


def _setup_windows_runtime():
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


def main():
    _setup_windows_runtime()

    if not SOURCE_PT.exists():
        raise FileNotFoundError(f"未找到源权重: {SOURCE_PT}")

    print("=" * 60)
    print("  bottle_train12 一次性 TensorRT 导出")
    print("=" * 60)
    print(f"  源权重   : {SOURCE_PT}")
    print(f"  输出文件 : {TARGET_ENGINE}")
    print(f"  精度模式 : {'FP16' if ENGINE_HALF else 'FP32'}")
    print(f"  动态输入 : {ENGINE_DYNAMIC}")
    print(f"  Batch    : {ENGINE_BATCH}")

    from ultralytics import YOLO

    model = YOLO(str(SOURCE_PT))
    exported_engine = model.export(
        format="engine",
        imgsz=IMGSZ,
        device=DEVICE,
        half=ENGINE_HALF,
        dynamic=ENGINE_DYNAMIC,
        batch=ENGINE_BATCH,
        verbose=True,
    )

    exported_path = Path(exported_engine)
    TARGET_ENGINE.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_ENGINE.exists():
        TARGET_ENGINE.unlink()

    if exported_path.resolve() != TARGET_ENGINE.resolve():
        shutil.move(str(exported_path), str(TARGET_ENGINE))

    print(f"[INFO] 导出完成: {TARGET_ENGINE}")


if __name__ == "__main__":
    main()