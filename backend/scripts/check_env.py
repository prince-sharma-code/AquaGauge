"""Non-failing environment preflight for the hackathon machine."""
import importlib.util
import shutil
import subprocess
import sys

print(f"Python: {sys.version.split()[0]}")
for name in ["fastapi", "uvicorn", "numpy", "cv2", "PIL", "pytest"]:
    ok = importlib.util.find_spec(name) is not None
    print(f"{'OK' if ok else 'MISSING'}: {name}")

if shutil.which("nvidia-smi"):
    try:
        p = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=8)
        print("nvidia-smi:", "OK" if p.returncode == 0 else "FAILED")
        if p.returncode == 0:
            print("  " + p.stdout.splitlines()[0])
    except Exception as e:
        print("nvidia-smi check failed:", e)
else:
    print("nvidia-smi: not found (CPU demo is still viable)")

if importlib.util.find_spec("torch") is not None:
    try:
        import torch
        print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("PyTorch import/CUDA check failed:", repr(e))
else:
    print("PyTorch: not installed (install optional vision stack only if needed)")
