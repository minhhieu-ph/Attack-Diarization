"""Check Python packages, GPU visibility and a tiny bf16 matmul on the GPU."""
import importlib
import os
import platform
import shutil
import subprocess
import sys

OK, BAD = "\033[32mOK\033[0m", "\033[31mMISSING\033[0m"
print(f"python {sys.version.split()[0]} on {platform.platform()}  cpus={os.cpu_count()}")
for m in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "yaml", "xgboost", "optuna", "tabulate", "torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<10} {getattr(mod, '__version__', '?'):<12} {OK}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:<10} {'':<12} {BAD} ({e.__class__.__name__})")

if shutil.which("nvidia-smi"):
    q = "name,memory.total,driver_version"
    print(subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())

try:
    import torch

    print(f"torch {torch.__version__}  built for CUDA {torch.version.cuda}  cuda available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  device: {p.name}  {p.total_memory / 2**30:.0f} GiB  capability {p.major}.{p.minor}")
        archs = torch.cuda.get_arch_list()
        print(f"  wheel archs: {archs}")
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        _ = (a @ a).float().sum().item()
        print(f"  bf16 matmul on GPU: {OK}")
    else:
        print("  WARNING: no CUDA device visible to torch. If nvidia-smi works, the wheel's CUDA is newer than "
              "the driver supports: rerun with TORCH_CUDA=cu128 bash setup_env.sh")
except Exception as e:  # noqa: BLE001
    print(f"torch check failed: {e}")
    print("  If the error mentions 'no kernel image' or sm_10x, install a wheel built for CUDA >= 12.8.")

try:
    import xgboost as xgb
    import numpy as np

    X = np.random.rand(1000, 5).astype("float32")
    y = (X[:, 0] > .5).astype(int)
    xgb.XGBClassifier(n_estimators=5, device="cuda", tree_method="hist").fit(X, y)
    print(f"xgboost on GPU: {OK}")
except Exception as e:  # noqa: BLE001
    print(f"xgboost GPU check: {e} (CPU fallback will be used)")
