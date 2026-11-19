"""Reports what hardware the task landed on."""
import json
import os
import platform
import shutil
import subprocess

info = {"provider": os.environ.get("FGPU_PROVIDER", ""), "python": platform.python_version(),
        "machine": platform.machine(), "cpus": os.cpu_count(), "gpu": None}
if shutil.which("nvidia-smi"):
    info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
os.makedirs("out", exist_ok=True)
with open("out/info.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info))
