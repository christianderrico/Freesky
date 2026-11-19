"""Your own machine: always free, used as the last resort (or first, if it has a GPU)."""
from __future__ import annotations

import os
import subprocess
import sys

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, RunResult


def detect_gpu() -> str:
    try:
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "none"
    if not name:
        return "none"
    n = name[0].upper()
    for g in ("H100", "A100", "A10G", "L4", "T4", "P100"):
        if g in n:
            return g
    return "local-gpu"


class LocalProvider(Provider):
    name = "local"
    title = "Questo computer"
    startup_s = 0
    signup = "-"
    install = "-"

    def __init__(self, config=None):
        super().__init__(config)
        self.gpus = [detect_gpu()]

    def pick_gpu(self, task):
        g = self.gpus[0]
        if not task.resources.needs_gpu:
            return "none"
        if g == "local-gpu":  # unknown model: accept 'any' only
            return g if task.resources.accelerator == "any" else None
        if g != "none" and task.resources.accepts(g):
            return g
        if self.config.get("allow_cpu_fallback") and task.resources.accelerator == "any":
            return "none"  # user accepts running GPU-optional tasks on CPU
        return None

    def launch(self, task, gpu, workdir):
        src, _ = make_driver(task, self.name, base=workdir)
        driver = os.path.join(workdir, "fgpu_driver.py")
        with open(driver, "w") as f:
            f.write(src)
        try:
            p = subprocess.run([sys.executable, driver], cwd=workdir, text=True,
                               capture_output=True, timeout=task.resources.max_runtime + 60)
        except subprocess.TimeoutExpired:
            raise ProviderError("task", f"superato max_runtime ({task.resources.max_runtime}s)")
        if not os.path.exists(os.path.join(workdir, RESULT_TGZ)):
            raise ProviderError("transient", (p.stderr or p.stdout)[-500:])
        return RunResult(self.name, p.returncode, 0, gpu, log_tail=p.stdout[-2000:])
