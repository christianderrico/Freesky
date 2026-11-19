"""Your own machine: always free. Last resort for GPU jobs, first choice if it has a fitting GPU."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Replica, RunResult


def detect_gpu() -> str:
    try:
        names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "none"
    if not names:
        return "none"
    n = names[0].upper()
    for g in ("H100", "A100", "A10G", "L4", "T4", "P100"):
        if g in n:
            return g
    return "local-gpu"


class LocalProvider(Provider):
    name = "local"
    title = "This computer"
    startup_s = 0
    can_serve = True
    signup = "-"
    install = "-"

    def __init__(self, config=None):
        super().__init__(config)
        self.gpus = [detect_gpu()]

    def pick_gpu(self, task):
        g = self.gpus[0]
        if not task.resources.needs_gpu:
            return "none"
        if g == "local-gpu":  # unknown model: only for accelerator: any
            return g if task.resources.accelerator == "any" else None
        if g != "none" and task.resources.accepts(g):
            return g
        if (self.config.get("allow_cpu_fallback") and task.resources.accelerator == "any"
                and not task.resources.min_gpu_memory_gb):  # a VRAM requirement means "really needs a GPU"
            return "none"
        return None

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        src, _ = make_driver(task, self.name, base=workdir, deadline_s=deadline_s, segment=segment,
                             grace_s=int(self.config.get("grace_s", 120)))
        driver = os.path.join(workdir, "fgpu_driver.py")
        with open(driver, "w") as f:
            f.write(src)
        limit = (deadline_s or task.resources.max_runtime) + 300
        try:
            p = subprocess.run([sys.executable, driver], cwd=workdir, text=True,
                               capture_output=True, timeout=limit)
        except subprocess.TimeoutExpired:
            raise ProviderError("task", f"exceeded max_runtime ({task.resources.max_runtime}s)")
        if not os.path.exists(os.path.join(workdir, RESULT_TGZ)):
            raise ProviderError("transient", (p.stderr or p.stdout)[-500:])
        return RunResult(self.name, p.returncode, 0, gpu, log_tail=p.stdout[-2000:])

    def serve(self, task, gpu, workdir):
        src, _ = make_driver(task, self.name, base=workdir, mode="serve")
        driver = os.path.join(workdir, "fgpu_driver.py")
        with open(driver, "w") as f:
            f.write(src)
        log = open(os.path.join(workdir, "service.log"), "w")
        proc = subprocess.Popen([sys.executable, driver], cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)

        def stop():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close()

        return Replica(self.name, gpu, f"http://127.0.0.1:{task.service.port}", time.time(), stop)
