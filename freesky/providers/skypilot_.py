"""Cloud trial credits through SkyPilot (GCP $300, Azure $200, AWS, OCI ... any cloud SkyPilot knows).

The credits are free money, but a VM keeps billing until it is torn down: this adapter
always runs `sky down` at the end. One instance per credit account, e.g.

  providers:
    gcp-trial:
      type: skypilot
      quota_limit: 300          # USD of trial credit
      quota_window: total
      rates: {T4: 0.40, L4: 0.75, A100: 3.0}
      resources: {infra: gcp}   # passed through to the SkyPilot task's `resources`

Setup:  pip install 'skypilot[gcp]' (or your cloud) && sky check
Experimental: written against SkyPilot's task YAML; check `sky check` before relying on it.
"""
from __future__ import annotations

import os
import time
import uuid

import yaml

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, classify, run_cmd, which


class SkyPilotProvider(Provider):
    name = "skypilot"
    title = "Crediti di prova cloud via SkyPilot (sperimentale)"
    gpus = ["T4", "L4", "A10G", "A100", "H100"]
    startup_s = 300
    signup = "GCP $300/90gg, Azure $200/30gg, AWS Free Tier, OCI $300/30gg"
    install = "pip install 'skypilot[gcp]'  &&  sky check"
    terms = "Spendi credito reale: il router chiude sempre la VM a fine lavoro."

    def __init__(self, config=None):
        config = config or {}
        self.quota = Quota("usd", float(config.get("quota_limit", 0)), config.get("quota_window", "total"),
                           dict(config.get("rates") or {"T4": 0.40, "L4": 0.75, "A10G": 1.0,
                                                         "A100": 3.0, "H100": 6.0, "none": 0.05}),
                           "credito di prova, una tantum")
        super().__init__(config)
        self.name = config.get("_instance", "skypilot")

    def available(self):
        if not which("sky"):
            return False, "SkyPilot non installato"
        if not self.quota.limit:
            return False, "imposta quota_limit (credito disponibile) nella config"
        return True, ""

    def launch(self, task, gpu, workdir):
        cluster = f"fgpu-{task.slug[:16]}-{uuid.uuid4().hex[:5]}"
        src, payload = make_driver(task, self.name, embed=False)
        drv, pay = os.path.join(workdir, "fgpu_driver.py"), os.path.join(workdir, "fgpu_payload.tgz")
        with open(drv, "w") as f:
            f.write(src)
        with open(pay, "wb") as f:
            f.write(payload)
        resources = dict(self.config.get("resources") or {})
        if gpu != "none":
            resources["accelerators"] = f"{gpu}:1"
        spec = {
            "name": cluster,
            "resources": resources,
            "file_mounts": {"~/fgpu/fgpu_driver.py": drv, "~/fgpu/fgpu_payload.tgz": pay},
            "run": "cd ~/fgpu && FGPU_BASE=$HOME/fgpu python3 fgpu_driver.py",
        }
        ypath = os.path.join(workdir, "sky-task.yaml")
        with open(ypath, "w") as f:
            yaml.safe_dump(spec, f)
        t0 = time.time()
        try:
            p = run_cmd(["sky", "launch", "-y", "-c", cluster, ypath],
                        timeout=task.resources.max_runtime + 1800, check=False)
            if p.returncode != 0 and "[fgpu] RESULT" not in p.stdout:
                # failed before our driver ran: provisioning problem (no capacity, credit, auth)
                raise ProviderError(classify(p.stderr + p.stdout), (p.stderr or p.stdout).strip()[-500:])
            run_cmd(["rsync", "-az", f"{cluster}:~/fgpu/{RESULT_TGZ}", os.path.join(workdir, RESULT_TGZ)],
                    timeout=900)
            return RunResult(self.name, p.returncode, time.time() - t0, gpu, log_tail=p.stdout[-2000:])
        finally:
            run_cmd(["sky", "down", "-y", cluster], timeout=900, check=False)
