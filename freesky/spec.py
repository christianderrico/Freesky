"""Task specification, in the spirit of a SkyPilot task YAML.

    name: kev-triage
    resources:
      accelerator: T4          # T4 | L4 | A10G | A100 | H100 | any | none
      min_gpu_memory_gb: 15    # optional
      max_runtime: 30m         # hard limit, also used to estimate quota use
      estimated_runtime: 5m    # optional, defaults to max_runtime
    image: python:3.12-slim    # used by container-based providers (modal, local --docker)
    workdir: .                 # uploaded to every provider
    setup: pip install -r requirements.txt
    run: python score.py --out out/
    outputs: [out]             # paths (relative to workdir) copied back
    env: {HF_HOME: /tmp/hf}
    providers: [modal, kaggle, colab, lightning, local]   # optional allow-list / order
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

ACCELERATORS = ["none", "T4", "P100", "L4", "A10G", "A100", "H100", "any"]
GPU_MEMORY_GB = {"T4": 16, "P100": 16, "L4": 24, "A10G": 24, "A100": 40, "H100": 80}


def parse_duration(value, default: int = 3600) -> int:
    """'90', '90s', '15m', '2h', '1h30m' -> seconds."""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    if s.isdigit():
        return int(s)
    total, matched = 0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        matched = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    if not matched:
        raise ValueError(f"durata non valida: {value!r} (usa es. 90s, 15m, 2h)")
    return int(total)


@dataclass
class Resources:
    accelerator: str = "any"
    min_gpu_memory_gb: float = 0
    max_runtime: int = 3600
    estimated_runtime: int = 0

    def __post_init__(self):
        acc = str(self.accelerator or "any")
        canon = {a.lower(): a for a in ACCELERATORS}
        if acc.lower() not in canon:
            raise ValueError(f"acceleratore sconosciuto {acc!r}; validi: {', '.join(ACCELERATORS)}")
        self.accelerator = canon[acc.lower()]
        if not self.estimated_runtime:
            self.estimated_runtime = self.max_runtime

    @property
    def needs_gpu(self) -> bool:
        return self.accelerator != "none"

    def accepts(self, gpu: str) -> bool:
        """Does a provider GPU of type `gpu` ('none' for CPU) satisfy this request?"""
        if self.accelerator == "none":
            return True  # a CPU job runs anywhere
        if gpu == "none":
            return False
        if self.accelerator != "any" and gpu != self.accelerator:
            return False
        return GPU_MEMORY_GB.get(gpu, 0) >= self.min_gpu_memory_gb


@dataclass
class Task:
    name: str
    run: str
    setup: str = ""
    workdir: str = "."
    outputs: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    image: str = "python:3.12-slim"
    providers: Optional[List[str]] = None
    resources: Resources = field(default_factory=Resources)
    base_dir: str = "."  # directory of the YAML file, workdir is relative to it

    @property
    def workdir_path(self) -> str:
        return os.path.normpath(os.path.join(self.base_dir, self.workdir))

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9-]+", "-", self.name.lower()).strip("-")[:40] or "task"


def load_task(path: str) -> Task:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return task_from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))


def task_from_dict(data: dict, base_dir: str = ".") -> Task:
    if not data.get("run"):
        raise ValueError("il task deve avere un comando 'run'")
    res = data.get("resources") or {}
    resources = Resources(
        accelerator=res.get("accelerator", "any"),
        min_gpu_memory_gb=float(res.get("min_gpu_memory_gb", 0) or 0),
        max_runtime=parse_duration(res.get("max_runtime"), 3600),
        estimated_runtime=parse_duration(res.get("estimated_runtime"), 0) if res.get("estimated_runtime") else 0,
    )
    outputs = data.get("outputs") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    for o in outputs:
        if os.path.isabs(o) or ".." in o.split("/"):
            raise ValueError(f"output {o!r}: usa percorsi relativi alla workdir")
    providers = data.get("providers")
    if isinstance(providers, str):
        providers = [providers]
    return Task(
        name=str(data.get("name") or "task"),
        run=str(data["run"]),
        setup=str(data.get("setup") or ""),
        workdir=str(data.get("workdir") or "."),
        outputs=list(outputs),
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        image=str(data.get("image") or "python:3.12-slim"),
        providers=list(providers) if providers else None,
        resources=resources,
        base_dir=base_dir,
    )
