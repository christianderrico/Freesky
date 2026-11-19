"""Task and service specification, in the spirit of a SkyPilot task YAML.

Batch task:

    name: train
    resources:
      accelerator: T4          # none | T4 | P100 | L4 | A10G | A100 | H100 | any
      min_gpu_memory_gb: 15
      max_runtime: 30h         # total budget; long jobs are split into segments
      estimated_runtime: 5m    # used to estimate quota use (defaults to max_runtime)
    image: python:3.12-slim    # container-based providers (Modal)
    workdir: .
    setup: pip install -r requirements.txt
    run: python train.py
    outputs: [out]
    env: {HF_HOME: /tmp/hf}
    checkpoint: ckpt           # enables resume: this dir travels from segment to segment
    providers: [modal, kaggle, colab, lightning, local]

Service (add a `service:` block, run with `freesky serve`):

    service:
      port: 8009
      health: /health          # GET path that answers when the server is ready
      ready_timeout: 20m
      idle_timeout: 10m        # stop the replica after this long without requests
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
    total, matched = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        matched = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    if not matched:
        raise ValueError(f"invalid duration {value!r} (use e.g. 90s, 15m, 2h)")
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
            raise ValueError(f"unknown accelerator {acc!r}; valid: {', '.join(ACCELERATORS)}")
        self.accelerator = canon[acc.lower()]
        if not self.estimated_runtime:
            self.estimated_runtime = self.max_runtime

    @property
    def needs_gpu(self) -> bool:
        return self.accelerator != "none"

    def accepts(self, gpu: str) -> bool:
        """Does a provider GPU of type `gpu` ('none' = CPU) satisfy this request?"""
        if self.accelerator == "none":
            return True
        if gpu == "none":
            return False
        if self.accelerator != "any" and gpu != self.accelerator:
            return False
        return GPU_MEMORY_GB.get(gpu, 0) >= self.min_gpu_memory_gb


@dataclass
class ServiceSpec:
    port: int = 8000
    health: str = "/"
    ready_timeout: int = 1200
    idle_timeout: int = 600
    max_lifetime: int = 6 * 3600


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
    checkpoint: str = ""
    max_segments: int = 20
    service: Optional[ServiceSpec] = None
    base_dir: str = "."

    @property
    def workdir_path(self) -> str:
        return os.path.normpath(os.path.join(self.base_dir, self.workdir))

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9-]+", "-", self.name.lower()).strip("-")[:40] or "task"


def _relative(path: str, what: str) -> str:
    if os.path.isabs(path) or ".." in path.replace("\\", "/").split("/"):
        raise ValueError(f"{what} {path!r}: use a path relative to the workdir")
    return path.strip("/")


def load_task(path: str) -> Task:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return task_from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))


def task_from_dict(data: dict, base_dir: str = ".") -> Task:
    if not data.get("run"):
        raise ValueError("a task needs a 'run' command")
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
    providers = data.get("providers")
    if isinstance(providers, str):
        providers = [providers]
    service = None
    if data.get("service"):
        s = data["service"] or {}
        service = ServiceSpec(
            port=int(s.get("port", 8000)),
            health=str(s.get("health", "/")),
            ready_timeout=parse_duration(s.get("ready_timeout"), 1200),
            idle_timeout=parse_duration(s.get("idle_timeout"), 600),
            max_lifetime=parse_duration(s.get("max_lifetime"), 6 * 3600),
        )
    resume = data.get("resume") or {}
    return Task(
        name=str(data.get("name") or "task"),
        run=str(data["run"]),
        setup=str(data.get("setup") or ""),
        workdir=str(data.get("workdir") or "."),
        outputs=[_relative(o, "output") for o in outputs],
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        image=str(data.get("image") or "python:3.12-slim"),
        providers=list(providers) if providers else None,
        resources=resources,
        checkpoint=_relative(str(data["checkpoint"]), "checkpoint") if data.get("checkpoint") else "",
        max_segments=int(resume.get("max_segments", 20)),
        service=service,
        base_dir=base_dir,
    )
