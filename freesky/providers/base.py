"""Common interface for every free GPU source."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..spec import Task


class ProviderError(Exception):
    """kind: capacity | quota | auth | transient | task (see ledger.COOLDOWN)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Quota:
    unit: str              # "hours" | "usd" | "credits"
    limit: float           # free amount per window
    window: str            # "day" | "week" | "month" | "total"
    rate: Dict[str, float] = field(default_factory=dict)  # cost per hour of each GPU, in `unit`
    note: str = ""


@dataclass
class RunResult:
    provider: str
    exit_code: int
    seconds: float
    gpu: str = ""
    log_tail: str = ""


@dataclass
class Replica:
    """A running service instance. `url` is the base URL the local proxy forwards to."""
    provider: str
    gpu: str
    url: str
    started: float
    stop: Callable[[], None]
    headers: Dict[str, str] = field(default_factory=dict)


class Provider:
    name = "base"
    title = ""
    gpus: List[str] = []
    startup_s = 60              # typical time before the task starts (cold start, queue)
    session_limit_s = 0         # longest single session the free tier allows (0 = no limit)
    can_serve = False           # can expose an HTTP port to the outside, within the free tier's rules
    quota: Optional[Quota] = None
    signup = ""
    install = ""
    terms = ""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        if self.quota and "quota_limit" in self.config:
            self.quota.limit = float(self.config["quota_limit"])
        if self.quota and "rates" in self.config:
            self.quota.rate.update({k: float(v) for k, v in self.config["rates"].items()})
        if "session_limit" in self.config:
            from ..spec import parse_duration
            self.session_limit_s = parse_duration(self.config["session_limit"])

    # --- capability -------------------------------------------------------
    def pick_gpu(self, task: Task) -> Optional[str]:
        if not task.resources.needs_gpu:
            return "none"
        for g in self.config.get("gpus") or self.gpus:
            if task.resources.accepts(g):
                return g
        return None

    def available(self) -> tuple[bool, str]:
        """Credentials/tools present? Must be cheap (no VM allocation)."""
        return True, ""

    def remaining(self, used: float) -> Optional[float]:
        if not self.quota:
            return None
        return max(0.0, self.quota.limit - used)

    def hourly(self, gpu: str) -> float:
        if not self.quota:
            return 0.0
        return self.quota.rate.get(gpu, self.quota.rate.get("*", 1.0))

    def estimate_cost(self, task: Task, gpu: str, seconds: Optional[float] = None) -> float:
        run = task.resources.estimated_runtime if seconds is None else seconds
        return (run + self.startup_s) / 3600 * self.hourly(gpu)

    def actual_cost(self, gpu: str, seconds: float) -> float:
        return seconds / 3600 * self.hourly(gpu)

    # --- execution --------------------------------------------------------
    def launch(self, task: Task, gpu: str, workdir: str, deadline_s: int = 0, segment: int = 1) -> RunResult:
        """Run one segment; leave fgpu_out.tgz in `workdir`. Raise ProviderError on failure."""
        raise NotImplementedError

    def serve(self, task: Task, gpu: str, workdir: str) -> Replica:
        """Start the service and return as soon as the process is launched (health is checked by the caller)."""
        raise ProviderError("capacity", f"{self.name} cannot serve endpoints")


# ---------------------------------------------------------------- helpers
def which(binary: str) -> bool:
    return shutil.which(binary) is not None


def run_cmd(args: List[str], timeout: Optional[int] = None, check: bool = True,
            env: Optional[dict] = None, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, **(env or {})), cwd=cwd,
                           stdin=subprocess.DEVNULL)  # never wait for interactive input
    except FileNotFoundError:
        raise ProviderError("auth", f"command not found: {args[0]}")
    except subprocess.TimeoutExpired:
        raise ProviderError("transient", f"timeout: {' '.join(args[:3])}")
    if check and p.returncode != 0:
        raise ProviderError(classify(p.stderr + p.stdout), (p.stderr or p.stdout).strip()[-500:])
    return p


def classify(text: str) -> str:
    """Map provider error text to a cooldown kind."""
    t = text.lower()
    if any(k in t for k in ("quota", "exceeded", "limit reached", "usage limit", "out of credits",
                            "insufficient credit", "billing", "spend limit")):
        return "quota"
    if any(k in t for k in ("unauthor", "401", "403", "forbidden", "credential", "token", "login",
                            "not logged", "permission denied", "api key")):
        return "auth"
    if any(k in t for k in ("no gpu", "unavailable", "capacity", "not available", "no backend",
                            "resource exhausted", "try again later", "429", "too many requests",
                            "400")):
        return "capacity"
    return "transient"


def scratch_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"freesky-{prefix}-")


def write_driver_files(workdir: str, src: str, payload: bytes) -> tuple[str, str]:
    drv, pay = os.path.join(workdir, "fgpu_driver.py"), os.path.join(workdir, "fgpu_payload.tgz")
    with open(drv, "w") as f:
        f.write(src)
    with open(pay, "wb") as f:
        f.write(payload)
    return drv, pay
