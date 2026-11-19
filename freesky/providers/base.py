"""Common interface for every free GPU source."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    outputs_dir: str = ""
    log_tail: str = ""


class Provider:
    name = "base"
    title = ""
    gpus: List[str] = []            # GPU types the free tier can give
    startup_s = 60                  # typical time before the task starts (cold start, queue)
    quota: Optional[Quota] = None
    signup = ""                     # where to get an account
    install = ""                    # pip extra / CLI needed
    terms = ""                      # usage-rule reminder shown in `freesky providers`

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        if self.quota and "quota_limit" in self.config:
            self.quota.limit = float(self.config["quota_limit"])

    # --- capability -------------------------------------------------------
    def pick_gpu(self, task: Task) -> Optional[str]:
        """The cheapest GPU of this provider that satisfies the task, or None."""
        if not task.resources.needs_gpu:
            return "none"
        order = self.config.get("gpus") or self.gpus
        for g in order:
            if task.resources.accepts(g):
                return g
        return None

    def available(self) -> tuple[bool, str]:
        """Credentials/tools present? Must be cheap (no VM allocation)."""
        return True, ""

    def remaining(self, used: float) -> Optional[float]:
        """Remaining free quota in quota units; providers with a real API override this."""
        if not self.quota:
            return None
        return max(0.0, self.quota.limit - used)

    def estimate_cost(self, task: Task, gpu: str) -> float:
        if not self.quota:
            return 0.0
        hours = (task.resources.estimated_runtime + self.startup_s) / 3600
        return hours * self.quota.rate.get(gpu, self.quota.rate.get("*", 1.0))

    def actual_cost(self, gpu: str, seconds: float) -> float:
        if not self.quota:
            return 0.0
        return seconds / 3600 * self.quota.rate.get(gpu, self.quota.rate.get("*", 1.0))

    # --- execution --------------------------------------------------------
    def launch(self, task: Task, gpu: str, workdir: str) -> RunResult:
        """Run the task; leave fgpu_out.tgz in `workdir` and return the result. Raise ProviderError."""
        raise NotImplementedError


# ---------------------------------------------------------------- helpers
def which(binary: str) -> bool:
    return shutil.which(binary) is not None


def run_cmd(args: List[str], timeout: Optional[int] = None, check: bool = True,
            env: Optional[dict] = None, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, **(env or {})), cwd=cwd)
    except FileNotFoundError:
        raise ProviderError("auth", f"comando non trovato: {args[0]}")
    except subprocess.TimeoutExpired:
        raise ProviderError("transient", f"timeout: {' '.join(args[:3])}")
    if check and p.returncode != 0:
        raise ProviderError(classify(p.stderr + p.stdout), (p.stderr or p.stdout).strip()[-500:])
    return p


def classify(text: str) -> str:
    """Map provider error text to a cooldown kind."""
    t = text.lower()
    if any(k in t for k in ("quota", "exceeded", "limit reached", "usage limit", "out of credits",
                            "insufficient credit", "billing")):
        return "quota"
    if any(k in t for k in ("unauthor", "401", "403", "forbidden", "credential", "token", "login",
                            "not logged", "permission denied", "api key")):
        return "auth"
    if any(k in t for k in ("no gpu", "unavailable", "capacity", "not available", "no backend",
                            "resource exhausted", "try again later", "429", "too many requests")):
        return "capacity"
    return "transient"


def scratch_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"freesky-{prefix}-")
