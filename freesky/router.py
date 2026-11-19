"""Picks a free GPU source for a task and fails over to the next one, like OmniRoute does for LLMs.

Long jobs (`checkpoint:` set) run as a chain of segments. Each segment gets a deadline equal
to the chosen provider's session limit; when it is hit, the job is asked to stop (SIGTERM),
its checkpoint directory comes back, and the next segment resumes from it on whichever
provider is available next - Kaggle's 12 h cap or a Colab preemption no longer ends the run.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import yaml

from .bundle import EXCLUDE_DIRS, RESULT_TGZ, unpack_result
from .ledger import Attempt, Ledger, home
from .providers.base import Provider, ProviderError, scratch_dir
from .spec import Task

POLICIES = ("fastest", "save-scarce", "order")
SEGMENT_MARGIN_S = 600   # leave room for startup and packing inside a provider session


def provider_types() -> Dict[str, type]:
    from .providers.beam_ import BeamProvider
    from .providers.colab import ColabProvider
    from .providers.kaggle_ import KaggleProvider
    from .providers.lightning_ import LightningProvider
    from .providers.local import LocalProvider
    from .providers.modal_ import ModalProvider
    from .providers.skypilot_ import SkyPilotProvider
    return {"local": LocalProvider, "modal": ModalProvider, "kaggle": KaggleProvider,
            "colab": ColabProvider, "lightning": LightningProvider, "skypilot": SkyPilotProvider,
            "beam": BeamProvider}


DEFAULT_CONFIG = {
    "policy": "fastest",
    "providers": {"modal": {}, "kaggle": {}, "colab": {}, "lightning": {}, "local": {}},
}


def load_config(path: Optional[str] = None) -> dict:
    path = path or os.environ.get("FREESKY_CONFIG") or os.path.join(home(), "config.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return dict(DEFAULT_CONFIG)


def build_providers(config: dict, types: Optional[Dict[str, type]] = None) -> List[Provider]:
    types = types or provider_types()
    out = []
    for name, cfg in (config.get("providers") or {}).items():
        cfg = dict(cfg or {})
        if cfg.get("enabled", True) is False:
            continue
        kind = cfg.get("type", name)
        if kind not in types:
            raise ValueError(f"provider {name!r}: unknown type {kind!r} (valid: {', '.join(types)})")
        cfg["_instance"] = name
        p = types[kind](cfg)
        p.name = name
        out.append(p)
    return out


@dataclass
class Candidate:
    provider: Provider
    gpu: Optional[str]
    estimate: float = 0.0
    remaining: Optional[float] = None
    skip: str = ""

    @property
    def usable(self) -> bool:
        return not self.skip


@dataclass
class Outcome:
    ok: bool
    provider: str = ""
    gpu: str = ""
    exit_code: int = -1
    seconds: float = 0.0
    outputs_dir: str = ""
    tried: List[str] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)
    message: str = ""


class Router:
    def __init__(self, providers: List[Provider], ledger: Optional[Ledger] = None,
                 policy: str = "fastest", order: Optional[List[str]] = None,
                 log: Callable[[str], None] = print):
        if policy not in POLICIES:
            raise ValueError(f"invalid policy {policy!r}: {', '.join(POLICIES)}")
        self.providers = providers
        self.ledger = ledger or Ledger()
        self.policy = policy
        self.order = order or [p.name for p in providers]
        self.log = log
        self.segment_margin_s = SEGMENT_MARGIN_S
        self.min_segment_s = 60

    # ------------------------------------------------------------ planning
    def segment_seconds(self, p: Provider, task: Task, budget: float) -> float:
        """How long one segment on p can run: capped by the session limit for resumable tasks."""
        if task.checkpoint and p.session_limit_s:
            return max(self.min_segment_s, min(budget, p.session_limit_s - self.segment_margin_s))
        return budget

    def plan(self, task: Task, only: Optional[List[str]] = None, serving: bool = False,
             budget: Optional[float] = None) -> List[Candidate]:
        allowed = only or task.providers
        budget = task.resources.max_runtime if budget is None else budget
        cands = []
        for p in self.providers:
            if allowed and p.name not in allowed:
                continue
            c = Candidate(p, p.pick_gpu(task))
            cands.append(c)
            if serving and not p.can_serve:
                c.skip = "batch only (cannot expose an endpoint)"
                continue
            if c.gpu is None:
                need = task.resources.accelerator
                if task.resources.min_gpu_memory_gb:
                    need += f", >= {task.resources.min_gpu_memory_gb:g} GB"
                c.skip = f"no suitable GPU ({need})"
                continue
            if (not serving and not task.checkpoint and p.session_limit_s
                    and task.resources.max_runtime > p.session_limit_s):
                c.skip = (f"session limit {p.session_limit_s / 3600:.1f} h < max_runtime "
                          f"(add `checkpoint:` to resume across sessions)")
                continue
            ok, why = p.available()
            if not ok:
                c.skip = why
                continue
            left, reason = self.ledger.cooldown(p.name)
            if left > 0:
                c.skip = f"cooling down for {int(left // 60)} min ({reason})"
                continue
            if p.quota:
                est_run = min(task.resources.estimated_runtime, self.segment_seconds(p, task, budget))
                c.estimate = p.estimate_cost(task, c.gpu, est_run)
                c.remaining = p.remaining(self.ledger.used(p.name, p.quota.window))
                if c.remaining is not None and c.remaining < c.estimate:
                    c.skip = (f"not enough quota: {c.remaining:.2f} {p.quota.unit} left, "
                              f"~{c.estimate:.2f} needed")
        return self._rank(cands, task, ordered=bool(allowed))

    def _rank(self, cands, task, ordered):
        def cpu_fallback(c):  # running a GPU task on CPU is always the very last resort
            return 1 if task.resources.needs_gpu and c.gpu == "none" else 0

        def frac(c):
            p = c.provider
            if not p.quota or c.remaining is None or not p.quota.limit:
                return 1.0
            return c.remaining / p.quota.limit

        names = task.providers if ordered and task.providers else self.order
        pos = {n: i for i, n in enumerate(names)}
        if ordered or self.policy == "order":
            key = lambda c: (not c.usable, cpu_fallback(c), pos.get(c.provider.name, 99))
        elif self.policy == "save-scarce":
            key = lambda c: (not c.usable, cpu_fallback(c), -frac(c), c.provider.startup_s)
        else:
            key = lambda c: (not c.usable, cpu_fallback(c), c.provider.startup_s, -frac(c))
        return sorted(cands, key=key)

    def describe(self, cands: List[Candidate]) -> None:
        for c in cands:
            p = c.provider
            if c.skip:
                info = c.skip
            elif p.quota and c.remaining is not None:
                info = f"{c.gpu}, ~{c.estimate:.2f} {p.quota.unit} of {c.remaining:.2f} left"
            else:
                info = c.gpu
            self.log(f"  {'->' if c.usable else '  '} {p.name:<12} {info}")

    # ------------------------------------------------------------ execution
    def run(self, task: Task, out_dir: str = "freesky-runs", only: Optional[List[str]] = None,
            dry_run: bool = False) -> Outcome:
        first = self.plan(task, only)
        self.describe(first)
        if dry_run or not any(c.usable for c in first):
            return Outcome(False, message="dry run" if dry_run else "no provider available")

        staging = None
        if task.checkpoint:  # work on a private copy: checkpoints are merged into it between segments
            staging = scratch_dir("staging")
            shutil.copytree(task.workdir_path, staging, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
            task = dataclasses.replace(task, base_dir=staging, workdir=".")
        try:
            return self._run_segments(task, out_dir, only, first, staging)
        finally:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)

    def _run_segments(self, task, out_dir, only, cands, staging) -> Outcome:
        outcome = Outcome(False)
        used_s = 0.0
        stamp = time.strftime("%Y%m%d-%H%M%S")
        segments = task.max_segments if task.checkpoint else 1
        for seg in range(1, segments + 1):
            budget = task.resources.max_runtime - used_s
            if budget < self.min_segment_s:
                outcome.message = f"max_runtime reached after {seg - 1} segments"
                return outcome
            if seg > 1:
                cands = self.plan(task, only, budget=budget)
                self.log(f"> segment {seg}: resuming from the checkpoint")
                self.describe(cands)
            done = self._try_segment(task, out_dir, stamp, seg, budget, cands, outcome, staging)
            if done is None:
                return outcome            # every provider failed, or the task itself failed
            used_s += done
            if outcome.ok:
                return outcome
        outcome.message = f"stopped after max_segments={segments}"
        return outcome

    def _try_segment(self, task, out_dir, stamp, seg, budget, cands, outcome, staging) -> Optional[float]:
        """Returns run seconds if the segment produced a result (finished or deadline), else None."""
        for c in [c for c in cands if c.usable]:
            p = c.provider
            outcome.tried.append(p.name)
            deadline = int(self.segment_seconds(p, task, budget)) if task.checkpoint else 0
            self.log(f"> trying {p.name} ({c.gpu})" + (f", segment {seg}, deadline {deadline}s" if deadline else ""))
            scratch = scratch_dir(p.name)
            t0 = time.time()
            try:
                res = p.launch(task, c.gpu, scratch, deadline_s=deadline, segment=seg)
                suffix = f"-seg{seg}" if task.checkpoint else ""
                dest = os.path.join(out_dir, f"{task.slug}-{stamp}{suffix}-{p.name}")
                tgz = os.path.join(scratch, RESULT_TGZ)
                info = unpack_result(tgz, dest) if os.path.exists(tgz) else {}
                code = int(info.get("exit_code", res.exit_code))
                hit = bool(info.get("deadline_hit"))
                seconds = float(info.get("seconds") or res.seconds or (time.time() - t0))
                billed = time.time() - t0
                finished = code == 0 and not hit
                self.ledger.record(Attempt(time.time(), p.name, task.name, billed, p.actual_cost(c.gpu, billed),
                                           ok=finished or hit, error_kind="" if finished or hit else "task",
                                           error="" if finished or hit else f"exit {code}"))
                outcome.segments.append({"segment": seg, "provider": p.name, "gpu": info.get("gpu") or c.gpu,
                                         "seconds": seconds, "exit_code": code, "deadline_hit": hit,
                                         "outputs": dest})
                outcome.provider, outcome.gpu, outcome.exit_code = p.name, info.get("gpu") or c.gpu, code
                outcome.seconds += seconds
                outcome.outputs_dir = dest
                if task.checkpoint and staging:
                    src = os.path.join(dest, "outputs", task.checkpoint)
                    if os.path.isdir(src):
                        target = os.path.join(staging, task.checkpoint)
                        shutil.rmtree(target, ignore_errors=True)
                        shutil.copytree(src, target)
                if finished:
                    outcome.ok, outcome.message = True, "ok"
                    return seconds
                if hit:
                    self.log(f"  segment {seg} reached its deadline on {p.name}; checkpoint saved")
                    return seconds
                # the user's command failed: not the provider's fault, don't burn other quotas on it
                outcome.message = f"the task exited with code {code}"
                return None
            except ProviderError as e:
                billed = time.time() - t0
                charge = p.actual_cost(c.gpu, billed) if e.kind not in ("auth", "capacity") else 0.0
                self.ledger.record(Attempt(time.time(), p.name, task.name, billed, charge,
                                           ok=False, error_kind=e.kind, error=str(e)))
                self.log(f"  x {p.name}: {e.kind} - {str(e)[:200]}")
                outcome.message = f"{p.name}: {e}"
                if e.kind == "task":
                    outcome.provider = p.name
                    return None
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        outcome.message = "every provider failed; last error: " + outcome.message
        return None
