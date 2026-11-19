"""Picks a free GPU source for a task and fails over to the next one, like OmniRoute does for LLMs."""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import yaml

from .bundle import RESULT_TGZ, unpack_result
from .ledger import HOME, Attempt, Ledger
from .providers.base import Provider, ProviderError, scratch_dir
from .spec import Task

POLICIES = ("fastest", "save-scarce", "order")


def provider_types() -> Dict[str, type]:
    from .providers.colab import ColabProvider
    from .providers.kaggle_ import KaggleProvider
    from .providers.lightning_ import LightningProvider
    from .providers.local import LocalProvider
    from .providers.modal_ import ModalProvider
    from .providers.skypilot_ import SkyPilotProvider
    return {"local": LocalProvider, "modal": ModalProvider, "kaggle": KaggleProvider,
            "colab": ColabProvider, "lightning": LightningProvider, "skypilot": SkyPilotProvider}


DEFAULT_CONFIG = {
    "policy": "fastest",
    "providers": {"modal": {}, "kaggle": {}, "colab": {}, "lightning": {}, "local": {}},
}


def load_config(path: Optional[str] = None) -> dict:
    path = path or os.environ.get("FREESKY_CONFIG") or os.path.join(HOME, "config.yaml")
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
            raise ValueError(f"fornitore {name!r}: tipo sconosciuto {kind!r} (validi: {', '.join(types)})")
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
    skip: str = ""          # why it can't be used now; empty = usable

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
    message: str = ""


class Router:
    def __init__(self, providers: List[Provider], ledger: Optional[Ledger] = None,
                 policy: str = "fastest", order: Optional[List[str]] = None,
                 log: Callable[[str], None] = print):
        if policy not in POLICIES:
            raise ValueError(f"politica {policy!r} non valida: {', '.join(POLICIES)}")
        self.providers = providers
        self.ledger = ledger or Ledger()
        self.policy = policy
        self.order = order or [p.name for p in providers]
        self.log = log

    # ------------------------------------------------------------ planning
    def plan(self, task: Task, only: Optional[List[str]] = None) -> List[Candidate]:
        allowed = only or task.providers
        cands = []
        for p in self.providers:
            if allowed and p.name not in allowed:
                continue
            c = Candidate(p, p.pick_gpu(task))
            if c.gpu is None:
                c.skip = f"nessuna GPU adatta ({task.resources.accelerator}"
                c.skip += f", >= {task.resources.min_gpu_memory_gb:g} GB)" if task.resources.min_gpu_memory_gb else ")"
                cands.append(c)
                continue
            ok, why = p.available()
            if not ok:
                c.skip = why
                cands.append(c)
                continue
            left, reason = self.ledger.cooldown(p.name)
            if left > 0:
                c.skip = f"in pausa per {int(left // 60)} min ({reason})"
                cands.append(c)
                continue
            if p.quota:
                c.estimate = p.estimate_cost(task, c.gpu)
                c.remaining = p.remaining(self.ledger.used(p.name, p.quota.window))
                if c.remaining is not None and c.remaining < c.estimate:
                    c.skip = (f"quota insufficiente: restano {c.remaining:.2f} {p.quota.unit}, "
                              f"servono ~{c.estimate:.2f}")
            cands.append(c)
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
        elif self.policy == "save-scarce":  # spend the most abundant quota first
            key = lambda c: (not c.usable, cpu_fallback(c), -frac(c), c.provider.startup_s)
        else:  # fastest start
            key = lambda c: (not c.usable, cpu_fallback(c), c.provider.startup_s, -frac(c))
        return sorted(cands, key=key)

    # ------------------------------------------------------------ execution
    def run(self, task: Task, out_dir: str = "freesky-runs", only: Optional[List[str]] = None,
            dry_run: bool = False) -> Outcome:
        cands = self.plan(task, only)
        usable = [c for c in cands if c.usable]
        for c in cands:
            mark = "->" if c.usable else "  "
            info = c.skip or (f"{c.gpu}, stima {c.estimate:.2f} {c.provider.quota.unit}, "
                              f"restano {c.remaining:.2f}" if c.provider.quota and c.remaining is not None
                              else f"{c.gpu}")
            self.log(f"  {mark} {c.provider.name:<12} {info}")
        if dry_run or not usable:
            return Outcome(False, message="nessun fornitore disponibile" if not usable else "dry-run")

        outcome = Outcome(False)
        for c in usable:
            p = c.provider
            outcome.tried.append(p.name)
            self.log(f"> provo {p.name} ({c.gpu})")
            scratch = scratch_dir(p.name)
            t0 = time.time()
            try:
                res = p.launch(task, c.gpu, scratch)
                dest = os.path.join(out_dir, f"{task.slug}-{time.strftime('%Y%m%d-%H%M%S')}-{p.name}")
                tgz = os.path.join(scratch, RESULT_TGZ)
                info = unpack_result(tgz, dest) if os.path.exists(tgz) else {}
                code = int(info.get("exit_code", res.exit_code))
                seconds = float(info.get("seconds") or res.seconds or (time.time() - t0))
                billed = time.time() - t0  # wall time incl. startup is what the free tier pays
                self.ledger.record(Attempt(time.time(), p.name, task.name, billed,
                                           p.actual_cost(c.gpu, billed), ok=code == 0,
                                           error_kind="" if code == 0 else "task",
                                           error="" if code == 0 else f"exit {code}"))
                outcome = Outcome(code == 0, p.name, info.get("gpu") or c.gpu, code, seconds, dest,
                                  outcome.tried, "ok" if code == 0 else f"il task e' terminato con codice {code}")
                # a failing user command is not the provider's fault: don't burn other quotas on it
                return outcome
            except ProviderError as e:
                billed = time.time() - t0
                self.ledger.record(Attempt(time.time(), p.name, task.name, billed,
                                           p.actual_cost(c.gpu, billed) if e.kind not in ("auth", "capacity") else 0.0,
                                           ok=False, error_kind=e.kind, error=str(e)))
                self.log(f"  x {p.name}: {e.kind} - {str(e)[:200]}")
                outcome.message = f"{p.name}: {e}"
                if e.kind == "task":  # the task itself is at fault (e.g. too slow): stop here
                    outcome.provider = p.name
                    return outcome
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        outcome.message = "tutti i fornitori hanno fallito; ultimo errore: " + outcome.message
        return outcome
