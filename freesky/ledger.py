"""Local bookkeeping: how much free quota each provider has left, and who is cooling down.

Like OmniRoute's per-key cooldowns, but for GPU quotas:
  ~/.freesky/ledger.jsonl  one line per attempt (provider, seconds, cost, ok, error)
  ~/.freesky/state.json    cooldowns and consecutive failures per provider
Quota windows are computed from the ledger (rolling week / calendar month / day), unless a
provider can report its real remaining quota (Kaggle does), which then wins.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

HOME = os.environ.get("FREESKY_HOME") or os.path.join(os.path.expanduser("~"), ".freesky")

# error kinds -> base cooldown seconds (doubled on each consecutive failure, capped)
COOLDOWN = {
    "capacity": 15 * 60,    # no GPU available right now: try again later
    "quota": 6 * 3600,      # quota exhausted for this window
    "auth": 24 * 3600,      # credentials missing/expired: needs `freesky check`
    "transient": 2 * 60,    # network blip, 5xx
    "task": 0,              # the user's code failed: not the provider's fault
}
MAX_COOLDOWN = 24 * 3600


def _now() -> float:
    return time.time()


def window_start(window: str, now: Optional[float] = None) -> float:
    now = now or _now()
    if window == "day":
        return now - 86400
    if window == "week":
        return now - 7 * 86400
    if window == "month":
        d = datetime.fromtimestamp(now, tz=timezone.utc)
        return datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp()
    return 0.0  # "total": one-off credits


@dataclass
class Attempt:
    ts: float
    provider: str
    task: str
    seconds: float
    cost: float          # in the provider's quota unit (hours, usd, credits)
    ok: bool
    error_kind: str = ""
    error: str = ""


class Ledger:
    def __init__(self, home: str = HOME):
        self.home = home
        os.makedirs(home, exist_ok=True)
        self.ledger_path = os.path.join(home, "ledger.jsonl")
        self.state_path = os.path.join(home, "state.json")

    # ---------------------------------------------------------------- attempts
    def record(self, a: Attempt) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(a.__dict__) + "\n")
        st = self._state()
        p = st.setdefault(a.provider, {"failures": 0, "cooldown_until": 0, "reason": ""})
        if a.ok or a.error_kind == "task":
            p.update(failures=0, cooldown_until=0, reason="")
        else:
            p["failures"] = p.get("failures", 0) + 1
            base = COOLDOWN.get(a.error_kind, COOLDOWN["transient"])
            wait = min(MAX_COOLDOWN, base * (2 ** (p["failures"] - 1)))
            p["cooldown_until"] = a.ts + wait
            p["reason"] = f"{a.error_kind}: {a.error}"[:200]
        self._save(st)

    def attempts(self, provider: Optional[str] = None, since: float = 0.0):
        if not os.path.exists(self.ledger_path):
            return []
        out = []
        with open(self.ledger_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (provider is None or d["provider"] == provider) and d["ts"] >= since:
                    out.append(Attempt(**d))
        return out

    def used(self, provider: str, window: str) -> float:
        return sum(a.cost for a in self.attempts(provider, window_start(window)))

    # ---------------------------------------------------------------- cooldowns
    def cooldown(self, provider: str) -> tuple[float, str]:
        """(seconds left, reason); 0 when the provider is usable."""
        p = self._state().get(provider) or {}
        left = p.get("cooldown_until", 0) - _now()
        return (left, p.get("reason", "")) if left > 0 else (0.0, "")

    def clear(self, provider: Optional[str] = None) -> None:
        st = self._state()
        for name in ([provider] if provider else list(st)):
            st.pop(name, None)
        self._save(st)

    def _state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, st: dict) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        os.replace(tmp, self.state_path)
