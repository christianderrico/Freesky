import json
import os
import sys

import pytest

from freesky.bundle import RESULT_TGZ, make_driver
from freesky.ledger import Ledger
from freesky.providers.base import Provider, ProviderError, Quota, RunResult
from freesky.providers.local import LocalProvider
from freesky.router import Router, build_providers
from freesky.spec import parse_duration, task_from_dict

HELLO = os.path.join(os.path.dirname(__file__), "..", "examples", "hello-gpu")


@pytest.fixture
def ledger(tmp_path):
    return Ledger(str(tmp_path / "home"))


def make_task(tmp_path, run="python3 main.py", **res):
    wd = tmp_path / "wd"
    wd.mkdir(exist_ok=True)
    (wd / "main.py").write_text(
        "import os, json\nos.makedirs('out', exist_ok=True)\n"
        "json.dump({'p': os.environ.get('FGPU_PROVIDER'), 'x': os.environ.get('X')}, open('out/r.json','w'))\n"
        "print('hello')\n")
    return task_from_dict({"name": "t", "run": run, "outputs": ["out"], "env": {"X": "1"},
                           "resources": {"accelerator": res.get("accelerator", "T4"),
                                         "max_runtime": "5m", **{k: v for k, v in res.items() if k != "accelerator"}}},
                          base_dir=str(tmp_path / "wd"))


class Fake(Provider):
    """Fake GPU provider: runs the real driver locally, but can be told to fail."""
    startup_s = 10

    def __init__(self, name, fail=None, gpus=("T4",), startup=10, quota=None, ok=True):
        self.name, self.fail, self.gpus, self.startup_s = name, fail, list(gpus), startup
        self.quota, self._ok, self.calls = quota, ok, 0
        super().__init__({})

    def available(self):
        return (self._ok, "" if self._ok else "no token")

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        self.calls += 1
        if self.fail:
            raise ProviderError(self.fail, f"{self.name} simulated {self.fail}")
        src, _ = make_driver(task, self.name, base=workdir, deadline_s=deadline_s, segment=segment, grace_s=5)
        path = os.path.join(workdir, "d.py")
        open(path, "w").write(src)
        import subprocess
        p = subprocess.run([sys.executable, path], cwd=workdir, capture_output=True, text=True)
        return RunResult(self.name, p.returncode, 1.0, gpu)


def test_durations():
    assert parse_duration("90") == 90 and parse_duration("15m") == 900 and parse_duration("1h30m") == 5400


def test_failover_capacity_then_success(tmp_path, ledger):
    a, b = Fake("a", fail="capacity", startup=5), Fake("b", startup=50)
    r = Router([a, b], ledger, log=lambda m: None)
    out = r.run(make_task(tmp_path), out_dir=str(tmp_path / "runs"))
    assert out.ok and out.provider == "b" and out.tried == ["a", "b"]
    data = json.load(open(os.path.join(out.outputs_dir, "outputs", "out", "r.json")))
    assert data == {"p": "b", "x": "1"}
    # a is now cooling down and skipped by the planner
    plan = r.plan(make_task(tmp_path))
    assert plan[0].provider.name == "b" and "cooling down" in [c for c in plan if c.provider.name == "a"][0].skip


def test_quota_exhausted_is_skipped(tmp_path, ledger):
    q = Quota("hours", 1.0, "week", {"T4": 1.0})
    a, b = Fake("a", startup=5, quota=q), Fake("b", startup=50)
    from freesky.ledger import Attempt
    ledger.record(Attempt(__import__("time").time(), "a", "x", 3600, 0.99, True))
    plan = Router([a, b], ledger, log=lambda m: None).plan(make_task(tmp_path))
    assert plan[0].provider.name == "b"
    assert "not enough quota" in plan[1].skip


def test_no_suitable_gpu_and_unavailable(tmp_path, ledger):
    a = Fake("a", gpus=("T4",))
    b = Fake("b", ok=False)
    plan = Router([a, b], ledger, log=lambda m: None).plan(make_task(tmp_path, accelerator="A100"))
    assert all(not c.usable for c in plan)
    assert "no suitable GPU" in plan[0].skip


def test_task_failure_does_not_failover(tmp_path, ledger):
    a, b = Fake("a", startup=5), Fake("b", startup=50)
    out = Router([a, b], ledger, log=lambda m: None).run(make_task(tmp_path, run="exit 3"),
                                                        out_dir=str(tmp_path / "runs"))
    assert not out.ok and out.exit_code == 3 and out.tried == ["a"] and b.calls == 0
    assert ledger.cooldown("a")[0] == 0  # the user's bug doesn't penalise the provider


def test_policy_save_scarce_prefers_abundant_quota(tmp_path, ledger):
    from freesky.ledger import Attempt
    qa, qb = Quota("hours", 10, "week", {"T4": 1}), Quota("hours", 10, "week", {"T4": 1})
    a, b = Fake("a", startup=5, quota=qa), Fake("b", startup=50, quota=qb)
    ledger.record(Attempt(__import__("time").time(), "a", "x", 1, 8.0, True))
    t = make_task(tmp_path)
    assert Router([a, b], ledger, policy="fastest", log=lambda m: None).plan(t)[0].provider.name == "a"
    assert Router([a, b], ledger, policy="save-scarce", log=lambda m: None).plan(t)[0].provider.name == "b"


def test_cpu_fallback_is_last(tmp_path, ledger):
    local = LocalProvider({"allow_cpu_fallback": True})
    if local.gpus[0] != "none":
        pytest.skip("this machine has a GPU")
    fake = Fake("gpu", startup=300)
    t = make_task(tmp_path, accelerator="any")
    plan = Router([local, fake], ledger, log=lambda m: None).plan(t)
    assert plan[0].provider.name == "gpu" and plan[1].provider.name == "local" and plan[1].gpu == "none"


def test_local_end_to_end_hello(tmp_path, ledger):
    from freesky.spec import load_task
    task = load_task(os.path.join(HELLO, "task.yaml"))
    r = Router([LocalProvider({"allow_cpu_fallback": True})], ledger, log=lambda m: None)
    out = r.run(task, out_dir=str(tmp_path / "runs"))
    assert out.ok, out.message
    info = json.load(open(os.path.join(out.outputs_dir, "outputs", "out", "info.json")))
    assert info["provider"] == "local"
    assert os.path.exists(os.path.join(out.outputs_dir, "fgpu.log"))
    assert ledger.attempts("local")[-1].ok


def test_build_providers_from_config():
    ps = build_providers({"providers": {"local": {}, "gcp-trial": {"type": "skypilot", "quota_limit": 300},
                                        "colab": {"enabled": False}}})
    names = [p.name for p in ps]
    assert names == ["local", "gcp-trial"] and ps[1].quota.limit == 300
