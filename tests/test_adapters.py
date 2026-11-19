"""Adapters against simulated providers that really execute the driver."""
import base64
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import types

import pytest

from freesky.ledger import Ledger
from freesky.router import Router
from freesky.spec import task_from_dict

FAKES = os.path.join(os.path.dirname(__file__), "fakes")


def task(tmp_path, accelerator="T4"):
    wd = tmp_path / "wd"
    wd.mkdir(exist_ok=True)
    (wd / "main.py").write_text("import os,json\nos.makedirs('out',exist_ok=True)\n"
                                "json.dump({'p':os.environ['FGPU_PROVIDER']},open('out/r.json','w'))\n")
    return task_from_dict({"name": "adapter test", "run": "python3 main.py", "outputs": ["out"],
                           "resources": {"accelerator": accelerator, "max_runtime": "5m"}}, base_dir=str(wd))


def provider_name_in_output(out):
    return json.load(open(os.path.join(out.outputs_dir, "outputs", "out", "r.json")))["p"]


def run_driver_in(remote_dir, driver_path, base_placeholder):
    src = open(driver_path).read()
    spec_marker = 'SPEC = json.loads(base64.b64decode("'
    i = src.index(spec_marker) + len(spec_marker)
    j = src.index('"', i)
    spec = json.loads(base64.b64decode(src[i:j]))
    spec["base"] = remote_dir
    if spec.get("payload_path"):
        spec["payload_path"] = spec["payload_path"].replace(base_placeholder, remote_dir)
    src = src[:i] + base64.b64encode(json.dumps(spec).encode()).decode() + src[j:]
    p = subprocess.run([sys.executable, "-c", src], cwd=remote_dir, capture_output=True, text=True,
                       env=dict(os.environ, FGPU_BASE=remote_dir))
    return p.returncode, p.stdout


# ------------------------------------------------------------------ colab
def test_colab_adapter(tmp_path, monkeypatch):
    from freesky.providers.colab import ColabProvider
    root = tmp_path / "vm"
    root.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shutil.copy(os.path.join(FAKES, "colab"), bindir / "colab")
    os.chmod(bindir / "colab", 0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(root))
    out = Router([ColabProvider({"auth": "adc"})], Ledger(str(tmp_path / "h")), log=lambda m: None).run(
        task(tmp_path), out_dir=str(tmp_path / "runs"))
    assert out.ok, out.message
    assert provider_name_in_output(out) == "colab"
    calls = [c for c in (root / "calls.log").read_text().splitlines() if c != "sessions"]
    assert calls[0].startswith("new -s fgpu-adapter-test") and "--gpu T4" in calls[0]
    assert calls[-1].startswith("stop -s")


def test_colab_no_gpu_is_capacity_and_stops_nothing(tmp_path, monkeypatch):
    from freesky.providers.colab import ColabProvider
    root = tmp_path / "vm"
    root.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shutil.copy(os.path.join(FAKES, "colab"), bindir / "colab")
    os.chmod(bindir / "colab", 0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(root))
    monkeypatch.setenv("FAKE_COLAB_NO_GPU", "1")
    led = Ledger(str(tmp_path / "h"))
    out = Router([ColabProvider()], led, log=lambda m: None).run(task(tmp_path), out_dir=str(tmp_path / "runs"))
    assert not out.ok
    assert led.attempts("colab")[-1].error_kind == "capacity"
    assert led.cooldown("colab")[0] > 0


# ------------------------------------------------------------------ kaggle
class FakeKaggleApi:
    def __init__(self, root):
        self.root, self.pushed = root, None

    def authenticate(self):
        pass

    def get_config_value(self, k):
        return "christian"

    def quota_view(self):
        q = types.SimpleNamespace(total_time_allowed=dt.timedelta(hours=30), time_used=dt.timedelta(hours=4.5))
        return types.SimpleNamespace(gpu_quota=q, tpu_quota=None)

    def kernels_push(self, folder, timeout=None, acc=None):
        self.pushed = (json.load(open(os.path.join(folder, "kernel-metadata.json"))), acc)
        work = os.path.join(self.root, "working")
        os.makedirs(work, exist_ok=True)
        self.code, _ = run_driver_in(work, os.path.join(folder, "fgpu_driver.py"), "/kaggle/working")

    def kernels_status(self, ref):
        return types.SimpleNamespace(status="KernelWorkerStatus.COMPLETE" if self.code == 0 else "ERROR")

    def kernels_output(self, ref, path, force=True, quiet=True):
        os.makedirs(path, exist_ok=True)
        shutil.copy(os.path.join(self.root, "working", "fgpu_out.tgz"), path)


def test_kaggle_adapter_and_real_quota(tmp_path, monkeypatch):
    from freesky.providers import kaggle_
    fake = FakeKaggleApi(str(tmp_path / "k"))
    p = kaggle_.KaggleProvider()
    p._api = fake
    monkeypatch.setattr(p, "available", lambda: (True, ""))
    assert p.remaining(used=0) == pytest.approx(25.5)  # from the API, not the ledger
    out = Router([p], Ledger(str(tmp_path / "h")), log=lambda m: None).run(task(tmp_path),
                                                                          out_dir=str(tmp_path / "runs"))
    assert out.ok, out.message
    assert provider_name_in_output(out) == "kaggle"
    meta, acc = fake.pushed
    assert meta["enable_gpu"] == "true" and meta["is_private"] == "true" and acc == "NvidiaTeslaT4"
    assert meta["id"] == "christian/freesky-adapter-test"


# ------------------------------------------------------------------ modal
def install_fake_modal(monkeypatch, tmp_path):
    remote = tmp_path / "modal-vm"
    remote.mkdir()
    created = {}

    def m(p):
        return str(remote) + p[len("/fgpu"):] if p.startswith("/fgpu") else p

    class FS:
        def make_directory(self, p):
            os.makedirs(m(p), exist_ok=True)

        def copy_from_local(self, local, rp):
            shutil.copy(local, m(rp))

        def copy_to_local(self, rp, local):
            shutil.copy(m(rp), local)

    class Proc:
        def __init__(self, code, out):
            self.code, self.stdout = code, types.SimpleNamespace(read=lambda: out)

        def wait(self):
            return self.code

    class Sandbox:
        @staticmethod
        def create(*args, **kw):
            created.update(kw)
            sb = Sandbox()
            sb.filesystem = FS()
            sb.terminated = False
            created["sb"] = sb
            return sb

        def exec(self, *args, timeout=None):
            code, out = run_driver_in(str(remote), m(args[1]), "/fgpu")
            return Proc(code, out)

        def terminate(self):
            self.terminated = True

    mod = types.ModuleType("modal")
    mod.App = types.SimpleNamespace(lookup=lambda name, create_if_missing=True: f"app:{name}")
    mod.Image = types.SimpleNamespace(from_registry=lambda ref: f"img:{ref}")
    mod.Sandbox = Sandbox
    monkeypatch.setitem(sys.modules, "modal", mod)
    return created


def test_modal_adapter(tmp_path, monkeypatch):
    from freesky.providers.modal_ import ModalProvider
    created = install_fake_modal(monkeypatch, tmp_path)
    monkeypatch.setenv("MODAL_TOKEN_ID", "x")
    led = Ledger(str(tmp_path / "h"))
    out = Router([ModalProvider()], led, log=lambda m: None).run(task(tmp_path, "L4"),
                                                                out_dir=str(tmp_path / "runs"))
    assert out.ok, out.message
    assert provider_name_in_output(out) == "modal"
    assert created["gpu"] == "L4" and created["image"] == "img:python:3.12-slim" and created["sb"].terminated
    assert led.attempts("modal")[-1].cost > 0  # USD charged against the $30 credit


# ------------------------------------------------------------------ lightning
def test_lightning_adapter(tmp_path, monkeypatch):
    remote = tmp_path / "studio"
    remote.mkdir()
    state = {}

    class Machine:
        T4, L4, CPU = "T4", "L4", "CPU"

    class Studio:
        def __init__(self, **kw):
            state["init"] = kw

        def start(self, machine, interruptible=None, max_runtime=None):
            state["machine"] = machine

        def run(self, cmd):
            os.makedirs(remote / "fgpu", exist_ok=True)
            return ""

        def upload_file(self, local, rp, progress_bar=True):
            shutil.copy(local, remote / rp)

        def run_with_exit_code(self, cmd):
            return run_driver_in(str(remote / "fgpu"), str(remote / "fgpu" / "fgpu_driver.py"), "")[::-1]

        def download_file(self, rp, local):
            shutil.copy(remote / rp, local)

        def stop(self):
            state["stopped"] = True

    mod = types.ModuleType("lightning_sdk")
    mod.Machine, mod.Studio = Machine, Studio
    monkeypatch.setitem(sys.modules, "lightning_sdk", mod)
    monkeypatch.setenv("LIGHTNING_API_KEY", "k")
    monkeypatch.setenv("LIGHTNING_USER_ID", "u")
    from freesky.providers.lightning_ import LightningProvider
    out = Router([LightningProvider({"teamspace": "ts"})], Ledger(str(tmp_path / "h")), log=lambda m: None).run(
        task(tmp_path), out_dir=str(tmp_path / "runs"))
    assert out.ok, out.message
    assert provider_name_in_output(out) == "lightning"
    assert state["machine"] == "T4" and state["stopped"] and state["init"]["teamspace"] == "ts"
