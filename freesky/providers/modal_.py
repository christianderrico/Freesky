"""Modal: serverless GPUs, $30/month of free credit on the Starter plan.

Batch: a Modal Sandbox runs the driver. Services: the Sandbox exposes the service port
through a Modal encrypted tunnel (an HTTPS URL), and is terminated when idle.
Setup:  pip install modal && modal setup
"""
from __future__ import annotations

import os
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, Replica, RunResult, classify, write_driver_files

# $/hour on Modal (pricing page, Nov 2026; A10G approximate). Override with `rates:` in the config.
RATES = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "A100": 2.10, "H100": 3.95, "none": 0.05}


def _error(e: Exception) -> ProviderError:
    if isinstance(e, ProviderError):
        return e
    name = type(e).__name__.lower()
    return ProviderError("auth" if "auth" in name else classify(f"{name} {e}"), f"{type(e).__name__}: {e}")


class ModalProvider(Provider):
    name = "modal"
    title = "Modal (Starter, $30/month)"
    gpus = ["T4", "L4", "A10G", "A100", "H100"]
    startup_s = 40
    session_limit_s = 24 * 3600     # Sandbox maximum lifetime
    can_serve = True
    signup = "https://modal.com/signup"
    install = "pip install 'freesky[modal]' && modal setup"
    terms = "The free credit covers metered usage; beyond $30 you pay."

    def __init__(self, config=None):
        self.quota = Quota("usd", 30.0, "month", dict(RATES), "Starter credit, renews monthly")
        super().__init__(config)

    def available(self):
        import importlib.util
        if importlib.util.find_spec("modal") is None:
            return False, "modal package not installed"
        has_token = os.environ.get("MODAL_TOKEN_ID") or os.path.exists(os.path.expanduser("~/.modal.toml"))
        return (True, "") if has_token else (False, "no token: run `modal setup`")

    def _sandbox(self, task, gpu, timeout, ports=()):
        import modal
        app = modal.App.lookup(self.config.get("app", "freesky"), create_if_missing=True)
        kw = dict(app=app, image=modal.Image.from_registry(task.image),
                  gpu=None if gpu == "none" else gpu, timeout=timeout)
        if ports:
            kw["encrypted_ports"] = list(ports)
        return modal.Sandbox.create("sleep", "infinity", **kw)

    def _upload(self, sb, drv, pay):
        sb.filesystem.make_directory("/fgpu")
        sb.filesystem.copy_from_local(drv, "/fgpu/fgpu_driver.py")
        sb.filesystem.copy_from_local(pay, "/fgpu/fgpu_payload.tgz")

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        src, payload = make_driver(task, self.name, base="/fgpu", embed=False,
                                   deadline_s=deadline_s, segment=segment)
        drv, pay = write_driver_files(workdir, src, payload)
        run_s = deadline_s or task.resources.max_runtime
        sb = None
        try:
            sb = self._sandbox(task, gpu, run_s + 600)
            self._upload(sb, drv, pay)
            t0 = time.time()
            proc = sb.exec("python3", "/fgpu/fgpu_driver.py", timeout=run_s + 300)
            out = proc.stdout.read()
            code = proc.wait()
            sb.filesystem.copy_to_local(f"/fgpu/{RESULT_TGZ}", os.path.join(workdir, RESULT_TGZ))
            return RunResult(self.name, code, time.time() - t0, gpu, log_tail=out[-2000:])
        except Exception as e:
            raise _error(e)
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass

    def serve(self, task, gpu, workdir):
        src, payload = make_driver(task, self.name, base="/fgpu", embed=False, mode="serve")
        drv, pay = write_driver_files(workdir, src, payload)
        port = task.service.port
        sb = None
        try:
            sb = self._sandbox(task, gpu, task.service.max_lifetime, ports=[port])
            self._upload(sb, drv, pay)
            sb.exec("python3", "/fgpu/fgpu_driver.py")  # runs in the background inside the sandbox
            url = sb.tunnels()[port].url
        except Exception as e:
            if sb is not None:  # never leave a half-started GPU sandbox billing
                try:
                    sb.terminate()
                except Exception:
                    pass
            raise _error(e)

        def stop():
            try:
                sb.terminate()
            except Exception:
                pass

        return Replica(self.name, gpu, url.rstrip("/"), time.time(), stop)
