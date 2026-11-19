"""Modal: serverless GPUs, $30/month of free credit on the Starter plan.

Runs the task in a Modal Sandbox (arbitrary commands, per-second billing, scale to zero).
Setup:  pip install modal && modal setup
"""
from __future__ import annotations

import os
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, classify

# $/hour on Modal (pricing page, Nov 2026). A10G is approximate; override in config if it changes.
RATES = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "A100": 2.10, "H100": 3.95, "none": 0.05}


class ModalProvider(Provider):
    name = "modal"
    title = "Modal (Starter, $30/mese)"
    gpus = ["T4", "L4", "A10G", "A100", "H100"]
    startup_s = 40
    signup = "https://modal.com/signup"
    install = "pip install 'freesky[modal]' && modal setup"
    terms = "Il credito gratuito copre l'uso a consumo; oltre i $30 si paga."

    def __init__(self, config=None):
        self.quota = Quota("usd", 30.0, "month", dict(RATES), "credito Starter, si rinnova ogni mese")
        super().__init__(config)

    def available(self):
        try:
            import modal  # noqa: F401
        except ImportError:
            return False, "pacchetto modal non installato"
        has_token = os.environ.get("MODAL_TOKEN_ID") or os.path.exists(os.path.expanduser("~/.modal.toml"))
        return (True, "") if has_token else (False, "nessun token: esegui `modal setup`")

    def launch(self, task, gpu, workdir):
        import modal

        src, payload = make_driver(task, self.name, base="/fgpu", embed=False)
        drv, pay = os.path.join(workdir, "fgpu_driver.py"), os.path.join(workdir, "fgpu_payload.tgz")
        with open(drv, "w") as f:
            f.write(src)
        with open(pay, "wb") as f:
            f.write(payload)
        sb = None
        try:
            app = modal.App.lookup(self.config.get("app", "freesky"), create_if_missing=True)
            image = modal.Image.from_registry(task.image)
            sb = modal.Sandbox.create(
                "sleep", "infinity", app=app, image=image,
                gpu=None if gpu == "none" else gpu,
                timeout=task.resources.max_runtime + 300,
            )
            sb.filesystem.make_directory("/fgpu")
            sb.filesystem.copy_from_local(drv, "/fgpu/fgpu_driver.py")
            sb.filesystem.copy_from_local(pay, "/fgpu/fgpu_payload.tgz")
            t0 = time.time()
            proc = sb.exec("python3", "/fgpu/fgpu_driver.py", timeout=task.resources.max_runtime + 60)
            out = proc.stdout.read()
            code = proc.wait()
            sb.filesystem.copy_to_local(f"/fgpu/{RESULT_TGZ}", os.path.join(workdir, RESULT_TGZ))
            return RunResult(self.name, code, time.time() - t0, gpu, log_tail=out[-2000:])
        except ProviderError:
            raise
        except Exception as e:  # modal raises its own exception types
            name = type(e).__name__.lower()
            kind = "auth" if "auth" in name else classify(f"{name} {e}")
            raise ProviderError(kind, f"{type(e).__name__}: {e}")
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass
