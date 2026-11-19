"""Lightning AI Studios: free monthly credits (~15 credits, roughly 20 T4-hours).

Flow: create/start a Studio on a T4 -> upload driver + payload -> run -> download -> stop.
Setup:  pip install lightning-sdk, then LIGHTNING_USER_ID / LIGHTNING_API_KEY
        (from lightning.ai -> Settings -> Keys), and a teamspace name in the config.
"""
from __future__ import annotations

import os
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, classify


class LightningProvider(Provider):
    name = "lightning"
    title = "Lightning AI Studio (crediti gratuiti)"
    gpus = ["T4", "L4"]
    startup_s = 120
    signup = "https://lightning.ai/sign-up"
    install = "pip install 'freesky[lightning]'  +  LIGHTNING_USER_ID / LIGHTNING_API_KEY"
    terms = "Crediti gratuiti mensili (verifica del telefono); oltre si paga."

    def __init__(self, config=None):
        # credits per hour: ~0.7 for a T4 (15 credits ~ 22 h); L4 is pricier. Adjust in config.
        self.quota = Quota("credits", 15.0, "month", {"T4": 0.68, "L4": 1.2, "none": 0.1},
                           "crediti mensili, stima del consumo per ora")
        super().__init__(config)

    def available(self):
        try:
            import lightning_sdk  # noqa: F401
        except ImportError:
            return False, "pacchetto lightning-sdk non installato"
        if not (os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID")):
            return False, "LIGHTNING_USER_ID / LIGHTNING_API_KEY mancanti"
        if not self.config.get("teamspace"):
            return False, "manca 'teamspace' nella config di freesky"
        return True, ""

    def launch(self, task, gpu, workdir):
        from lightning_sdk import Machine, Studio

        machine = {"T4": Machine.T4, "L4": Machine.L4, "none": Machine.CPU}.get(gpu)
        if machine is None:
            raise ProviderError("capacity", f"GPU {gpu} non gestita")
        src, payload = make_driver(task, self.name, base="", embed=False)
        drv, pay = os.path.join(workdir, "fgpu_driver.py"), os.path.join(workdir, "fgpu_payload.tgz")
        with open(drv, "w") as f:
            f.write(src)
        with open(pay, "wb") as f:
            f.write(payload)
        studio = None
        try:
            studio = Studio(name=self.config.get("studio", "freesky"), teamspace=self.config["teamspace"],
                            user=self.config.get("user"), org=self.config.get("org"), create_ok=True)
            studio.start(machine, interruptible=self.config.get("interruptible", False),
                         max_runtime=task.resources.max_runtime + 600)
            remote = "fgpu"
            studio.run(f"rm -rf {remote} && mkdir -p {remote}")
            studio.upload_file(drv, f"{remote}/fgpu_driver.py", progress_bar=False)
            studio.upload_file(pay, f"{remote}/fgpu_payload.tgz", progress_bar=False)
            t0 = time.time()
            out, code = studio.run_with_exit_code(f"cd {remote} && FGPU_BASE=$PWD python fgpu_driver.py")
            studio.download_file(f"{remote}/{RESULT_TGZ}", os.path.join(workdir, RESULT_TGZ))
            return RunResult(self.name, code, time.time() - t0, gpu, log_tail=str(out)[-2000:])
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(classify(str(e)), f"{type(e).__name__}: {e}")
        finally:
            if studio is not None:
                try:
                    studio.stop()
                except Exception:
                    pass
