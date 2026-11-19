"""Lightning AI Studios: free monthly credits (~15 credits, roughly 20 T4-hours).

Flow: create/start a Studio on a T4 -> upload driver + payload -> run -> download -> stop.
Setup:  pip install lightning-sdk, LIGHTNING_USER_ID / LIGHTNING_API_KEY (lightning.ai ->
        Settings -> Keys), and `teamspace:` in the freesky config.
"""
from __future__ import annotations

import os
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, classify, write_driver_files


class LightningProvider(Provider):
    name = "lightning"
    title = "Lightning AI Studio (free credits)"
    gpus = ["T4", "L4"]
    startup_s = 120
    session_limit_s = 4 * 3600 - 600   # free Studios restart every 4 h
    signup = "https://lightning.ai/sign-up"
    install = "pip install 'freesky[lightning]'  +  LIGHTNING_USER_ID / LIGHTNING_API_KEY"
    terms = "Monthly free credits (phone verification); beyond them you pay."

    def __init__(self, config=None):
        # credits per hour: ~0.7 for a T4 (15 credits ~ 22 h); L4 is pricier. Override with `rates:`.
        self.quota = Quota("credits", 15.0, "month", {"T4": 0.68, "L4": 1.2, "none": 0.1},
                           "monthly credits, estimated hourly burn")
        super().__init__(config)

    def available(self):
        import importlib.util
        if importlib.util.find_spec("lightning_sdk") is None:
            return False, "lightning-sdk package not installed"
        if not (os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID")):
            return False, "LIGHTNING_USER_ID / LIGHTNING_API_KEY not set"
        if not self.config.get("teamspace"):
            return False, "set `teamspace:` in the freesky config"
        return True, ""

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        from lightning_sdk import Machine, Studio

        machine = {"T4": Machine.T4, "L4": Machine.L4, "none": Machine.CPU}.get(gpu)
        if machine is None:
            raise ProviderError("capacity", f"GPU {gpu} not handled")
        src, payload = make_driver(task, self.name, embed=False, deadline_s=deadline_s, segment=segment)
        drv, pay = write_driver_files(workdir, src, payload)
        studio = None
        try:
            studio = Studio(name=self.config.get("studio", "freesky"), teamspace=self.config["teamspace"],
                            user=self.config.get("user"), org=self.config.get("org"), create_ok=True)
            studio.start(machine, interruptible=self.config.get("interruptible", False),
                         max_runtime=(deadline_s or task.resources.max_runtime) + 900)
            studio.run("rm -rf fgpu && mkdir -p fgpu")
            studio.upload_file(drv, "fgpu/fgpu_driver.py", progress_bar=False)
            studio.upload_file(pay, "fgpu/fgpu_payload.tgz", progress_bar=False)
            t0 = time.time()
            out, code = studio.run_with_exit_code("cd fgpu && FGPU_BASE=$PWD python fgpu_driver.py")
            studio.download_file(f"fgpu/{RESULT_TGZ}", os.path.join(workdir, RESULT_TGZ))
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
