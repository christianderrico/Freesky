"""Beam (beam.cloud): serverless GPUs with free signup hours (one-off; verify the current offer).

Batch: a Beam Sandbox runs the driver. Services: `expose_port` gives a public URL.
Setup:  pip install beam-client && beam configure  (or BEAM_TOKEN)
Experimental: written against beam-client 0.2.x, not yet tested on a real account.
"""
from __future__ import annotations

import os
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, Replica, RunResult, classify, write_driver_files

REMOTE = "/workspace/fgpu"


def _error(e):
    if isinstance(e, ProviderError):
        return e
    return ProviderError(classify(f"{type(e).__name__} {e}"), f"{type(e).__name__}: {e}")


class BeamProvider(Provider):
    name = "beam"
    title = "Beam (free signup hours, experimental)"
    gpus = ["T4", "A10G", "L4", "A100", "H100"]
    startup_s = 60
    can_serve = True
    signup = "https://beam.cloud"
    install = "pip install 'freesky[beam]' && beam configure"
    terms = "Free hours are a one-off signup credit; afterwards usage is billed."

    def __init__(self, config=None):
        self.quota = Quota("hours", 15.0, "total", {"*": 1.0, "none": 0.1},
                           "one-off signup hours (secondary sources: verify on your account)")
        super().__init__(config)

    def available(self):
        import importlib.util
        if importlib.util.find_spec("beam") is None:
            return False, "beam-client not installed"
        if not (os.environ.get("BEAM_TOKEN") or os.path.exists(os.path.expanduser("~/.beam/config.ini"))
                or os.path.exists(os.path.expanduser("~/.beta9/config.ini"))):
            return False, "no credentials: run `beam configure`"
        return True, ""

    def _sandbox(self, task, gpu, ports=(), keep_warm=600):
        import beam
        return beam.Sandbox(gpu="" if gpu == "none" else gpu, image=beam.Image(base_image=task.image),
                            ports=list(ports), keep_warm_seconds=keep_warm, name=f"freesky-{task.slug}").create()

    def _upload(self, sb, drv, pay):
        sb.fs.create_directory(REMOTE)
        sb.fs.upload_file(drv, f"{REMOTE}/fgpu_driver.py")
        sb.fs.upload_file(pay, f"{REMOTE}/fgpu_payload.tgz")

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        src, payload = make_driver(task, self.name, base=REMOTE, embed=False, deadline_s=deadline_s, segment=segment)
        drv, pay = write_driver_files(workdir, src, payload)
        sb = None
        try:
            sb = self._sandbox(task, gpu, keep_warm=(deadline_s or task.resources.max_runtime) + 600)
            self._upload(sb, drv, pay)
            t0 = time.time()
            proc = sb.process.exec("python3", f"{REMOTE}/fgpu_driver.py", cwd=REMOTE, env={"FGPU_BASE": REMOTE})
            code = proc.wait()
            out = proc.stdout.read()
            sb.fs.download_file(f"{REMOTE}/{RESULT_TGZ}", os.path.join(workdir, RESULT_TGZ))
            return RunResult(self.name, code, time.time() - t0, gpu, log_tail=str(out)[-2000:])
        except Exception as e:
            raise _error(e)
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass

    def serve(self, task, gpu, workdir):
        src, payload = make_driver(task, self.name, base=REMOTE, embed=False, mode="serve")
        drv, pay = write_driver_files(workdir, src, payload)
        sb = None
        try:
            sb = self._sandbox(task, gpu, ports=[task.service.port], keep_warm=task.service.max_lifetime)
            self._upload(sb, drv, pay)
            sb.process.exec("python3", f"{REMOTE}/fgpu_driver.py", cwd=REMOTE, env={"FGPU_BASE": REMOTE})
            url = sb.expose_port(task.service.port)
        except Exception as e:
            if sb is not None:
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

        return Replica(self.name, gpu, str(url).rstrip("/"), time.time(), stop)
