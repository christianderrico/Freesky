"""Kaggle Notebooks: ~30 GPU-hours/week (T4 x2), real quota readable through the API.

Flow: push a private script kernel with the driver (small workdirs are embedded, larger
ones go to a private dataset attached to the kernel) -> poll status -> download output.
Setup:  pip install kaggle, then `kaggle auth login` (OAuth) or a token in KAGGLE_API_TOKEN /
        ~/.kaggle/access_token (legacy ~/.kaggle/kaggle.json still accepted).
Internet inside the kernel (pip install, model download) needs a phone-verified account.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, classify

EMBED_LIMIT = 512 * 1024
ACCELERATOR = {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100"}


class KaggleProvider(Provider):
    name = "kaggle"
    title = "Kaggle Notebooks (free)"
    gpus = ["T4"]  # T4 x2; P100 is retired and silently replaced
    startup_s = 240
    session_limit_s = 12 * 3600 - 600   # 12 h per run, minus room to pack the outputs
    signup = "https://www.kaggle.com/account/login"
    install = "pip install kaggle  +  kaggle auth login (or KAGGLE_API_TOKEN)"
    terms = "Batch jobs on private kernels; one account per person."

    def __init__(self, config=None):
        self.quota = Quota("hours", 30.0, "week", {"T4": 1.0, "none": 0.0},
                           "real quota read from the API when possible")
        super().__init__(config)
        self._api = None

    def api(self):
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                raise ProviderError("auth", "kaggle package not installed")
            api = KaggleApi()
            try:
                api.authenticate()
            except Exception as e:
                raise ProviderError("auth", f"Kaggle authentication failed: {e}")
            self._api = api
        return self._api

    def username(self):
        return self.config.get("username") or os.environ.get("KAGGLE_USERNAME") or \
            self.api().get_config_value("username")

    def available(self):
        if importlib.util.find_spec("kaggle") is None:  # importing kaggle triggers auth: avoid it here
            return False, "kaggle package not installed"
        kdir = os.environ.get("KAGGLE_CONFIG_DIR") or os.path.expanduser("~/.kaggle")
        sources = [os.environ.get("KAGGLE_API_TOKEN"), os.environ.get("KAGGLE_KEY"),
                   os.path.exists(os.path.join(kdir, "kaggle.json")),
                   os.path.exists(os.path.join(kdir, "access_token")),
                   self.config.get("oauth")]  # set `oauth: true` after `kaggle auth login`
        if not any(sources):
            return False, "no credentials: `kaggle auth login` or KAGGLE_API_TOKEN"
        return True, ""

    def remaining(self, used):
        """Real weekly GPU quota from Kaggle; falls back to the local ledger."""
        try:
            q = self.api().quota_view().gpu_quota
            if q is not None:
                return max(0.0, (q.total_time_allowed - q.time_used).total_seconds() / 3600)
        except Exception:
            pass
        return super().remaining(used)

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        api, user = self.api(), self.username()
        slug = f"freesky-{task.slug}"[:50]
        kdir = os.path.join(workdir, "kernel")
        os.makedirs(kdir, exist_ok=True)
        opts = dict(deadline_s=deadline_s, segment=segment)
        _, payload = make_driver(task, self.name)
        dataset_sources = []
        if len(payload) <= EMBED_LIMIT:
            src, _ = make_driver(task, self.name, base="/kaggle/working", **opts)
        else:
            ds = self._upload_payload(api, user, slug, payload, workdir)
            dataset_sources = [ds]
            src, _ = make_driver(task, self.name, base="/kaggle/working", embed=False,
                                 payload_path=f"/kaggle/input/{ds.split('/')[1]}/fgpu_payload.tgz", **opts)
        with open(os.path.join(kdir, "fgpu_driver.py"), "w") as f:
            f.write(src)
        meta = {
            "id": f"{user}/{slug}", "title": slug, "code_file": "fgpu_driver.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true" if gpu != "none" else "false", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": ACCELERATOR.get(gpu, ""),
            "dataset_sources": dataset_sources, "competition_sources": [],
            "kernel_sources": [], "model_sources": [],
        }
        with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        t0 = time.time()
        run_s = deadline_s or task.resources.max_runtime
        try:
            api.kernels_push(kdir, timeout=str(run_s + 300),
                             acc=ACCELERATOR.get(gpu) if gpu != "none" else None)
        except Exception as e:
            raise ProviderError(classify(str(e)), f"push failed: {e}")
        status = self._wait(api, f"{user}/{slug}", run_s + 1800)
        outdir = os.path.join(workdir, "kaggle-output")
        try:
            api.kernels_output(f"{user}/{slug}", outdir, force=True, quiet=True)
        except Exception as e:
            raise ProviderError("transient", f"output download failed: {e}")
        res = os.path.join(outdir, RESULT_TGZ)
        if not os.path.exists(res):
            raise ProviderError("task" if status == "error" else "transient",
                                f"kernel finished ({status}) without {RESULT_TGZ}")
        os.replace(res, os.path.join(workdir, RESULT_TGZ))
        return RunResult(self.name, 0 if status == "complete" else 1, time.time() - t0, gpu)

    def _wait(self, api, ref, timeout):
        deadline, delay = time.time() + timeout, 10.0
        while time.time() < deadline:
            try:
                st = str(getattr(api.kernels_status(ref), "status", "")).lower()
            except Exception as e:
                st = ""
                if classify(str(e)) == "auth":
                    raise ProviderError("auth", str(e))
            if "complete" in st:
                return "complete"
            if "error" in st or "cancel" in st:
                return "error"
            time.sleep(delay)
            delay = min(60.0, delay * 1.5)
        raise ProviderError("transient", f"timed out waiting for kernel {ref}")

    def _upload_payload(self, api, user, slug, payload, workdir):
        ddir = os.path.join(workdir, "dataset")
        os.makedirs(ddir, exist_ok=True)
        with open(os.path.join(ddir, "fgpu_payload.tgz"), "wb") as f:
            f.write(payload)
        ref = f"{user}/{slug}-payload"
        with open(os.path.join(ddir, "dataset-metadata.json"), "w") as f:
            json.dump({"title": f"{slug}-payload", "id": ref, "licenses": [{"name": "CC0-1.0"}]}, f)
        try:
            try:
                api.dataset_create_version(ddir, version_notes="freesky", quiet=True)
            except Exception:
                api.dataset_create_new(ddir, public=False, quiet=True)
        except Exception as e:
            raise ProviderError(classify(str(e)), f"dataset upload failed: {e}")
        time.sleep(30)  # datasets need a moment to be processed before a kernel can mount them
        return ref
