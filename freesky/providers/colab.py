"""Google Colab free tier, driven by Google's official Colab CLI (google-colab-cli).

Flow: colab new (T4) -> upload payload -> exec driver -> download outputs -> colab stop.
The free T4 comes "when available" and is preemptible: a refused allocation is a
capacity error and the router moves on. Batch only: serving from the free tier is
outside its intended use.
Setup:  pip install google-colab-cli, then (headless) Application Default Credentials:
  gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, run_cmd, which, write_driver_files


class ColabProvider(Provider):
    name = "colab"
    title = "Google Colab (free)"
    gpus = ["T4"]
    startup_s = 90
    session_limit_s = 11 * 3600      # free sessions are reclaimed around 12 h
    signup = "https://colab.research.google.com"
    install = "pip install google-colab-cli  (then gcloud ADC login, see docs)"
    terms = ("Script jobs started and stopped with the official CLI only; no always-on services. "
             "The free quota is not published: the limit is an adjustable estimate.")

    def __init__(self, config=None):
        self.quota = Quota("hours", 12.0, "week", {"T4": 1.0, "none": 0.0},
                           "estimate: Colab does not publish the free quota")
        super().__init__(config)

    def _base(self, scratch):
        args = ["colab"]
        if self.config.get("auth"):
            args.append(f"--auth={self.config['auth']}")
        args += ["--config", os.path.join(scratch, "colab-sessions.json")]
        return args

    def available(self):
        if not which("colab"):
            return False, "`colab` CLI not installed (pip install google-colab-cli)"
        if not hasattr(self, "_auth_ok"):
            # read-only call: lists sessions, allocates nothing; fails fast without credentials
            args = ["colab"] + ([f"--auth={self.config['auth']}"] if self.config.get("auth") else [])
            try:
                p = run_cmd(args + ["sessions"], timeout=45, check=False, env={"BROWSER": "none"})
                self._auth_ok = (p.returncode == 0, _last_error_line(p.stderr or p.stdout))
            except ProviderError as e:
                self._auth_ok = (False, str(e))
        ok, why = self._auth_ok
        return (True, "") if ok else (False, f"not logged in to Colab: {why or 'see docs'}")

    def launch(self, task, gpu, workdir, deadline_s=0, segment=1):
        session = f"fgpu-{task.slug[:20]}-{uuid.uuid4().hex[:6]}"
        base = self._base(workdir)
        src, payload = make_driver(task, self.name, base="/content", embed=False, exit_with_code=False,
                                   deadline_s=deadline_s, segment=segment)
        drv, pay = write_driver_files(workdir, src, payload)
        run_cmd(base + ["new", "-s", session] + ([] if gpu == "none" else ["--gpu", gpu]), timeout=600)
        try:
            t0 = time.time()
            run_cmd(base + ["upload", "-s", session, pay, "/content/fgpu_payload.tgz"], timeout=900)
            p = run_cmd(base + ["exec", "-s", session, "-f", drv],
                        timeout=(deadline_s or task.resources.max_runtime) + 300, check=False)
            run_cmd(base + ["download", "-s", session, f"/content/{RESULT_TGZ}",
                            os.path.join(workdir, RESULT_TGZ)], timeout=900)
            return RunResult(self.name, exit_code_from_log(p.stdout), time.time() - t0, gpu,
                             log_tail=p.stdout[-2000:])
        finally:
            run_cmd(base + ["stop", "-s", session], timeout=120, check=False)


def _last_error_line(text: str) -> str:
    """The most informative line of a CLI error (skips rich traceback frames and URLs)."""
    box = "\u256d\u2502\u2570\u2500\u256e\u256f"
    lines = [l.strip().strip(box).strip() for l in (text or "").splitlines()]
    lines = [l for l in lines if l and set(l) - set(box + " ") and "http" not in l and "\u2771" not in l
             and "\u2502" not in l and not l.startswith(("Traceback", "File ", "/"))]
    for l in reversed(lines):
        if re.match(r"^[A-Za-z_.]+(Error|Exception):", l):
            return l[:160]
    for l in reversed(lines):
        if any(k in l.lower() for k in ("credential", "not found", "error:", "denied", "expired")):
            return l[:160]
    if any("authorization code" in l.lower() for l in lines):
        return "needs an interactive OAuth login; use `auth: adc` (see docs)"
    return lines[-1][:160] if lines else ""


def exit_code_from_log(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        if line.startswith("[fgpu] RESULT "):
            try:
                return int(json.loads(line[len("[fgpu] RESULT "):])["exit_code"])
            except (ValueError, KeyError):
                break
    raise ProviderError("transient", "the driver did not report a result (session lost?)")
