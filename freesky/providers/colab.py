"""Google Colab free tier, driven by Google's official Colab CLI (google-colab-cli).

Flow: colab new (T4) -> upload payload -> exec driver -> download outputs -> colab stop.
The free T4 comes "when available" and is preemptible: a refused allocation is a
capacity error and the router moves on.
Setup:  pip install google-colab-cli, then authenticate once (see `freesky check colab`).
"""
from __future__ import annotations

import json
import os
import time
import uuid

from ..bundle import RESULT_TGZ, make_driver
from .base import Provider, ProviderError, Quota, RunResult, run_cmd, which  # noqa: F401


class ColabProvider(Provider):
    name = "colab"
    title = "Google Colab (gratuito)"
    gpus = ["T4"]
    startup_s = 90
    signup = "https://colab.research.google.com"
    install = "pip install google-colab-cli  (poi: gcloud auth application-default login --scopes=...)"
    terms = ("Solo lavori a script avviati e chiusi dalla CLI ufficiale; niente servizi sempre accesi. "
             "La quota gratuita non e' pubblica: il limite qui sotto e' una stima regolabile.")

    def __init__(self, config=None):
        self.quota = Quota("hours", 12.0, "week", {"T4": 1.0, "none": 0.0},
                           "stima: Colab non pubblica la quota gratuita")
        super().__init__(config)

    def _base(self, scratch):
        args = ["colab"]
        if self.config.get("auth"):
            args.append(f"--auth={self.config['auth']}")  # 'adc' is the reliable headless choice
        args += ["--config", os.path.join(scratch, "colab-sessions.json")]
        return args

    def available(self):
        if not which("colab"):
            return False, "CLI `colab` non installata (pip install google-colab-cli)"
        if not hasattr(self, "_auth_ok"):
            # read-only call: lists sessions, allocates nothing; fails fast without credentials
            args = ["colab"] + ([f"--auth={self.config['auth']}"] if self.config.get("auth") else [])
            try:
                p = run_cmd(args + ["sessions"], timeout=45, check=False,
                            env={"BROWSER": "none"})  # never open a consent page from the router
                self._auth_ok = (p.returncode == 0, (p.stderr or p.stdout).strip()[-160:])
            except ProviderError as e:
                self._auth_ok = (False, str(e))
        ok, why = self._auth_ok
        return (True, "") if ok else (False, f"login Colab non valido: {why or 'esegui il login (vedi README)'}")

    def launch(self, task, gpu, workdir):
        session = f"fgpu-{task.slug[:20]}-{uuid.uuid4().hex[:6]}"
        base = self._base(workdir)
        src, payload = make_driver(task, self.name, base="/content", embed=False, exit_with_code=False)
        drv, pay = os.path.join(workdir, "fgpu_driver.py"), os.path.join(workdir, "fgpu_payload.tgz")
        with open(drv, "w") as f:
            f.write(src)
        with open(pay, "wb") as f:
            f.write(payload)
        new = base + ["new", "-s", session] + ([] if gpu == "none" else ["--gpu", gpu])
        run_cmd(new, timeout=600)  # 400 here = no free T4 right now -> capacity
        try:
            t0 = time.time()
            run_cmd(base + ["upload", "-s", session, pay, "/content/fgpu_payload.tgz"], timeout=900)
            p = run_cmd(base + ["exec", "-s", session, "-f", drv],
                        timeout=task.resources.max_runtime + 120, check=False)
            run_cmd(base + ["download", "-s", session, f"/content/{RESULT_TGZ}",
                            os.path.join(workdir, RESULT_TGZ)], timeout=900)
            code = _exit_code_from_log(p.stdout)
            return RunResult(self.name, code, time.time() - t0, gpu, log_tail=p.stdout[-2000:])
        finally:
            run_cmd(base + ["stop", "-s", session], timeout=120, check=False)


def _exit_code_from_log(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        if line.startswith("[fgpu] RESULT "):
            try:
                return int(json.loads(line[len("[fgpu] RESULT "):])["exit_code"])
            except (ValueError, KeyError):
                break
    raise ProviderError("transient", "il driver non ha riportato un risultato")
