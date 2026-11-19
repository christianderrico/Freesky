"""Long jobs resumed across providers, and endpoints with failover."""
import json
import os
import socket
import sys
import time
import urllib.request

from freesky.ledger import Ledger
from freesky.providers.base import ProviderError
from freesky.providers.local import LocalProvider
from freesky.router import Router
from freesky.serve import ServiceManager, make_server
from freesky.spec import task_from_dict

sys.path.insert(0, os.path.dirname(__file__))
from test_router import Fake  # noqa: E402

TRAIN = r'''
import json, os, signal, sys, time
ck = os.environ["FREESKY_CHECKPOINT_DIR"]
state_f = os.path.join(ck, "state.json")
state = json.load(open(state_f)) if os.path.exists(state_f) else {"step": 0, "segments": []}
state["segments"].append(os.environ["FREESKY_PROVIDER"] + ":" + os.environ["FREESKY_SEGMENT"])
def save(*_):
    json.dump(state, open(state_f, "w"))
    sys.exit(0 if state["step"] >= 12 else 1)
signal.signal(signal.SIGTERM, save)
while state["step"] < 12:
    time.sleep(0.25)
    state["step"] += 1
    json.dump(state, open(state_f, "w"))
os.makedirs("out", exist_ok=True)
json.dump(state, open("out/final.json", "w"))
'''


def test_long_job_resumes_across_providers(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "train.py").write_text(TRAIN)
    task = task_from_dict({"name": "long", "run": "python3 train.py", "outputs": ["out"], "checkpoint": "ckpt",
                           "resources": {"accelerator": "T4", "max_runtime": "60s", "estimated_runtime": "5s"}},
                          base_dir=str(wd))
    a, b = Fake("kaggle-like", startup=5), Fake("colab-like", startup=10)
    a.session_limit_s = b.session_limit_s = 1.2      # each "session" holds ~4 steps of 12
    r = Router([a, b], Ledger(str(tmp_path / "h")), log=lambda m: None)
    r.segment_margin_s, r.min_segment_s = 0, 1
    out = r.run(task, out_dir=str(tmp_path / "runs"))
    assert out.ok, out.message
    assert len(out.segments) >= 2 and all(s["deadline_hit"] for s in out.segments[:-1])
    final = json.load(open(os.path.join(out.outputs_dir, "outputs", "out", "final.json")))
    assert final["step"] == 12
    assert len(final["segments"]) == len(out.segments)   # every segment resumed from the previous one
    assert (wd / "ckpt").exists() is False                # the user's workdir is never modified


def test_long_job_without_checkpoint_skips_short_sessions(tmp_path):
    a = Fake("short")
    a.session_limit_s = 3600
    task = task_from_dict({"name": "x", "run": "true", "resources": {"accelerator": "T4", "max_runtime": "5h"}},
                          base_dir=str(tmp_path))
    plan = Router([a], Ledger(str(tmp_path / "h")), log=lambda m: None).plan(task)
    assert "session limit" in plan[0].skip


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


SERVER = r'''
import http.server, json, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"provider": os.environ.get("FREESKY_PROVIDER"), "path": self.path}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    do_POST = do_GET
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


class BrokenServe(Fake):
    can_serve = True

    def serve(self, task, gpu, workdir):
        raise ProviderError("capacity", "no GPU for endpoints right now")


def service_task(tmp_path, port, idle="60s"):
    wd = tmp_path / "svc"
    wd.mkdir(exist_ok=True)
    (wd / "server.py").write_text(SERVER)
    return task_from_dict({"name": "svc", "run": f"python3 server.py {port}", "resources": {"accelerator": "any"},
                           "service": {"port": port, "health": "/health", "ready_timeout": "30s",
                                       "idle_timeout": idle}}, base_dir=str(wd))


def test_endpoint_cold_start_failover_and_idle_shutdown(tmp_path):
    port, proxy_port = free_port(), free_port()
    task = service_task(tmp_path, port, idle="2s")
    broken = BrokenServe("modal-like", startup=1)
    local = LocalProvider({"allow_cpu_fallback": True})
    led = Ledger(str(tmp_path / "h"))
    r = Router([broken, local], led, log=lambda m: None)
    m = ServiceManager(r, task, log=lambda m: None, health_interval=1)
    srv = make_server(m, "127.0.0.1", proxy_port)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=m.monitor, daemon=True).start()
    try:
        # first request: cold start; modal-like fails, the replica comes up locally
        with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/v1/test?x=1", timeout=60) as resp:
            data = json.load(resp)
        assert data == {"provider": "local", "path": "/v1/test?x=1"}
        st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/_freesky/status"))
        assert st["replica"]["provider"] == "local"
        assert led.cooldown("modal-like")[0] > 0
        # idle shutdown
        deadline = time.time() + 15
        while time.time() < deadline and not m.history:
            time.sleep(0.5)
        assert m.replica is None and m.history[-1]["reason"].startswith("idle")
    finally:
        srv.shutdown()
        m.shutdown()


def test_endpoint_restarts_after_replica_death(tmp_path):
    port, proxy_port = free_port(), free_port()
    task = service_task(tmp_path, port)
    local = LocalProvider({"allow_cpu_fallback": True})
    r = Router([local], Ledger(str(tmp_path / "h")), log=lambda m: None)
    m = ServiceManager(r, task, log=lambda m: None)
    srv = make_server(m, "127.0.0.1", proxy_port)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/a", timeout=60).read()
        m.replica.stop()                      # simulate a preempted replica
        time.sleep(0.5)
        r.ledger.cooldown = lambda name: (0.0, "")   # single provider: let it be retried at once
        with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/b", timeout=60) as resp:
            assert json.load(resp)["path"] == "/b"
        assert m.history and m.history[-1]["reason"] == "unreachable"
    finally:
        srv.shutdown()
        m.shutdown()
