"""Endpoints on free GPUs: a local proxy in front of one warm replica, with failover.

    freesky serve kev.yaml --port 9000      # http://127.0.0.1:9000 -> replica on Modal / local / ...

- Scale to zero: no replica runs until the first request (or `--warm` starts one now).
- Cold start: the request waits while a replica is started on the best available
  provider and its health check (`service.health`) passes.
- Idle shutdown: after `service.idle_timeout` without requests the replica is stopped,
  so free credit is only spent while the endpoint is actually used.
- Failover: if the replica dies (connection errors / failing health checks), the provider
  is put on cooldown, a replica starts on the next provider, and the request is retried once.
- Accounting: replica uptime is charged to the provider's quota in the shared ledger.
"""
from __future__ import annotations

import http.client
import json
import shutil
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from .ledger import Attempt
from .providers.base import ProviderError, Replica, scratch_dir
from .router import Router
from .spec import Task

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
               "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class NoReplica(Exception):
    pass


class ServiceManager:
    def __init__(self, router: Router, task: Task, only=None, log: Callable[[str], None] = print,
                 health_interval: float = 30.0):
        if not task.service:
            raise ValueError("this task has no `service:` block")
        self.router, self.task, self.only, self.log = router, task, only, log
        self.svc = task.service
        self.replica: Optional[Replica] = None
        self.gpu = ""
        self.scratch = ""
        self.lock = threading.RLock()
        self.last_request = time.time()
        self.health_interval = health_interval
        self.stopping = threading.Event()
        self.history = []

    # ------------------------------------------------------------ lifecycle
    def ensure(self) -> Replica:
        with self.lock:
            if self.replica:
                return self.replica
            cands = [c for c in self.router.plan(self.task, self.only, serving=True) if c.usable]
            if not cands:
                raise NoReplica("no provider can serve this endpoint right now")
            last = ""
            for c in cands:
                p = c.provider
                self.log(f"> starting replica on {p.name} ({c.gpu})")
                scratch = scratch_dir(f"serve-{p.name}")
                t0 = time.time()
                rep = None
                try:
                    rep = p.serve(self.task, c.gpu, scratch)
                    self._wait_ready(rep)
                    self.replica, self.gpu, self.scratch = rep, c.gpu, scratch
                    self.log(f"  replica ready on {p.name} in {time.time() - t0:.0f}s: {rep.url}")
                    return rep
                except (ProviderError, TimeoutError, OSError) as e:
                    kind = getattr(e, "kind", "capacity")
                    last = f"{p.name}: {e}"
                    self.log(f"  x {last}")
                    if rep:
                        rep.stop()
                    self._charge(p.name, c.gpu, time.time() - t0, ok=False, kind=kind, err=str(e))
                    shutil.rmtree(scratch, ignore_errors=True)
            raise NoReplica(f"every provider failed; last error: {last}")

    def _wait_ready(self, rep: Replica):
        deadline = time.time() + self.svc.ready_timeout
        while time.time() < deadline:
            if self._healthy(rep):
                return
            time.sleep(3)
        raise TimeoutError(f"health check {self.svc.health} not passing after {self.svc.ready_timeout}s")

    def _healthy(self, rep: Replica) -> bool:
        try:
            req = urllib.request.Request(rep.url + self.svc.health, headers=rep.headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status < 500
        except urllib.error.HTTPError as e:  # the server answered: it is up
            return e.code < 500
        except Exception:
            return False

    def retire(self, reason: str, failed: bool = False):
        with self.lock:
            rep = self.replica
            if not rep:
                return
            self.replica = None
            self.log(f"> stopping replica on {rep.provider}: {reason}")
            try:
                rep.stop()
            finally:
                up = time.time() - rep.started
                self._charge(rep.provider, self.gpu, up, ok=not failed,
                             kind="transient" if failed else "", err=reason if failed else "")
                self.history.append({"provider": rep.provider, "gpu": self.gpu, "uptime_s": round(up),
                                     "reason": reason})
                shutil.rmtree(self.scratch, ignore_errors=True)

    def _charge(self, provider, gpu, seconds, ok, kind="", err=""):
        p = next((x for x in self.router.providers if x.name == provider), None)
        cost = p.actual_cost(gpu, seconds) if p and kind not in ("auth", "capacity") else 0.0
        self.router.ledger.record(Attempt(time.time(), provider, f"serve:{self.task.name}", seconds,
                                          cost, ok=ok, error_kind=kind, error=err))

    def monitor(self):
        """Background thread: idle shutdown, health checks, max lifetime."""
        last_check = 0.0
        while not self.stopping.wait(min(self.health_interval, 5)):
            rep = self.replica
            if not rep:
                continue
            now = time.time()
            if now - self.last_request > self.svc.idle_timeout:
                self.retire(f"idle for {self.svc.idle_timeout}s")
            elif now - rep.started > self.svc.max_lifetime - 120:
                self.retire("max lifetime reached")
            elif now - last_check >= self.health_interval:
                last_check = now
                if not self._healthy(rep):
                    self.retire("health check failed", failed=True)

    def status(self) -> dict:
        rep = self.replica
        return {"service": self.task.name,
                "replica": None if not rep else {"provider": rep.provider, "gpu": self.gpu, "url": rep.url,
                                                 "uptime_s": round(time.time() - rep.started)},
                "idle_s": round(time.time() - self.last_request), "history": self.history[-20:]}

    def shutdown(self):
        self.stopping.set()
        self.retire("proxy shutting down")

    # ------------------------------------------------------------ forwarding
    def forward(self, method, path, headers, body) -> tuple[int, list, bytes]:
        self.last_request = time.time()
        for attempt in (1, 2):
            rep = self.ensure()
            try:
                return _http(method, rep.url + path, {**headers, **rep.headers}, body)
            except (ConnectionError, OSError, http.client.HTTPException) as e:
                if attempt == 2:
                    raise
                self.log(f"  replica on {rep.provider} unreachable ({e}); failing over")
                self.retire("unreachable", failed=True)  # records a failure: the provider cools down
        raise NoReplica("unreachable")


def _http(method, url, headers, body, timeout=600):
    u = urllib.parse.urlsplit(url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port, timeout=timeout)
    target = (u.path or "/") + (f"?{u.query}" if u.query else "")
    try:
        conn.request(method, target, body=body, headers=headers)
        r = conn.getresponse()
        data = r.read()
        return r.status, [(k, v) for k, v in r.getheaders() if k.lower() not in HOP_HEADERS], data
    finally:
        conn.close()


def make_server(manager: ServiceManager, host="127.0.0.1", port=9000) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _handle(self):
            if self.path == "/_freesky/status":
                return self._reply(200, [("Content-Type", "application/json")],
                                   json.dumps(manager.status()).encode())
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
            try:
                code, hdrs, data = manager.forward(self.command, self.path, headers, body)
            except NoReplica as e:
                return self._reply(503, [("Content-Type", "application/json")],
                                   json.dumps({"error": str(e)}).encode())
            except Exception as e:
                return self._reply(502, [("Content-Type", "application/json")],
                                   json.dumps({"error": f"upstream: {e}"}).encode())
            self._reply(code, hdrs, data)

        def _reply(self, code, headers, data):
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    return ThreadingHTTPServer((host, port), Handler)


def serve_forever(manager: ServiceManager, host="127.0.0.1", port=9000, warm=False):
    srv = make_server(manager, host, port)
    threading.Thread(target=manager.monitor, daemon=True).start()
    if warm:
        threading.Thread(target=lambda: _safe(manager.ensure, manager.log), daemon=True).start()
    manager.log(f"freesky endpoint on http://{host}:{port} (status: /_freesky/status)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        manager.shutdown()


def _safe(fn, log):
    try:
        fn()
    except Exception as e:
        log(f"  warm start failed: {e}")
