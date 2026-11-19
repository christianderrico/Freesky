"""freesky: run jobs and endpoints on whichever free GPU is available right now.

  freesky init                      write ~/.freesky/config.yaml
  freesky providers [-v]            world catalog + which providers are ready here
  freesky plan task.yaml            the order providers would be tried in, and why others are skipped
  freesky launch task.yaml          run it, failing over across free providers (resumes long jobs)
  freesky serve task.yaml           local endpoint backed by a warm replica on a free GPU
  freesky quota                     free quota left per provider
  freesky history [-n 20]           past attempts
  freesky reset [provider]          clear cooldowns
  freesky dashboard                 local web dashboard
  freesky mcp                       MCP server (stdio) for coding agents
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .catalog import CATALOG, STATUS_ORDER
from .providers.base import Provider
from .ledger import Ledger, home
from .router import POLICIES, Router, build_providers, load_config
from .spec import load_task

CONFIG_TEMPLATE = """\
# freesky providers. With policy "order" they are tried in this order.
policy: fastest          # fastest | save-scarce | order

providers:
  modal: {}              # pip install modal && modal setup
  kaggle: {}             # kaggle auth login (then oauth: true) or KAGGLE_API_TOKEN
  colab:
    auth: adc            # gcloud auth application-default login --scopes=...colaboratory
    quota_limit: 12      # estimated hours/week (Colab does not publish its free quota)
  lightning:
    teamspace: my-teamspace   # + LIGHTNING_USER_ID / LIGHTNING_API_KEY
  local:
    allow_cpu_fallback: true  # tasks with accelerator: any may run on CPU as a last resort

  # beam: {}            # pip install beam-client && beam configure (free signup hours)

  # cloud trial credits through SkyPilot (one entry per account)
  # gcp-trial:
  #   type: skypilot
  #   quota_limit: 300
  #   quota_window: total
  #   resources: {infra: gcp}
"""


def make_router(config_path=None, policy=None, log=print) -> Router:
    cfg = load_config(config_path)
    providers = build_providers(cfg)
    return Router(providers, Ledger(), policy=policy or cfg.get("policy", "fastest"),
                  order=[p.name for p in providers], log=log)


def quota_rows(config_path=None) -> list:
    cfg = load_config(config_path)
    led = Ledger()
    rows = []
    for p in build_providers(cfg):
        left, reason = led.cooldown(p.name)
        ok, why = p.available()
        row = {"provider": p.name, "ready": ok, "why": why, "cooldown_min": int(left // 60),
               "cooldown_reason": reason, "unit": None, "window": None, "used": None,
               "remaining": None, "limit": None, "serve": p.can_serve,
               "session_limit_h": round(p.session_limit_s / 3600, 1) if p.session_limit_s else None}
        if p.quota:
            used = led.used(p.name, p.quota.window)
            # real quota APIs need credentials: without them fall back to the local ledger
            rem = p.remaining(used) if ok else Provider.remaining(p, used)
            row.update(unit=p.quota.unit, window=p.quota.window, used=round(used, 3),
                       remaining=round(rem, 3) if rem is not None else None, limit=p.quota.limit)
        rows.append(row)
    return rows


def cmd_init(args):
    path = args.config or os.path.join(home(), "config.yaml")
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
        return 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(CONFIG_TEMPLATE)
    print(f"wrote {path}")
    return 0


def cmd_providers(args):
    cfg = load_config(args.config)
    ready = {}
    for p in build_providers(cfg):
        ok, why = p.available()
        ready[p.name] = "ready" if ok else f"not ready: {why}"
    rows = sorted(CATALOG, key=lambda r: (STATUS_ORDER[r[1]], r[0]))
    print(f"{'provider':<22}{'status':<13}{'region':<8}{'free offer':<40}local setup")
    for name, status, region, offer, how, note in rows:
        local = ready.get(name, "")
        if not local and status in ("supported", "experimental"):
            local = "not configured"
        print(f"{name:<22}{status:<13}{region:<8}{offer[:38]:<40}{local}")
        if args.verbose:
            print(f"{'':<22}via: {how}; notes: {note}")
    for n in [n for n in ready if n not in {r[0] for r in CATALOG}]:
        print(f"{n:<22}{'config':<13}{'-':<8}{'(custom entry)':<40}{ready[n]}")
    return 0


def _header(task, r):
    extra = f", resumable via {task.checkpoint}/" if task.checkpoint else ""
    print(f"task {task.name}: {task.resources.accelerator}, max {task.resources.max_runtime}s{extra} "
          f"(policy {r.policy})")


def cmd_plan(args):
    task = load_task(args.task)
    r = make_router(args.config, args.policy)
    _header(task, r)
    if task.service:
        print("(endpoint: only providers that can expose a port)")
        r.describe(r.plan(task, args.provider, serving=True))
    else:
        r.run(task, dry_run=True, only=args.provider)
    return 0


def cmd_launch(args):
    task = load_task(args.task)
    r = make_router(args.config, args.policy)
    _header(task, r)
    out = r.run(task, out_dir=args.out, only=args.provider)
    if len(out.segments) > 1:
        chain = " -> ".join(f"{s['provider']}" for s in out.segments)
        print(f"segments: {chain}")
    if out.ok:
        print(f"OK on {out.provider} ({out.gpu or 'cpu'}) in {out.seconds:.0f}s -> {out.outputs_dir}")
        return 0
    print(f"FAILED: {out.message}")
    if out.outputs_dir:
        print(f"logs and partial outputs in {out.outputs_dir}")
    return out.exit_code if out.exit_code > 0 else 2


def cmd_serve(args):
    from .serve import ServiceManager, serve_forever
    task = load_task(args.task)
    r = make_router(args.config, args.policy)
    if not task.service:
        print("error: the task has no `service:` block", file=sys.stderr)
        return 2
    m = ServiceManager(r, task, only=args.provider)
    serve_forever(m, host=args.host, port=args.port, warm=args.warm)
    return 0


def cmd_quota(args):
    print(f"{'provider':<14}{'used':>9}{'left':>9}{'limit':>8}  unit/window      session  state")
    for row in quota_rows(args.config):
        state = (f"cooldown {row['cooldown_min']} min" if row["cooldown_min"] else
                 ("ok" if row["ready"] else "not ready"))
        sess = f"{row['session_limit_h']}h" if row["session_limit_h"] else "-"
        if row["unit"] is None:
            print(f"{row['provider']:<14}{'-':>9}{'-':>9}{'-':>8}  unlimited        {sess:<8} {state}")
            continue
        left = f"{row['remaining']:.2f}" if row["remaining"] is not None else "?"
        if not row["ready"]:
            left += "*"
        print(f"{row['provider']:<14}{row['used']:>9.2f}{left:>9}{row['limit']:>8.1f}  "
              f"{row['unit'] + '/' + row['window']:<16} {sess:<8} {state}")
    return 0


def cmd_history(args):  # noqa: D103
    rows = Ledger().attempts()[-args.n:]
    for a in rows:
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(a.ts))
        status = "ok" if a.ok else (a.error_kind or "error")
        print(f"{t}  {a.provider:<12} {a.task[:20]:<20} {a.seconds:>7.0f}s  {a.cost:>7.3f}  {status} {a.error[:60]}")
    if not rows:
        print("no attempts recorded yet")
    return 0


def cmd_reset(args):
    Ledger().clear(args.provider)
    print("cooldowns cleared" + (f" for {args.provider}" if args.provider else ""))
    return 0


def cmd_dashboard(args):
    from .dashboard import serve_dashboard
    serve_dashboard(args.config, host=args.host, port=args.port)
    return 0


def cmd_mcp(args):
    from .mcp_server import main as mcp_main
    return mcp_main(args.config)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="freesky", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config path (default ~/.freesky/config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write a config template")
    s.add_argument("--force", action="store_true")
    s.set_defaults(f=cmd_init)
    s = sub.add_parser("providers", help="world catalog of free GPUs")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(f=cmd_providers)
    for name, f, hlp in (("plan", cmd_plan, "dry run"), ("launch", cmd_launch, "run a task"),
                         ("serve", cmd_serve, "serve an endpoint")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("task")
        s.add_argument("-p", "--provider", action="append", help="restrict to these providers (repeatable)")
        s.add_argument("--policy", choices=POLICIES)
        if name == "serve":
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=9000)
            s.add_argument("--warm", action="store_true", help="start a replica now instead of on first request")
        else:
            s.add_argument("--out", default="freesky-runs")
        s.set_defaults(f=f)
    s = sub.add_parser("quota", help="free quota left")
    s.set_defaults(f=cmd_quota)
    s = sub.add_parser("history", help="past attempts")
    s.add_argument("-n", type=int, default=20)
    s.set_defaults(f=cmd_history)
    s = sub.add_parser("reset", help="clear cooldowns")
    s.add_argument("provider", nargs="?")
    s.set_defaults(f=cmd_reset)
    s = sub.add_parser("dashboard", help="local web dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8790)
    s.set_defaults(f=cmd_dashboard)
    s = sub.add_parser("mcp", help="MCP server over stdio")
    s.set_defaults(f=cmd_mcp)

    args = ap.parse_args(argv)
    if not hasattr(args, "policy"):
        args.policy = None
    try:
        return args.f(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
