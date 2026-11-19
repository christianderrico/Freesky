"""freesky: run a task on whichever free GPU is available right now.

  freesky init                      write ~/.freesky/config.yaml
  freesky providers                 world catalog + which ones are ready on this machine
  freesky plan task.yaml            show the order in which providers would be tried
  freesky launch task.yaml          run it (failover across free providers)
  freesky quota                     free quota left per provider
  freesky history [-n 20]           past attempts
  freesky reset [provider]          clear cooldowns
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .catalog import CATALOG, STATUS_ORDER
from .ledger import HOME, Ledger, window_start
from .router import POLICIES, Router, build_providers, load_config
from .spec import load_task

CONFIG_TEMPLATE = """\
# freesky - fornitori di GPU gratuite, in ordine di preferenza per la politica "order"
policy: fastest          # fastest | save-scarce | order

providers:
  modal: {}              # pip install modal && modal setup
  kaggle: {}             # kaggle auth login  (poi oauth: true)  oppure KAGGLE_API_TOKEN
  colab:
    auth: adc            # gcloud auth application-default login --scopes=...colaboratory
    quota_limit: 12      # ore/settimana stimate (Colab non pubblica la quota)
  lightning:
    teamspace: my-teamspace   # + LIGHTNING_USER_ID / LIGHTNING_API_KEY
  local:
    allow_cpu_fallback: true  # task con accelerator: any possono girare su CPU come ultima risorsa

  # crediti di prova dei cloud, via SkyPilot (una voce per account)
  # gcp-trial:
  #   type: skypilot
  #   quota_limit: 300
  #   quota_window: total
  #   resources: {infra: gcp}
"""


def _router(args) -> Router:
    cfg = load_config(args.config)
    providers = build_providers(cfg)
    order = [p.name for p in providers]
    return Router(providers, Ledger(), policy=args.policy or cfg.get("policy", "fastest"), order=order)


def cmd_init(args):
    path = args.config or os.path.join(HOME, "config.yaml")
    if os.path.exists(path) and not args.force:
        print(f"{path} esiste gia' (usa --force per sovrascrivere)")
        return 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(CONFIG_TEMPLATE)
    print(f"scritto {path}")
    return 0


def cmd_providers(args):
    cfg = load_config(args.config)
    ready = {}
    for p in build_providers(cfg):
        ok, why = p.available()
        ready[p.name] = "pronto" if ok else f"non pronto: {why}"
    rows = sorted(CATALOG, key=lambda r: (STATUS_ORDER[r[1]], r[0]))
    print(f"{'fornitore':<22}{'stato':<13}{'regione':<8}{'cosa offre':<44}configurazione locale")
    for name, status, region, offer, _how, note in rows:
        local = ready.get(name, "")
        if not local and status in ("supported", "experimental"):
            local = "non configurato"
        print(f"{name:<22}{status:<13}{region:<8}{offer[:42]:<44}{local}")
        if args.verbose:
            print(f"{'':<22}come: {_how}; note: {note}")
    extra = [n for n in ready if n not in {r[0] for r in CATALOG}]
    for n in extra:
        print(f"{n:<22}{'config':<13}{'-':<8}{'(voce personalizzata)':<44}{ready[n]}")
    return 0


def cmd_plan(args):
    task = load_task(args.task)
    r = _router(args)
    print(f"task {task.name}: {task.resources.accelerator}, max {task.resources.max_runtime}s "
          f"(politica {r.policy})")
    r.run(task, dry_run=True, only=args.provider)
    return 0


def cmd_launch(args):
    task = load_task(args.task)
    r = _router(args)
    print(f"task {task.name}: {task.resources.accelerator}, max {task.resources.max_runtime}s "
          f"(politica {r.policy})")
    out = r.run(task, out_dir=args.out, only=args.provider)
    if out.ok:
        print(f"OK su {out.provider} ({out.gpu or 'cpu'}) in {out.seconds:.0f}s -> {out.outputs_dir}")
        return 0
    print(f"FALLITO: {out.message}")
    if out.outputs_dir:
        print(f"log e output parziali in {out.outputs_dir}")
    return out.exit_code if out.exit_code > 0 else 2


def cmd_quota(args):
    cfg = load_config(args.config)
    led = Ledger()
    print(f"{'fornitore':<14}{'usato':>10}{'restante':>11}{'limite':>9}  unita'/finestra   stato")
    for p in build_providers(cfg):
        left, reason = led.cooldown(p.name)
        state = f"pausa {int(left // 60)} min" if left > 0 else "ok"
        if not p.quota:
            print(f"{p.name:<14}{'-':>10}{'-':>11}{'-':>9}  illimitato        {state}")
            continue
        used = led.used(p.name, p.quota.window)
        rem = p.remaining(used) if p.available()[0] else None
        rem_s = f"{rem:.2f}" if rem is not None else "?"
        print(f"{p.name:<14}{used:>10.2f}{rem_s:>11}{p.quota.limit:>9.1f}  "
              f"{p.quota.unit}/{p.quota.window:<9} {state}")
    return 0


def cmd_history(args):
    led = Ledger()
    rows = led.attempts()[-args.n:]
    for a in rows:
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(a.ts))
        status = "ok" if a.ok else (a.error_kind or "errore")
        print(f"{t}  {a.provider:<12} {a.task:<20} {a.seconds:>7.0f}s  {a.cost:>7.3f}  {status} {a.error[:60]}")
    if not rows:
        print("nessun tentativo registrato")
    return 0


def cmd_reset(args):
    Ledger().clear(args.provider)
    print("pause azzerate" + (f" per {args.provider}" if args.provider else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="freesky", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="percorso config (default ~/.freesky/config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init"); s.add_argument("--force", action="store_true"); s.set_defaults(f=cmd_init)
    s = sub.add_parser("providers"); s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(f=cmd_providers)
    for name, f in (("plan", cmd_plan), ("launch", cmd_launch)):
        s = sub.add_parser(name)
        s.add_argument("task")
        s.add_argument("-p", "--provider", action="append", help="limita a questi fornitori (ripetibile)")
        s.add_argument("--policy", choices=POLICIES)
        s.add_argument("--out", default="freesky-runs")
        s.set_defaults(f=f)
    s = sub.add_parser("quota"); s.set_defaults(f=cmd_quota)
    s = sub.add_parser("history"); s.add_argument("-n", type=int, default=20); s.set_defaults(f=cmd_history)
    s = sub.add_parser("reset"); s.add_argument("provider", nargs="?"); s.set_defaults(f=cmd_reset)

    args = ap.parse_args(argv)
    if not hasattr(args, "policy"):
        args.policy = None
    try:
        return args.f(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"errore: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
