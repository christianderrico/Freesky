"""Minimal resumable loop. The contract with freesky:
- read/write state in $FREESKY_CHECKPOINT_DIR (it comes back on the next segment);
- on SIGTERM, save and exit quickly (you get a grace period, 120 s by default).
Replace the loop body with your training step and torch.save/torch.load.
"""
import json
import os
import signal
import sys
import time

CKPT = os.path.join(os.environ.get("FREESKY_CHECKPOINT_DIR", "ckpt"), "state.json")
TOTAL_STEPS = 100_000

state = {"step": 0}
if os.path.exists(CKPT):
    state = json.load(open(CKPT))
    print(f"resuming from step {state['step']} on {os.environ.get('FREESKY_PROVIDER')}", flush=True)


def save():
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tmp = CKPT + ".tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, CKPT)          # atomic: a kill mid-write never corrupts the checkpoint


def on_term(*_):
    save()
    print(f"stopped at step {state['step']}, checkpoint saved", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, on_term)
while state["step"] < TOTAL_STEPS:
    time.sleep(1)                  # your training step here
    state["step"] += 1
    if state["step"] % 300 == 0:
        save()                     # periodic saves also protect against hard preemption
save()
os.makedirs("out", exist_ok=True)
json.dump(state, open("out/final.json", "w"))
