# freesky

**Like SkyPilot, but only free GPUs.** Describe a job (or an endpoint) once; freesky runs it on whichever free tier has capacity right now — Modal, Kaggle, Colab, Lightning AI, Beam, cloud trial credits, your own machine — and fails over to the next one when a provider is out of GPUs, out of quota or logged out. Like OmniRoute does for LLM APIs, for GPU compute.

Illustrative run of a 40-hour job on free tiers:

```
$ freesky launch examples/resumable-training/task.yaml
task resumable-training: any, max 144000s, resumable via ckpt/ (policy fastest)
  -> modal        T4, ~0.12 usd of 29.41 left
  -> lightning    T4, ~2.31 credits of 14.20 left
  -> colab        T4, ~11.00 hours of 12.00 left
  -> kaggle       T4, ~11.73 hours of 25.50 left
> trying lightning (T4), segment 1, deadline 13800s
  segment 1 reached its deadline on lightning; checkpoint saved
> segment 2: resuming from the checkpoint
> trying kaggle (T4), segment 2, deadline 42600s
...
segments: lightning -> kaggle -> colab -> kaggle
OK on kaggle (Tesla T4) in 143212s -> freesky-runs/resumable-training-20261119-150416-seg4-kaggle
```

## What it does

- **Jobs** — one task YAML for every provider; outputs, logs and exit code come back to `freesky-runs/`.
- **Jobs longer than any free session** — add `checkpoint: ckpt`. freesky splits the run at each provider's session limit (Kaggle 12 h, Colab ~12 h, Lightning 4 h), sends SIGTERM so your code can save, brings the checkpoint back and resumes on whichever provider is free next.
- **Endpoints on free GPUs** — `freesky serve` gives you a local URL backed by one warm replica (Modal tunnels, Beam ports, or local). It starts on the first request, stops after `idle_timeout` so credit is only spent while used, and moves to another provider if the replica dies.
- **One quota pool** — a ledger of hours / dollars / credits per provider and window (day, week, month, one-off). Real remaining quota is read where the provider exposes it (Kaggle). Failing providers cool down with backoff by error kind (no capacity, quota, auth, transient).
- **Policies** — `fastest` (shortest start), `save-scarce` (spend the most abundant quota first), `order` (yours).
- **Never leaks a GPU** — sandboxes, sessions, studios and VMs are torn down at the end, including on errors.
- **Dashboard and agents** — `freesky dashboard` shows free compute left everywhere; `freesky mcp` lets Claude Code and other agents plan and launch jobs on free GPUs.
- **World catalog** — `freesky providers` lists 23 free GPU sources across the US, Europe and China with their status.

## How it compares

| | freesky | omnirun | SkyPilot |
|---|---|---|---|
| Focus | free tiers only | your clusters + free notebooks + marketplaces | paid clouds and Kubernetes |
| Modal, Lightning, Beam, cloud trial credits | yes | no | trials via its cloud backends |
| Resume across providers past session limits | yes (`checkpoint:`) | no (non-goal) | managed-job recovery from spot preemption |
| Endpoints with scale to zero + failover | yes (`serve`) | no | SkyServe (paid clouds) |
| Failover on a direct launch | yes | queue mode only | yes |
| Needs a pushed git commit | no (ships the workdir) | yes | no |
| Dashboard / agent integration | yes / MCP server | no / agent skill | yes / agent skill |

omnirun is excellent for research clusters (Slurm, SSH) and has a far more battle-tested scheduler; use it there. freesky is for getting real work done for $0.

## Install

```bash
pip install -e '.[all]'          # or pick extras: modal, kaggle, colab, lightning, beam, skypilot, mcp
freesky init                     # writes ~/.freesky/config.yaml
freesky providers                # what is ready on this machine
```

Log in once per provider, **with your own account** (free tiers are personal; sharing one account breaks their terms):

| Provider | Login |
|---|---|
| Modal | `modal setup` |
| Kaggle | `kaggle auth login` (then `oauth: true` in the config) or `KAGGLE_API_TOKEN` |
| Colab | `gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory` and `auth: adc` in the config |
| Lightning | `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY` (Settings → Keys) and `teamspace:` in the config |
| Beam | `beam configure` |
| Cloud trials | `pip install 'skypilot[gcp]' && sky check`, then one `type: skypilot` entry per account |

## Jobs

```yaml
name: train
resources:
  accelerator: any            # none | T4 | P100 | L4 | A10G | A100 | H100 | any
  min_gpu_memory_gb: 16
  max_runtime: 40h            # total budget across segments
  estimated_runtime: 40h      # used to estimate quota use
image: python:3.12-slim       # container-based providers (Modal, Beam)
workdir: .                    # shipped to the provider (up to 30 MB; download big data in setup)
setup: pip install -r requirements.txt
run: python train.py
outputs: [out]
checkpoint: ckpt              # optional: resume across sessions and providers
env: {HF_HOME: /tmp/hf}
providers: [modal, kaggle, colab, lightning, local]   # optional allow-list and order
```

Your code sees `FREESKY_PROVIDER`, `FREESKY_SEGMENT`, `FREESKY_CHECKPOINT_DIR` and `FREESKY_RESUME`. For resumable jobs: load state from the checkpoint dir if it exists, save periodically and on SIGTERM. See `examples/resumable-training/train.py`.

```bash
freesky plan task.yaml           # order of providers, and why the others are skipped
freesky launch task.yaml         # run with failover (and resume)
freesky quota                    # free quota used / left
freesky history                  # past attempts
freesky reset [provider]         # clear cooldowns
```

## Endpoints

```yaml
name: kev-endpoint
resources: {accelerator: any, min_gpu_memory_gb: 16}
setup: ...
run: python -m kev.serve --host 0.0.0.0 --port 8009
service:
  port: 8009
  health: /v1/models          # any answer below 500 counts as up
  ready_timeout: 25m
  idle_timeout: 10m
providers: [modal, beam, local]
```

```bash
freesky serve examples/kev-endpoint/task.yaml --port 9000 [--warm]
curl localhost:9000/v1/systemone -d '{...}'
curl localhost:9000/_freesky/status
```

Only providers whose free tier allows exposing a port are used for endpoints (Modal, Beam, local). Colab and Kaggle are batch-only. **Modal and Beam endpoint URLs are public**: anyone with the URL can call your model while the replica runs — put your own auth in the server if that matters.

## Dashboard and MCP

```bash
freesky dashboard                       # http://127.0.0.1:8790
claude mcp add freesky -- freesky mcp   # tools: list_providers, quota, plan, launch, history
```

## How it works

Every provider runs the same generated **Python driver**: it carries the workdir, runs `setup` and `run` with bash in their own process group, enforces the segment deadline (SIGTERM, then SIGKILL after a grace period), and packs outputs, checkpoint, log and `fgpu_result.json` into one archive. Adapters only have to upload a file, run Python, and download a file.

| Provider | Batch | Endpoint | Mechanism |
|---|---|---|---|
| `local` | ✓ | ✓ | subprocess; GPU detected with `nvidia-smi`; optional CPU fallback |
| `modal` | ✓ | ✓ | `modal.Sandbox` + filesystem API; endpoints through encrypted tunnels |
| `kaggle` | ✓ | | private script kernel (`kernels push`, T4), big workdirs via a private dataset, real quota via `quota_view` |
| `colab` | ✓ | | official Colab CLI: `new --gpu T4`, `upload`, `exec`, `download`, `stop` |
| `lightning` | ✓ | | `lightning_sdk.Studio`: start, upload, `run_with_exit_code`, download, stop |
| `beam` | ✓ | ✓ | `beam.Sandbox`: fs upload/download, `process.exec`, `expose_port` (experimental) |
| `skypilot` | ✓ | | `sky launch` on any cloud with trial credit, `rsync`, `sky down` (experimental) |

## World catalog (November 2026)

| Status | Providers |
|---|---|
| **supported** | local, Modal ($30/month), Kaggle (~30 GPU-h/week), Colab (T4 when available), Lightning AI (monthly credits) |
| **experimental** | Beam (signup hours), cloud trials via SkyPilot: GCP ($300/90 d), Azure ($200/30 d), OCI ($300/30 d), AWS Free Tier |
| **planned** | Hugging Face ZeroGPU, Inferless, Cerebrium, Baseten |
| **manual** (no job API) | Saturn Cloud, Intel Tiber AI Cloud, Paperspace, Baidu AI Studio, ModelScope, Tianchi, Google TPU Research Cloud, GitHub Student Pack |
| **closed** | SageMaker Studio Lab (closed to new users since 2026-07-30) |

Free tiers change often; default limits are estimates you can override per provider (`quota_limit`, `rates`, `session_limit`).

## Rules

freesky uses each free tier the way its provider intends: script jobs started and stopped with official tools, no always-on services on free notebooks, no mining, **one account per person per provider**. Trial credits are real money: freesky always tears VMs down, but check your console.

## Status

v0.2. The core (spec, ledger, router, segments and resume, endpoints and failover, CLI, dashboard, MCP) is tested end to end with real execution locally. The Modal, Kaggle, Colab, Lightning and Beam adapters are written against the current SDKs (modal 1.6, kaggle 2.2, lightning-sdk 2026.11, google-colab-cli 0.8, beam-client 0.2) and tested against simulators that really execute the driver — **not yet against real accounts**. SkyPilot is experimental.

Next: real-account test matrix, real quota readers for Modal and Lightning, HF ZeroGPU adapter, periodic checkpoint sync during a segment (to survive hard preemption), more than one replica per endpoint.

```bash
pip install -e '.[dev]' && pytest
```

## Sources for the catalog

- Modal pricing (Starter, $30/month): https://modal.com/pricing
- Colab CLI: https://github.com/googlecolab/google-colab-cli
- Kaggle API (kernels, quota): https://github.com/Kaggle/kaggle-api
- Free tier overviews 2026: https://aimultiple.com/free-cloud-gpu, https://gpuperhour.com/blog/free-cloud-gpus-and-credits, https://amitpandeytiktok.github.io/awesome-free-compute-2026/compute.html
- SageMaker Studio Lab closing: https://dev.to/billiem/free-compute-worth-claiming-in-august-2026-154d
- omnirun: https://github.com/flyingleafe/omnirun

License: Apache-2.0.
