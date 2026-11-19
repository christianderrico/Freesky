"""World catalog of free GPU sources (snapshot November 2026; free tiers change often).

status:
  supported     adapter implemented, routable
  experimental  adapter implemented, not battle-tested
  planned       programmable in principle, adapter not written yet
  manual        free GPU exists but no API/CLI to submit jobs: listed for reference
  closed        no longer available to new users
"""

CATALOG = [
    # name, status, region, what you get, how it is driven, notes
    ("local", "supported", "-", "your own GPU or CPU", "subprocess", "always free; last resort"),
    ("modal", "supported", "global", "$30/month credit (T4..H100)", "Python SDK, Sandbox + tunnels",
     "scales to zero, ~40 s cold start; batch and endpoints"),
    ("kaggle", "supported", "global", "~30 GPU-h/week, T4 x2", "kaggle API (kernels push)",
     "real quota via API; 12 h sessions; internet needs phone verification"),
    ("colab", "supported", "global", "T4 when available (quota not published)", "official Colab CLI",
     "preemptible; script jobs only"),
    ("lightning", "supported", "global", "~15 credits/month (~20 T4-h)", "lightning-sdk (Studio)",
     "phone verification; free Studio restarts every 4 h"),
    ("gcp-trial", "experimental", "global", "$300 for 90 days", "SkyPilot", "one-off credit for new accounts"),
    ("azure-trial", "experimental", "global", "$200 for 30 days", "SkyPilot", "one-off"),
    ("oci-trial", "experimental", "global", "$300 for 30 days", "SkyPilot", "Always Free has no GPU"),
    ("aws-free-tier", "experimental", "global", "$100 + up to $100 earned, 6 months", "SkyPilot",
     "default GPU quota is zero: request it"),
    ("hf-zerogpu", "planned", "global", "shared GPU for short calls", "Gradio Space + gradio_client",
     "calls < 120 s; hosting a ZeroGPU Space needs PRO"),
    ("beam", "experimental", "global", "free signup hours (~15 h, one-off)", "beam-client Sandbox + expose_port",
     "batch and endpoints; verify the current offer"),
    ("inferless", "planned", "global", "10 h / $30, no card", "API", "joining Baseten"),
    ("cerebrium", "planned", "global", "signup credits", "CLI", "amount to verify"),
    ("baseten", "planned", "global", "signup credits", "Truss CLI", "amount to verify"),
    ("saturn-cloud", "manual", "global", "Hosted Free plan", "web notebooks", "public notebooks; moving into DigitalOcean"),
    ("intel-tiber", "manual", "global", "Intel Gaudi / Data Center GPU Max", "JupyterLab, batch",
     "oneAPI/XPU stack, not CUDA"),
    ("paperspace", "manual", "global", "free machines rarely available", "web notebooks", "in practice needs Pro"),
    ("baidu-ai-studio", "manual", "china", "free V100 hours daily", "web notebooks (PaddlePaddle)",
     "Chinese phone number; quotas to verify"),
    ("modelscope", "manual", "china", "free notebook GPU hours", "web notebooks / Studio", "Alibaba; variable quotas"),
    ("tianchi", "manual", "china", "free GPU notebooks", "web notebooks", "Alibaba; small disk quota"),
    ("google-trc", "manual", "global", "TPU v2-v5 for ~30 days", "application", "needs a research proposal"),
    ("github-student", "manual", "global", "Azure $100 without card, other credits", "SkyPilot (Azure)",
     "verified students only"),
    ("sagemaker-studio-lab", "closed", "global", "T4, 4 h/day", "web notebooks",
     "closed to new users since 2026-07-30"),
]

STATUS_ORDER = {"supported": 0, "experimental": 1, "planned": 2, "manual": 3, "closed": 4}
