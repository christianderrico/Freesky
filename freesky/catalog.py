"""World catalog of free GPU sources (snapshot November 2026; free tiers change often).

status:
  supported     adapter implemented, can be routed to
  experimental  adapter implemented, not battle-tested
  planned       programmable in principle, adapter not written yet
  manual        free GPU exists but no API/CLI to submit jobs: listed for reference
  closed        no longer available to new users
"""

CATALOG = [
    # name, status, region, what you get, how it is driven, notes
    ("local", "supported", "-", "la tua GPU o CPU", "subprocess", "sempre gratis, ultima risorsa"),
    ("modal", "supported", "global", "$30/mese di credito (T4..H100)", "SDK Python, Sandbox",
     "scala a zero, avvio ~40 s"),
    ("kaggle", "supported", "global", "~30 h GPU/settimana, T4 x2", "API kaggle (kernels push)",
     "quota reale via API; internet richiede verifica telefono"),
    ("colab", "supported", "global", "T4 quando disponibile (quota non pubblica)", "Colab CLI ufficiale",
     "interrompibile; solo lavori a script"),
    ("lightning", "supported", "global", "~15 crediti/mese (~20 h T4)", "lightning-sdk (Studio)",
     "verifica telefono; Studio gratuito si riavvia ogni 4 h"),
    ("gcp-trial", "experimental", "global", "$300 per 90 giorni", "SkyPilot", "credito una tantum per nuovi account"),
    ("azure-trial", "experimental", "global", "$200 per 30 giorni", "SkyPilot", "una tantum"),
    ("oci-trial", "experimental", "global", "$300 per 30 giorni", "SkyPilot", "l'Always Free non ha GPU"),
    ("aws-free-tier", "experimental", "global", "$100 + fino a $100 guadagnati, 6 mesi", "SkyPilot",
     "quota GPU di default a zero: va richiesta"),
    ("hf-zerogpu", "planned", "global", "GPU condivisa per chiamate brevi", "Gradio Space + gradio_client",
     "chiamate <120 s; ospitare uno Space ZeroGPU richiede PRO"),
    ("beam", "planned", "global", "15 h una tantum", "SDK/CLI Beam", "avvii a freddo non fatturati"),
    ("inferless", "planned", "global", "10 h / $30, senza carta", "API", "in fusione con Baseten"),
    ("cerebrium", "planned", "global", "crediti di benvenuto", "CLI", "importo da verificare"),
    ("baseten", "planned", "global", "crediti di benvenuto", "Truss CLI", "importo da verificare"),
    ("saturn-cloud", "manual", "global", "piano Hosted Free", "notebook web", "notebook pubblici; in integrazione con DigitalOcean"),
    ("intel-tiber", "manual", "global", "Intel Gaudi / Data Center GPU Max", "JupyterLab, batch",
     "stack oneAPI/XPU, non CUDA"),
    ("paperspace", "manual", "global", "macchine gratuite quasi mai disponibili", "notebook web",
     "di fatto serve il piano Pro"),
    ("baidu-ai-studio", "manual", "cina", "GPU V100 gratuite a ore giornaliere", "notebook web (PaddlePaddle)",
     "registrazione con numero cinese; quote da verificare"),
    ("modelscope", "manual", "cina", "ore GPU gratuite su notebook", "notebook/Studio web",
     "Alibaba; quote variabili"),
    ("tianchi", "manual", "cina", "notebook GPU gratuiti", "notebook web", "Alibaba; spazio disco ridotto"),
    ("google-trc", "manual", "global", "TPU v2-v5 per ~30 giorni", "su domanda", "serve una proposta di ricerca"),
    ("github-student", "manual", "global", "Azure $100 senza carta e altri crediti", "SkyPilot (Azure)",
     "solo studenti verificati"),
    ("sagemaker-studio-lab", "closed", "global", "T4, 4 h/giorno", "notebook web",
     "chiuso ai nuovi utenti dal 30/07/2026"),
]

STATUS_ORDER = {"supported": 0, "experimental": 1, "planned": 2, "manual": 3, "closed": 4}
