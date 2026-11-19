# freesky

**Come SkyPilot, ma solo con GPU gratuite.** Descrivi il lavoro in un file YAML. freesky lo esegue sul primo fornitore gratuito disponibile (Modal, Kaggle, Colab, Lightning, i crediti di prova dei cloud, il tuo PC). Se un fornitore non ha GPU libere, ha finito la quota o dà errore, passa al successivo, come fa OmniRoute con i modelli LLM.

```
$ freesky launch examples/kev-triage/task.yaml
task kev-triage: any, max 2400s (politica fastest)
  -> modal        T4, stima 0.14 usd, restano 29.41
  -> lightning    T4, stima 0.16 credits, restano 14.20
  -> colab        T4, stima 0.23 hours, restano 12.00
  -> kaggle       T4, stima 0.27 hours, restano 25.50
     local        nessuna GPU adatta (any, >= 16 GB)
> provo modal (T4)
  x modal: capacity - no GPU available
> provo lightning (T4)
OK su lightning (T4) in 412s -> freesky-runs/kev-triage-20261119-150416-lightning
```

## Perché

Le GPU gratuite ci sono, ma sono sparse: un po' di credito qui, qualche ora alla settimana là, ognuna con la sua CLI, le sue quote e i suoi orari di disponibilità. freesky le tratta come un'unica riserva:

- **un solo formato di lavoro** (`task.yaml`) per tutti i fornitori;
- un **registro delle quote** (ore, dollari, crediti) per fornitore e per finestra (giorno, settimana, mese, una tantum). Dove il fornitore la espone, legge la quota reale: per esempio Kaggle tramite API;
- **pause automatiche** quando un fornitore fallisce. La durata dipende dal tipo di errore (nessuna GPU libera, quota esaurita, login scaduto, errore di rete) e cresce a ogni fallimento consecutivo;
- **tre politiche** di scelta: `fastest` (avvio più rapido), `save-scarce` (consuma prima la quota più abbondante), `order` (l'ordine che decidi tu);
- **pulizia garantita**: le VM e i sandbox vengono sempre spenti a fine lavoro, anche in caso di errore.

## Installazione

```bash
pip install -e '.[all]'          # oppure solo gli extra che ti servono: modal, kaggle, colab, lightning, skypilot
freesky init                     # scrive ~/.freesky/config.yaml
freesky providers                # catalogo mondiale + cosa è pronto su questa macchina
```

Login, una volta per fornitore:

| Fornitore | Login |
|---|---|
| Modal | `modal setup` |
| Kaggle | `kaggle auth login` (poi `oauth: true` nella config) oppure `KAGGLE_API_TOKEN` |
| Colab | `gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory` e `auth: adc` nella config |
| Lightning | `LIGHTNING_USER_ID` e `LIGHTNING_API_KEY` (Settings → Keys) e `teamspace:` nella config |
| Cloud trial | `pip install 'skypilot[gcp]' && sky check`, poi una voce `type: skypilot` per account |

## Il task

```yaml
name: kev-triage
resources:
  accelerator: any            # none | T4 | P100 | L4 | A10G | A100 | H100 | any
  min_gpu_memory_gb: 16
  max_runtime: 40m            # limite rigido, usato anche per stimare la quota
  estimated_runtime: 12m
image: python:3.12-slim       # per i fornitori a container (Modal)
workdir: .                    # caricata sul fornitore (fino a 30 MB; i dati grossi scaricali in setup)
setup: pip install -r requirements.txt
run: python score.py
outputs: [out]                # riportati in freesky-runs/<task>-<data>-<fornitore>/outputs/
env: {HF_HOME: /tmp/hf}
providers: [modal, kaggle, colab, lightning, local]   # facoltativo: elenco consentito e ordine
```

Comandi:

```bash
freesky plan task.yaml           # in che ordine proverebbe i fornitori, e perché scarta gli altri
freesky launch task.yaml         # esegue, con failover
freesky launch task.yaml -p kaggle --policy save-scarce
freesky quota                    # quota usata e restante per fornitore
freesky history                  # tentativi passati
freesky reset [fornitore]        # azzera le pause
```

## Come funziona

Ogni fornitore esegue la stessa cosa: un unico **driver Python** generato da freesky. Il driver contiene la workdir (incorporata oppure caricata accanto), la estrae, esegue `setup` e `run` con bash, poi impacchetta gli output, il log e un `fgpu_result.json` con codice di uscita, tempi e GPU vista. Così gli adattatori restano sottili: devono solo saper caricare un file, eseguire Python e scaricare un file.

| Fornitore | Come lo usa freesky |
|---|---|
| `local` | sottoprocesso; GPU rilevata con `nvidia-smi`; con `allow_cpu_fallback` la CPU diventa l'ultima risorsa |
| `modal` | `modal.Sandbox` con la GPU richiesta; i file passano dal filesystem del sandbox; terminato sempre |
| `kaggle` | kernel privato di tipo script (`kernels push` con acceleratore T4); workdir oltre 512 KB in un dataset privato; attesa con `kernels status` e download con `kernels output`; quota reale con `quota_view` |
| `colab` | CLI ufficiale di Google: `new --gpu T4`, `upload`, `exec`, `download`, `stop` |
| `lightning` | `lightning_sdk.Studio`: `start(Machine.T4)`, `upload_file`, `run_with_exit_code`, `download_file`, `stop` |
| `skypilot` | `sky launch` del driver su qualsiasi cloud con crediti di prova, `rsync` degli output, `sky down` |

Il router scarta un fornitore se non ha la GPU richiesta, se mancano le credenziali, se è in pausa o se la quota restante è inferiore alla stima (durata stimata più avvio, moltiplicate per il costo orario della GPU). Ordina i candidati secondo la politica scelta e li prova uno dopo l'altro. Se il **tuo** comando fallisce (codice di uscita diverso da zero), freesky si ferma lì: non consuma la quota di altri fornitori per un errore che si ripeterebbe uguale, e non mette in pausa il fornitore.

## Catalogo mondiale (novembre 2026)

`freesky providers -v` mostra la tabella completa. In sintesi:

| Stato | Fornitori |
|---|---|
| **supported** | local, Modal ($30/mese), Kaggle (~30 h GPU/settimana), Colab (T4 quando disponibile), Lightning AI (crediti mensili) |
| **experimental** | crediti di prova cloud via SkyPilot: GCP ($300/90 gg), Azure ($200/30 gg), OCI ($300/30 gg), AWS Free Tier |
| **planned** | Hugging Face ZeroGPU, Beam, Inferless, Cerebrium, Baseten |
| **manual** (niente API per lanciare lavori) | Saturn Cloud, Intel Tiber AI Cloud, Paperspace, Baidu AI Studio, ModelScope, Tianchi, Google TPU Research Cloud, GitHub Student Pack |
| **closed** | SageMaker Studio Lab (chiuso ai nuovi utenti dal 30/07/2026) |

Le quote gratuite cambiano spesso. I limiti predefiniti sono stime da verificare, e puoi correggerli per fornitore con `quota_limit:` nella config.

## Regole d'uso

freesky usa ogni piano gratuito **come previsto da chi lo offre**: lavori a script avviati e chiusi con gli strumenti ufficiali, niente servizi sempre accesi sui notebook gratuiti, niente mining. Usa **un account per fornitore**: creare più account per moltiplicare la quota viola i termini di servizio di tutti. I crediti cloud sono soldi veri, quindi freesky spegne sempre le VM, ma controlla comunque la console del fornitore.

## Stato del progetto

- Il core (spec, registro delle quote, router, failover, CLI) e l'adattatore `local` sono testati, anche con esecuzione reale.
- Gli adattatori Modal, Kaggle, Colab e Lightning sono scritti sulle API attuali degli SDK (Modal 1.6, kaggle 2.2, lightning-sdk 2026.11, google-colab-cli 0.8). Sono testati contro simulatori che eseguono davvero il driver, **non ancora contro account reali**.
- SkyPilot è sperimentale.

Prossimi passi: provare gli adattatori su account veri, leggere la quota reale anche da Modal e Lightning, scrivere l'adattatore ZeroGPU, aggiungere una modalità `serve` per endpoint che si spengono da soli (oggi c'è solo su Modal).

```bash
pip install -e '.[dev]' && pytest
```

## Fonti del catalogo

- Modal Starter, $30/mese: https://modal.com/pricing
- Colab CLI: https://github.com/googlecolab/google-colab-cli
- Kaggle API (kernels, quota): https://github.com/Kaggle/kaggle-api
- Panoramica dei piani gratuiti 2026: https://aimultiple.com/free-cloud-gpu, https://gpuperhour.com/blog/free-cloud-gpus-and-credits, https://amitpandeytiktok.github.io/awesome-free-compute-2026/compute.html
- Chiusura di SageMaker Studio Lab: https://dev.to/billiem/free-compute-worth-claiming-in-august-2026-154d

Licenza: Apache-2.0.
