"""Local web dashboard: free quota left everywhere, readiness, cooldowns, recent runs, world catalog.

    freesky dashboard        # http://127.0.0.1:8790
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .catalog import CATALOG
from .ledger import Ledger

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>freesky</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--text:#e6e8ee;--mute:#8a92a6;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
@media (max-width:600px){header{flex-wrap:wrap;padding:16px}main{padding:16px}}
@media (prefers-color-scheme: light){:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6eb;--text:#1b1f24;--mute:#5f6b7a}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--text)}
header{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px}
h1{margin:0;font-size:20px}header span{color:var(--mute)}
main{padding:20px 24px;display:grid;grid-template-columns:minmax(0,1fr);gap:20px;max-width:1200px}main>*{min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .small{overflow-wrap:anywhere}.card h3{margin:0 0 6px;font-size:15px;display:flex;justify-content:space-between}
.pill{font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--mute)}
.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.warn{color:var(--warn);border-color:var(--warn)}.pill.bad{color:var(--bad);border-color:var(--bad)}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;margin:8px 0 4px}.bar i{display:block;height:100%;background:var(--acc)}
.small{color:var(--mute);font-size:12px}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-size:13px;white-space:nowrap}td.small{white-space:normal;min-width:180px}
th{color:var(--mute);font-weight:500}h2{font-size:15px;margin:0 0 10px}
.total{font-size:28px;font-weight:600}
</style></head><body>
<header><h1>freesky</h1><span>free GPUs, one pool</span><span id="updated" class="small"></span></header>
<main>
 <section class="card"><h2>Free compute left</h2><div id="totals" class="grid"></div></section>
 <section><h2>Providers</h2><div id="providers" class="grid"></div></section>
 <section class="card"><h2>Recent runs</h2><div class="scroll"><table id="history"><thead><tr><th>when</th><th>provider</th><th>task</th><th>time</th><th>cost</th><th>result</th></tr></thead><tbody></tbody></table></div></section>
 <section class="card"><h2>World catalog</h2><div class="scroll"><table id="catalog"><thead><tr><th>provider</th><th>status</th><th>region</th><th>free offer</th><th>notes</th></tr></thead><tbody></tbody></table></div></section>
</main>
<script>
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function load(){
  const [q,h,c]=await Promise.all(["quota","history","catalog"].map(p=>fetch("/api/"+p).then(r=>r.json())));
  const units={}, unitOf={};
  q.forEach(r=>{unitOf[r.provider]=r.unit||"";if(r.ready&&r.unit&&r.remaining!=null){units[r.unit]=(units[r.unit]||0)+r.remaining}});
  document.getElementById("totals").innerHTML=Object.keys(units).length?Object.entries(units).map(([u,v])=>
    `<div><div class="total">${v.toFixed(1)}</div><div class="small">${esc(u)} left across ready providers</div></div>`).join(""):
    `<div class="small">No provider is ready yet: run <code>freesky providers</code> to see what to set up.</div>`;
  document.getElementById("providers").innerHTML=q.map(r=>{
    const state=r.cooldown_min?`<span class="pill warn">cooldown ${r.cooldown_min}m</span>`:r.ready?`<span class="pill ok">ready</span>`:`<span class="pill bad">not ready</span>`;
    let quota=`<div class="small">no quota limit</div>`;
    if(r.unit){const left=r.remaining, pct=left!=null&&r.limit?Math.max(0,Math.min(100,100*left/r.limit)):0;
      quota=`<div class="bar"><i style="width:${pct}%"></i></div><div class="small">${left!=null?left.toFixed(2):"?"} of ${r.limit} ${esc(r.unit)} left (${r.window==="total"?"one-off":"per "+esc(r.window)})</div>`}
    const extra=[r.session_limit_h?`sessions ${r.session_limit_h} h`:null, r.serve?"endpoints":null].filter(Boolean).join(" · ");
    return `<div class="card"><h3>${esc(r.provider)} ${state}</h3>${quota}<div class="small">${esc(extra)}</div>${!r.ready?`<div class="small">${esc(r.why)}</div>`:""}${r.cooldown_reason?`<div class="small">${esc(r.cooldown_reason)}</div>`:""}</div>`}).join("");
  document.querySelector("#history tbody").innerHTML=h.slice().reverse().map(a=>`<tr><td>${new Date(a.ts*1000).toLocaleString([], {dateStyle:"short", timeStyle:"short"})}</td><td>${esc(a.provider)}</td><td>${esc(a.task)}</td><td>${Math.round(a.seconds)} s</td><td>${a.cost?a.cost.toFixed(2)+" "+esc(unitOf[a.provider]||""):"-"}</td><td>${a.ok?'<span class="pill ok">ok</span>':`<span class="pill bad">${esc(a.error_kind||"error")}</span> <div class="small" style="white-space:normal">${esc((a.error||"").slice(0,80))}</div>`}</td></tr>`).join("")||`<tr><td colspan="6" class="small">No runs yet.</td></tr>`;
  document.querySelector("#catalog tbody").innerHTML=c.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="pill ${x.status==="supported"?"ok":x.status==="closed"?"bad":""}">${esc(x.status)}</span></td><td>${esc(x.region)}</td><td>${esc(x.offer)}</td><td class="small">${esc(x.notes)}</td></tr>`).join("");
  document.getElementById("updated").textContent="updated "+new Date().toLocaleTimeString();
}
load();setInterval(load,15000);
</script></body></html>"""


def api(path: str, config_path=None):
    from .cli import quota_rows
    if path == "/api/quota":
        return quota_rows(config_path)
    if path == "/api/history":
        return [a.__dict__ for a in Ledger().attempts()[-50:]]
    if path == "/api/catalog":
        from .catalog import STATUS_ORDER
        return [{"name": n, "status": s, "region": r, "offer": o, "via": h, "notes": t}
                for n, s, r, o, h, t in sorted(CATALOG, key=lambda x: (STATUS_ORDER[x[1]], x[0]))]
    return None


def make_dashboard(config_path=None, host="127.0.0.1", port=8790) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                data = api(self.path.split("?")[0], config_path)
                if data is None:
                    self.send_error(404)
                    return
                body, ctype = json.dumps(data).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)


def serve_dashboard(config_path=None, host="127.0.0.1", port=8790):
    srv = make_dashboard(config_path, host, port)
    print(f"freesky dashboard on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
