"""Asks Kev every question about every chunk; writes answers with timings."""
import json
import sys
import time
import urllib.request

chunks_path, questions_path, out_path = sys.argv[1:4]
chunks = json.load(open(chunks_path))
questions = json.load(open(questions_path))

results = []
t0 = time.time()
for ch in chunks:
    body = json.dumps({"state": ch["text"], "model": "kev-latest", "questions": questions}).encode()
    req = urllib.request.Request("http://127.0.0.1:8009/v1/systemone", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        ans = json.load(r)
    results.append({"t": ch.get("t"), "text": ch["text"][:200], "answers": ans["answers"],
                    "latency_ms": ans.get("latency_ms")})
json.dump({"seconds": round(time.time() - t0, 2), "results": results}, open(out_path, "w"), indent=2)
print(f"scored {len(chunks)} chunks x {len(questions)} questions in {time.time() - t0:.1f}s")
