#!/usr/bin/env bash
set -euo pipefail
HERE=$PWD
mkdir -p out
(cd /tmp/kev && uv run --extra serve python -m kev.serve --run "$KEV_MODEL" --port 8009 > "$HERE/out/kev-server.log" 2>&1) &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do            # up to 30 min for the first weights download
  if curl -s -o /dev/null -m 2 http://127.0.0.1:8009/; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "Kev server exited"; tail -50 out/kev-server.log; exit 1; fi
  sleep 10
done
python3 score.py chunks.json questions.json out/answers.json
