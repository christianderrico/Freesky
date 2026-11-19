import json
import threading
import urllib.request

import pytest

from freesky.dashboard import make_dashboard


def test_dashboard_serves_page_and_api(tmp_path, monkeypatch):
    monkeypatch.setenv("FREESKY_HOME", str(tmp_path))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("providers:\n  local: {}\n  kaggle: {}\n")
    srv = make_dashboard(str(cfg), "127.0.0.1", 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "freesky" in page and "/api/" in page
        q = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/quota"))
        assert {r["provider"] for r in q} == {"local", "kaggle"}
        cat = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/catalog"))
        assert any(c["name"] == "modal" for c in cat)
    finally:
        srv.shutdown()


def test_mcp_tools_registered(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    import asyncio
    from freesky.mcp_server import build
    server = build(None)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"list_providers", "quota", "plan", "launch", "history"} <= names
