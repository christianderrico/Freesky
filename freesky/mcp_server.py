"""MCP server: lets coding agents (Claude Code, Cursor, ...) run work on free GPUs.

    claude mcp add freesky -- freesky mcp

Tools: list_providers, quota, plan, launch, history. Launching spends free quota on the
user's own accounts, so agents should call `plan` first and say where it will run.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from .catalog import CATALOG
from .ledger import Ledger


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x
        return FastMCP


def build(config_path=None):
    from .cli import make_router, quota_rows
    from .spec import load_task

    server = _server_class()(
        name="freesky",
        instructions=("Run batch jobs on free GPU tiers (Modal, Kaggle, Colab, Lightning, Beam, cloud trial "
                      "credits, local) with automatic failover and checkpoint resume. Jobs are described by "
                      "a task YAML. Call `plan` before `launch` and tell the user which provider will be used."),
    )

    @server.tool()
    def list_providers() -> str:
        """World catalog of free GPU sources and their status (supported/experimental/planned/manual/closed)."""
        return json.dumps([{"name": n, "status": s, "region": r, "offer": o, "via": h, "notes": t}
                           for n, s, r, o, h, t in CATALOG], indent=1)

    @server.tool()
    def quota() -> str:
        """Free quota used/left per configured provider, readiness and cooldowns."""
        return json.dumps(quota_rows(config_path), indent=1)

    @server.tool()
    def plan(task_yaml: str) -> str:
        """Dry run: the order in which providers would be tried for a task YAML file, and why others are skipped."""
        lines = []
        r = make_router(config_path, log=lines.append)
        r.run(load_task(task_yaml), dry_run=True)
        return "\n".join(lines)

    @server.tool()
    def launch(task_yaml: str, provider: str = "", out_dir: str = "freesky-runs") -> str:
        """Run a task YAML on the best available free GPU, failing over across providers.
        Blocks until the job ends (long jobs may take hours). Returns provider, segments and output dir."""
        lines = []
        r = make_router(config_path, log=lines.append)
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = r.run(load_task(task_yaml), out_dir=out_dir, only=[provider] if provider else None)
        return json.dumps({"ok": out.ok, "provider": out.provider, "gpu": out.gpu, "exit_code": out.exit_code,
                           "seconds": round(out.seconds, 1), "outputs_dir": out.outputs_dir,
                           "segments": out.segments, "message": out.message, "log": lines[-40:]}, indent=1)

    @server.tool()
    def history(n: int = 20) -> str:
        """Most recent attempts recorded in the local ledger."""
        return json.dumps([a.__dict__ for a in Ledger().attempts()[-n:]], indent=1)

    return server


def main(config_path=None):
    try:
        server = build(config_path)
    except ImportError:
        print("the MCP server needs the `mcp` package: pip install 'freesky[mcp]'")
        return 2
    server.run("stdio")
    return 0
