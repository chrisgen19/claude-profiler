#!/usr/bin/env python3
"""
claude-profiler — Multi-Account Token Profiler, Hook Logger & Post-Mortem Analyzer for Claude Code
==================================================================================================
Tracks token usage, burn rate, actor attribution (Main vs Subagents vs MCP vs Tools),
week-over-week comparisons, and automated 5-mode post-mortem failure diagnostics
across one or multiple Claude Code accounts (~/.claude, ~/.claude-work, ~/.claude-personal, ~/.claude-dev, etc.).

Zero external dependencies (Python 3.8+ standard library + SQLite WAL).

Usage:
  Hook mode (configured in settings.json):
    python3 ~/.claude/hooks/token_profiler.py hook [event_name]

  CLI mode:
    claude-profiler                                  # Full dashboard
    claude-profiler burn                             # Burn rate & 5h window velocity
    claude-profiler who                              # MCP vs Subagent vs Tool attribution
    claude-profiler wow                              # Week-over-Week comparison
    claude-profiler postmortem                       # Scan for the 5 failure modes
    claude-profiler postmortem <session_id_prefix>   # Deep-dive a specific session
    claude-profiler sync [--force]                   # Sync JSONL transcripts to SQLite
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HOME = Path.home()
DB_PATH = Path(os.environ.get("CLAUDE_PROFILER_DB", str(HOME / ".claude" / "usage_analytics.db")))


def discover_account_dirs() -> Tuple[Dict[str, Path], Dict[str, str]]:
    """
    Dynamically discovers ~/.claude and any ~/.claude-* multi-account directories
    (e.g., ~/.claude-work, ~/.claude-personal, ~/.claude-dev).
    """
    dirs: Dict[str, Path] = {}
    aliases: Dict[str, str] = {}

    for p in sorted(HOME.glob(".claude*")):
        if not p.is_dir():
            continue
        name = p.name
        if name == ".claude":
            acct = "claude-default"
            dirs[acct] = p
            aliases["default"] = acct
            aliases["claude"] = acct
            aliases[acct] = acct
        elif name.startswith(".claude-"):
            suffix = name[len(".claude-") :]
            if not suffix or suffix.startswith("."):
                continue
            acct = f"claude-{suffix}"
            dirs[acct] = p
            aliases[suffix.lower()] = acct
            aliases[acct.lower()] = acct

    if not dirs:
        acct = "claude-default"
        dirs[acct] = HOME / ".claude"
        aliases["default"] = acct
        aliases[acct] = acct

    return dirs, aliases


ACCOUNT_DIRS, ACCOUNT_ALIASES = discover_account_dirs()

# Model pricing per 1M tokens: (input, output, cache_write_5m, cache_write_1h, cache_read)
MODEL_PRICING: Dict[str, Tuple[float, float, float, float, float]] = {
    "opus-5": (5.00, 25.00, 6.25, 10.00, 0.50),
    "opus-4-6": (5.00, 25.00, 6.25, 10.00, 0.50),
    "opus-4-5": (5.00, 25.00, 6.25, 10.00, 0.50),
    "opus-4": (15.00, 75.00, 18.75, 30.00, 1.50),
    "opus": (5.00, 25.00, 6.25, 10.00, 0.50),
    "sonnet-5": (3.00, 15.00, 3.75, 6.00, 0.30),
    "sonnet-4": (3.00, 15.00, 3.75, 6.00, 0.30),
    "sonnet": (3.00, 15.00, 3.75, 6.00, 0.30),
    "haiku-5-5": (0.10, 0.50, 0.125, 0.20, 0.01),
    "haiku-5": (0.10, 0.50, 0.125, 0.20, 0.01),
    "haiku-4-5": (1.00, 5.00, 1.25, 2.00, 0.10),
    "haiku": (0.80, 4.00, 1.00, 1.60, 0.08),
}

# ANSI styling helpers
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    if not USE_COLOR:
        return str(text)
    return f"\033[{code}m{text}\033[0m"


def bold(text: Any) -> str:
    return c(str(text), "1")


def dim(text: Any) -> str:
    return c(str(text), "2")


def red(text: Any) -> str:
    return c(str(text), "31;1")


def green(text: Any) -> str:
    return c(str(text), "32;1")


def yellow(text: Any) -> str:
    return c(str(text), "33;1")


def blue(text: Any) -> str:
    return c(str(text), "34;1")


def magenta(text: Any) -> str:
    return c(str(text), "35;1")


def cyan(text: Any) -> str:
    return c(str(text), "36;1")


def fmt_num(n: Optional[float]) -> str:
    if n is None:
        return "0"
    n = float(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 10_000:
        return f"{n / 1_000:.1f}k"
    if abs(n) >= 1_000:
        return f"{n:,.0f}"
    if n == int(n):
        return str(int(n))
    return f"{n:.1f}"


def fmt_usd(val: Optional[float]) -> str:
    if val is None:
        return "$0.00"
    if abs(val) < 0.01 and val != 0:
        return f"${val:.4f}"
    return f"${val:,.2f}"


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_5m_tokens: int,
    cache_1h_tokens: int,
    cache_read_tokens: int,
    web_searches: int = 0,
) -> float:
    m = (model or "").lower()
    rates = (3.00, 15.00, 3.75, 6.00, 0.30)
    for key, r in MODEL_PRICING.items():
        if key in m:
            rates = r
            break
    inp_r, out_r, cw5_r, cw1h_r, cr_r = rates
    cost = (
        input_tokens * inp_r
        + output_tokens * out_r
        + cache_5m_tokens * cw5_r
        + cache_1h_tokens * cw1h_r
        + cache_read_tokens * cr_r
    ) / 1_000_000.0
    cost += web_searches * 0.01
    return cost


def detect_account_from_path(path_str: Optional[str]) -> str:
    if not path_str:
        cfg = os.environ.get("CLAUDE_CONFIG_DIR", "")
        if cfg:
            path_str = cfg
    if path_str:
        p = str(path_str)
        # Match specific ~/.claude-<name> accounts first before ~/.claude fallback
        for acct, base_path in sorted(ACCOUNT_DIRS.items(), key=lambda x: -len(str(x[1]))):
            if str(base_path) in p or f"/{base_path.name}/" in p:
                return acct
    return "claude-default"


def parse_iso_ts(ts_val: Any) -> Optional[float]:
    if not ts_val:
        return None
    if isinstance(ts_val, (int, float)):
        return float(ts_val) / 1000.0 if ts_val > 1e11 else float(ts_val)
    if isinstance(ts_val, str):
        try:
            s = ts_val.replace("Z", "+00:00")
            return dt.datetime.fromisoformat(s).timestamp()
        except Exception:
            return None
    return None


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS synced_files (
            file_path TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            session_id TEXT NOT NULL,
            is_subagent INTEGER NOT NULL DEFAULT 0,
            mtime REAL NOT NULL,
            file_size INTEGER NOT NULL,
            synced_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL,
            account TEXT NOT NULL,
            project_slug TEXT,
            file_path TEXT,
            start_ts REAL,
            end_ts REAL,
            turn_count INTEGER DEFAULT 0,
            subagent_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            thinking_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0,
            cache_5m_tokens INTEGER DEFAULT 0,
            cache_1h_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            subagent_total_tokens INTEGER DEFAULT 0,
            subagent_cost_usd REAL DEFAULT 0.0,
            bg_haiku_tokens INTEGER DEFAULT 0,
            bg_haiku_cost_usd REAL DEFAULT 0.0,
            total_cost_usd REAL DEFAULT 0.0,
            PRIMARY KEY (session_id, account)
        );

        CREATE TABLE IF NOT EXISTS api_requests (
            file_path TEXT NOT NULL,
            request_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            account TEXT NOT NULL,
            is_subagent INTEGER NOT NULL DEFAULT 0,
            agent_id TEXT,
            agent_type TEXT,
            agent_desc TEXT,
            seq_idx INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL,
            model TEXT,
            stop_reason TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            thinking_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0,
            cache_5m_tokens INTEGER DEFAULT 0,
            cache_1h_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            web_searches INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            tool_names TEXT,
            prev_tool_actor TEXT,
            prev_tool_name TEXT,
            prev_tool_result_len INTEGER DEFAULT 0,
            had_streaming_null INTEGER DEFAULT 0,
            PRIMARY KEY (file_path, request_id)
        );

        CREATE TABLE IF NOT EXISTS tool_invocations (
            file_path TEXT NOT NULL,
            tool_use_id TEXT NOT NULL,
            request_id TEXT,
            session_id TEXT NOT NULL,
            account TEXT NOT NULL,
            is_subagent INTEGER NOT NULL DEFAULT 0,
            agent_id TEXT,
            agent_type TEXT,
            timestamp REAL NOT NULL,
            result_timestamp REAL,
            duration_sec REAL,
            tool_name TEXT NOT NULL,
            actor_category TEXT NOT NULL,
            mcp_server TEXT,
            mcp_tool TEXT,
            target_hint TEXT,
            input_chars INTEGER DEFAULT 0,
            result_chars INTEGER DEFAULT 0,
            est_result_tokens INTEGER DEFAULT 0,
            induced_cache_write_tokens INTEGER DEFAULT 0,
            is_error INTEGER DEFAULT 0,
            error_snippet TEXT,
            PRIMARY KEY (file_path, tool_use_id)
        );

        CREATE TABLE IF NOT EXISTS hook_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            account TEXT NOT NULL,
            session_id TEXT,
            hook_event TEXT NOT NULL,
            tool_name TEXT,
            actor_category TEXT,
            agent_id TEXT,
            agent_type TEXT,
            input_chars INTEGER DEFAULT 0,
            response_chars INTEGER DEFAULT 0,
            is_error INTEGER DEFAULT 0,
            transcript_path TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_req_ts ON api_requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_req_acct_ts ON api_requests(account, timestamp);
        CREATE INDEX IF NOT EXISTS idx_req_session ON api_requests(session_id, account);
        CREATE INDEX IF NOT EXISTS idx_tool_ts ON tool_invocations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tool_acct_ts ON tool_invocations(account, timestamp);
        CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_invocations(session_id, account);
        """
    )


def classify_tool_actor(tool_name: str, tool_input: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Returns (actor_category, mcp_server, mcp_tool, target_hint)
    """
    name = tool_name or "unknown"
    hint = ""
    if isinstance(tool_input, dict):
        for k in ("command", "file_path", "path", "pattern", "url", "description", "prompt", "query"):
            if k in tool_input and tool_input[k]:
                val = str(tool_input[k]).strip().replace("\n", " ")
                hint = val[:80]
                break

    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        server = parts[1] if len(parts) > 1 else "unknown"
        mtool = parts[2] if len(parts) > 2 else name
        return (f"mcp:{server}", server, mtool, hint)

    if name in ("Task", "Agent", "dispatch_agent"):
        sub_type = "general-purpose"
        if isinstance(tool_input, dict):
            sub_type = str(tool_input.get("subagent_type") or tool_input.get("agent_type") or "general-purpose")
        return (f"subagent_spawn:{sub_type}", None, None, hint)

    if name in ("SubagentHandback", "TaskStop", "SendMessage", "TeamCreate"):
        return (f"subagent_ctrl:{name}", None, None, hint)

    return (f"tool:{name}", None, None, hint)


def parse_jsonl_file(
    conn: sqlite3.Connection,
    file_path: Path,
    account: str,
    session_id: str,
    project_slug: str,
    is_subagent: bool,
) -> None:
    """
    Parses a main or subagent .jsonl transcript file, deduplicating multi-block assistant
    records by requestId, linking tool_use -> tool_result, and attributing cache_creation
    spikes to the preceding tool call.
    """
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_desc: Optional[str] = None

    if is_subagent:
        agent_id = file_path.stem.replace("agent-", "")
        meta_path = file_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
                agent_type = meta.get("agentType") or "general-purpose"
                agent_desc = meta.get("description") or ""
            except Exception:
                agent_type = "general-purpose"
        else:
            agent_type = "general-purpose"

    requests_by_id: Dict[str, Dict[str, Any]] = {}
    request_order: List[str] = []
    tools_by_id: Dict[str, Dict[str, Any]] = {}
    cost_state_latest: Optional[Dict[str, Any]] = None

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, raw_line in enumerate(f):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                except Exception:
                    continue

                rtype = rec.get("type")
                ts = parse_iso_ts(rec.get("timestamp"))

                if rtype == "cost-state" and not is_subagent:
                    cost_state_latest = rec
                    continue

                if rtype == "assistant":
                    msg = rec.get("message") or {}
                    model = msg.get("model") or ""
                    if model == "<synthetic>":
                        continue

                    req_id = rec.get("requestId") or msg.get("id") or f"line_{line_num}"
                    usage = msg.get("usage") or {}
                    inp = int(usage.get("input_tokens") or 0)
                    out = int(usage.get("output_tokens") or 0)
                    out_details = usage.get("output_tokens_details") or {}
                    think = int(out_details.get("thinking_tokens") or 0)
                    cc_total = int(usage.get("cache_creation_input_tokens") or 0)
                    cc_obj = usage.get("cache_creation") or {}
                    cc_5m = int(cc_obj.get("ephemeral_5m_input_tokens") or 0)
                    cc_1h = int(cc_obj.get("ephemeral_1h_input_tokens") or 0)
                    if cc_total > 0 and (cc_5m + cc_1h) == 0:
                        cc_5m = cc_total
                    cr = int(usage.get("cache_read_input_tokens") or 0)
                    srv_tool = usage.get("server_tool_use") or {}
                    web_searches = int(srv_tool.get("web_search_requests") or 0)
                    stop_reason = msg.get("stop_reason")

                    if req_id not in requests_by_id:
                        requests_by_id[req_id] = {
                            "request_id": req_id,
                            "seq_idx": len(request_order),
                            "timestamp": ts or time.time(),
                            "model": model,
                            "stop_reason": stop_reason,
                            "input_tokens": inp,
                            "output_tokens": out,
                            "thinking_tokens": think,
                            "cache_creation_tokens": cc_total,
                            "cache_5m_tokens": cc_5m,
                            "cache_1h_tokens": cc_1h,
                            "cache_read_tokens": cr,
                            "web_searches": web_searches,
                            "tool_ids": [],
                            "tool_names": [],
                            "had_streaming_null": 1 if (stop_reason is None and out <= 1) else 0,
                        }
                        request_order.append(req_id)
                    else:
                        entry = requests_by_id[req_id]
                        if ts and ts > entry["timestamp"]:
                            entry["timestamp"] = ts
                        if stop_reason is not None:
                            entry["stop_reason"] = stop_reason
                        elif out <= 1:
                            entry["had_streaming_null"] = 1
                        entry["input_tokens"] = max(entry["input_tokens"], inp)
                        entry["output_tokens"] = max(entry["output_tokens"], out)
                        entry["thinking_tokens"] = max(entry["thinking_tokens"], think)
                        entry["cache_creation_tokens"] = max(entry["cache_creation_tokens"], cc_total)
                        entry["cache_5m_tokens"] = max(entry["cache_5m_tokens"], cc_5m)
                        entry["cache_1h_tokens"] = max(entry["cache_1h_tokens"], cc_1h)
                        entry["cache_read_tokens"] = max(entry["cache_read_tokens"], cr)
                        entry["web_searches"] = max(entry["web_searches"], web_searches)

                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "thinking":
                                t_text = block.get("thinking") or ""
                                if t_text and requests_by_id[req_id]["thinking_tokens"] == 0:
                                    requests_by_id[req_id]["thinking_tokens"] = max(1, len(t_text) // 4)
                            elif btype == "tool_use":
                                tid = block.get("id") or f"{req_id}_tool_{len(tools_by_id)}"
                                tname = block.get("name") or "unknown"
                                tinput = block.get("input") if isinstance(block.get("input"), dict) else {}
                                input_str = json.dumps(tinput, ensure_ascii=False)
                                actor_cat, mcp_srv, mcp_tl, hint = classify_tool_actor(tname, tinput)
                                if tid not in tools_by_id:
                                    tools_by_id[tid] = {
                                        "tool_use_id": tid,
                                        "request_id": req_id,
                                        "timestamp": ts or requests_by_id[req_id]["timestamp"],
                                        "result_timestamp": None,
                                        "duration_sec": None,
                                        "tool_name": tname,
                                        "actor_category": actor_cat,
                                        "mcp_server": mcp_srv,
                                        "mcp_tool": mcp_tl,
                                        "target_hint": hint,
                                        "input_chars": len(input_str),
                                        "result_chars": 0,
                                        "est_result_tokens": 0,
                                        "induced_cache_write_tokens": 0,
                                        "is_error": 0,
                                        "error_snippet": None,
                                    }
                                    requests_by_id[req_id]["tool_ids"].append(tid)
                                    requests_by_id[req_id]["tool_names"].append(tname)

                elif rtype == "user":
                    msg = rec.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                tid = block.get("tool_use_id")
                                is_err = 1 if block.get("is_error") else 0
                                raw_c = block.get("content")
                                c_len = 0
                                est_tok = 0
                                c_str = ""
                                if isinstance(raw_c, str):
                                    c_str = raw_c
                                    c_len = len(c_str)
                                    est_tok = max(1, c_len // 4) if c_len > 0 else 0
                                elif isinstance(raw_c, list):
                                    for sub in raw_c:
                                        if isinstance(sub, dict):
                                            if sub.get("type") == "image" or "image" in str(sub.get("source", {}).get("media_type", "")):
                                                # Vision image tokens are ~1,600 tokens per screenshot regardless of base64 byte size
                                                b64_data = str(sub.get("source", {}).get("data", ""))
                                                c_len += len(b64_data)
                                                est_tok += 1600
                                            else:
                                                s_txt = str(sub.get("text") or json.dumps(sub, ensure_ascii=False))
                                                c_len += len(s_txt)
                                                est_tok += max(1, len(s_txt) // 4)
                                                if not c_str:
                                                    c_str = s_txt[:200]
                                else:
                                    c_str = json.dumps(raw_c, ensure_ascii=False) if raw_c is not None else ""
                                    c_len = len(c_str)
                                    est_tok = max(1, c_len // 4) if c_len > 0 else 0

                                if tid and tid in tools_by_id:
                                    t_entry = tools_by_id[tid]
                                    t_entry["result_timestamp"] = ts
                                    if ts and t_entry["timestamp"]:
                                        t_entry["duration_sec"] = max(0.0, ts - t_entry["timestamp"])
                                    t_entry["result_chars"] = c_len
                                    t_entry["est_result_tokens"] = est_tok
                                    t_entry["is_error"] = is_err
                                    if is_err:
                                        t_entry["error_snippet"] = c_str[:200].replace("\n", " ")
    except Exception:
        return

    # Link consecutive requests to attribute cache_creation_tokens to the preceding tool call(s)
    prev_tool_ids: List[str] = []
    for req_id in request_order:
        req = requests_by_id[req_id]
        req["cost_usd"] = estimate_cost(
            req["model"],
            req["input_tokens"],
            req["output_tokens"],
            req["cache_5m_tokens"],
            req["cache_1h_tokens"],
            req["cache_read_tokens"],
            req["web_searches"],
        )
        if prev_tool_ids:
            valid_prev = [tools_by_id[tid] for tid in prev_tool_ids if tid in tools_by_id]
            if valid_prev:
                primary_tool = max(valid_prev, key=lambda x: x["result_chars"])
                req["prev_tool_actor"] = primary_tool["actor_category"]
                req["prev_tool_name"] = primary_tool["tool_name"]
                req["prev_tool_result_len"] = sum(t["result_chars"] for t in valid_prev)

                # Distribute induced cache_creation_tokens across previous turn's tools proportionally by result size,
                # capped to each tool's plausible token footprint so 5m TTL cold-cache rewrites aren't blamed on tiny tools.
                total_res_chars = sum(max(1, t["result_chars"]) for t in valid_prev)
                cc_new = req["cache_creation_tokens"]
                for t in valid_prev:
                    share = max(1, t["result_chars"]) / total_res_chars
                    raw_induced = int(round(cc_new * share))
                    max_plausible = int(t["est_result_tokens"] * 1.4 + (t["input_chars"] // 4) + 512)
                    t["induced_cache_write_tokens"] = min(raw_induced, max_plausible)
            else:
                req["prev_tool_actor"] = None
                req["prev_tool_name"] = None
                req["prev_tool_result_len"] = 0
        else:
            req["prev_tool_actor"] = "session_init_or_user" if req["seq_idx"] == 0 else "user_prompt"
            req["prev_tool_name"] = None
            req["prev_tool_result_len"] = 0

        prev_tool_ids = req["tool_ids"]

    f_str = str(file_path)
    conn.execute("DELETE FROM api_requests WHERE file_path = ?", (f_str,))
    conn.execute("DELETE FROM tool_invocations WHERE file_path = ?", (f_str,))

    req_rows = [
        (
            f_str,
            r["request_id"],
            session_id,
            account,
            1 if is_subagent else 0,
            agent_id,
            agent_type if is_subagent else "main",
            agent_desc,
            r["seq_idx"],
            r["timestamp"],
            r["model"],
            r["stop_reason"],
            r["input_tokens"],
            r["output_tokens"],
            r["thinking_tokens"],
            r["cache_creation_tokens"],
            r["cache_5m_tokens"],
            r["cache_1h_tokens"],
            r["cache_read_tokens"],
            r["web_searches"],
            r["cost_usd"],
            ",".join(r["tool_names"]),
            r.get("prev_tool_actor"),
            r.get("prev_tool_name"),
            r.get("prev_tool_result_len", 0),
            r["had_streaming_null"],
        )
        for r in requests_by_id.values()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO api_requests (
            file_path, request_id, session_id, account, is_subagent,
            agent_id, agent_type, agent_desc, seq_idx, timestamp,
            model, stop_reason, input_tokens, output_tokens, thinking_tokens,
            cache_creation_tokens, cache_5m_tokens, cache_1h_tokens, cache_read_tokens,
            web_searches, cost_usd, tool_names, prev_tool_actor, prev_tool_name,
            prev_tool_result_len, had_streaming_null
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        req_rows,
    )

    tool_rows = [
        (
            f_str,
            t["tool_use_id"],
            t["request_id"],
            session_id,
            account,
            1 if is_subagent else 0,
            agent_id,
            agent_type if is_subagent else "main",
            t["timestamp"],
            t["result_timestamp"],
            t["duration_sec"],
            t["tool_name"],
            t["actor_category"],
            t["mcp_server"],
            t["mcp_tool"],
            t["target_hint"],
            t["input_chars"],
            t["result_chars"],
            t["est_result_tokens"],
            t["induced_cache_write_tokens"],
            t["is_error"],
            t["error_snippet"],
        )
        for t in tools_by_id.values()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO tool_invocations (
            file_path, tool_use_id, request_id, session_id, account,
            is_subagent, agent_id, agent_type, timestamp, result_timestamp,
            duration_sec, tool_name, actor_category, mcp_server, mcp_tool,
            target_hint, input_chars, result_chars, est_result_tokens,
            induced_cache_write_tokens, is_error, error_snippet
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tool_rows,
    )

    # Extract background Haiku usage from cost-state if present on main session
    bg_haiku_tokens = 0
    bg_haiku_cost = 0.0
    if cost_state_latest and isinstance(cost_state_latest.get("modelUsage"), dict):
        for m_name, m_stats in cost_state_latest["modelUsage"].items():
            if "haiku" in m_name.lower() and isinstance(m_stats, dict):
                bg_haiku_tokens += (
                    int(m_stats.get("inputTokens") or 0)
                    + int(m_stats.get("outputTokens") or 0)
                    + int(m_stats.get("cacheReadInputTokens") or 0)
                    + int(m_stats.get("cacheCreationInputTokens") or 0)
                )
                bg_haiku_cost += float(m_stats.get("costUSD") or 0.0)

    rebuild_session_rollups(conn, session_id, account, project_slug, f_str if not is_subagent else None, bg_haiku_tokens, bg_haiku_cost)


def rebuild_session_rollups(
    conn: sqlite3.Connection,
    session_id: str,
    account: str,
    project_slug: str,
    main_file_path: Optional[str],
    bg_haiku_tokens: int = 0,
    bg_haiku_cost: float = 0.0,
) -> None:
    row = conn.execute(
        """
        SELECT
            MIN(timestamp) AS start_ts,
            MAX(timestamp) AS end_ts,
            SUM(CASE WHEN is_subagent = 0 THEN 1 ELSE 0 END) AS turn_count,
            COUNT(DISTINCT CASE WHEN is_subagent = 1 THEN agent_id END) AS subagent_count,
            SUM(input_tokens) AS input_tokens,
            SUM(output_tokens) AS output_tokens,
            SUM(thinking_tokens) AS thinking_tokens,
            SUM(cache_creation_tokens) AS cache_creation_tokens,
            SUM(cache_5m_tokens) AS cache_5m_tokens,
            SUM(cache_1h_tokens) AS cache_1h_tokens,
            SUM(cache_read_tokens) AS cache_read_tokens,
            SUM(CASE WHEN is_subagent = 1 THEN (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) ELSE 0 END) AS subagent_total_tokens,
            SUM(CASE WHEN is_subagent = 1 THEN cost_usd ELSE 0.0 END) AS subagent_cost_usd,
            SUM(cost_usd) AS req_cost_usd
        FROM api_requests
        WHERE session_id = ? AND account = ?
        """,
        (session_id, account),
    ).fetchone()

    if not row or row["start_ts"] is None:
        return

    existing = conn.execute(
        "SELECT file_path, bg_haiku_tokens, bg_haiku_cost_usd FROM sessions WHERE session_id = ? AND account = ?",
        (session_id, account),
    ).fetchone()

    fp = main_file_path or (existing["file_path"] if existing else None)
    haiku_tok = bg_haiku_tokens if bg_haiku_tokens > 0 else (existing["bg_haiku_tokens"] if existing else 0)
    haiku_cost = bg_haiku_cost if bg_haiku_cost > 0 else (existing["bg_haiku_cost_usd"] if existing else 0.0)
    total_cost = float(row["req_cost_usd"] or 0.0) + float(haiku_cost or 0.0)

    conn.execute(
        """
        INSERT OR REPLACE INTO sessions (
            session_id, account, project_slug, file_path, start_ts, end_ts,
            turn_count, subagent_count, input_tokens, output_tokens, thinking_tokens,
            cache_creation_tokens, cache_5m_tokens, cache_1h_tokens, cache_read_tokens,
            subagent_total_tokens, subagent_cost_usd, bg_haiku_tokens, bg_haiku_cost_usd, total_cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            account,
            project_slug,
            fp,
            row["start_ts"],
            row["end_ts"],
            row["turn_count"] or 0,
            row["subagent_count"] or 0,
            row["input_tokens"] or 0,
            row["output_tokens"] or 0,
            row["thinking_tokens"] or 0,
            row["cache_creation_tokens"] or 0,
            row["cache_5m_tokens"] or 0,
            row["cache_1h_tokens"] or 0,
            row["cache_read_tokens"] or 0,
            row["subagent_total_tokens"] or 0,
            row["subagent_cost_usd"] or 0.0,
            haiku_tok,
            haiku_cost,
            total_cost,
        ),
    )


def sync_all_accounts(conn: sqlite3.Connection, force: bool = False, quiet: bool = False) -> int:
    """
    Scans all discovered account directories for main and subagent .jsonl files and incrementally
    updates SQLite for any new or modified files.
    """
    synced_map: Dict[str, Tuple[float, int]] = {}
    if not force:
        for r in conn.execute("SELECT file_path, mtime, file_size FROM synced_files"):
            synced_map[r["file_path"]] = (r["mtime"], r["file_size"])

    updated_count = 0
    now = time.time()

    for account, base_dir in ACCOUNT_DIRS.items():
        projects_dir = base_dir / "projects"
        if not projects_dir.is_dir():
            continue

        for root, _dirs, files in os.walk(projects_dir):
            for fname in files:
                if not fname.endswith(".jsonl"):
                    continue
                fpath = Path(root) / fname
                try:
                    st = fpath.stat()
                except OSError:
                    continue

                f_str = str(fpath)
                prev = synced_map.get(f_str)
                if prev and abs(prev[0] - st.st_mtime) < 0.001 and prev[1] == st.st_size:
                    continue

                rel = fpath.relative_to(projects_dir)
                parts = rel.parts
                project_slug = parts[0] if len(parts) > 0 else "unknown"
                is_subagent = "subagents" in parts
                if is_subagent:
                    session_id = parts[1] if len(parts) > 2 else fpath.stem
                else:
                    session_id = fpath.stem

                parse_jsonl_file(conn, fpath, account, session_id, project_slug, is_subagent)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO synced_files (
                        file_path, account, session_id, is_subagent, mtime, file_size, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (f_str, account, session_id, 1 if is_subagent else 0, st.st_mtime, st.st_size, now),
                )
                updated_count += 1

    if updated_count > 0:
        conn.commit()
    return updated_count


def handle_hook(event_arg: Optional[str]) -> int:
    """
    Entry point for Claude Code lifecycle hooks. Reads JSON from stdin, logs hook telemetry,
    and incrementally syncs the active session's JSONL + subagent JSONLs.
    Never raises or blocks Claude Code.
    """
    try:
        raw = ""
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    try:
        event_name = payload.get("hook_event_name") or event_arg or "unknown"
        session_id = payload.get("session_id") or ""
        transcript_path = payload.get("transcript_path") or ""
        account = detect_account_from_path(transcript_path)
        tool_name = payload.get("tool_name")
        tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
        tool_response = payload.get("tool_response")
        agent_id = payload.get("agent_id")
        agent_type = payload.get("agent_type")

        actor_cat = None
        if tool_name:
            actor_cat, _, _, _ = classify_tool_actor(tool_name, tool_input)
        elif agent_type or event_name in ("SubagentStart", "SubagentStop"):
            actor_cat = f"subagent:{agent_type or 'general-purpose'}"

        inp_chars = len(json.dumps(tool_input, ensure_ascii=False)) if tool_input else 0
        resp_str = json.dumps(tool_response, ensure_ascii=False) if tool_response is not None else ""
        resp_chars = len(resp_str)
        is_err = 0
        if isinstance(tool_response, dict) and (tool_response.get("is_error") or tool_response.get("error")):
            is_err = 1

        conn = get_db()
        conn.execute(
            """
            INSERT INTO hook_events (
                timestamp, account, session_id, hook_event, tool_name,
                actor_category, agent_id, agent_type, input_chars,
                response_chars, is_error, transcript_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                account,
                session_id,
                event_name,
                tool_name,
                actor_cat,
                agent_id,
                agent_type,
                inp_chars,
                resp_chars,
                is_err,
                transcript_path,
            ),
        )

        # Incrementally sync the active transcript & any sibling subagent transcripts
        if transcript_path:
            t_path = Path(transcript_path).expanduser()
            if t_path.is_file():
                is_sub = "subagents" in t_path.parts
                proj_slug = t_path.parent.name if not is_sub else t_path.parent.parent.parent.name
                sess_id = session_id or (t_path.stem if not is_sub else t_path.parent.parent.name)
                parse_jsonl_file(conn, t_path, account, sess_id, proj_slug, is_sub)
                st = t_path.stat()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO synced_files (
                        file_path, account, session_id, is_subagent, mtime, file_size, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(t_path), account, sess_id, 1 if is_sub else 0, st.st_mtime, st.st_size, time.time()),
                )
                sub_dir = t_path.with_suffix("") / "subagents"
                if sub_dir.is_dir():
                    for sub_f in sub_dir.glob("agent-*.jsonl"):
                        parse_jsonl_file(conn, sub_f, account, sess_id, proj_slug, True)
                        sst = sub_f.stat()
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO synced_files (
                                file_path, account, session_id, is_subagent, mtime, file_size, synced_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (str(sub_f), account, sess_id, 1, sst.st_mtime, sst.st_size, time.time()),
                        )

        conn.commit()
        conn.close()
    except Exception:
        pass
    return 0


def resolve_account_filter(acct_arg: Optional[str]) -> Optional[str]:
    if not acct_arg or acct_arg.lower() == "all":
        return None
    low = acct_arg.lower()
    if low in ACCOUNT_ALIASES:
        return ACCOUNT_ALIASES[low]
    if not low.startswith("claude-"):
        return f"claude-{low}"
    return acct_arg


def compute_active_burn_rate(conn: sqlite3.Connection, since_ts: float, account_filter: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Computes active session duration (summing inter-request intervals <= 15 minutes
    within each session, with a minimum 8s per request) and token burn rates per account.
    """
    sql = """
        SELECT account, session_id, timestamp,
               (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
               (input_tokens + output_tokens + cache_creation_tokens) AS fresh_tok,
               cost_usd
        FROM api_requests
        WHERE timestamp >= ?
    """
    params: List[Any] = [since_ts]
    if account_filter:
        sql += " AND account = ?"
        params.append(account_filter)
    sql += " ORDER BY account, session_id, timestamp ASC"

    rows = conn.execute(sql, params).fetchall()
    by_acct_sess: Dict[Tuple[str, str], List[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_acct_sess[(r["account"], r["session_id"])].append(r)

    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"active_sec": 0.0, "total_tok": 0.0, "fresh_tok": 0.0, "cost_usd": 0.0, "requests": 0.0, "sessions": 0.0}
    )

    for (acct, _sess), srows in by_acct_sess.items():
        stats[acct]["sessions"] += 1
        prev_ts: Optional[float] = None
        sess_active = 0.0
        for r in srows:
            ts = float(r["timestamp"])
            if prev_ts is not None:
                gap = ts - prev_ts
                if 0 < gap <= 900:  # <= 15m counts as continuous active work
                    sess_active += gap
                else:
                    sess_active += 8.0
            else:
                sess_active += 8.0
            prev_ts = ts
            stats[acct]["total_tok"] += float(r["total_tok"] or 0)
            stats[acct]["fresh_tok"] += float(r["fresh_tok"] or 0)
            stats[acct]["cost_usd"] += float(r["cost_usd"] or 0.0)
            stats[acct]["requests"] += 1
        stats[acct]["active_sec"] += max(15.0, sess_active)

    return stats


def print_header(title: str) -> None:
    line = "═" * 92
    print(f"\n{cyan(line)}")
    print(f"  {bold(title)}")
    print(f"{cyan(line)}")


def cmd_burn(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    acct_filter = resolve_account_filter(args.account)
    now = time.time()

    print_header(f"🔥 TOKEN BURN RATE & QUOTA VELOCITY ({acct_filter or 'ALL ACCOUNTS'})")

    windows = [
        ("Last 1 Hour", now - 3600),
        ("Rolling 5h Block", now - 5 * 3600),
        ("Last 24 Hours", now - 24 * 3600),
        (f"Last {args.days} Days", now - args.days * 86400),
    ]

    for w_label, since_ts in windows:
        stats = compute_active_burn_rate(conn, since_ts, acct_filter)
        print(f"\n{bold(f'▸ {w_label}')}")
        if not stats:
            print(f"  {dim('No activity recorded in this window.')}")
            continue

        print(
            f"  {'Account':<17} {'Sessions':>8} {'Turns':>7} {'Active Time':>12} "
            f"{'Total Tokens':>13} {'Fresh/Write':>12} {'Burn (Tok/m)':>14} {'Burn ($/hr)':>12} {'Est. Cost':>10}"
        )
        print("  " + "─" * 115)

        tot_sec = 0.0
        tot_tok = 0.0
        tot_fresh = 0.0
        tot_cost = 0.0
        tot_reqs = 0.0
        tot_sess = 0.0

        for acct in sorted(stats.keys()):
            s = stats[acct]
            active_min = max(0.25, s["active_sec"] / 60.0)
            active_hr = active_min / 60.0
            tok_per_min = s["total_tok"] / active_min
            fresh_per_min = s["fresh_tok"] / active_min
            usd_per_hr = s["cost_usd"] / active_hr

            tot_sec += s["active_sec"]
            tot_tok += s["total_tok"]
            tot_fresh += s["fresh_tok"]
            tot_cost += s["cost_usd"]
            tot_reqs += s["requests"]
            tot_sess += s["sessions"]

            hrs = int(s["active_sec"] // 3600)
            mins = int((s["active_sec"] % 3600) // 60)
            t_str = f"{hrs}h {mins:02d}m" if hrs > 0 else f"{mins}m"

            burn_str = f"{fmt_num(tok_per_min)}/m ({fmt_num(fresh_per_min)} new)"
            print(
                f"  {cyan(f'{acct:<17}')} {int(s['sessions']):>8} {int(s['requests']):>7} {t_str:>12} "
                f"{fmt_num(s['total_tok']):>13} {fmt_num(s['fresh_tok']):>12} "
                f"{yellow(f'{burn_str:>14}')} {magenta(f'{fmt_usd(usd_per_hr)}/h'):>12} {green(f'{fmt_usd(s['cost_usd']):>10}')}"
            )

        if len(stats) > 1:
            tot_min = max(0.25, tot_sec / 60.0)
            hrs = int(tot_sec // 3600)
            mins = int((tot_sec % 3600) // 60)
            t_str = f"{hrs}h {mins:02d}m" if hrs > 0 else f"{mins}m"
            print("  " + "─" * 115)
            print(
                f"  {bold(f'{"TOTAL (COMBINED)":<17}')} {int(tot_sess):>8} {int(tot_reqs):>7} {t_str:>12} "
                f"{bold(f'{fmt_num(tot_tok):>13}')} {fmt_num(tot_fresh):>12} "
                f"{bold(f'{fmt_num(tot_tok / tot_min)}/m'):>14} {bold(f'{fmt_usd(tot_cost / (tot_min / 60.0))}/h'):>12} {bold(f'{fmt_usd(tot_cost):>10}')}"
            )

    # Top fastest-burning recent sessions
    since_recent = now - args.days * 86400
    sql_top = """
        SELECT account, session_id, project_slug, start_ts, end_ts, turn_count, subagent_count,
               (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
               (input_tokens + output_tokens + cache_creation_tokens) AS fresh_tok,
               subagent_total_tokens, total_cost_usd
        FROM sessions
        WHERE end_ts >= ? AND turn_count >= 2
    """
    p_top: List[Any] = [since_recent]
    if acct_filter:
        sql_top += " AND account = ?"
        p_top.append(acct_filter)
    sql_top += " ORDER BY total_cost_usd DESC LIMIT 8"
    top_rows = conn.execute(sql_top, p_top).fetchall()

    if top_rows:
        print(f"\n{bold(f'▸ Top Highest-Burn Sessions (Last {args.days} Days)')}")
        print(
            f"  {'Account':<16} {'Session ID':<12} {'When':<16} {'Turns':>6} {'SubAg':>6} "
            f"{'Total Tok':>11} {'Subagent %':>11} {'Est. Cost':>10}  {'Project'}"
        )
        print("  " + "─" * 115)
        for r in top_rows:
            when = dt.datetime.fromtimestamp(r["end_ts"]).strftime("%m-%d %H:%M")
            tot = max(1, int(r["total_tok"] or 1))
            sub_pct = 100.0 * float(r["subagent_total_tokens"] or 0) / tot
            proj = (r["project_slug"] or "")[-28:]
            print(
                f"  {r['account']:<16} {r['session_id'][:10]:<12} {when:<16} {r['turn_count']:>6} {r['subagent_count']:>6} "
                f"{fmt_num(tot):>11} {sub_pct:>10.1f}% {fmt_usd(r['total_cost_usd']):>10}  {dim(proj)}"
            )


def cmd_who(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    acct_filter = resolve_account_filter(args.account)
    since_ts = time.time() - args.days * 86400

    print_header(f"🕵️ WHO USES TOKENS: ACTOR, SUBAGENT & MCP ATTRIBUTION (Last {args.days} Days)")

    # 1. High-level Actor Split (Main Agent vs Subagents)
    sql_actors = """
        SELECT
            account,
            CASE WHEN is_subagent = 1 THEN 'Subagents (' || COALESCE(agent_type, 'general') || ')' ELSE 'Main Session Agent' END AS actor,
            COUNT(*) AS reqs,
            SUM(input_tokens) AS inp,
            SUM(output_tokens) AS out_tok,
            SUM(thinking_tokens) AS think_tok,
            SUM(cache_creation_tokens) AS cw_tok,
            SUM(cache_read_tokens) AS cr_tok,
            SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
            SUM(cost_usd) AS cost
        FROM api_requests
        WHERE timestamp >= ?
    """
    params: List[Any] = [since_ts]
    if acct_filter:
        sql_actors += " AND account = ?"
        params.append(acct_filter)
    sql_actors += " GROUP BY account, actor ORDER BY cost DESC"
    actor_rows = conn.execute(sql_actors, params).fetchall()

    grand_cost = sum(float(r["cost"] or 0) for r in actor_rows) or 1.0
    print(f"\n{bold('▸ 1. Execution Context Split (Main Thread vs. Subagents)')}")
    print(
        f"  {'Account':<16} {'Actor / Agent Type':<30} {'Turns':>7} {'Total Tok':>11} "
        f"{'Cache Write':>12} {'Output+Think':>13} {'Cost ($)':>10} {'Cost %':>8}"
    )
    print("  " + "─" * 113)
    for r in actor_rows:
        c_val = float(r["cost"] or 0.0)
        pct = 100.0 * c_val / grand_cost
        out_and_think = f"{fmt_num(r['out_tok'])} ({fmt_num(r['think_tok'])} thk)"
        print(
            f"  {r['account']:<16} {r['actor'][:30]:<30} {r['reqs']:>7} {fmt_num(r['total_tok']):>11} "
            f"{fmt_num(r['cw_tok']):>12} {out_and_think:>13} {fmt_usd(c_val):>10} {pct:>7.1f}%"
        )

    # 2. MCP Server Breakdown
    sql_mcp = """
        SELECT
            account,
            mcp_server,
            mcp_tool,
            COUNT(*) AS calls,
            SUM(is_error) AS errors,
            SUM(est_result_tokens) AS payload_tokens,
            MAX(est_result_tokens) AS max_payload_tokens,
            SUM(induced_cache_write_tokens) AS induced_cw,
            AVG(duration_sec) AS avg_sec
        FROM tool_invocations
        WHERE timestamp >= ? AND mcp_server IS NOT NULL
    """
    p_mcp: List[Any] = [since_ts]
    if acct_filter:
        sql_mcp += " AND account = ?"
        p_mcp.append(acct_filter)
    sql_mcp += " GROUP BY account, mcp_server, mcp_tool ORDER BY induced_cw DESC, payload_tokens DESC LIMIT 20"
    mcp_rows = conn.execute(sql_mcp, p_mcp).fetchall()

    print(f"\n{bold('▸ 2. MCP Server Token Attribution (Payload Injected + Induced Cache Writes)')}")
    if not mcp_rows:
        print(f"  {dim('No MCP tool calls recorded in this window.')}")
    else:
        print(
            f"  {'Account':<16} {'MCP Server':<20} {'MCP Tool':<26} {'Calls':>6} {'Err':>4} "
            f"{'Payload Tok':>12} {'Max Call':>9} {'Induced CacheWrite':>18} {'Avg Time':>9}"
        )
        print("  " + "─" * 126)
        for r in mcp_rows:
            avg_s = f"{r['avg_sec']:.1f}s" if r["avg_sec"] is not None else "-"
            err_s = red(str(r["errors"])) if r["errors"] > 0 else "0"
            print(
                f"  {r['account']:<16} {cyan(r['mcp_server'][:20]):<20} {r['mcp_tool'][:26]:<26} "
                f"{r['calls']:>6} {err_s:>4} {fmt_num(r['payload_tokens']):>12} "
                f"{fmt_num(r['max_payload_tokens']):>9} {yellow(f'{fmt_num(r['induced_cw']):>18}')} {avg_s:>9}"
            )

    # 3. Built-in Tools & Subagent Spawners Breakdown
    sql_tools = """
        SELECT
            actor_category,
            tool_name,
            COUNT(*) AS calls,
            SUM(is_error) AS errors,
            SUM(est_result_tokens) AS payload_tokens,
            MAX(est_result_tokens) AS max_payload_tokens,
            SUM(induced_cache_write_tokens) AS induced_cw,
            AVG(duration_sec) AS avg_sec
        FROM tool_invocations
        WHERE timestamp >= ? AND mcp_server IS NULL
    """
    p_tools: List[Any] = [since_ts]
    if acct_filter:
        sql_tools += " AND account = ?"
        p_tools.append(acct_filter)
    sql_tools += " GROUP BY actor_category, tool_name ORDER BY induced_cw DESC, payload_tokens DESC LIMIT 15"
    tool_rows = conn.execute(sql_tools, p_tools).fetchall()

    print(f"\n{bold('▸ 3. Built-in Tool & Subagent Spawner Token Impact')}")
    if tool_rows:
        print(
            f"  {'Category / Actor':<28} {'Tool Name':<18} {'Calls':>7} {'Errors':>7} "
            f"{'Payload Tok':>13} {'Max Single':>11} {'Induced CacheWrite':>18} {'Avg Time':>9}"
        )
        print("  " + "─" * 118)
        for r in tool_rows:
            avg_s = f"{r['avg_sec']:.1f}s" if r["avg_sec"] is not None else "-"
            err_s = red(f"{r['errors']:>7}") if r["errors"] > 0 else f"{0:>7}"
            print(
                f"  {r['actor_category'][:28]:<28} {r['tool_name'][:18]:<18} {r['calls']:>7} {err_s} "
                f"{fmt_num(r['payload_tokens']):>13} {fmt_num(r['max_payload_tokens']):>11} "
                f"{fmt_num(r['induced_cw']):>18} {avg_s:>9}"
            )


def get_period_metrics(conn: sqlite3.Connection, start_ts: float, end_ts: float, acct_filter: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    sql = """
        SELECT
            account,
            COUNT(DISTINCT session_id) AS sessions,
            COUNT(*) AS turns,
            SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tokens,
            SUM(input_tokens + cache_creation_tokens) AS fresh_input_tokens,
            SUM(cache_creation_tokens) AS cache_write_tokens,
            SUM(cache_read_tokens) AS cache_read_tokens,
            SUM(output_tokens) AS output_tokens,
            SUM(thinking_tokens) AS thinking_tokens,
            SUM(CASE WHEN is_subagent = 1 THEN (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) ELSE 0 END) AS subagent_tokens,
            SUM(cost_usd) AS cost_usd
        FROM api_requests
        WHERE timestamp >= ? AND timestamp < ?
    """
    params: List[Any] = [start_ts, end_ts]
    if acct_filter:
        sql += " AND account = ?"
        params.append(acct_filter)
    sql += " GROUP BY account"

    res: Dict[str, Dict[str, float]] = {}
    for r in conn.execute(sql, params):
        acct = r["account"]
        res[acct] = {k: float(r[k] or 0.0) for k in r.keys() if k != "account"}

    # Add MCP payload & tool error metrics
    sql_t = """
        SELECT
            account,
            SUM(CASE WHEN mcp_server IS NOT NULL THEN est_result_tokens ELSE 0 END) AS mcp_tokens,
            SUM(CASE WHEN mcp_server IS NOT NULL THEN 1 ELSE 0 END) AS mcp_calls,
            SUM(is_error) AS tool_errors,
            COUNT(*) AS tool_calls
        FROM tool_invocations
        WHERE timestamp >= ? AND timestamp < ?
    """
    if acct_filter:
        sql_t += " AND account = ?"
    sql_t += " GROUP BY account"

    for r in conn.execute(sql_t, params):
        acct = r["account"]
        if acct not in res:
            continue
        res[acct]["mcp_tokens"] = float(r["mcp_tokens"] or 0.0)
        res[acct]["mcp_calls"] = float(r["mcp_calls"] or 0.0)
        res[acct]["tool_errors"] = float(r["tool_errors"] or 0.0)
        res[acct]["tool_calls"] = float(r["tool_calls"] or 0.0)

    # Compute ALL combined
    if res:
        combined: Dict[str, float] = defaultdict(float)
        for acct_metrics in res.values():
            for k, v in acct_metrics.items():
                combined[k] += v
        res["ALL ACCOUNTS"] = dict(combined)

    return res


def fmt_delta(curr: float, prev: float, invert_color: bool = False) -> str:
    if prev == 0 and curr == 0:
        return dim("0.0% ▬")
    if prev == 0:
        return yellow("+NEW ▲")
    pct = ((curr - prev) / prev) * 100.0
    arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "▬")
    s = f"{pct:+.1f}% {arrow}"
    if abs(pct) < 2.0:
        return dim(s)
    if invert_color:
        return green(s) if pct > 0 else red(s)
    return red(s) if pct > 15.0 else (green(s) if pct < -5.0 else yellow(s))


def cmd_wow(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    acct_filter = resolve_account_filter(args.account)
    now_dt = dt.datetime.now()
    now_ts = now_dt.timestamp()

    monday_this_week = (now_dt - dt.timedelta(days=now_dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    monday_last_week = monday_this_week - dt.timedelta(days=7)

    r7_start = now_ts - 7 * 86400
    prev7_start = now_ts - 14 * 86400

    print_header(f"📊 WEEK-OVER-WEEK (WoW) USAGE COMPARISON ({acct_filter or 'ALL ACCOUNTS'})")

    periods = [
        (
            f"Calendar Week: This Week (since {monday_this_week.strftime('%b %d')}) vs. Last Week ({monday_last_week.strftime('%b %d')}–{(monday_this_week - dt.timedelta(days=1)).strftime('%b %d')})",
            monday_this_week.timestamp(),
            now_ts,
            monday_last_week.timestamp(),
            monday_this_week.timestamp(),
        ),
        (
            "Rolling 7-Day Window: Last 7 Days vs. Previous 7 Days (Normalized Equal Duration)",
            r7_start,
            now_ts,
            prev7_start,
            r7_start,
        ),
    ]

    for title, cur_s, cur_e, prv_s, prv_e in periods:
        cur_map = get_period_metrics(conn, cur_s, cur_e, acct_filter)
        prv_map = get_period_metrics(conn, prv_s, prv_e, acct_filter)
        all_accts = sorted(set(cur_map.keys()) | set(prv_map.keys()), key=lambda x: (x == "ALL ACCOUNTS", x))

        print(f"\n{bold('▸ ' + title)}")
        if not all_accts:
            print(f"  {dim('No data available for these windows.')}")
            continue

        for acct in all_accts:
            c_m = cur_map.get(acct, defaultdict(float))
            p_m = prv_map.get(acct, defaultdict(float))

            c_cache_hit = (
                100.0 * c_m.get("cache_read_tokens", 0) / max(1.0, c_m.get("cache_read_tokens", 0) + c_m.get("fresh_input_tokens", 0))
            )
            p_cache_hit = (
                100.0 * p_m.get("cache_read_tokens", 0) / max(1.0, p_m.get("cache_read_tokens", 0) + p_m.get("fresh_input_tokens", 0))
            )

            print(f"\n  {cyan(bold(f'[{acct}]'))}")
            print(f"  {'Metric':<30} {'Last Week / Prev':>18} {'This Week / Curr':>18} {'WoW Delta':>16}")
            print("  " + "─" * 86)
            rows = [
                ("Active Sessions", fmt_num(p_m.get("sessions", 0)), fmt_num(c_m.get("sessions", 0)), fmt_delta(c_m.get("sessions", 0), p_m.get("sessions", 0))),
                ("API Turns (Requests)", fmt_num(p_m.get("turns", 0)), fmt_num(c_m.get("turns", 0)), fmt_delta(c_m.get("turns", 0), p_m.get("turns", 0))),
                ("Total Tokens Processed", fmt_num(p_m.get("total_tokens", 0)), fmt_num(c_m.get("total_tokens", 0)), fmt_delta(c_m.get("total_tokens", 0), p_m.get("total_tokens", 0))),
                ("Cache Creation (Write) Tok", fmt_num(p_m.get("cache_write_tokens", 0)), fmt_num(c_m.get("cache_write_tokens", 0)), fmt_delta(c_m.get("cache_write_tokens", 0), p_m.get("cache_write_tokens", 0))),
                ("Cache Read Hit Rate (%)", f"{p_cache_hit:.1f}%", f"{c_cache_hit:.1f}%", fmt_delta(c_cache_hit, p_cache_hit, invert_color=True)),
                ("Output + Thinking Tokens", fmt_num(p_m.get("output_tokens", 0)), fmt_num(c_m.get("output_tokens", 0)), fmt_delta(c_m.get("output_tokens", 0), p_m.get("output_tokens", 0))),
                ("Subagent Token Volume", fmt_num(p_m.get("subagent_tokens", 0)), fmt_num(c_m.get("subagent_tokens", 0)), fmt_delta(c_m.get("subagent_tokens", 0), p_m.get("subagent_tokens", 0))),
                ("MCP Payload Tokens", fmt_num(p_m.get("mcp_tokens", 0)), fmt_num(c_m.get("mcp_tokens", 0)), fmt_delta(c_m.get("mcp_tokens", 0), p_m.get("mcp_tokens", 0))),
                ("Tool Errors", fmt_num(p_m.get("tool_errors", 0)), fmt_num(c_m.get("tool_errors", 0)), fmt_delta(c_m.get("tool_errors", 0), p_m.get("tool_errors", 0))),
                ("Estimated API Cost ($)", fmt_usd(p_m.get("cost_usd", 0)), fmt_usd(c_m.get("cost_usd", 0)), fmt_delta(c_m.get("cost_usd", 0), p_m.get("cost_usd", 0))),
            ]
            for label, pv, cv, dlt in rows:
                print(f"  {label:<30} {pv:>18} {cv:>18} {dlt:>16}")


def analyze_postmortem_anomalies(
    conn: sqlite3.Connection, since_ts: float, acct_filter: Optional[str] = None, session_prefix: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detects the 5 classic Claude Code token & failure modes across sessions:
      1. [MCP_OR_TOOL_PAYLOAD_BLOAT] Single tool/MCP response >= 8k est tokens
      2. [CACHE_BUST_SPIRAL]         Mid-session cache_creation_tokens >= 35k after >5m gap or repeated cold-cache writes
      3. [SUBAGENT_AMPLIFICATION]    Subagents consumed >= 40% of session tokens or >= $2.00 in a single session
      4. [THRASHING_RETRY_LOOP]      >= 3 consecutive tool errors
      5. [AFK_APPROVAL_OR_STALL]     Subagent unfinished streaming dropout or tool stall >= 10m
    """
    anomalies: List[Dict[str, Any]] = []

    where_clauses = ["timestamp >= ?"]
    params: List[Any] = [since_ts]
    if acct_filter:
        where_clauses.append("account = ?")
        params.append(acct_filter)
    if session_prefix:
        where_clauses.append("session_id LIKE ?")
        params.append(f"{session_prefix}%")
    where_sql = " AND ".join(where_clauses)

    # Mode 1: Unbounded MCP / Tool Payload Bloat
    bloat_rows = conn.execute(
        f"""
        SELECT account, session_id, timestamp, tool_name, actor_category, target_hint,
               result_chars, est_result_tokens, induced_cache_write_tokens, is_subagent
        FROM tool_invocations
        WHERE {where_sql} AND est_result_tokens >= 8000
        ORDER BY est_result_tokens DESC
        LIMIT 15
        """,
        params,
    ).fetchall()
    for r in bloat_rows:
        sev = "HIGH" if r["est_result_tokens"] >= 20000 else "MED"
        anomalies.append(
            {
                "mode": "1. MCP_OR_TOOL_PAYLOAD_BLOAT",
                "severity": sev,
                "account": r["account"],
                "session_id": r["session_id"],
                "timestamp": r["timestamp"],
                "summary": (
                    f"{r['actor_category']} ({r['tool_name']}) injected ~{fmt_num(r['est_result_tokens'])} tok "
                    f"({fmt_num(r['result_chars'])} chars) -> induced +{fmt_num(r['induced_cache_write_tokens'])} cache-write tokens"
                ),
                "remediation": f"Target: '{r['target_hint'][:55]}'. Restrict output with head/tail/jq, line ranges, or lower screenshot resolution.",
            }
        )

    # Mode 2: Cache-Busting / Cold-Cache Spirals (5m TTL expiry or mid-conversation cache invalidation)
    req_rows = conn.execute(
        f"""
        SELECT account, session_id, is_subagent, agent_id, seq_idx, timestamp, model,
               cache_creation_tokens, cache_5m_tokens, cache_1h_tokens, cache_read_tokens,
               prev_tool_actor, cost_usd, had_streaming_null
        FROM api_requests
        WHERE {where_sql}
        ORDER BY account, session_id, is_subagent, agent_id, seq_idx ASC
        """,
        params,
    ).fetchall()

    by_stream: Dict[Tuple[str, str, int, Optional[str]], List[sqlite3.Row]] = defaultdict(list)
    for r in req_rows:
        by_stream[(r["account"], r["session_id"], r["is_subagent"], r["agent_id"])].append(r)

    for (acct, sess_id, is_sub, ag_id), srows in by_stream.items():
        prev_ts: Optional[float] = None
        cold_rewrites = 0
        wasted_cw_tokens = 0
        max_gap_min = 0.0
        first_bust_ts = srows[0]["timestamp"] if srows else 0.0

        for idx, r in enumerate(srows):
            cw = int(r["cache_creation_tokens"] or 0)
            cr = int(r["cache_read_tokens"] or 0)
            ts = float(r["timestamp"])
            if idx > 0 and prev_ts is not None:
                gap_min = (ts - prev_ts) / 60.0
                if cw >= 35000 and (gap_min >= 5.0 or cr < cw):
                    cold_rewrites += 1
                    wasted_cw_tokens += cw
                    if gap_min > max_gap_min:
                        max_gap_min = gap_min
                    if cold_rewrites == 1:
                        first_bust_ts = ts
            prev_ts = ts

            if r["had_streaming_null"] and r["cache_read_tokens"] > 0 and is_sub:
                anomalies.append(
                    {
                        "mode": "5. SUBAGENT_STREAM_DROPOUT",
                        "severity": "LOW",
                        "account": acct,
                        "session_id": sess_id,
                        "timestamp": ts,
                        "summary": f"Subagent {ag_id or ''} had incomplete streaming snapshot (output_tokens=1, stop_reason=null)",
                        "remediation": "Deduplicated by requestId in profiler; check if subagent hit timeout or max_tokens.",
                    }
                )

        if cold_rewrites >= 2 or wasted_cw_tokens >= 90000:
            anomalies.append(
                {
                    "mode": "2. CACHE_BUST_SPIRAL",
                    "severity": "HIGH" if wasted_cw_tokens >= 180000 else "MED",
                    "account": acct,
                    "session_id": sess_id,
                    "timestamp": first_bust_ts,
                    "summary": (
                        f"{cold_rewrites} cold-cache re-writes totaling {fmt_num(wasted_cw_tokens)} cache_creation tokens "
                        f"(max idle gap: {max_gap_min:.1f}m)"
                    ),
                    "remediation": "Pauses >5m expire ephemeral_5m cache, re-billing full context at 1.25x. Use /compact or /clear after long breaks.",
                }
            )

    # Mode 3: Subagent Fork / Heavy Model Amplification
    sess_where = ["end_ts >= ?"]
    s_params: List[Any] = [since_ts]
    if acct_filter:
        sess_where.append("account = ?")
        s_params.append(acct_filter)
    if session_prefix:
        sess_where.append("session_id LIKE ?")
        s_params.append(f"{session_prefix}%")

    sub_amp_rows = conn.execute(
        f"""
        SELECT account, session_id, end_ts, subagent_count,
               (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
               subagent_total_tokens, subagent_cost_usd, total_cost_usd
        FROM sessions
        WHERE {' AND '.join(sess_where)} AND subagent_count > 0
        """,
        s_params,
    ).fetchall()

    for r in sub_amp_rows:
        tot = max(1, int(r["total_tok"] or 1))
        sub_tok = int(r["subagent_total_tokens"] or 0)
        sub_pct = 100.0 * sub_tok / tot
        sub_cost = float(r["subagent_cost_usd"] or 0.0)
        if (sub_pct >= 40.0 and sub_tok >= 250000) or sub_cost >= 2.00:
            anomalies.append(
                {
                    "mode": "3. SUBAGENT_AMPLIFICATION",
                    "severity": "HIGH" if sub_cost >= 5.0 else "MED",
                    "account": r["account"],
                    "session_id": r["session_id"],
                    "timestamp": r["end_ts"],
                    "summary": (
                        f"{r['subagent_count']} subagents consumed {fmt_num(sub_tok)} tokens "
                        f"({sub_pct:.1f}% of session, {fmt_usd(sub_cost)} subagent cost)"
                    ),
                    "remediation": "Set model: sonnet or haiku in subagent definitions instead of inheriting Opus for search/exploration tasks.",
                }
            )

    # Mode 4 & 5b: Tool Retry Thrashing Loops & AFK Permission Stalls
    tool_stream_rows = conn.execute(
        f"""
        SELECT account, session_id, is_subagent, timestamp, duration_sec,
               tool_name, actor_category, target_hint, is_error, error_snippet
        FROM tool_invocations
        WHERE {where_sql}
        ORDER BY account, session_id, timestamp ASC
        """,
        params,
    ).fetchall()

    tools_by_sess: Dict[Tuple[str, str], List[sqlite3.Row]] = defaultdict(list)
    for r in tool_stream_rows:
        tools_by_sess[(r["account"], r["session_id"])].append(r)

    for (acct, sess_id), trows in tools_by_sess.items():
        consec_err = 0
        err_tools: List[str] = []
        last_err_snip = ""
        for r in trows:
            dur = float(r["duration_sec"] or 0.0)
            if dur >= 600.0:
                anomalies.append(
                    {
                        "mode": "5. AFK_APPROVAL_OR_STALL",
                        "severity": "MED",
                        "account": acct,
                        "session_id": sess_id,
                        "timestamp": r["timestamp"],
                        "summary": f"{r['tool_name']} paused for {dur / 60.0:.1f} min waiting for user approval or hung process",
                        "remediation": f"Target: '{r['target_hint'][:50]}'. An idle wait >5m also forces a cold cache rewrite on resume!",
                    }
                )

            if r["is_error"]:
                consec_err += 1
                err_tools.append(r["tool_name"])
                last_err_snip = r["error_snippet"] or ""
            else:
                if consec_err >= 3:
                    anomalies.append(
                        {
                            "mode": "4. THRASHING_RETRY_LOOP",
                            "severity": "HIGH" if consec_err >= 5 else "MED",
                            "account": acct,
                            "session_id": sess_id,
                            "timestamp": r["timestamp"],
                            "summary": f"{consec_err} consecutive failing tool calls ({', '.join(err_tools[:4])})",
                            "remediation": f"Last error: '{last_err_snip[:70]}'. Interrupt and fix environment/path when agent loops.",
                        }
                    )
                consec_err = 0
                err_tools = []

        if consec_err >= 3:
            anomalies.append(
                {
                    "mode": "4. THRASHING_RETRY_LOOP",
                    "severity": "HIGH" if consec_err >= 5 else "MED",
                    "account": acct,
                    "session_id": sess_id,
                    "timestamp": trows[-1]["timestamp"],
                    "summary": f"{consec_err} consecutive failing tool calls at end of session ({', '.join(err_tools[:4])})",
                    "remediation": f"Last error: '{last_err_snip[:70]}'.",
                }
            )

    sev_rank = {"HIGH": 0, "MED": 1, "LOW": 2}
    anomalies.sort(key=lambda x: (sev_rank.get(x["severity"], 9), -float(x["timestamp"] or 0)))
    return anomalies


def cmd_postmortem(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    acct_filter = resolve_account_filter(args.account)
    since_ts = time.time() - args.days * 86400
    sess_prefix = getattr(args, "session_id", None)

    title_target = f"Session {sess_prefix}" if sess_prefix else f"Last {args.days} Days ({acct_filter or 'ALL ACCOUNTS'})"
    print_header(f"🔬 AUTOMATED 'WHAT WENT WRONG' POST-MORTEM DIAGNOSTICS — {title_target}")

    if sess_prefix:
        srow = conn.execute(
            """
            SELECT * FROM sessions WHERE session_id LIKE ? ORDER BY end_ts DESC LIMIT 1
            """,
            (f"{sess_prefix}%",),
        ).fetchone()
        if srow:
            sid = srow["session_id"]
            acct = srow["account"]
            tot_tok = (
                int(srow["input_tokens"] or 0)
                + int(srow["output_tokens"] or 0)
                + int(srow["cache_creation_tokens"] or 0)
                + int(srow["cache_read_tokens"] or 0)
            )
            print(f"\n{bold('▸ Session Overview')}")
            print(f"  Account      : {cyan(acct)}")
            print(f"  Session ID   : {bold(sid)}")
            print(f"  Project      : {srow['project_slug']}")
            print(f"  Time Window  : {dt.datetime.fromtimestamp(srow['start_ts']).strftime('%Y-%m-%d %H:%M:%S')} -> {dt.datetime.fromtimestamp(srow['end_ts']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Turns / SubAg: {srow['turn_count']} main turns | {srow['subagent_count']} subagents")
            print(f"  Token Split  : Total={fmt_num(tot_tok)} | CacheRead={fmt_num(srow['cache_read_tokens'])} | CacheWrite={fmt_num(srow['cache_creation_tokens'])} (5m={fmt_num(srow['cache_5m_tokens'])}, 1h={fmt_num(srow['cache_1h_tokens'])}) | Out={fmt_num(srow['output_tokens'])}")
            print(f"  Total Cost   : {green(fmt_usd(srow['total_cost_usd']))} (Subagents: {fmt_usd(srow['subagent_cost_usd'])}, Background Haiku: {fmt_usd(srow['bg_haiku_cost_usd'])})")

            t_rows = conn.execute(
                """
                SELECT seq_idx, is_subagent, agent_type, timestamp, model,
                       cache_creation_tokens, cache_read_tokens, output_tokens, thinking_tokens,
                       cost_usd, tool_names, prev_tool_actor, prev_tool_result_len
                FROM api_requests
                WHERE session_id = ? AND account = ?
                ORDER BY cost_usd DESC LIMIT 10
                """,
                (sid, acct),
            ).fetchall()
            print(f"\n{bold('▸ Top 10 Most Expensive Turns in Session')}")
            print(
                f"  {'#':>4} {'Time':<9} {'Context':<18} {'CacheWrite':>11} {'CacheRead':>11} "
                f"{'Out(Thk)':>12} {'Cost':>9}  {'Triggered By -> Tools Called'}"
            )
            print("  " + "─" * 112)
            for tr in t_rows:
                t_str = dt.datetime.fromtimestamp(tr["timestamp"]).strftime("%H:%M:%S")
                ctx = f"sub:{tr['agent_type'][:13]}" if tr["is_subagent"] else "main"
                out_s = f"{fmt_num(tr['output_tokens'])}({fmt_num(tr['thinking_tokens'])})"
                trig = f"{tr['prev_tool_actor'] or 'user'} ({fmt_num(tr['prev_tool_result_len'])}c) -> {tr['tool_names'] or 'reply'}"
                print(
                    f"  {tr['seq_idx']:>4} {t_str:<9} {ctx:<18} {fmt_num(tr['cache_creation_tokens']):>11} "
                    f"{fmt_num(tr['cache_read_tokens']):>11} {out_s:>12} {fmt_usd(tr['cost_usd']):>9}  {trig[:42]}"
                )

            # Subagent Breakdown Table for this session
            sub_rows = conn.execute(
                """
                SELECT agent_id, agent_type, MAX(agent_desc) AS desc, model,
                       COUNT(*) AS turns,
                       SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS tot_tok,
                       SUM(cache_creation_tokens) AS cw_tok,
                       SUM(cost_usd) AS cost
                FROM api_requests
                WHERE session_id = ? AND account = ? AND is_subagent = 1
                GROUP BY agent_id
                ORDER BY cost DESC
                """,
                (sid, acct),
            ).fetchall()
            if sub_rows:
                print(f"\n{bold(f'▸ Subagents Breakdown ({len(sub_rows)} Subagents Spawned)')}")
                print(
                    f"  {'#':>2} {'Agent ID':<18} {'Type':<25} {'Model':<17} {'Turns':>5} "
                    f"{'Total Tok':>10} {'CacheWrite':>10} {'Cost':>9}  {'Task Description'}"
                )
                print("  " + "─" * 128)
                for idx_s, sr in enumerate(sub_rows, 1):
                    print(
                        f"  {idx_s:>2} {(sr['agent_id'] or '')[:18]:<18} {(sr['agent_type'] or '')[:25]:<25} "
                        f"{(sr['model'] or '')[:17]:<17} {sr['turns']:>5} {fmt_num(sr['tot_tok']):>10} "
                        f"{fmt_num(sr['cw_tok']):>10} {fmt_usd(sr['cost']):>9}  {(sr['desc'] or '')[:42]}"
                    )

            # Cold-Cache Bust Turns Timeline
            bust_rows = conn.execute(
                """
                SELECT seq_idx, is_subagent, agent_id, agent_type, timestamp,
                       cache_creation_tokens, cache_5m_tokens, cache_1h_tokens, cache_read_tokens,
                       cost_usd, prev_tool_actor, tool_names
                FROM api_requests
                WHERE session_id = ? AND account = ? AND cache_creation_tokens >= 35000
                ORDER BY timestamp ASC
                """,
                (sid, acct),
            ).fetchall()
            if bust_rows:
                print(f"\n{bold(f'▸ Cold-Cache Write Spikes (Turns with CacheWrite >= 35k Tokens)')}")
                print(
                    f"  {'Timestamp':<19} {'Actor / Subagent':<30} {'Turn#':>5} {'CacheWrite (5m / 1h TTL)':<28} "
                    f"{'CacheRead':>10} {'Cost':>8}  {'Trigger'}"
                )
                print("  " + "─" * 124)
                for br in bust_rows:
                    ts_s = dt.datetime.fromtimestamp(br["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    who_s = f"sub:{(br['agent_id'] or '')[:8]}({(br['agent_type'] or '')[:14]})" if br["is_subagent"] else "MAIN THREAD"
                    cw_s = f"{fmt_num(br['cache_creation_tokens'])} (5m:{fmt_num(br['cache_5m_tokens'])}/1h:{fmt_num(br['cache_1h_tokens'])})"
                    trig_s = f"{br['prev_tool_actor'] or 'user'} -> {br['tool_names'] or 'reply'}"
                    print(
                        f"  {ts_s:<19} {who_s:<30} {br['seq_idx']:>5} {cw_s:<28} "
                        f"{fmt_num(br['cache_read_tokens']):>10} {fmt_usd(br['cost_usd']):>8}  {trig_s[:32]}"
                    )

    anomalies = analyze_postmortem_anomalies(conn, 0.0 if sess_prefix else since_ts, acct_filter, sess_prefix)

    mode_counts: Dict[str, int] = defaultdict(int)
    for a in anomalies:
        mode_counts[a["mode"]] += 1

    print(f"\n{bold('▸ 5 Failure Modes Summary')}")
    all_modes = [
        "1. MCP_OR_TOOL_PAYLOAD_BLOAT",
        "2. CACHE_BUST_SPIRAL",
        "3. SUBAGENT_AMPLIFICATION",
        "4. THRASHING_RETRY_LOOP",
        "5. AFK_APPROVAL_OR_STALL",
    ]
    for m in all_modes:
        cnt = mode_counts.get(m, 0)
        badge = red(f"{cnt} detected") if cnt > 0 else green("0 clean")
        print(f"  • {m:<32}: {badge}")

    if not anomalies:
        print(f"\n  {green('✓ No high-burn anomalies or failure loops detected in this window!')}")
        return

    print(f"\n{bold('▸ Detailed Incident Log & Root-Cause Remediations')}")
    for idx, a in enumerate(anomalies[:25], 1):
        sev = a["severity"]
        sev_badge = red(f"[{sev}]") if sev == "HIGH" else (yellow(f"[{sev}]") if sev == "MED" else dim(f"[{sev}]"))
        when = dt.datetime.fromtimestamp(a["timestamp"]).strftime("%Y-%m-%d %H:%M") if a["timestamp"] else "-"
        print(
            f"\n  {idx:>2}. {sev_badge} {bold(a['mode'])} | {cyan(a['account'])} | "
            f"Session {bold(a['session_id'][:10])} ({when})"
        )
        print(f"      {bold('What happened:')} {a['summary']}")
        print(f"      {dim('How to fix   :')} {a['remediation']}")


def cmd_dashboard(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    cmd_burn(conn, args)
    cmd_who(conn, args)
    cmd_wow(conn, args)
    cmd_postmortem(conn, args)


def build_html_report_data(conn: sqlite3.Connection, days: int = 14) -> Dict[str, Any]:
    now = time.time()
    now_dt = dt.datetime.fromtimestamp(now)
    since_ts = now - days * 86400

    # Discover all accounts present in DB
    accts_in_db = [
        r["account"]
        for r in conn.execute("SELECT DISTINCT account FROM api_requests ORDER BY account").fetchall()
        if r["account"]
    ]
    if not accts_in_db:
        accts_in_db = sorted(ACCOUNT_DIRS.keys())

    # 1. Daily time-series per account for the last `days` days
    start_day_dt = (now_dt - dt.timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_labels: List[str] = []
    for i in range(days):
        d = start_day_dt + dt.timedelta(days=i)
        day_labels.append(d.strftime("%Y-%m-%d"))

    daily_rows = conn.execute(
        """
        SELECT
            account,
            strftime('%Y-%m-%d', timestamp, 'unixepoch', 'localtime') AS day_str,
            SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
            SUM(input_tokens + output_tokens + cache_creation_tokens) AS fresh_tok,
            SUM(cache_creation_tokens) AS cw_tok,
            SUM(CASE WHEN is_subagent = 1 THEN (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) ELSE 0 END) AS sub_tok,
            SUM(cost_usd) AS cost_usd,
            COUNT(*) AS turns
        FROM api_requests
        WHERE timestamp >= ?
        GROUP BY account, day_str
        ORDER BY day_str ASC
        """,
        (start_day_dt.timestamp(),),
    ).fetchall()

    daily_by_acct: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"total_tok": 0.0, "fresh_tok": 0.0, "cw_tok": 0.0, "sub_tok": 0.0, "cost_usd": 0.0, "turns": 0.0})
    )
    for r in daily_rows:
        d_str = r["day_str"]
        if not d_str:
            continue
        acct = r["account"]
        for k in ("total_tok", "fresh_tok", "cw_tok", "sub_tok", "cost_usd", "turns"):
            val = float(r[k] or 0.0)
            daily_by_acct[acct][d_str][k] += val
            daily_by_acct["ALL"][d_str][k] += val

    # 2. Predictive Forecasting Engine (7-Day Forward Forecast using Weighted EMA + Linear Trend)
    forecast_days = 7
    future_labels: List[str] = [
        (now_dt + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, forecast_days + 1)
    ]

    forecasts_by_acct: Dict[str, Dict[str, Any]] = {}
    for acct_key in ["ALL"] + accts_in_db:
        cost_series = [daily_by_acct[acct_key][d]["cost_usd"] for d in day_labels]
        tok_series = [daily_by_acct[acct_key][d]["total_tok"] for d in day_labels]
        fresh_series = [daily_by_acct[acct_key][d]["fresh_tok"] for d in day_labels]

        def project_series(series: List[float]) -> Tuple[List[float], List[float], List[float], float, float]:
            n = len(series)
            if n == 0 or sum(series) == 0:
                zeros = [0.0] * forecast_days
                return zeros, zeros, zeros, 0.0, 0.0

            # Exclude or normalize the partial current day (series[-1]) so early-day runs
            # do not treat a few hours of usage as a full-day drop.
            if n > 1:
                completed = list(series[:-1])
                today_raw = series[-1]
                elapsed_frac = max(0.15, (now_dt.hour * 3600 + now_dt.minute * 60 + now_dt.second) / 86400.0)
                hist_mean = sum(completed) / len(completed)
                if today_raw > 0:
                    today_est = min(max(today_raw, hist_mean * 1.5), today_raw / elapsed_frac)
                    fit_series = completed + [0.5 * today_est + 0.5 * hist_mean]
                else:
                    fit_series = completed
            else:
                fit_series = list(series)

            fn = len(fit_series)
            if fn == 0 or sum(fit_series) == 0:
                zeros = [0.0] * forecast_days
                return zeros, zeros, zeros, 0.0, 0.0

            # Compute Exponential Moving Average (alpha = 0.35)
            alpha = 0.35
            ema = fit_series[0]
            for val in fit_series[1:]:
                ema = alpha * val + (1.0 - alpha) * ema
            # Compute linear regression slope over the last min(fn, 10) completed/normalized days
            w = min(fn, 10)
            sub = fit_series[-w:]
            x_mean = (w - 1) / 2.0
            y_mean = sum(sub) / w
            num = sum((i - x_mean) * (sub[i] - y_mean) for i in range(w))
            den = sum((i - x_mean) ** 2 for i in range(w)) or 1.0
            slope = num / den
            # Standard deviation of active days for upper/lower confidence bands
            variance = sum((v - y_mean) ** 2 for v in sub) / max(1, w)
            std_dev = variance ** 0.5

            expected: List[float] = []
            high_band: List[float] = []
            low_band: List[float] = []
            for step in range(1, forecast_days + 1):
                # Damped trend blended with EMA
                pred = max(0.0, 0.65 * ema + 0.35 * y_mean + slope * step * 0.4)
                expected.append(round(pred, 2))
                high_band.append(round(pred + 0.75 * std_dev, 2))
                low_band.append(round(max(0.0, pred - 0.5 * std_dev), 2))
            proj_7d = sum(expected)
            proj_30d = proj_7d * (30.0 / 7.0)
            return expected, low_band, high_band, proj_7d, proj_30d

        c_exp, c_low, c_high, c_7d, c_30d = project_series(cost_series)
        t_exp, t_low, t_high, t_7d, t_30d = project_series(tok_series)
        f_exp, _, _, f_7d, f_30d = project_series(fresh_series)

        forecasts_by_acct[acct_key] = {
            "cost_expected": c_exp,
            "cost_low": c_low,
            "cost_high": c_high,
            "cost_7d": round(c_7d, 2),
            "cost_30d": round(c_30d, 2),
            "tok_expected": t_exp,
            "tok_low": t_low,
            "tok_high": t_high,
            "tok_7d": round(t_7d, 0),
            "tok_30d": round(t_30d, 0),
            "fresh_expected": f_exp,
            "fresh_7d": round(f_7d, 0),
            "fresh_30d": round(f_30d, 0),
        }

    # 3. 5-Hour Rolling Window & P90 Empirical Quota Exhaustion Prediction
    # Group historical requests into 5-hour buckets over the last 30 days to find empirical P90 & Max 5h block ceiling per account
    hist_5h_rows = conn.execute(
        """
        SELECT
            account,
            CAST(timestamp / 18000 AS INTEGER) AS bucket_5h,
            SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
            SUM(input_tokens + output_tokens + cache_creation_tokens) AS fresh_tok,
            SUM(cost_usd) AS cost_usd
        FROM api_requests
        WHERE timestamp >= ?
        GROUP BY account, bucket_5h
        """,
        (now - 30 * 86400,),
    ).fetchall()

    buckets_by_acct: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    combined_5h_buckets: Dict[int, Dict[str, float]] = defaultdict(lambda: {"total_tok": 0.0, "fresh_tok": 0.0, "cost_usd": 0.0})
    for r in hist_5h_rows:
        entry = {"total_tok": float(r["total_tok"] or 0), "fresh_tok": float(r["fresh_tok"] or 0), "cost_usd": float(r["cost_usd"] or 0)}
        buckets_by_acct[r["account"]].append(entry)
        b_id = int(r["bucket_5h"] or 0)
        combined_5h_buckets[b_id]["total_tok"] += entry["total_tok"]
        combined_5h_buckets[b_id]["fresh_tok"] += entry["fresh_tok"]
        combined_5h_buckets[b_id]["cost_usd"] += entry["cost_usd"]
    buckets_by_acct["ALL"] = list(combined_5h_buckets.values())

    def percentile(vals: List[float], q: float, default_val: float) -> float:
        if not vals:
            return default_val
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return max(s[idx], default_val)

    burn_5h = compute_active_burn_rate(conn, now - 5 * 3600)
    burn_1h = compute_active_burn_rate(conn, now - 3600)
    burn_window = compute_active_burn_rate(conn, since_ts)

    quota_predictions: Dict[str, Dict[str, Any]] = {}
    for acct_key in ["ALL"] + accts_in_db:
        b_list = buckets_by_acct.get(acct_key, [])
        p90_cost = percentile([x["cost_usd"] for x in b_list], 0.90, 45.0)
        max_cost = percentile([x["cost_usd"] for x in b_list], 1.00, 60.0)
        p90_tok = percentile([x["total_tok"] for x in b_list], 0.90, 80_000_000.0)

        if acct_key == "ALL":
            cur_5h_cost = sum(v["cost_usd"] for v in burn_5h.values())
            cur_5h_tok = sum(v["total_tok"] for v in burn_5h.values())
            cur_5h_fresh = sum(v["fresh_tok"] for v in burn_5h.values())
            cur_5h_sec = sum(v["active_sec"] for v in burn_5h.values())
            cur_1h_cost = sum(v["cost_usd"] for v in burn_1h.values())
            cur_1h_tok = sum(v["total_tok"] for v in burn_1h.values())
            cur_1h_sec = sum(v["active_sec"] for v in burn_1h.values())
        else:
            s5 = burn_5h.get(acct_key, {"cost_usd": 0.0, "total_tok": 0.0, "fresh_tok": 0.0, "active_sec": 0.0})
            s1 = burn_1h.get(acct_key, {"cost_usd": 0.0, "total_tok": 0.0, "active_sec": 0.0})
            cur_5h_cost = s5["cost_usd"]
            cur_5h_tok = s5["total_tok"]
            cur_5h_fresh = s5["fresh_tok"]
            cur_5h_sec = s5["active_sec"]
            cur_1h_cost = s1["cost_usd"]
            cur_1h_tok = s1["total_tok"]
            cur_1h_sec = s1["active_sec"]

        active_min_5h = max(0.25, cur_5h_sec / 60.0) if cur_5h_sec > 0 else 0.0
        tok_per_min = (cur_5h_tok / active_min_5h) if active_min_5h > 0 else 0.0
        usd_per_hr = (cur_5h_cost / (active_min_5h / 60.0)) if active_min_5h > 0 else 0.0

        fill_pct = min(100.0, 100.0 * cur_5h_cost / max(1.0, p90_cost))
        rem_cost = max(0.0, p90_cost - cur_5h_cost)
        if cur_1h_cost > 0.25 and usd_per_hr > 0.5:
            hrs_left = rem_cost / usd_per_hr
            if hrs_left <= 0.05:
                eta_str = "Near/At P90 Cap!"
                status_level = "critical"
            elif hrs_left < 5.0:
                h = int(hrs_left)
                m = int((hrs_left - h) * 60)
                eta_str = f"~{h}h {m:02d}m to P90 cap" if h > 0 else f"~{m}m to P90 cap"
                status_level = "warning" if hrs_left < 2.0 else "active"
            else:
                eta_str = ">5h headroom (Safe)"
                status_level = "safe"
        else:
            eta_str = "Idle / Safe"
            status_level = "safe"

        quota_predictions[acct_key] = {
            "cur_5h_cost": round(cur_5h_cost, 2),
            "cur_5h_tok": round(cur_5h_tok, 0),
            "cur_5h_fresh": round(cur_5h_fresh, 0),
            "p90_5h_cost": round(p90_cost, 2),
            "max_5h_cost": round(max_cost, 2),
            "p90_5h_tok": round(p90_tok, 0),
            "tok_per_min": round(tok_per_min, 0),
            "usd_per_hr": round(usd_per_hr, 2),
            "fill_pct": round(fill_pct, 1),
            "eta_str": eta_str,
            "status_level": status_level,
        }

    # 4. Hourly Velocity over Last 48 Hours
    h48_start = (now_dt - dt.timedelta(hours=47)).replace(minute=0, second=0, microsecond=0)
    hour_labels: List[str] = [
        (h48_start + dt.timedelta(hours=i)).strftime("%m-%d %H:00") for i in range(48)
    ]
    h48_rows = conn.execute(
        """
        SELECT
            account,
            strftime('%m-%d %H:00', timestamp, 'unixepoch', 'localtime') AS hr_str,
            SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
            SUM(cost_usd) AS cost_usd
        FROM api_requests
        WHERE timestamp >= ?
        GROUP BY account, hr_str
        """,
        (h48_start.timestamp(),),
    ).fetchall()
    hourly_by_acct: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"total_tok": 0.0, "cost_usd": 0.0})
    )
    for r in h48_rows:
        hs = r["hr_str"]
        if not hs:
            continue
        hourly_by_acct[r["account"]][hs]["total_tok"] += float(r["total_tok"] or 0.0)
        hourly_by_acct[r["account"]][hs]["cost_usd"] += float(r["cost_usd"] or 0.0)
        hourly_by_acct["ALL"][hs]["total_tok"] += float(r["total_tok"] or 0.0)
        hourly_by_acct["ALL"][hs]["cost_usd"] += float(r["cost_usd"] or 0.0)

    # 5. Actors & Subagents Breakdown
    actor_rows = conn.execute(
        """
        SELECT
            account,
            CASE WHEN is_subagent = 1 THEN 'Subagent: ' || COALESCE(agent_type, 'general') ELSE 'Main Session Agent' END AS actor,
            COUNT(*) AS turns,
            SUM(input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
            SUM(cache_creation_tokens) AS cw_tok,
            SUM(output_tokens) AS out_tok,
            SUM(thinking_tokens) AS think_tok,
            SUM(cost_usd) AS cost_usd
        FROM api_requests
        WHERE timestamp >= ?
        GROUP BY account, actor
        ORDER BY cost_usd DESC
        """,
        (since_ts,),
    ).fetchall()

    actors_by_acct: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    actors_all_agg: Dict[str, Dict[str, Any]] = {}
    for r in actor_rows:
        item = {
            "actor": r["actor"],
            "turns": int(r["turns"] or 0),
            "total_tok": float(r["total_tok"] or 0),
            "cw_tok": float(r["cw_tok"] or 0),
            "out_tok": float(r["out_tok"] or 0),
            "think_tok": float(r["think_tok"] or 0),
            "cost_usd": round(float(r["cost_usd"] or 0.0), 2),
        }
        actors_by_acct[r["account"]].append(item)
        if r["actor"] not in actors_all_agg:
            actors_all_agg[r["actor"]] = dict(item)
        else:
            for k in ("turns", "total_tok", "cw_tok", "out_tok", "think_tok", "cost_usd"):
                actors_all_agg[r["actor"]][k] = round(actors_all_agg[r["actor"]][k] + item[k], 2)
    actors_by_acct["ALL"] = sorted(actors_all_agg.values(), key=lambda x: -x["cost_usd"])

    # 6. Top MCP Servers & Tools Attribution
    tool_rows = conn.execute(
        """
        SELECT
            account,
            actor_category,
            tool_name,
            mcp_server,
            mcp_tool,
            COUNT(*) AS calls,
            SUM(is_error) AS errors,
            SUM(est_result_tokens) AS payload_tok,
            SUM(induced_cache_write_tokens) AS induced_cw,
            AVG(duration_sec) AS avg_sec
        FROM tool_invocations
        WHERE timestamp >= ?
        GROUP BY account, actor_category, tool_name
        ORDER BY induced_cw DESC, payload_tok DESC
        """,
        (since_ts,),
    ).fetchall()

    tools_by_acct: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tools_all_agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in tool_rows:
        label = f"mcp:{r['mcp_server']}/{r['mcp_tool']}" if r["mcp_server"] else r["tool_name"]
        item = {
            "label": label[:36],
            "category": r["actor_category"],
            "is_mcp": bool(r["mcp_server"]),
            "calls": int(r["calls"] or 0),
            "errors": int(r["errors"] or 0),
            "payload_tok": int(r["payload_tok"] or 0),
            "induced_cw": int(r["induced_cw"] or 0),
            "avg_sec": round(float(r["avg_sec"] or 0.0), 1),
        }
        tools_by_acct[r["account"]].append(item)
        key = (r["actor_category"], label[:36])
        if key not in tools_all_agg:
            tools_all_agg[key] = dict(item)
        else:
            for k in ("calls", "errors", "payload_tok", "induced_cw"):
                tools_all_agg[key][k] += item[k]
    tools_by_acct["ALL"] = sorted(tools_all_agg.values(), key=lambda x: -(x["induced_cw"] + x["payload_tok"]))[:25]
    for acct in accts_in_db:
        tools_by_acct[acct] = sorted(tools_by_acct[acct], key=lambda x: -(x["induced_cw"] + x["payload_tok"]))[:25]

    # 7. Week-over-Week (Rolling 7d vs Previous 7d)
    r7_start = now - 7 * 86400
    prev7_start = now - 14 * 86400
    cur_wow = get_period_metrics(conn, r7_start, now)
    prv_wow = get_period_metrics(conn, prev7_start, r7_start)

    # 8. Top Sessions (up to 20 per account so no account tab is starved) & Post-Mortem Anomalies
    sess_rows = conn.execute(
        """
        WITH ranked_sessions AS (
            SELECT account, session_id, project_slug, start_ts, end_ts, turn_count, subagent_count,
                   (input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens) AS total_tok,
                   cache_creation_tokens, cache_read_tokens, output_tokens, subagent_total_tokens,
                   subagent_cost_usd, total_cost_usd,
                   ROW_NUMBER() OVER (PARTITION BY account ORDER BY total_cost_usd DESC) AS rn
            FROM sessions
            WHERE end_ts >= ? AND turn_count >= 1
        )
        SELECT * FROM ranked_sessions
        WHERE rn <= 20
        ORDER BY total_cost_usd DESC
        """,
        (since_ts,),
    ).fetchall()
    top_sessions = [
        {
            "account": r["account"],
            "session_id": r["session_id"],
            "project": r["project_slug"] or "",
            "end_time": dt.datetime.fromtimestamp(r["end_ts"]).strftime("%Y-%m-%d %H:%M"),
            "turns": int(r["turn_count"] or 0),
            "subagents": int(r["subagent_count"] or 0),
            "total_tok": int(r["total_tok"] or 0),
            "cw_tok": int(r["cache_creation_tokens"] or 0),
            "sub_tok": int(r["subagent_total_tokens"] or 0),
            "sub_cost": round(float(r["subagent_cost_usd"] or 0.0), 2),
            "total_cost": round(float(r["total_cost_usd"] or 0.0), 2),
        }
        for r in sess_rows
    ]

    raw_anomalies = analyze_postmortem_anomalies(conn, since_ts)
    anomalies_list = [
        {
            "mode": a["mode"],
            "severity": a["severity"],
            "account": a["account"],
            "session_id": a["session_id"],
            "time_str": dt.datetime.fromtimestamp(a["timestamp"]).strftime("%Y-%m-%d %H:%M") if a["timestamp"] else "-",
            "summary": a["summary"],
            "remediation": a["remediation"],
        }
        for a in raw_anomalies[:50]
    ]

    # Serialize daily & hourly series cleanly
    daily_serialized: Dict[str, Dict[str, List[float]]] = {}
    for acct_key in ["ALL"] + accts_in_db:
        daily_serialized[acct_key] = {
            "cost_usd": [round(daily_by_acct[acct_key][d]["cost_usd"], 2) for d in day_labels],
            "total_tok": [round(daily_by_acct[acct_key][d]["total_tok"], 0) for d in day_labels],
            "fresh_tok": [round(daily_by_acct[acct_key][d]["fresh_tok"], 0) for d in day_labels],
            "sub_tok": [round(daily_by_acct[acct_key][d]["sub_tok"], 0) for d in day_labels],
            "turns": [int(daily_by_acct[acct_key][d]["turns"]) for d in day_labels],
        }

    hourly_serialized: Dict[str, Dict[str, List[float]]] = {}
    for acct_key in ["ALL"] + accts_in_db:
        hourly_serialized[acct_key] = {
            "cost_usd": [round(hourly_by_acct[acct_key][h]["cost_usd"], 2) for h in hour_labels],
            "total_tok": [round(hourly_by_acct[acct_key][h]["total_tok"], 0) for h in hour_labels],
        }

    return {
        "generated_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "days": days,
        "accounts": accts_in_db,
        "day_labels": day_labels,
        "future_labels": future_labels,
        "daily": daily_serialized,
        "forecasts": forecasts_by_acct,
        "quota_predictions": quota_predictions,
        "hour_labels": hour_labels,
        "hourly": hourly_serialized,
        "actors": dict(actors_by_acct),
        "tools": dict(tools_by_acct),
        "wow_curr": cur_wow,
        "wow_prev": prv_wow,
        "top_sessions": top_sessions,
        "anomalies": anomalies_list,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Claude Code Token Profiler — Analytics & Predictive Forecast</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0b0f17;
      --card: #131b2e;
      --card-border: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --purple: #a855f7;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
      padding: 24px 32px 64px;
    }
    .container { max-width: 1480px; margin: 0 auto; }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }
    .title-group h1 {
      font-size: 1.55rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .title-group p { color: var(--muted); font-size: 0.85rem; margin-top: 4px; }
    .tabs {
      display: flex;
      gap: 8px;
      background: #0f172a;
      padding: 5px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }
    .tab-btn {
      background: transparent;
      color: var(--muted);
      border: none;
      padding: 8px 16px;
      border-radius: 7px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active {
      background: linear-gradient(135deg, #0284c7, #6366f1);
      color: #fff;
      box-shadow: 0 2px 10px rgba(56, 189, 248, 0.25);
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }
    .kpi-val { font-size: 1.75rem; font-weight: 700; margin: 6px 0 4px; color: #f8fafc; }
    .kpi-sub { font-size: 0.82rem; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 600;
    }
    .badge-safe { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-active { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-critical { background: rgba(244, 63, 94, 0.18); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.35); }
    .progress-bar {
      width: 100%;
      height: 8px;
      background: #0f172a;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 10px;
      border: 1px solid #1e293b;
    }
    .progress-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.4s ease;
    }
    .charts-grid-2 {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }
    .charts-grid-equal {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 24px;
    }
    @media (max-width: 1080px) {
      .charts-grid-2, .charts-grid-equal { grid-template-columns: 1fr; }
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .card-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; }
    .card-subtitle { font-size: 0.78rem; color: var(--muted); }
    .chart-box { position: relative; height: 320px; width: 100%; }
    .metric-switch {
      display: flex;
      gap: 6px;
      background: #0f172a;
      padding: 3px;
      border-radius: 7px;
      border: 1px solid var(--card-border);
    }
    .metric-btn {
      background: transparent;
      color: var(--muted);
      border: none;
      padding: 4px 10px;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 5px;
      cursor: pointer;
    }
    .metric-btn.active { background: #1e293b; color: #38bdf8; }
    table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #1e293b;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: rgba(15, 23, 42, 0.5);
    }
    tr:hover td { background: rgba(30, 41, 59, 0.35); }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .up-bad { color: #fb7185; font-weight: 600; }
    .down-good { color: #34d399; font-weight: 600; }
    .anomaly-list { display: flex; flex-direction: column; gap: 10px; max-height: 480px; overflow-y: auto; padding-right: 4px; }
    .anomaly-item {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-left: 4px solid #f43f5e;
      border-radius: 8px;
      padding: 12px 14px;
    }
    .anomaly-item.MED { border-left-color: #f59e0b; }
    .anomaly-item.LOW { border-left-color: #38bdf8; }
    .anomaly-top { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; color: var(--muted); }
    .anomaly-what { font-size: 0.88rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
    .anomaly-fix { font-size: 0.8rem; color: #94a3b8; }
    code { background: #1e293b; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; color: #38bdf8; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>⚡ Claude Code Multi-Account Token Profiler & Forecast</h1>
        <p id="report-meta">Generated: —</p>
      </div>
      <div class="tabs" id="account-tabs"></div>
    </header>

    <!-- Predictive & Burn Rate KPI Cards -->
    <div class="kpi-grid" id="kpi-container"></div>

    <!-- Row 1: Daily Usage + 7-Day Forecast & 48h Rolling Burn Trajectory -->
    <div class="charts-grid-2">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">📈 Daily Consumption & 7-Day Predictive Forecast (EMA + Trend)</div>
            <div class="card-subtitle">Solid bars = Historical daily usage | Dashed line & shaded band = Next 7 days forecast</div>
          </div>
          <div class="metric-switch">
            <button class="metric-btn active" onclick="setForecastMetric('cost_usd', this)">Cost ($)</button>
            <button class="metric-btn" onclick="setForecastMetric('total_tok', this)">Total Tokens</button>
            <button class="metric-btn" onclick="setForecastMetric('fresh_tok', this)">Fresh CacheWrite</button>
          </div>
        </div>
        <div class="chart-box"><canvas id="forecastChart"></canvas></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🔥 Last 48 Hours Burn Velocity</div>
            <div class="card-subtitle">Hourly token & cost spikes across active sessions</div>
          </div>
        </div>
        <div class="chart-box"><canvas id="hourlyChart"></canvas></div>
      </div>
    </div>

    <!-- Row 2: Who Uses Tokens (Actors/Subagents) & Top MCP/Tool Attribution -->
    <div class="charts-grid-equal">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🕵️ Who Uses Tokens: Main Agent vs. Subagent Types</div>
            <div class="card-subtitle">Share of API-equivalent cost by execution context</div>
          </div>
        </div>
        <div class="chart-box"><canvas id="actorChart"></canvas></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🔌 Top MCP Servers & Tools Token Attribution</div>
            <div class="card-subtitle">Direct Payload Tokens Injected vs. Induced Prompt-Cache Writes</div>
          </div>
        </div>
        <div class="chart-box"><canvas id="toolChart"></canvas></div>
      </div>
    </div>

    <!-- Row 3: Week-over-Week Comparison & 5-Mode Post-Mortem Diagnostics -->
    <div class="charts-grid-equal">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">📊 Week-over-Week (Rolling 7d vs. Previous 7d)</div>
            <div class="card-subtitle">Normalized 7-day window comparison</div>
          </div>
        </div>
        <div id="wow-table-wrap"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🔬 Automated "What Went Wrong" Post-Mortem Incidents</div>
            <div class="card-subtitle">5 failure modes: MCP bloat, 5m TTL cold-cache busts, Opus subagent amplification, retry loops & AFK stalls</div>
          </div>
        </div>
        <div class="anomaly-list" id="anomaly-container"></div>
      </div>
    </div>

    <!-- Row 4: Top Highest-Burn Sessions Table -->
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">🏆 Top Highest-Burn Sessions</div>
          <div class="card-subtitle">Tip: Run <code>claude-profiler postmortem &lt;session_id&gt;</code> in your terminal for turn-by-turn & subagent tables</div>
        </div>
      </div>
      <div style="overflow-x:auto" id="sessions-table-wrap"></div>
    </div>
  </div>

  <script>
    const DATA = __REPORT_DATA_JSON__;
    let currentAccount = 'ALL';
    let currentMetric = 'cost_usd';
    let forecastChartInst = null;
    let hourlyChartInst = null;
    let actorChartInst = null;
    let toolChartInst = null;

    const ACCT_COLORS = {
      'claude-work': '#38bdf8',
      'claude-dev': '#a855f7',
      'claude-personal': '#10b981',
      'claude-default': '#f59e0b'
    };

    function fmtNum(n) {
      if (n == null || isNaN(n)) return '0';
      const abs = Math.abs(n);
      if (abs >= 1e9) return (n / 1e9).toFixed(2) + 'B';
      if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M';
      if (abs >= 1e4) return (n / 1e3).toFixed(1) + 'k';
      if (abs >= 1e3) return Math.round(n).toLocaleString();
      return Number.isInteger(n) ? String(n) : n.toFixed(1);
    }

    function fmtUsd(n) {
      if (n == null || isNaN(n)) return '$0.00';
      return '$' + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function esc(v) {
      return String(v ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[ch]));
    }

    function showChartOfflineFallback(canvasId) {
      const canvas = document.getElementById(canvasId);
      if (!canvas || !canvas.parentElement) return;
      canvas.parentElement.innerHTML =
        '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);font-size:0.85rem;text-align:center;padding:16px;">Chart.js CDN unavailable offline — KPI cards, forecasts, WoW tables, post-mortem incidents, and session tables remain active below.</div>';
    }

    function init() {
      document.getElementById('report-meta').textContent =
        `Generated: ${DATA.generated_at} • Lookback: Last ${DATA.days} Days • Accounts: ${DATA.accounts.join(', ')}`;

      const tabsEl = document.getElementById('account-tabs');
      const allTabs = ['ALL', ...DATA.accounts];
      tabsEl.innerHTML = allTabs.map((acct, idx) => `
        <button class="tab-btn ${acct === currentAccount ? 'active' : ''}" data-acct-idx="${idx}">
          ${acct === 'ALL' ? 'All Accounts' : esc(acct)}
        </button>
      `).join('');
      tabsEl.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const acct = allTabs[Number(btn.getAttribute('data-acct-idx'))] || 'ALL';
          selectAccount(acct, btn);
        });
      });

      renderAll();
    }

    function selectAccount(acct, btnEl) {
      currentAccount = acct;
      document.querySelectorAll('#account-tabs .tab-btn').forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');
      renderAll();
    }

    function setForecastMetric(metric, btnEl) {
      currentMetric = metric;
      document.querySelectorAll('.metric-switch .metric-btn').forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');
      renderForecastChart();
    }

    function renderAll() {
      renderKPIs();
      renderForecastChart();
      renderHourlyChart();
      renderActorChart();
      renderToolChart();
      renderWoWTable();
      renderAnomalies();
      renderSessionsTable();
    }

    function renderKPIs() {
      const q = DATA.quota_predictions[currentAccount] || DATA.quota_predictions['ALL'];
      const f = DATA.forecasts[currentAccount] || DATA.forecasts['ALL'];
      const wowKey = currentAccount === 'ALL' ? 'ALL ACCOUNTS' : currentAccount;
      const curW = (DATA.wow_curr && DATA.wow_curr[wowKey]) || {};
      const prvW = (DATA.wow_prev && DATA.wow_prev[wowKey]) || {};

      const badgeClass = `badge badge-${q.status_level || 'safe'}`;
      const barColor = q.fill_pct >= 85 ? '#f43f5e' : (q.fill_pct >= 60 ? '#f59e0b' : '#38bdf8');

      const cur7Cost = curW.cost_usd || 0;
      const prv7Cost = prvW.cost_usd || 0;
      const wowPct = prv7Cost > 0 ? (((cur7Cost - prv7Cost) / prv7Cost) * 100).toFixed(1) : '0.0';

      document.getElementById('kpi-container').innerHTML = `
        <div class="card">
          <div class="kpi-label">⏱️ Rolling 5h Window & Exhaustion ETA</div>
          <div class="kpi-val">${fmtUsd(q.cur_5h_cost)} <span style="font-size:0.95rem;color:var(--muted);font-weight:500">/ ${fmtUsd(q.p90_5h_cost)} P90</span></div>
          <div class="kpi-sub">
            <span>${fmtNum(q.cur_5h_tok)} tok (${fmtNum(q.cur_5h_fresh)} write)</span>
            <span class="${badgeClass}">${q.eta_str}</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" style="width:${Math.min(100, q.fill_pct)}%;background:${barColor}"></div>
          </div>
        </div>

        <div class="card">
          <div class="kpi-label">🔥 Live Active Burn Velocity</div>
          <div class="kpi-val">${fmtNum(q.tok_per_min)} <span style="font-size:1rem;color:var(--muted)">tok/min</span></div>
          <div class="kpi-sub">
            <span>Cost Velocity: <strong style="color:#38bdf8">${fmtUsd(q.usd_per_hr)}/hr</strong></span>
            <span>5h Fill: <strong>${q.fill_pct}%</strong></span>
          </div>
        </div>

        <div class="card">
          <div class="kpi-label">🔮 Next 7-Day Predictive Forecast</div>
          <div class="kpi-val">${fmtUsd(f.cost_7d)}</div>
          <div class="kpi-sub">
            <span>Projected Tokens: <strong>${fmtNum(f.tok_7d)}</strong></span>
            <span>Fresh Write: <strong>${fmtNum(f.fresh_7d)}</strong></span>
          </div>
        </div>

        <div class="card">
          <div class="kpi-label">📅 30-Day Run-Rate Projection</div>
          <div class="kpi-val">${fmtUsd(f.cost_30d)}</div>
          <div class="kpi-sub">
            <span>30d Tokens: <strong>${fmtNum(f.tok_30d)}</strong></span>
            <span>7d WoW Trend: <strong class="${Number(wowPct) > 0 ? 'up-bad' : 'down-good'}">${Number(wowPct) > 0 ? '+' : ''}${wowPct}%</strong></span>
          </div>
        </div>
      `;
    }

    function renderForecastChart() {
      if (typeof Chart === 'undefined') {
        showChartOfflineFallback('forecastChart');
        return;
      }
      const canvas = document.getElementById('forecastChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (forecastChartInst) forecastChartInst.destroy();

      const histLabels = DATA.day_labels.map(d => d.slice(5));
      const futLabels = DATA.future_labels.map(d => d.slice(5) + ' (proj)');
      const allLabels = [...histLabels, ...futLabels];
      const histLen = histLabels.length;
      const futLen = futLabels.length;

      const fData = DATA.forecasts[currentAccount] || DATA.forecasts['ALL'];
      const expKey = currentMetric === 'cost_usd' ? 'cost_expected' : (currentMetric === 'total_tok' ? 'tok_expected' : 'fresh_expected');
      const highKey = currentMetric === 'cost_usd' ? 'cost_high' : 'tok_high';
      const lowKey = currentMetric === 'cost_usd' ? 'cost_low' : 'tok_low';

      const datasets = [];
      if (currentAccount === 'ALL') {
        DATA.accounts.forEach(acct => {
          const s = (DATA.daily[acct] && DATA.daily[acct][currentMetric]) || [];
          datasets.push({
            type: 'bar',
            label: acct,
            data: [...s, ...Array(futLen).fill(null)],
            backgroundColor: ACCT_COLORS[acct] || '#38bdf8',
            stack: 'hist',
            borderRadius: 4
          });
        });
      } else {
        const s = (DATA.daily[currentAccount] && DATA.daily[currentAccount][currentMetric]) || [];
        datasets.push({
          type: 'bar',
          label: currentAccount,
          data: [...s, ...Array(futLen).fill(null)],
          backgroundColor: ACCT_COLORS[currentAccount] || '#38bdf8',
          stack: 'hist',
          borderRadius: 4
        });
      }

      const totHist = (DATA.daily[currentAccount] && DATA.daily[currentAccount][currentMetric]) || [];
      const lastActual = totHist.length ? totHist[totHist.length - 1] : 0;
      const forecastLine = [...Array(Math.max(0, histLen - 1)).fill(null), lastActual, ...(fData[expKey] || [])];

      datasets.push({
        type: 'line',
        label: '7-Day Forecast (EMA + Trend)',
        data: forecastLine,
        stack: 'forecast',
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.15)',
        borderDash: [6, 4],
        borderWidth: 2.5,
        pointRadius: 3,
        tension: 0.25
      });

      if (currentMetric !== 'fresh_tok' && fData[highKey]) {
        const highLine = [...Array(Math.max(0, histLen - 1)).fill(null), lastActual, ...fData[highKey]];
        datasets.push({
          type: 'line',
          label: 'High-Burst Band (+0.75σ)',
          data: highLine,
          stack: 'band',
          borderColor: 'rgba(244, 63, 94, 0.45)',
          borderDash: [3, 3],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.25
        });
      }

      forecastChartInst = new Chart(ctx, {
        data: { labels: allLabels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { labels: { color: '#cbd5e1', font: { size: 11 } } },
            tooltip: {
              callbacks: {
                label: (c) => `${c.dataset.label}: ${currentMetric === 'cost_usd' ? fmtUsd(c.raw) : fmtNum(c.raw)}`
              }
            }
          },
          scales: {
            x: { stacked: true, ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: '#1e293b' } },
            y: {
              stacked: true,
              ticks: {
                color: '#94a3b8',
                callback: (v) => currentMetric === 'cost_usd' ? '$' + v : fmtNum(v)
              },
              grid: { color: '#1e293b' }
            }
          }
        }
      });
    }

    function renderHourlyChart() {
      if (typeof Chart === 'undefined') {
        showChartOfflineFallback('hourlyChart');
        return;
      }
      const canvas = document.getElementById('hourlyChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (hourlyChartInst) hourlyChartInst.destroy();

      const hData = DATA.hourly[currentAccount] || DATA.hourly['ALL'];
      hourlyChartInst = new Chart(ctx, {
        type: 'line',
        data: {
          labels: DATA.hour_labels.map(h => h.slice(6)),
          datasets: [
            {
              label: 'Hourly Cost ($)',
              data: hData ? hData.cost_usd : [],
              borderColor: '#38bdf8',
              backgroundColor: 'rgba(56, 189, 248, 0.15)',
              fill: true,
              tension: 0.3,
              pointRadius: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#cbd5e1' } },
            tooltip: { callbacks: { label: (c) => `Hourly Burn: ${fmtUsd(c.raw)}` } }
          },
          scales: {
            x: { ticks: { color: '#94a3b8', maxTicksLimit: 8 }, grid: { color: '#1e293b' } },
            y: { ticks: { color: '#94a3b8', callback: v => '$' + v }, grid: { color: '#1e293b' } }
          }
        }
      });
    }

    function renderActorChart() {
      if (typeof Chart === 'undefined') {
        showChartOfflineFallback('actorChart');
        return;
      }
      const canvas = document.getElementById('actorChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (actorChartInst) actorChartInst.destroy();

      const list = (DATA.actors[currentAccount] || DATA.actors['ALL'] || []).slice(0, 8);
      actorChartInst = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: list.map(x => `${x.actor} (${fmtUsd(x.cost_usd)})`),
          datasets: [{
            data: list.map(x => x.cost_usd),
            backgroundColor: ['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#f43f5e', '#6366f1', '#14b8a6', '#eab308'],
            borderColor: '#131b2e',
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { color: '#cbd5e1', font: { size: 11 } } }
          }
        }
      });
    }

    function renderToolChart() {
      if (typeof Chart === 'undefined') {
        showChartOfflineFallback('toolChart');
        return;
      }
      const canvas = document.getElementById('toolChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (toolChartInst) toolChartInst.destroy();

      const list = (DATA.tools[currentAccount] || DATA.tools['ALL'] || []).slice(0, 10);
      toolChartInst = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: list.map(x => x.label),
          datasets: [
            {
              label: 'Induced Cache-Write Tokens',
              data: list.map(x => x.induced_cw),
              backgroundColor: '#f59e0b',
              borderRadius: 4
            },
            {
              label: 'Direct Payload Tokens',
              data: list.map(x => x.payload_tok),
              backgroundColor: '#38bdf8',
              borderRadius: 4
            }
          ]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#cbd5e1', font: { size: 11 } } },
            tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtNum(c.raw)} tok` } }
          },
          scales: {
            x: { ticks: { color: '#94a3b8', callback: v => fmtNum(v) }, grid: { color: '#1e293b' } },
            y: { ticks: { color: '#cbd5e1', font: { size: 11 } }, grid: { display: false } }
          }
        }
      });
    }

    function renderWoWTable() {
      const key = currentAccount === 'ALL' ? 'ALL ACCOUNTS' : currentAccount;
      const cur = (DATA.wow_curr && DATA.wow_curr[key]) || {};
      const prv = (DATA.wow_prev && DATA.wow_prev[key]) || {};

      const rows = [
        ['Active Sessions', prv.sessions || 0, cur.sessions || 0, fmtNum],
        ['API Turns (Requests)', prv.turns || 0, cur.turns || 0, fmtNum],
        ['Total Tokens Processed', prv.total_tokens || 0, cur.total_tokens || 0, fmtNum],
        ['Cache Creation (Write) Tok', prv.cache_write_tokens || 0, cur.cache_write_tokens || 0, fmtNum],
        ['Output + Thinking Tokens', prv.output_tokens || 0, cur.output_tokens || 0, fmtNum],
        ['Subagent Token Volume', prv.subagent_tokens || 0, cur.subagent_tokens || 0, fmtNum],
        ['MCP Payload Tokens', prv.mcp_tokens || 0, cur.mcp_tokens || 0, fmtNum],
        ['Tool Errors', prv.tool_errors || 0, cur.tool_errors || 0, fmtNum],
        ['Estimated API Cost ($)', prv.cost_usd || 0, cur.cost_usd || 0, fmtUsd]
      ];

      const html = `
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th class="num">Prev 7 Days</th>
              <th class="num">Last 7 Days</th>
              <th class="num">WoW Delta</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(([label, p, c, fn]) => {
              let deltaStr = '0.0%';
              let cls = '';
              if (p === 0 && c > 0) { deltaStr = '+NEW ▲'; cls = 'up-bad'; }
              else if (p > 0) {
                const pct = ((c - p) / p) * 100;
                deltaStr = `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}% ${pct >= 0 ? '▲' : '▼'}`;
                cls = pct > 5 ? 'up-bad' : (pct < -5 ? 'down-good' : '');
              }
              return `
                <tr>
                  <td>${label}</td>
                  <td class="num">${fn(p)}</td>
                  <td class="num"><strong>${fn(c)}</strong></td>
                  <td class="num ${cls}">${deltaStr}</td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      `;
      document.getElementById('wow-table-wrap').innerHTML = html;
    }

    function renderAnomalies() {
      const list = DATA.anomalies.filter(a => currentAccount === 'ALL' || a.account === currentAccount);
      const el = document.getElementById('anomaly-container');
      if (!list.length) {
        el.innerHTML = `<div style="color:#34d399;padding:16px;">✓ No anomalies detected for ${esc(currentAccount)}!</div>`;
        return;
      }
      el.innerHTML = list.slice(0, 20).map(a => `
        <div class="anomaly-item ${esc(a.severity)}">
          <div class="anomaly-top">
            <span><strong>[${esc(a.severity)}] ${esc(a.mode)}</strong> • ${esc(a.account)}</span>
            <span>Session <code>${esc(a.session_id.slice(0, 10))}</code> • ${esc(a.time_str)}</span>
          </div>
          <div class="anomaly-what">${esc(a.summary)}</div>
          <div class="anomaly-fix">💡 ${esc(a.remediation)}</div>
        </div>
      `).join('');
    }

    function renderSessionsTable() {
      const list = DATA.top_sessions.filter(s => currentAccount === 'ALL' || s.account === currentAccount).slice(0, 15);
      document.getElementById('sessions-table-wrap').innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Account</th>
              <th>Session ID</th>
              <th>Last Active</th>
              <th class="num">Turns</th>
              <th class="num">Subagents</th>
              <th class="num">Total Tokens</th>
              <th class="num">Cache Write</th>
              <th class="num">Subagent Share</th>
              <th class="num">Total Cost</th>
              <th>Project</th>
            </tr>
          </thead>
          <tbody>
            ${list.map(s => {
              const subPct = s.total_tok > 0 ? ((s.sub_tok / s.total_tok) * 100).toFixed(1) : '0.0';
              return `
                <tr>
                  <td><span class="badge badge-active">${esc(s.account)}</span></td>
                  <td><code>${esc(s.session_id.slice(0, 10))}</code></td>
                  <td>${esc(s.end_time)}</td>
                  <td class="num">${s.turns}</td>
                  <td class="num">${s.subagents}</td>
                  <td class="num">${fmtNum(s.total_tok)}</td>
                  <td class="num">${fmtNum(s.cw_tok)}</td>
                  <td class="num">${subPct}% (${fmtUsd(s.sub_cost)})</td>
                  <td class="num"><strong style="color:#34d399">${fmtUsd(s.total_cost)}</strong></td>
                  <td style="color:var(--muted);font-size:0.78rem">${esc(s.project.slice(-38))}</td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      `;
    }

    init();
  </script>
</body>
</html>
"""


def cmd_html(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    report_data = build_html_report_data(conn, days=args.days)
    out_path = Path(args.output).expanduser() if getattr(args, "output", None) else (HOME / ".claude" / "reports" / "usage_dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    safe_json_payload = (
        json.dumps(report_data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    html_str = HTML_TEMPLATE.replace("__REPORT_DATA_JSON__", safe_json_payload)
    out_path.write_text(html_str, encoding="utf-8")

    print_header("📊 INTERACTIVE HTML ANALYTICS & PREDICTIVE FORECAST EXPORT")
    print(f"  {green('✓ Generated HTML dashboard:')} {bold(str(out_path))}")
    print(f"  {cyan('▸ File URL :')} file://{out_path}")

    # Print quick terminal summary of the predictive forecast
    f_all = report_data["forecasts"].get("ALL", {})
    q_all = report_data["quota_predictions"].get("ALL", {})
    print(f"\n  {bold('🔮 Predictive Forecast Summary (All Accounts Combined):')}")
    print(f"  • Rolling 5h Window Burn : {fmt_usd(q_all.get('cur_5h_cost', 0))} / {fmt_usd(q_all.get('p90_5h_cost', 0))} P90 Ceiling ({q_all.get('fill_pct', 0)}% fill — {yellow(q_all.get('eta_str', 'Safe'))})")
    print(f"  • Next 7-Day Projection  : {green(fmt_usd(f_all.get('cost_7d', 0)))} ({fmt_num(f_all.get('tok_7d', 0))} total tokens, {fmt_num(f_all.get('fresh_7d', 0))} fresh/cache-write)")
    print(f"  • 30-Day Run-Rate Proj   : {magenta(fmt_usd(f_all.get('cost_30d', 0)))} ({fmt_num(f_all.get('tok_30d', 0))} total tokens)")

    print(f"\n  {bold('▸ Per-Account 5h Quota & 7-Day Forecast Breakdown:')}")
    print(f"  {'Account':<17} {'5h Burn ($)':>12} {'5h P90 Cap':>12} {'5h Fill %':>10} {'Exhaustion ETA':>20} {'Next 7d Proj':>14} {'30d Run-Rate':>14}")
    print("  " + "─" * 105)
    for acct in report_data["accounts"]:
        qa = report_data["quota_predictions"].get(acct, {})
        fa = report_data["forecasts"].get(acct, {})
        print(
            f"  {cyan(f'{acct:<17}')} {fmt_usd(qa.get('cur_5h_cost', 0)):>12} {fmt_usd(qa.get('p90_5h_cost', 0)):>12} "
            f"{qa.get('fill_pct', 0):>9.1f}% {qa.get('eta_str', 'Safe'):>20} "
            f"{green(f'{fmt_usd(fa.get('cost_7d', 0)):>14}')} {fmt_usd(fa.get('cost_30d', 0)):>14}"
        )

    if getattr(args, "serve", False):
        import http.server
        import socketserver

        port = int(getattr(args, "port", 3821) or 3821)
        serve_dir = str(out_path.parent)
        fname = out_path.name

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *h_args: Any, **h_kwargs: Any) -> None:
                super().__init__(*h_args, directory=serve_dir, **h_kwargs)

            def _restrict(self) -> bool:
                req_path = self.path.split("?", 1)[0]
                if req_path in ("/", f"/{fname}"):
                    self.path = f"/{fname}"
                    return True
                self.send_error(404)
                return False

            def do_GET(self) -> None:
                if self._restrict():
                    super().do_GET()

            def do_HEAD(self) -> None:
                if self._restrict():
                    super().do_HEAD()

        print(f"\n  {green(f'🌐 Serving live at http://localhost:{port}/{fname} (Press Ctrl+C to stop)')}")
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n  Stopped local server.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="claude-profiler: Multi-Account Token Profiler, Burn Rate & Post-Mortem Analyzer for Claude Code"
    )
    p.add_argument(
        "command",
        nargs="?",
        default="dashboard",
        choices=["dashboard", "summary", "burn", "who", "actors", "wow", "compare", "postmortem", "investigate", "html", "export", "sync", "hook"],
        help="Subcommand to execute (default: dashboard)",
    )
    p.add_argument(
        "extra",
        nargs="?",
        default=None,
        help="Optional session_id prefix (for postmortem/investigate) or hook event name (for hook)",
    )
    p.add_argument(
        "-a",
        "--account",
        default="all",
        help="Filter by account name or suffix: all, work, personal, dev, default (default: all)",
    )
    p.add_argument(
        "-d",
        "--days",
        type=int,
        default=14,
        help="Lookback window in days for burn/who/postmortem/html reports (default: 14)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for HTML report (default: ~/.claude/reports/usage_dashboard.html)",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Serve the generated HTML dashboard over HTTP (default port 3821)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=3821,
        help="Port for --serve (default: 3821)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force full resync of all JSONL transcripts",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "hook":
        return handle_hook(args.extra)

    conn = get_db()
    t0 = time.time()
    synced = sync_all_accounts(conn, force=args.force, quiet=False)
    elapsed = time.time() - t0
    if synced > 0 or args.command == "sync":
        print(dim(f"[Synced {synced} updated transcript files across accounts in {elapsed:.2f}s -> {DB_PATH}]"))

    if args.command == "sync":
        conn.close()
        return 0

    if args.command in ("dashboard", "summary"):
        cmd_dashboard(conn, args)
    elif args.command == "burn":
        cmd_burn(conn, args)
    elif args.command in ("who", "actors"):
        cmd_who(conn, args)
    elif args.command in ("wow", "compare"):
        cmd_wow(conn, args)
    elif args.command in ("postmortem", "investigate"):
        args.session_id = args.extra
        cmd_postmortem(conn, args)
    elif args.command in ("html", "export"):
        cmd_html(conn, args)

    print()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
