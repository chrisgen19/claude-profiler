# claude-profiler

**Multi-Account Token Profiler, Burn Rate Monitor, MCP/Subagent Attribution & Automated Post-Mortem Analyzer for Claude Code.**

`claude-profiler` combines **non-blocking Claude Code lifecycle hooks** (`PostToolUse`, `SubagentStop`, `Stop`, `SessionEnd`) with a fast, incremental **SQLite WAL analytics engine** that parses both main session transcripts (`~/.claude*/projects/**/*.jsonl`) and nested subagent execution trees (`subagents/agent-*.jsonl` + `.meta.json`).

Zero external dependencies — runs on standard Python 3.8+ (`sqlite3`).

---

## ✨ Key Features

1. **Multi-Account Auto-Discovery (`~/.claude*`)**
   * Automatically discovers `~/.claude` and any multi-account `CLAUDE_CONFIG_DIR` profiles such as `~/.claude-work`, `~/.claude-personal`, and `~/.claude-dev`.
   * Filter any report with `-a work`, `-a personal`, `-a dev`, or view combined cross-account rollups (`-a all`).
2. **Burn Rate & Rolling 5-Hour Quota Velocity (`claude-profiler burn`)**
   * Tracks active session duration, **tokens/minute burn velocity**, **fresh/cache-write token velocity**, and **$/hr API-equivalent burn rate** across 1-hour, rolling 5-hour block, 24-hour, and N-day windows.
3. **Who Uses Tokens: Main vs. Subagents vs. MCP Servers (`claude-profiler who`)**
   * **Execution Context Split:** Attributes token consumption and cost between the Main Session Agent and every spawned Subagent type (`general-purpose`, `feature-dev:code-explorer`, `workflow-subagent`, `Plan`, etc.).
   * **MCP Server Attribution:** Breaks down every `mcp__<server>__<tool>` call by call count, error count, response payload tokens injected into context, and **induced prompt-cache write tokens** on the subsequent turn.
   * **Built-in Tool Impact:** Tracks `Bash`, `Read`, `Edit`, `Write`, `Grep`, `WebFetch`, `Agent`/`Task`, and `SubagentHandback` overhead.
4. **Week-over-Week (WoW) Comparisons (`claude-profiler wow`)**
   * Side-by-side comparison of **Calendar ISO Week (This Week vs. Last Week)** and **Normalized Rolling 7-Day Windows (Last 7d vs. Previous 7d)** per account with percentage deltas (`▲` / `▼`).
5. **Automated 5-Mode "What Went Wrong" Post-Mortem Detector (`claude-profiler postmortem [session_id]`)**
   * Automatically detects and diagnoses the 5 classic Claude Code quota-drain failure modes:
     1. **`[MCP_OR_TOOL_PAYLOAD_BLOAT]`** — Unbounded MCP responses or large file/command outputs injecting massive payloads into context.
     2. **`[CACHE_BUST_SPIRAL]`** — Cold-cache rewrites when resuming a session or subagent after the 5-minute ephemeral cache TTL expires (re-billing 300k–700k tokens at 1.25x–2.0x input rates).
     3. **`[SUBAGENT_AMPLIFICATION]`** — Spawning dozens of parallel subagents inheriting heavy models (`opus`) instead of `sonnet` or `haiku`.
     4. **`[THRASHING_RETRY_LOOP]`** — Consecutive failing `Bash`/`Edit` tool calls burning 100k+ context per retry.
     5. **`[AFK_APPROVAL_OR_STALL]`** — Interactive permission prompts (`AskUserQuestion`, `ExitPlanMode`, `Bash`) stalling a session for >10 minutes (and guaranteeing a cold-cache rewrite on resume), plus subagent streaming dropouts.

---

## 🚀 Why Raw JSONL Sums Are Often Wrong (And How `claude-profiler` Fixes Them)

* **Multi-Block Request Deduplication:** Claude Code logs separate `assistant` JSONL lines for thinking blocks, text blocks, and `tool_use` blocks that share the same `requestId` and cumulative `usage` object. Naive scripts triple-count those turns; `claude-profiler` deduplicates by `(file_path, requestId)` while unioning all tool calls.
* **5-Minute vs. 1-Hour Ephemeral Cache Pricing:** Differentiates `ephemeral_5m_input_tokens` (1.25x input cost) from `ephemeral_1h_input_tokens` (2.0x input cost).
* **Vision Screenshot Token Normalization:** Base64 screenshots inside `tool_result` blocks can be 600KB+ of characters (~150k text tokens if divided by 4), but Anthropic's vision tokenizer bills ~1,600 tokens per image. `claude-profiler` detects image blocks so cold-cache spikes aren't falsely blamed on base64 length.
* **Nested Subagent Tree Linking:** Links `~/.claude*/projects/<slug>/<session-id>/subagents/agent-<id>.jsonl` and `agent-<id>.meta.json` (`agentType`, `description`) back to the parent session.

---

## 📦 Installation

```bash
git clone https://github.com/chrisgen19/claude-profiler.git ~/projects/claude-profiler
cd ~/projects/claude-profiler
./install.sh
```

What `./install.sh` does:
1. Symlinks `token_profiler.py` to `~/.local/bin/claude-profiler` and `~/.claude*/hooks/token_profiler.py`.
2. Non-destructively registers async (`"async": true`) hooks for `PostToolUse`, `SubagentStop`, `Stop`, and `SessionEnd` across all discovered `~/.claude*/settings.json` files.
3. Backfills your existing `.jsonl` session and subagent transcripts into `~/.claude/usage_analytics.db`.

---

## 🖥️ CLI Usage

```bash
# Full dashboard (Burn Rate + Who Uses Tokens + Week-over-Week + Post-Mortem)
claude-profiler

# 1. Live burn rate (1h, rolling 5h block, 24h, N days) & highest-burn sessions
claude-profiler burn
claude-profiler burn -a work          # Filter to claude-work (or personal / dev)

# 2. Who uses tokens (Main Agent vs Subagent types vs MCP servers vs Built-in tools)
claude-profiler who --days 14

# 3. Week-over-Week comparison (Calendar Week & Rolling 7-Day windows per account)
claude-profiler wow

# 4. Automated "What Went Wrong" 5-mode post-mortem detector across sessions
claude-profiler postmortem --days 14

# 5. Deep-dive a specific session (turn-by-turn cost + all subagents + cold-cache bust timeline)
claude-profiler postmortem <session_id_prefix>

# 6. Force full resync of all transcript files
claude-profiler sync --force
```

---

## 📄 License

MIT © Christian Diomampo
