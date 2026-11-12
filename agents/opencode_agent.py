"""OpenCodeAgent: STATE-Bench's agent slot filled by a real harness, OpenCode.

Three pieces live here:

1. Tool bridge. A small MCP server inside this Python process. It lists the
   domain's tools to OpenCode and runs each call through STATE-Bench's own tool
   handlers for that task, so actions land in the task's environment and are
   recorded and scored as usual.
2. Driver. One `opencode serve` per task, with one session. Each customer
   message is sent to the session; OpenCode's reply goes back to STATE-Bench.
3. Lock-down. OpenCode's built-in tools (bash, read, edit, web fetch and so on)
   are switched off, so the store's tools are the only ones it has. The domain's
   system prompt is the OpenCode agent's prompt.

Two classes, chosen with --agent-class:

    OpenCodeAgent      OpenCode alone
    OpenCodeAgAgent    the same OpenCode, launched through Actor-Gatherer
                       (`python -m ag.cli --gatherer <g> opencode serve`), so ag's
                       OpenCode plugin edits the context before every model call.
                       Each task run gets a new gatherer, statebench-oc-<id>.

Everything else is identical between the two, so their difference is ag's.

The model is the run's --agent-model-name (an OpenAI model) and the key is
STATE_BENCH_AGENT_API_KEY. The OpenCode binary is OPENCODE_BIN if set, else
`opencode` on PATH, else npm's global install.

    python -m state_bench.scripts.run_batch --domain travel --agent-class OpenCodeAgent \\
        --agent-provider openai --agent-model-name gpt-6.1-sol \\
        --agent-model-reasoning-level medium --output-dir outputs/<name>/travel

On Windows, set PYTHONUTF8=1 for the run: STATE-Bench saves trajectories in the
system's default encoding, and a reply containing a character such as "→" then
fails to save and the task is counted as an error.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from state_bench.agents.state_bench import StateBenchAgent

logger = logging.getLogger(__name__)

MCP_NAME = "statebench"            # OpenCode shows the tools as statebench_<tool>
AGENT_NAME = "statebench"          # the OpenCode agent the domain prompt runs as
START_TIMEOUT_S = 120              # waiting for `opencode serve` to listen
TURN_TIMEOUT_S = 900               # one customer message, all its tool rounds included


# --------------------------------------------------------------------------- tool bridge


class _BridgeSession:
    """One task's tools as the bridge serves them."""

    def __init__(self, tools: list[dict[str, Any]], handlers: dict[str, Callable]):
        self.tools = tools
        self.handlers = handlers
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def take_calls(self) -> list[dict[str, Any]]:
        with self.lock:
            calls, self.calls = self.calls, []
        return calls

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        handler = self.handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}, True
        try:
            result = handler(arguments)
            failed = False
        except Exception as exc:  # a handler that raises is reported to the model, not fatal
            result, failed = {"error": f"{type(exc).__name__}: {exc}"}, True
        with self.lock:
            self.calls.append({"name": name, "arguments": arguments, "result": result})
        return result, failed


class _Bridge:
    """MCP over HTTP (JSON responses, no streaming): /mcp/<token> serves one task's tools."""

    PROTOCOL = "2025-06-18"

    def __init__(self):
        self.sessions: dict[str, _BridgeSession] = {}
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, code: int, body: Any | None = None) -> None:
                data = b"" if body is None else json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(code)
                if body is not None:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - no server-initiated stream
                self.send_response(405)
                self.send_header("Allow", "POST")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_DELETE(self):  # noqa: N802
                self._reply(200)

            def do_POST(self):  # noqa: N802
                session = bridge.sessions.get(self.path.rstrip("/").rsplit("/", 1)[-1])
                if session is None:
                    self._reply(404, {"error": "unknown session"})
                    return
                try:
                    message = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                except ValueError:
                    self._reply(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
                    return
                if isinstance(message, list):
                    replies = [r for r in (bridge.handle(session, m) for m in message) if r is not None]
                    self._reply(200, replies) if replies else self._reply(202)
                    return
                reply = bridge.handle(session, message)
                self._reply(200, reply) if reply is not None else self._reply(202)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name="statebench-mcp", daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"

    def handle(self, session: _BridgeSession, message: Any) -> dict | None:
        if not isinstance(message, dict) or "method" not in message:
            return None  # a response or junk; nothing to answer
        if "id" not in message:
            return None  # notification
        method, params, mid = message["method"], message.get("params") or {}, message["id"]
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion") or self.PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "statebench-bridge", "version": "1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "inputSchema": t.get("parameters") or {"type": "object", "properties": {}},
                    }
                    for t in session.tools
                ]
            }
        elif method == "tools/call":
            output, failed = session.call(params.get("name", ""), params.get("arguments") or {})
            result = {
                "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False, default=str)}],
                "isError": failed,
            }
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def open(self, tools, handlers) -> tuple[str, _BridgeSession]:
        token = uuid.uuid4().hex
        session = _BridgeSession(tools, handlers)
        self.sessions[token] = session
        return f"{self.base_url}/{token}", session

    def close(self, url: str) -> None:
        self.sessions.pop(url.rsplit("/", 1)[-1], None)


_bridge: _Bridge | None = None
_bridge_lock = threading.Lock()


def _get_bridge() -> _Bridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = _Bridge()
        return _bridge


# --------------------------------------------------------------------------- OpenCode process


def _opencode_bin() -> str:
    if os.environ.get("OPENCODE_BIN"):
        return os.environ["OPENCODE_BIN"]
    found = shutil.which("opencode")
    if found:
        return found
    if os.name == "nt" and os.environ.get("APPDATA"):
        exe = Path(os.environ["APPDATA"]) / "npm" / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
        if exe.is_file():
            return str(exe)
    raise RuntimeError("OpenCode not found: install it (npm install -g opencode-ai) or set OPENCODE_BIN")


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


_live: set[subprocess.Popen] = set()
atexit.register(lambda: [_kill_tree(p) for p in list(_live)])


def _http(method: str, url: str, body: Any | None = None, timeout: float = 60) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenCode {method} {url} -> {exc.code}: {exc.read()[:500]!r}") from exc
    return json.loads(raw) if raw.strip() else None


# --------------------------------------------------------------------------- agents


class OpenCodeAgent(StateBenchAgent):
    """STATE-Bench agent backed by an OpenCode session (no ag)."""

    use_ag = False
    ag_seed = False     # start the task's gatherer from the domain's seeded training gatherer

    # Named parameters: run_task passes agent_reasoning_effort only to a constructor that names it.
    def __init__(
        self,
        client,
        system_prompt,
        tools,
        tool_handlers,
        runtime_context=None,
        retrieve_learnings_top_k: int = 3,
        agent_reasoning_effort: str | None = None,
    ):
        super().__init__(
            client,
            system_prompt,
            tools,
            tool_handlers,
            runtime_context=runtime_context,
            retrieve_learnings_top_k=retrieve_learnings_top_k,
            agent_reasoning_effort=agent_reasoning_effort,
        )
        self.model = (
            getattr(client, "deployment", None)
            or getattr(client, "model", None)
            or os.environ.get("STATE_BENCH_AGENT_MODEL", "")
        ).strip('"')
        self.gatherer = f"statebench-oc-{uuid.uuid4().hex[:12]}" if self.use_ag else None
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.session_id: str | None = None
        self.workdir: str | None = None
        self.log_path: str | None = None
        self.bridge_url: str | None = None
        self.bridge_session: _BridgeSession | None = None
        self.counted_messages: set[str] = set()
        self.turns = 0
        self.started_s: float | None = None

    # ---- setup

    def _config(self) -> dict[str, Any]:
        model_options: dict[str, Any] = {}
        if self.agent_reasoning_effort:
            model_options["reasoningEffort"] = self.agent_reasoning_effort
        return {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "snapshot": False,
            "model": f"openai/{self.model}",
            "default_agent": AGENT_NAME,
            "agent": {
                AGENT_NAME: {
                    "mode": "primary",
                    "description": "STATE-Bench customer service agent",
                    "prompt": self.system_prompt,
                    # Only the store's tools: every built-in off, the bridge's on.
                    "tools": {"*": False, f"{MCP_NAME}_*": True},
                    "permission": {"*": "deny", f"{MCP_NAME}_*": "allow"},
                },
            },
            "provider": {"openai": {"models": {self.model: {"options": model_options}}}},
            "mcp": {MCP_NAME: {"type": "remote", "url": self.bridge_url, "oauth": False, "enabled": True}},
        }

    def _start(self) -> None:
        self.bridge_url, self.bridge_session = _get_bridge().open(self.tools, self.tool_handlers)
        self.workdir = tempfile.mkdtemp(prefix="statebench-oc-")   # empty: no project files to find
        self.log_path = os.path.join(self.workdir, "opencode.log")
        opencode = _opencode_bin()

        env = dict(os.environ)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(self._config())
        key = os.environ.get("STATE_BENCH_AGENT_API_KEY", "").strip('"')
        if key:
            env["OPENAI_API_KEY"] = key
        env["PATH"] = str(Path(opencode).parent) + os.pathsep + env.get("PATH", "")

        serve = ["serve", "--hostname", "127.0.0.1", "--port", "0"]
        if self.use_ag:
            if self.ag_seed:
                copy_gatherer(seed_gatherer_name(self.runtime_context.domain), self.gatherer)
            env["AG_PYTHON"] = sys.executable
            cmd = [sys.executable, "-m", "ag.cli", "--gatherer", self.gatherer, "opencode", *serve]
        else:
            cmd = [opencode, *serve]

        log = open(self.log_path, "wb")
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        self.started_s = time.monotonic()
        self.proc = subprocess.Popen(cmd, cwd=self.workdir, env=env, stdout=log, stderr=subprocess.STDOUT, **flags)
        _live.add(self.proc)
        log.close()

        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            text = Path(self.log_path).read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "listening on http" in line:
                    self.url = line[line.index("http") :].split()[0].rstrip("/")
                    break
            if self.url:
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"opencode exited before listening:\n{text[-2000:]}")
            time.sleep(0.2)
        else:
            raise RuntimeError(f"opencode did not start within {START_TIMEOUT_S}s:\n{text[-2000:]}")

        self.session_id = _http("POST", f"{self.url}/session", {"title": "statebench"})["id"]

    # ---- one turn

    def act(self, conversation: list[Any]) -> tuple[str, list[dict[str, Any]], list[Any]]:
        if self.proc is None:
            self._start()
        message = conversation[-1]
        text_in = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text_in, str):
            raise RuntimeError("OpenCodeAgent expects the conversation to end with the customer's message")

        reply = _http(
            "POST",
            f"{self.url}/session/{self.session_id}/message",
            {
                "parts": [{"type": "text", "text": text_in}],
                "agent": AGENT_NAME,
                "model": {"providerID": "openai", "modelID": self.model},
            },
            timeout=TURN_TIMEOUT_S,
        )
        self.turns += 1
        error = (reply or {}).get("info", {}).get("error")
        if error:
            raise RuntimeError(f"OpenCode turn failed: {json.dumps(error)[:1000]}")
        texts = self._collect_turn()
        if not texts:  # usage could not be read: fall back to the final step's text
            texts = [p.get("text", "") for p in (reply or {}).get("parts", []) if p.get("type") == "text"]
        text = "\n\n".join(t.strip() for t in texts if t and t.strip())
        tool_calls = self.bridge_session.take_calls()
        return text, tool_calls, [{"role": "assistant", "content": text}]

    def _collect_turn(self) -> list[str]:
        """For every assistant message OpenCode made since the last turn (one per model call):
        add its tokens and cost, and return its text. A turn's reply is all the text the agent
        wrote during it, as OpenCode shows it, including text written before a tool call."""
        try:
            messages = _http("GET", f"{self.url}/session/{self.session_id}/message")
        except Exception as exc:
            logger.warning("could not read OpenCode messages: %s", exc)
            return []
        texts: list[str] = []
        for m in messages or []:
            info = m.get("info", {})
            if info.get("role") != "assistant" or info.get("id") in self.counted_messages:
                continue
            self.counted_messages.add(info["id"])
            texts.extend(p.get("text", "") for p in m.get("parts", []) if p.get("type") == "text")
            tokens = info.get("tokens") or {}
            cache = tokens.get("cache") or {}
            # OpenCode splits the prompt into uncached input, cache reads and cache writes;
            # all three are input tokens (OpenAI reports them as input, with reads as cached).
            self.add_token_usage(
                input_tokens=int(tokens.get("input", 0)) + int(cache.get("read", 0)) + int(cache.get("write", 0)),
                output_tokens=int(tokens.get("output", 0)) + int(tokens.get("reasoning", 0)),
                cached_input_tokens=int(cache.get("read", 0)),
                reasoning_output_tokens=int(tokens.get("reasoning", 0)),
                category="agent_turn",
            )
            if info.get("cost"):
                self.add_cost_usd(float(info["cost"]), category="agent_turn")
        return texts

    # ---- end of task

    def close(self) -> None:
        if self.proc is not None:
            _kill_tree(self.proc)
            _live.discard(self.proc)
        if self.bridge_url:
            _get_bridge().close(self.bridge_url)
            self.bridge_url = None
        if self.workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def ingest_trajectory(self, trajectory) -> None:
        trajectory.metadata["opencode"] = {
            "ag": self.use_ag,
            "gatherer": self.gatherer,
            "ag_seed": seed_gatherer_name(self.runtime_context.domain) if self.ag_seed else None,
            "training_lookup": self.retrieval_enabled,
            "session": self.session_id,
            "model": self.model,
            "reasoning_effort": self.agent_reasoning_effort,
            "turns": self.turns,
            "model_calls": len(self.counted_messages),
        }
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class OpenCodeAgAgent(OpenCodeAgent):
    """The same OpenCode agent, launched through Actor-Gatherer."""

    use_ag = True


# --------------------------------------------------------------------------- training conversations
#
# The Agent Learning Track's 100 training conversations per domain
# (datasets/train_task_trajectories/<domain>/), given to the agent two ways:
#
#   retrieve_learnings   STATE-Bench's own hook. Defining it makes StateBenchAgent add the
#                        retrieve_learnings tool, its system-prompt instruction and the run's
#                        --retrieve-learnings-top-k; OpenCode gets the tool through the bridge.
#                        It returns the training conversations most similar to the query (BM25).
#   seeded gatherer      for ag: tools/seed_ag_training.py loads a domain's training conversations
#                        into the gatherer statebench-train-<domain>, one channel each; every task
#                        run starts from a copy of it, so no task sees another test task.

TRAIN_DIR = Path(__file__).resolve().parent.parent / "datasets" / "train_task_trajectories"
CONVERSATION_CHARS = 6000          # one returned conversation, condensed, at most
TOOL_RESULT_CHARS = 300


def condense_conversation(conversation: list[dict[str, Any]]) -> str:
    """A training conversation as plain text: the customer's and agent's messages, and each tool call."""
    lines: list[str] = []
    for m in conversation:
        role = m.get("role")
        if role == "user" and m.get("content"):
            lines.append(f"Customer: {m['content']}")
        elif role == "assistant":
            for c in m.get("tool_calls") or []:
                result = json.dumps(c.get("result"), ensure_ascii=False, default=str)
                if len(result) > TOOL_RESULT_CHARS:
                    result = result[:TOOL_RESULT_CHARS] + "..."
                lines.append(f"  [tool] {c.get('name')}({json.dumps(c.get('arguments'), ensure_ascii=False)}) -> {result}")
            if m.get("content"):
                lines.append(f"Agent: {m['content']}")
    text = "\n".join(lines)
    return text if len(text) <= CONVERSATION_CHARS else text[:CONVERSATION_CHARS] + "\n[... conversation cut]"


def load_training(domain: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """[(task id, conversation)] for a domain's training conversations, in file-name order."""
    out = []
    for path in sorted((TRAIN_DIR / domain).glob("*.json")):
        out.append((path.stem, json.loads(path.read_text(encoding="utf-8"))["conversation"]))
    if not out:
        raise RuntimeError(f"no training conversations under {TRAIN_DIR / domain}")
    return out


class _TrainingIndex:
    """BM25 over one domain's condensed training conversations."""

    def __init__(self, domain: str, k1: float = 1.5, b: float = 0.75):
        import math
        import re
        from collections import Counter

        self._re, self._math, self.k1, self.b = re.compile(r"[a-z0-9]+"), math, k1, b
        self.docs = [(tid, condense_conversation(conv)) for tid, conv in load_training(domain)]
        self.tfs = [Counter(self._tokens(text)) for _, text in self.docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _tokens(self, text: str) -> list[str]:
        return self._re.findall(text.lower())

    def search(self, query: str, top_k: int) -> list[str]:
        terms = set(self._tokens(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
            scores.append((s, i))
        best = [i for s, i in sorted(scores, reverse=True)[:top_k] if s > 0]
        return [f"Past conversation ({self.docs[i][0]}):\n{self.docs[i][1]}" for i in best]


_indexes: dict[str, _TrainingIndex] = {}
_index_lock = threading.Lock()


def training_index(domain: str) -> _TrainingIndex:
    with _index_lock:
        if domain not in _indexes:
            _indexes[domain] = _TrainingIndex(domain)
        return _indexes[domain]


def seed_gatherer_name(domain: str) -> str:
    return f"statebench-train-{domain}"


def copy_gatherer(source: str, target: str) -> None:
    """A new gatherer `target` with `source`'s settings and a consistent copy of its database."""
    import sqlite3

    from ag.ag_harness.store.paths import gatherer_dir

    src, dst = Path(gatherer_dir(source)), Path(gatherer_dir(target))
    if not (src / "context.db").is_file():
        raise RuntimeError(f"gatherer {source} not found: run tools/seed_ag_training.py first")
    if dst.exists():
        raise RuntimeError(f"gatherer {target} already exists")
    dst.mkdir(parents=True)
    shutil.copy2(src / "gatherer.toml", dst / "gatherer.toml")
    with sqlite3.connect(f"file:{src / 'context.db'}?mode=ro", uri=True) as a, sqlite3.connect(dst / "context.db") as b:
        a.backup(b)


class _TrainingLookup:
    """Mixin: STATE-Bench's retrieve_learnings hook over this domain's training conversations."""

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        return training_index(self.runtime_context.domain).search(query, top_k)


class OpenCodeTrainAgent(_TrainingLookup, OpenCodeAgent):
    """OpenCode with the training conversations available through retrieve_learnings."""


class OpenCodeAgTrainAgent(_TrainingLookup, OpenCodeAgAgent):
    """The same, through ag, with each task's gatherer starting as a copy of the domain's seeded one."""

    ag_seed = True
