"""Local run tracker for STATE-Bench.

Serves a small web page that watches a run_batch output folder and shows:
  - headline metrics (progress, pass@1, pass^N, UX score, cost)
  - each task's result across runs
  - a live feed of finished trajectories
  - per-trajectory detail: the conversation, every tool call, and the judges' reasoning

Usage:
    .venv/Scripts/python tools/tracker.py --outputs outputs
    # then open http://localhost:8765

Read-only: it never writes to the outputs folder. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO / "state_bench" / "domains"
PAGE = Path(__file__).resolve().parent / "tracker.html"
RUN_DIR_RE = re.compile(r"^run(\d+)$")

_cache: dict[Path, tuple[float, dict]] = {}


def _split_task_ids(domain: str, split: str) -> list[str]:
    path = DOMAINS_DIR / domain / "splits" / "train_test.json"
    try:
        splits = json.loads(path.read_text(encoding="utf-8"))["splits"]
    except (OSError, ValueError, KeyError):
        return []
    if split == "all":
        return [tid for ids in splits.values() for tid in ids]
    return list(splits.get(split, []))


def _status(traj: dict) -> str:
    if traj.get("error"):
        return "error"
    passed = traj.get("task_completion_pass")
    if passed is None:
        # run_batch saves the trajectory first, then rewrites it once the judges finish.
        return "judging"
    return "pass" if passed in (1, True) else "fail"


def _last_agent_activity(conversation: list) -> tuple[str, str]:
    """Return (last agent message, last tool name) for the activity feed."""
    for msg in reversed(conversation or []):
        if msg.get("role") == "assistant":
            calls = msg.get("tool_calls") or []
            tool = calls[-1].get("name", "") if calls else ""
            return str(msg.get("content") or "")[:240], tool
    return "", ""


def _read_json(path: Path):
    """A trajectory file as JSON. run_batch writes with the system's default encoding,
    which on Windows is cp1252 rather than UTF-8, so try UTF-8 first and fall back."""
    data = path.read_bytes()
    try:
        return json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(data.decode("cp1252"))


def _summarize(path: Path, root: Path, run: int) -> dict | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        traj = _read_json(path)
    except (OSError, ValueError):
        # File may be mid-write; show it as in progress rather than failing.
        return {
            "task_id": path.stem,
            "run": run,
            "status": "judging",
            "path": path.relative_to(root).as_posix(),
            "mtime": mtime,
        }
    last_msg, last_tool = _last_agent_activity(traj.get("conversation", []))
    usage = traj.get("token_usage") or {}
    summary = {
        "task_id": traj.get("task_id", path.stem),
        "task_summary": traj.get("task_summary", ""),
        "run": run,
        "status": _status(traj),
        "ux_score": traj.get("ux_score"),
        "turns": traj.get("turns"),
        "tool_calls": traj.get("tool_calls"),
        "tool_errors": traj.get("tool_errors"),
        "cost_usd": traj.get("cost_usd") or usage.get("total_cost_usd"),
        "total_tokens": traj.get("total_tokens") or usage.get("total_tokens"),
        "reasoning_tokens": traj.get("reasoning_output_tokens") or usage.get("reasoning_output_tokens"),
        "last_message": last_msg,
        "last_tool": last_tool,
        "path": path.relative_to(root).as_posix(),
        "mtime": mtime,
    }
    _cache[path] = (mtime, summary)
    return summary


def _metrics(rows: list[dict], expected_tasks: int, num_runs: int) -> dict:
    scored = [r for r in rows if r["status"] in ("pass", "fail")]
    by_task: dict[str, list[dict]] = {}
    for r in scored:
        by_task.setdefault(r["task_id"], []).append(r)
    complete = [runs for runs in by_task.values() if len(runs) >= num_runs]
    ux = [r["ux_score"] for r in rows if isinstance(r.get("ux_score"), (int, float))]
    costs = [r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), (int, float))]
    return {
        "finished": len(rows),
        "expected": expected_tasks * num_runs,
        "scored": len(scored),
        "judging": sum(r["status"] == "judging" for r in rows),
        "errors": sum(r["status"] == "error" for r in rows),
        "pass_at_1": sum(r["status"] == "pass" for r in scored) / len(scored) if scored else None,
        "pass_all": (
            sum(all(r["status"] == "pass" for r in runs) for runs in complete) / len(complete) if complete else None
        ),
        "pass_all_tasks": len(complete),
        "mean_ux": sum(ux) / len(ux) if ux else None,
        "total_cost": sum(costs) if costs else None,
        "cost_per_traj": sum(costs) / len(costs) if costs else None,
    }


def build_summary(root: Path, split: str, num_runs_arg: int | None) -> dict:
    groups: dict[str, dict[int, Path]] = {}
    if root.is_dir():
        for run_dir in sorted(p for p in root.rglob("run*") if p.is_dir() and RUN_DIR_RE.match(p.name)):
            key = run_dir.parent.relative_to(root).as_posix()
            key = root.name if key == "." else key
            groups.setdefault(key, {})[int(RUN_DIR_RE.match(run_dir.name).group(1))] = run_dir

    domains = []
    all_rows: list[dict] = []
    for key, run_dirs in sorted(groups.items()):
        domain_name = key.rsplit("/", 1)[-1]
        rows = []
        for run, run_dir in sorted(run_dirs.items()):
            for f in sorted(run_dir.glob("*.json")):
                s = _summarize(f, root, run)
                if s:
                    s["domain"] = key
                    rows.append(s)
        num_runs = num_runs_arg or max(run_dirs)
        # A tasks.txt next to the run folders (one task id per line) overrides the split.
        tasks_file = next(iter(run_dirs.values())).parent / "tasks.txt"
        if tasks_file.is_file():
            expected_ids = [line.strip() for line in tasks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            expected_ids = _split_task_ids(domain_name, split)
        seen = {r["task_id"] for r in rows}
        task_ids = expected_ids + sorted(seen - set(expected_ids))
        summaries = {r["task_id"]: r.get("task_summary", "") for r in rows}
        tasks = []
        for tid in task_ids:
            runs = {str(r["run"]): r for r in rows if r["task_id"] == tid}
            tasks.append({"task_id": tid, "task_summary": summaries.get(tid, ""), "runs": runs})
        metrics_file = root / ("" if key == root.name else key) / "metrics.json"
        official = None
        if metrics_file.is_file():
            try:
                official = json.loads(metrics_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                official = None
        domains.append(
            {
                "name": key,
                "num_runs": num_runs,
                "tasks": tasks,
                "metrics": _metrics(rows, len(task_ids), num_runs),
                "official_metrics": official,
            }
        )
        all_rows.extend(rows)

    total_expected = sum(d["metrics"]["expected"] for d in domains)
    overall = _metrics(all_rows, 0, max((d["num_runs"] for d in domains), default=1))
    overall["expected"] = total_expected
    # pass_all is per-domain; recompute across domains by task key
    complete = []
    for d in domains:
        for t in d["tasks"]:
            scored = [r for r in t["runs"].values() if r["status"] in ("pass", "fail")]
            if scored and len(scored) >= d["num_runs"]:
                complete.append(all(r["status"] == "pass" for r in scored))
    overall["pass_all"] = sum(complete) / len(complete) if complete else None
    overall["pass_all_tasks"] = len(complete)

    activity = sorted(all_rows, key=lambda r: r["mtime"], reverse=True)[:25]
    return {
        "root": str(root),
        "generated_at": time.time(),
        "overall": overall,
        "domains": domains,
        "activity": activity,
    }


def make_handler(root: Path, split: str, num_runs: int | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/summary":
                self._json(build_summary(root, split, num_runs))
            elif url.path == "/api/trajectory":
                rel = parse_qs(url.query).get("path", [""])[0]
                target = (root / rel).resolve()
                if root.resolve() not in target.parents or target.suffix != ".json" or not target.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    self._json(_read_json(target))
                except ValueError:
                    self._json({"error": "file is still being written, try again"}, 409)
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local tracker UI for STATE-Bench runs.")
    parser.add_argument("--outputs", default="outputs", help="run_batch output folder (default: outputs)")
    parser.add_argument("--split", default="test", help="split used to list expected tasks (default: test)")
    parser.add_argument("--num-runs", type=int, default=None, help="runs per task (default: highest runN found)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    root = Path(args.outputs).resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root, args.split, args.num_runs))
    url = f"http://localhost:{args.port}"
    print(f"STATE-Bench tracker watching {root}\nOpen {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
