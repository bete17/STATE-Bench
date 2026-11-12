"""Load the Agent Learning Track's training conversations into Actor-Gatherer.

For each domain, makes the gatherer statebench-train-<domain> and inserts that
domain's 100 training conversations (datasets/train_task_trajectories/<domain>/),
one after another as a single past session (channel train-history), with ag's
insert-only operation: the events are stored, nothing is built and no model is
called. tools/prebuild_ag_training.py then builds that session once.

OpenCodeAgTrainAgent starts every task's gatherer as a copy of this one, so each
test task sees the training conversations and never another test task.

Usage:
    python tools/seed_ag_training.py                    # all three domains
    python tools/seed_ag_training.py --domain travel
    python tools/seed_ag_training.py --force            # replace existing seeds
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOMAINS = ("travel", "customer_support", "shopping_assistant")
TRAIN_CHANNEL = "train-history"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def seed(domain: str, force: bool) -> None:
    from ag.ag_harness.store.paths import gatherer_dir
    from ag.integrations.harnesses.codex.adaptor.adaptor import CodexAdaptor
    from ag.integrations.tools.ag_connection.ag_client import AgClient
    from ag.integrations.tools.ag_orchestrator.adaptor import CHANNEL
    from ag.integrations.tools.ag_orchestrator.call_ag import call_ag

    oc = _load("_seed_opencode_agent", REPO / "agents" / "opencode_agent.py")
    ag_agent = _load("_seed_ag_agent", REPO / "agents" / "ag_agent.py")

    name = oc.seed_gatherer_name(domain)
    target = Path(gatherer_dir(name))
    if target.exists():
        if not force:
            print(f"{name}: exists, skipped (--force to replace)")
            return
        shutil.rmtree(target)

    # All the conversations, one after another, as ONE past session (channel TRAIN_CHANNEL): ag
    # splits a session into tasks when it builds, and a new session joins from another session's
    # ready build. A conversation alone is one task, which ag never builds, so as 100 sessions of
    # their own nothing would be ready to join.
    training = oc.load_training(domain)
    conversation = [m for _, conv in training for m in conv if m.get("role") != "system"]
    items = ag_agent.to_items(conversation)
    client, adaptor = AgClient("state_bench_seed"), CodexAdaptor()
    try:
        sent, note = await call_ag(client, name, {CHANNEL: TRAIN_CHANNEL}, None, items, adaptor, op="context")
    finally:
        await client.stop()
    if sent is None:
        raise SystemExit(f"{name}: not stored: {note}")
    print(f"{name}: {len(training)} conversations ({len(items)} items) stored as session {TRAIN_CHANNEL} in {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", choices=DOMAINS, action="append", help="domain to seed (default: all)")
    parser.add_argument("--force", action="store_true", help="replace an existing seeded gatherer")
    args = parser.parse_args()
    for domain in args.domain or DOMAINS:
        asyncio.run(seed(domain, args.force))


if __name__ == "__main__":
    main()
