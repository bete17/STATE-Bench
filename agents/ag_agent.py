"""AgAgent: STATE-Bench's built-in agent with every model call passed through Actor-Gatherer (ag).

The agent is the Actor. Before each model call it sends the conversation to ag's Gatherer
(through ag's Codex adaptor, since both speak OpenAI Responses API items) and sends the model
whatever context ag returns. STATE-Bench still executes and records the domain tools.

Each task run gets a new, empty gatherer (`statebench-<id>` under ~/.ag/gatherers/).

Run:
    uv run python -m state_bench.scripts.run_batch --domain travel --agent-class AgAgent \
        --agent-model-name <model> --num-runs 1 --output-dir outputs/ag/travel/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import uuid
from collections import Counter
from typing import Any

from ag.integrations.harnesses.codex.adaptor.adaptor import CodexAdaptor
from ag.integrations.tools.ag_connection.ag_client import AgClient
from ag.integrations.tools.ag_orchestrator.adaptor import CHANNEL
from ag.integrations.tools.ag_orchestrator.call_ag import call_ag

from state_bench.agents.base import AgentTurnResponse
from state_bench.agents.state_bench import StateBenchAgent

logger = logging.getLogger(__name__)

# One event loop and one ag server shared by all of run_batch's worker threads.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, name="ag-loop", daemon=True).start()


def _run(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


async def _new_client() -> AgClient:
    return AgClient("state_bench")


_AG: AgClient | None = None
_AG_LOCK = threading.Lock()
_ADAPTOR = CodexAdaptor()


def _ag() -> AgClient:
    """The shared ag client, started on first use. STATE-Bench imports every file in
    agents/ when loading any agent, so starting it at import would run ag for other agents too."""
    global _AG
    with _AG_LOCK:
        if _AG is None:
            _AG = _run(_new_client())
        return _AG


def to_items(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """STATE-Bench's conversation as Responses API input items.

    Call ids hash the call's content plus its occurrence, so the same call keeps its id
    across tool rounds (where calls sit on separate assistant messages) and later turns
    (where one assistant message holds all of a turn's calls).
    """
    items: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for msg in conversation:
        if msg.get("role") == "tool":  # a temporary copy of calls already on the assistant message
            continue
        for call in msg.get("tool_calls") or []:
            key = json.dumps([call["name"], call["arguments"], call.get("result")], sort_keys=True, default=str)
            digest = hashlib.sha1(key.encode()).hexdigest()[:16]
            seen[digest] += 1
            call_id = f"sb_{digest}_{seen[digest]}"
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"]),
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(call.get("result"), ensure_ascii=False, default=str),
                }
            )
        if msg.get("content"):
            items.append({"role": msg["role"], "content": msg["content"]})
    return items


class AgAgent(StateBenchAgent):
    """StateBenchAgent whose context comes from ag's Gatherer on every model call."""

    # The parameters are named, not *args/**kwargs: run_task passes agent_reasoning_effort
    # only to a constructor whose signature names it.
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
        self.gatherer = f"statebench-{uuid.uuid4().hex[:12]}"
        self.ag_calls = 0
        self.ag_fallbacks = 0

    def generate_next_turn(self, *, system_prompt, conversation, tools):
        items = to_items(conversation)
        sent, note = _run(
            call_ag(
                _ag(),
                self.gatherer,
                {CHANNEL: "actor"},
                None,
                items,
                _ADAPTOR,
                preamble=[
                    {"name": "system", "text": system_prompt},
                    {"name": "tools", "text": json.dumps(tools)},
                ],
            )
        )
        self.ag_calls += 1
        if sent is None:  # ag could not answer: send our own items, and say so
            self.ag_fallbacks += 1
            logger.warning("[ag fallback] %s: %s", self.gatherer, note)
            sent = items
        response = self.client.complete_with_tools(
            instructions=system_prompt,
            input=sent,
            tools=tools,
            reasoning_effort=self.agent_reasoning_effort,
        )
        if response.usage:
            self.add_response_usage(response.usage, category="agent_turn")
        calls = [
            {"name": o.name, "arguments": json.loads(o.arguments)}
            for o in response.output
            if o.type == "function_call"
        ]
        return AgentTurnResponse(text=response.output_text or "", tool_calls=calls)

    def ingest_trajectory(self, trajectory: Any) -> None:
        trajectory.metadata["ag"] = {
            "gatherer": self.gatherer,
            "calls": self.ag_calls,
            "fallbacks": self.ag_fallbacks,
        }
