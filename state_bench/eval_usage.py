"""Token usage and cost of the locked evaluation model, per task (local addition, not upstream).

STATE-Bench records the agent's tokens but not those of the evaluation model, which plays the
simulated user and runs the judges, though every task bills it too. This module counts them.

The evaluation client (client._build_eval_client) reports each response's usage to `record`.
run_batch wraps a task's conversation in `tracking("simulator", ...)` and its scoring in
`tracking("judge", ...)`, so calls are counted per task and per role. Calls made outside a
`tracking` block (or in another thread) are not counted. Counting changes nothing the model
is sent or returns.

Cost uses the prices per 1M tokens in STATE_BENCH_EVAL_PRICE_INPUT, STATE_BENCH_EVAL_PRICE_CACHED_INPUT
and STATE_BENCH_EVAL_PRICE_OUTPUT; when they are not set, the defaults below for known models.
Cached input tokens are part of input tokens and are billed at the cached price instead.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

#: USD per 1M tokens: (input, cached input, output). OpenAI's list prices.
DEFAULT_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.4": (2.50, 0.25, 15.00),
}

_FIELDS = ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
_local = threading.local()


def empty() -> dict[str, int]:
    return {f: 0 for f in _FIELDS}


@contextmanager
def tracking(role: str, into: dict[str, dict[str, int]]) -> Iterator[dict[str, int]]:
    """Count the evaluation-model calls this thread makes inside the block into `into[role]`."""
    previous = getattr(_local, "target", None)
    _local.target = into.setdefault(role, empty())
    try:
        yield _local.target
    finally:
        _local.target = previous


def record(usage: Any) -> None:
    """Add one response's usage (a Responses API `usage` object) to the block being tracked, if any."""
    target = getattr(_local, "target", None)
    if target is None or usage is None:
        return
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    target["calls"] += 1
    target["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
    target["cached_input_tokens"] += int(getattr(input_details, "cached_tokens", 0) or 0)
    target["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
    target["reasoning_output_tokens"] += int(getattr(output_details, "reasoning_tokens", 0) or 0)


def _price(var: str) -> float | None:
    raw = os.environ.get(var, "").strip().strip('"')
    return float(raw) if raw else None


def prices(model: str | None) -> tuple[float, float, float] | None:
    """(input, cached input, output) USD per 1M tokens: the environment's, else the model's default."""
    base = DEFAULT_PRICES.get((model or "").strip())
    given = (_price("STATE_BENCH_EVAL_PRICE_INPUT"), _price("STATE_BENCH_EVAL_PRICE_CACHED_INPUT"),
             _price("STATE_BENCH_EVAL_PRICE_OUTPUT"))
    if base is None and None in (given[0], given[2]):
        return None
    base = base or (given[0], given[0], given[2])
    return tuple(g if g is not None else b for g, b in zip(given, base))  # type: ignore[return-value]


def cost(bucket: dict[str, int], price: tuple[float, float, float]) -> float:
    uncached = bucket["input_tokens"] - bucket["cached_input_tokens"]
    return (uncached * price[0] + bucket["cached_input_tokens"] * price[1] + bucket["output_tokens"] * price[2]) / 1e6


def summary(usage: dict[str, dict[str, int]], model: str | None = None) -> dict[str, Any]:
    """What a trajectory stores: tokens per role and in total, and cost when prices are known."""
    model = model or os.environ.get("STATE_BENCH_EVAL_MODEL", "").strip().strip('"') or None
    total = empty()
    for bucket in usage.values():
        for f in _FIELDS:
            total[f] += bucket[f]
    out: dict[str, Any] = {"model": model, **{role: dict(b) for role, b in usage.items()}, "total": total}
    price = prices(model)
    if price is not None:
        out["prices_per_1m"] = {"input": price[0], "cached_input": price[1], "output": price[2]}
        for role, bucket in usage.items():
            out[role]["cost_usd"] = round(cost(bucket, price), 6)
        total["cost_usd"] = round(cost(total, price), 6)
    return out
