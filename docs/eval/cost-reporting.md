# Reporting Avg. Cost Per Task

Average cost per task is one of the four headline metrics. Reporting cost is **optional** but strongly encouraged — without reported cost, your submission will show `$0` cost in `metrics.json` and cannot be compared on the cost axis. Other metrics (completion, reliability, UX) still score correctly.

## Report cost from your agent

Providers and SDKs report token buckets differently, so STATE-Bench does not compute cost from token counts. Your custom agent should calculate cost using the accounting model that is correct for your provider, then report the dollar amount:

```python
self.add_cost_usd(cost_usd)
```

Use categories when you want a breakdown in metrics:

```python
self.add_cost_usd(agent_call_cost_usd, category="agent_turn")
self.add_cost_usd(memory_build_cost_usd, category="memory_ingestion")
self.add_cost_usd(memory_lookup_cost_usd, category="memory_retrieval")
```

## Where token counts come from

### OOTB StateBenchAgent (Azure AI Foundry/OpenAI)

Token counts come straight from the Responses API usage metadata. No extra code is required; the agent records usage automatically.

### Custom client + agent

Token reporting is optional telemetry. If your provider exposes useful counts, call `self.add_token_usage(...)` after each provider LLM call returns:

```python
self.add_token_usage(
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    cached_input_tokens=cached_input_tokens,
)
```

- `input_tokens` and `output_tokens` are required for usage telemetry to be recorded.
- `cached_input_tokens` is optional.
- If either input or output tokens are missing, STATE-Bench skips usage telemetry for that call.
- Token counts do not affect cost. Use `self.add_cost_usd(...)` for cost.

See the full signature in [`state_bench/agents/base.py`](../../state_bench/agents/base.py).

## Simulated user and judge cost (local addition)

The cost above is the agent's only. Every task also calls the locked evaluation model (GPT-5.4) as the simulated user and as the judges, and that is often the larger share of the bill. `run_batch` records it per task in each result file under `eval_usage`:

For example, a short two-turn travel task:

```json
"eval_usage": {
  "model": "gpt-5.4",
  "simulator": {"calls": 1, "input_tokens": 1521, "cached_input_tokens": 0, "output_tokens": 49, "reasoning_output_tokens": 0, "cost_usd": 0.004537},
  "judge": {"calls": 2, "input_tokens": 8983, "cached_input_tokens": 1792, "output_tokens": 3929, "reasoning_output_tokens": 3538, "cost_usd": 0.07736},
  "total": {"calls": 3, "input_tokens": 10504, "cached_input_tokens": 1792, "output_tokens": 3978, "reasoning_output_tokens": 3538, "cost_usd": 0.081898},
  "prices_per_1m": {"input": 2.5, "cached_input": 0.25, "output": 15.0}
}
```

- `simulator` counts the calls made during the conversation, `judge` those made while scoring it. Retried attempts are included, since they are billed too.
- Cost uses `STATE_BENCH_EVAL_PRICE_INPUT`, `STATE_BENCH_EVAL_PRICE_CACHED_INPUT` and `STATE_BENCH_EVAL_PRICE_OUTPUT` (USD per 1M tokens) when set, else GPT-5.4's list prices when `STATE_BENCH_EVAL_MODEL` is `gpt-5.4`; otherwise only tokens are saved.
- Counting does not change what the evaluation model is sent or returns, and it is not part of the official metrics.
- Results made before this change have no `eval_usage`. Scoring with `state_bench.scripts.score` on its own does not record it.

The tracker (`tools/tracker.py`) shows it as "Customer + judge cost", and per result in its detail view.
