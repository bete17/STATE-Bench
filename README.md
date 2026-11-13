<p align="center">
  <img src="assets/state-bench-banner.svg" alt="STATE-Bench: Benchmark For Enterprise Workflows" width="100%" />
</p>

<p align="center">
  <a href="https://microsoft.github.io/STATE-Bench/leaderboard/"><img src="assets/leaderboard-live-badge.svg" alt="Leaderboard Live" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  <a href="https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/"><img src="https://img.shields.io/badge/Blog-Read-blue" alt="Blog" /></a>
</p>

<p align="center">
  <a href="docs/RUN_BENCHMARK.md">Main Track</a> &nbsp;·&nbsp; <a href="docs/AGENT_LEARNING_TRACK.md">Agent Learning Track</a>
</p>

> [!NOTE]
> **This is the `ag_experiments` branch.** It holds our Actor-Gatherer (ag) experiments on STATE-Bench: the results of the runs we have done, and everything needed to look at them or run your own. Start with [Actor-Gatherer Experiments](#actor-gatherer-experiments) below. The rest of this README is STATE-Bench's own.

## Actor-Gatherer Experiments

We test whether [Actor-Gatherer](https://github.com/bete17/actor-gatherer) (ag), a context manager that edits what the model sees before every call, helps an agent that already runs inside a harness ([OpenCode](https://opencode.ai)). The benchmark itself is unchanged: same simulated customers, judges, tools, tasks and scoring.

### Experiment `pass1_75`: four agent setups, same 75 tasks

75 test tasks (25 per domain), the same tasks for every run, one attempt each. Agent model: `gpt-6.1-sol` at medium reasoning. Judge and simulated customer: STATE-Bench's locked `gpt-5.4`.

| Run (folder) | Setup | Pass@1 | Travel | Support | Shopping | Mean UX (1–5) | Agent cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `builtin` | STATE-Bench's built-in agent, no harness | 66.7% | 64% | 72% | 64% | 3.83 | $1.18 |
| `opencode` | OpenCode harness | 62.7% | 60% | 68% | 60% | 3.75 | $1.12 |
| `opencode_train` | OpenCode + a lookup tool over the 100 training conversations per domain | 69.3% | 76% | 72% | 60% | 4.19 | $2.22 |
| `opencode_ag_train` | OpenCode + ag + the same lookup; ag also adds related training conversations to every model call | 72.0% | 76% | 68% | 72% | 4.44 | $9.08 |

Things to keep in mind:

- One attempt per task, so a difference of a few tasks is not yet reliable (run 3 vs run 4: 7 tasks passed only with ag, 5 only without).
- Cost is the agent model only. ag's own model calls (Gemini), the judge and the simulated customer are not included.
- `opencode_ag_train` goes beyond the Agent Learning Track's rules: the track lets learnings reach the agent only through the `retrieve_learnings` lookup, and ag adds a second route. `opencode_train` follows them.

### Look at the results on your computer

The results are in this branch, so you only need Python 3.12 or newer. The tracker uses the standard library only.

```bash
git clone https://github.com/bete17/STATE-Bench.git
cd STATE-Bench
git checkout ag_experiments
python tools/tracker.py --outputs experiments/pass1_75
```

It opens http://localhost:8765 (change it with `--port`). Each run and domain has its own tab, such as `opencode_ag_train/travel`. The filter shows only failed or passed tasks, and the "Mean UX, tasks shown" tile follows it. Click a result to read the whole conversation, every tool call and the judges' reasoning.

### Where the files go

Each experiment is one folder under `experiments/`. Point the tracker at that folder.

```text
experiments/
  run_experiment.sh                   # starts one run of an experiment (below)
  pass1_75/                           # one experiment
    tasks/                            # the fixed task lists: one task id per line, per domain
      travel.txt  customer_support.txt  shopping_assistant.txt  SHA256SUMS
    builtin/                          # one folder per run (its label)
      travel/
        tasks.txt                     # copy of the task list: tells the tracker what to expect
        run1/<task_id>.json           # one scored conversation per task and attempt
      customer_support/ ...
      shopping_assistant/ ...
    opencode/ ...  opencode_train/ ...  opencode_ag_train/ ...
```

If someone sends you results, put each run folder (for example `opencode_ag_train/`, with its three domain folders inside) into the matching experiment folder, then refresh the tracker. Run logs (`*.log`) and ag's memory folders stay out of git.

**ag's memory for `opencode_ag_train` (optional).** It is about 270 MB, so it is not in git. If you were given it, put it at `experiments/pass1_75/ag_home_run4/` and open it in ag's dashboard, from an environment with Actor-Gatherer and its dashboard extras installed:

```bash
python -m ag.cli dashboard --home experiments/pass1_75/ag_home_run4
```

It lists one memory per test task (`statebench-oc-...`) and the three training memories (`statebench-train-<domain>`).

### Run your own experiment

**1. Set up STATE-Bench** as in its guide: `uv sync`, copy `.env.example` to `.env`, and configure the [locked evaluation client](docs/setup/eval-client.md) and the [agent's model key](docs/agents/builtin.md) (`STATE_BENCH_AGENT_API_KEY`).

**2. For the OpenCode runs**, install OpenCode (`npm install -g opencode-ai`, then `opencode --version`). See [Harness Agents](docs/RUN_BENCHMARK.md#7-harness-agents-opencode-and-actor-gatherer) for details.

**3. For the ag run**, install Actor-Gatherer into the same environment (`uv pip install -e ../actor-gatherer`). It needs its `windows-support` branch or newer, which has the wording settings the setup script uses. Set `GEMINI_API_KEY`, then load and prepare the training conversations once:

```bash
uv run python tools/seed_ag_training.py
uv run python tools/prebuild_ag_training.py --domain travel --domain customer_support --domain shopping_assistant
```

The first stores each domain's 100 training conversations in an ag memory (`statebench-train-<domain>`). The second writes the run's ag settings and processes those conversations once, so every task starts from a ready copy. The settings are: a lower build threshold (3,000 tokens), Gemini Flash for the quick per-turn steps, and wording that calls them other customers' conversations.

**4. Run.** Each command runs one setup on the experiment's task lists, all three domains at once:

```bash
experiments/run_experiment.sh pass1_75 builtin
experiments/run_experiment.sh pass1_75 opencode OpenCodeAgent
experiments/run_experiment.sh pass1_75 opencode_train OpenCodeTrainAgent
experiments/run_experiment.sh pass1_75 opencode_ag_train OpenCodeAgTrainAgent
```

The script refuses to overwrite a run that already has results, so use a new label (for example `opencode_ag_train_v2`) or a new experiment. To try another model or more attempts, set `MODEL`, `REASONING`, `RUNS` or `WORKERS` before the command (for example `RUNS=3 WORKERS=2 experiments/run_experiment.sh pass1_75 opencode_v2 OpenCodeAgent`). If `uv` isn't available, set `PYTHON` to your Python (for example `PYTHON=.venv/bin/python`).

**A new experiment with other tasks.** Make `experiments/<name>/tasks/<domain>.txt`, one task id per line, using ids from the test split (`state_bench/domains/<domain>/splits/train_test.json`). Then run the script with `<name>`. Use the same task lists for every run you want to compare.

**Tips**

- Lower `WORKERS` if you hit your provider's rate limit. Run 4 used the most tokens and hit OpenAI's 1M-tokens-per-minute limit at 3 workers per domain.
- Each OpenCode instance uses about 300 MB of memory. On a 16 GB machine, keep it to 3 workers per domain.
- Every ag task leaves a memory under `~/.ag/gatherers/` (`statebench-oc-*`). They can be deleted after a run.

STATE-Bench evaluates AI agents on realistic, multi-step enterprise workflows across three domains: **travel**, **customer support**, and **shopping assistant**.

Each task gives the agent a task-local sandbox database, domain-specific tools, and a simulated user. To pass a task, the agent must do multi-step reasoning by gathering the right information with domain tools, applying the correct policy, taking actions to update the database to the right final state when needed, and following the required procedure in conversation.

## Overview

STATE-Bench includes 450 challenging enterprise tasks across three domains.

| Domain | Tasks | Description |
| --- | ---: | --- |
| **Travel** | 150 | Flight, hotel, and car rental bookings; cancellations, updates, fee and policy reasoning, cross-product trip planning |
| **Customer Support** | 150 | Returns, refunds, exchanges, warranty claims, cancellations, shipping issues, and order changes |
| **Shopping Assistant** | 150 | Product search, cart updates, applying promos, loyalty redemption, shipping options, and compatibility checks |

## Choose Your Benchmark Track

Start with the track that matches what you want to evaluate. Each track guide links to the setup and reference docs only when you need them.

| Goal | Start here |
| --- | --- |
| Evaluate an agent or model directly on the provided enterprise benchmark tasks | **[Main Track](docs/RUN_BENCHMARK.md)** |
| Evaluate agentic memory, skills, or prompt optimization | **[Agent Learning Track](docs/AGENT_LEARNING_TRACK.md)** |

The **Main Track** is the default benchmark path. The **Agent Learning Track** uses the same simulator, domain tools, judges, and metrics, but adds train trajectories and a retrieval hook for reusable learnings such as memories, skills, or prompt optimizations.

For run commands and flag selection, start with the relevant track guide. The shared [`run_batch` reference](docs/eval/run-batch.md) lists which flags are required for the built-in agent, custom clients, and learning-track runs.

<br/>

<p align="center">
  <img src="assets/chat_bubble_2.svg" alt="Sample task trajectory from the Travel domain" width="55%" />
  <br/>
  <em>Sample task trajectory from the Travel domain.</em>
</p>

## Metrics

STATE-Bench reports four headline metrics:

| Metric | What it measures |
| --- | --- |
| **Task Completion pass@1** | Average task completion rate across five runs per task. |
| **Task Completion pass^5** | Percentage of tasks completed successfully on all five runs. |
| **UX Score** | LLM-judged conversation quality on a 1-5 scale. |
| **Cost Per Task** | Average agent cost from user-reported dollars per task. |

## License

STATE-Bench is released under the MIT License. See [LICENSE](LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Disclosures

Datasets provided in this benchmark were synthetically generated using large language models. The benchmark is intended for research purposes and users should exercise caution and consider the limitations of synthetic data when interpreting results.
