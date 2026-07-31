# Evidence-grounded market signal agent

Answers financial-market questions over the RBA cash-rate, ASX-18 price, and AFR
news datasets. Qwen plans and requests tools, application code executes them
against the raw data, and a fine-tuned Nemotron adapter writes the final answer
from the verified results.

- **Team:** [team name]
- **Endpoints:** `GET /health`, `POST /query`

## Contents

- [Architecture](#architecture)
- [Why it is built this way](#why-it-is-built-this-way)
- [Repository layout](#repository-layout)
- [Running the agent](#running-the-agent)
- [Configuration](#configuration)
- [Verification](#verification)
- [Fine-tuning](#fine-tuning)
- [Results](#results)
- [Known limitations](#known-limitations)

## Architecture

```
POST /query
   │  deadline = now + 50s   (the 60s slow-response penalty is the design constraint)
   ▼
┌─ plan ────────────────────────────────────────────────────────────┐
│  Qwen3.6-35B-A3B-FP8 via LiteLLM `agent-brain`                    │
│  system prompt = metric catalog + routing rules                   │
│  emits OpenAI tool_calls                                          │
└───────────────────────────────────────────────────────────────────┘
   │ tool_calls
   ▼
┌─ act ─────────────────────────────────────────────────────────────┐
│  application code validates and executes, in parallel             │
│  query_data  -> exact RBA / ASX / AFR computations                │
│  retrieve    -> one AFR article by headline and date              │
│                                                                   │
│  each result splits in two:                                       │
│    brain_view  <=700 chars  -> back to Qwen (4096-token context)  │
│    evidence    full facts   -> held for synthesis only            │
└───────────────────────────────────────────────────────────────────┘
   │ results
   ▼
┌─ review ──────────────────────────────────────────────────────────┐
│  Qwen sees the results and judges relevance: replies DONE, or     │
│  requests the missing calls. Same call as the next plan turn, so  │
│  the check costs no extra latency.                                │
└───────────────────────────────────────────────────────────────────┘
   │ done (max 3 brain turns)
   ▼
┌─ synthesize ──────────────────────────────────────────────────────┐
│  fine-tuned Nemotron via `domain-ft`, DOMAIN_PREDICT_MODE=llm     │
│  receives QUESTION + EVIDENCE json with pre-rendered fact strings │
└───────────────────────────────────────────────────────────────────┘
   ▼
{"answer": ..., "steps": N, "tool_trace": [...]}
```

The graph is an explicit state machine in [src/graph.py](src/graph.py): typed
`AgentState`, pure node functions, one router. It deliberately has no orchestration
framework dependency — the cluster is provisioned offline from a USB image, and an
agent that cannot import its scheduler cannot start. The node functions are shaped
so they drop into a `StateGraph` unchanged if one is available.

### Model roles

| Component | Responsibility | Fine-tuned? |
|---|---|---|
| Qwen3.6-35B-A3B-FP8 (`agent-brain`) | Plans, selects tools, emits tool calls, reviews results | No — supplied as-is |
| Agent runtime (`src/tools.py`) | Validates and executes every tool call | n/a |
| Nemotron-8B (`domain-ft`) | Writes the final grounded answer | **Yes** — LoRA r32 |

Neither model reads a file. Qwen never states a figure; Nemotron never picks a tool.

## Why it is built this way

**Every graded number is computed, not generated.** The public calibration
answers reproduce exactly from deterministic Python over the raw files, so the
tool layer — not the model — is what earns the hidden-question score. All 15 are
locked in [tests/test_golden.py](tests/test_golden.py).

**The brain context is only 4,096 tokens.** The supplied vLLM deployment serves
Qwen with `--max-model-len 4096`, so a retrieved article (3–8 KB) would consume
the entire planning budget. Tool results therefore split: a ≤700-character
`brain_view` for planning, and the full `evidence` bundle — including complete
article text — passed straight to Nemotron, which has a 128 K context.

**Evidence is pre-rendered for the writer.** Each fact carries a `text` string
with the value already formatted (`"41 of the 175 decision records changed the
rate"`). The synthesis model composes sentences from those strings instead of
reformatting raw numbers, which is what keeps counts, dates, and signs exact.

**Nemotron's reasoning traces are switched off.** Nemotron-Nano toggles its
`<think>` monologue from the system prompt; left on, that monologue lands in the
graded `answer` field and scores zero. The synthesis prompt begins with
`detailed thinking off` and any residual block is stripped.

**AFR search is indexed.** A brute-force regex scan of the 219,538-article corpus
takes ~15 seconds. [src/afr_index.py](src/afr_index.py) builds a doc-level
inverted index; the index proposes a superset of candidates and the caller's real
regex decides, so results are identical to a full scan.
[tests/test_afr_index.py](tests/test_afr_index.py) proves that equivalence.

**Nothing returns empty.** Brain unreachable → deterministic keyword planner.
Synthesis unreachable or unusable → answer composed from the facts. Unhandled
exception → a stated limitation. A malformed or missing `answer` scores zero, so
that outcome is unreachable by construction.

## Repository layout

```
src/
  app.py           FastAPI service: /health, /query, JSONL run logs
  config.py        environment-driven settings, no hard-coded endpoints
  graph.py         AgentState + plan / act / review / synthesize nodes
  brain.py         Qwen client, tool-call parsing, fallback planner
  synth.py         Nemotron client, thinking-trace stripping, fact composer
  tools.py         tool schemas and dispatch - the only path to the data
  data_rba.py      RBA metrics       data_asx.py  ASX metrics
  data_afr.py      AFR metrics       afr_index.py inverted index
  prompts.py       brain and synthesis prompts
  schemas.py       API contract and the Fact/ToolResult evidence format
scripts/
  build_afr_index.py   one-off index build (~80s)
  smoke_test.py        pre-flight checks, including 3 concurrent /query
  eval_public.py       score against the 15 calibration questions
training/
  make_dataset.py      generates the SFT set from the tool layer
  label_sentiment.py   Qwen-distilled AFR sentiment labels (run on cluster)
  configs/lora_r32.yaml  run_train.sh  eval_base_vs_ft.py  MODEL_CARD.md
tests/               golden values, index equivalence, API contract, concurrency
logs/                per-request JSONL traces
```

## Running the agent

```bash
# 1. dependencies (the Atom environment already supplies these)
pip install -r requirements.txt

# 2. environment
source ~/team.env
export DATA_ROOT="/home/$USER/Downloads/Jasonl format DataSets"
export DOMAIN_PREDICT_MODE=llm     # required before official evaluation

# 3. build the AFR index once (~80s; the agent also builds it on first start)
python scripts/build_afr_index.py

# 4. serve, bound to all interfaces so the harness can reach it
python -m uvicorn src.app:app --host 0.0.0.0 --port 5000
```

Check it:

```bash
curl http://<node-0-lan-ip>:5000/health
curl -X POST http://<node-0-lan-ip>:5000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many RBA decisions changed the cash rate?"}'
```

**Use node 0's LAN address in `submission.json`, not `10.0.1.10`.** The
`10.0.1.x` addresses live on the direct cable between the two nodes; the harness
runs off-cluster and cannot route there. `localhost` fails for the same reason.

**Port.** The submission template and execution guide specify `:5000`; the portal
setup page says the harness connects to port `8001` on the head node. Port 8001
is free on node 0 (8001 belongs to the model node), so both can be served at
once until an organizer confirms which is authoritative:

```bash
AGENT_PORT=5000 python -m uvicorn src.app:app --host 0.0.0.0 --port 5000 &
AGENT_PORT=8001 python -m uvicorn src.app:app --host 0.0.0.0 --port 8001 &
```

## Configuration

All settings come from the environment; no endpoint or credential is committed.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `./data set` | Root holding `RBA Rates/`, `ASX/`, `AFR/` |
| `ARTIFACTS_DIR` | `./artifacts` | Built AFR index |
| `LITELLM_BASE_URL` | `http://localhost:4000/v1` | LiteLLM proxy |
| `LITELLM_KEY` | `sk-local-cluster` | Event credential |
| `BRAIN_MODEL` | `agent-brain` | Qwen planning alias |
| `DOMAIN_FT_MODEL` | `domain-ft` | Fine-tuned Nemotron alias |
| `DOMAIN_PREDICT_MODE` | `mock` | **Set to `llm` before evaluation** |
| `MAX_AGENT_STEPS` | `3` | Brain turns per question |
| `AGENT_DEADLINE_S` | `50` | Hard budget under the 60s penalty |
| `AGENT_PORT` | `5000` | Listen port |

With `DOMAIN_PREDICT_MODE=mock` the agent composes answers deterministically
from the facts. That is the bootstrap default and a genuine fallback, but it does
**not** use the fine-tuned model — startup logs a warning while it is set.

## Verification

```bash
python scripts/smoke_test.py                                    # pre-flight
python scripts/smoke_test.py --endpoint http://<lan-ip>:5000    # plus live server
python -m pytest tests/ -q                                      # 51 tests
python scripts/eval_public.py                                   # calibration score
```

The smoke test checks datasets, the known-correct tool values, model
reachability, an end-to-end answer, the 60-second budget, `/health`, the response
contract, and three concurrent `/query` requests returning distinct answers.

## Fine-tuning

```bash
python training/make_dataset.py                  # ~2,000 examples from the tool layer
python training/label_sentiment.py --count 300   # on the cluster; adds sentiment
```

Then pick a training profile. Both produce a genuinely fine-tuned adapter; they
differ only in how long you wait.

```bash
bash training/run_train.sh --quick    # ~30-45 min: 25 steps, 600 examples
bash training/run_train.sh            # ~2-3 hours: 100 steps, full set
bash training/run_train.sh --dry-run  # print resolved settings, train nothing
```

| | `--quick` | default |
|---|---:|---:|
| Steps | 25 | 100 |
| Warmup steps | 10 | 50 |
| Checkpoint every | 5 | 20 |
| Training examples | 600 (stratified) | 1,688 |
| Wall time | 30–45 min | 2–3 hours |

Two details make the quick profile work rather than just finish early:

- **Warmup is scaled down to 10.** The reference baseline warms up over 50
  steps, so a 25-step run on the original setting would end before the learning
  rate ever reached its peak and would barely train at all.
- **The 600-example subset is stratified, not sampled.** Proportional sampling
  would drop the rare slices — a quick run that never saw a limitation case
  would learn to invent an answer instead of stating the limitation.

Use `--quick` to get an adapter serving early so the full pipeline can be
measured end to end, then start the full run if time allows and report whichever
checkpoint measures better. `--steps N` overrides either profile and rescales
warmup and checkpointing to match.

```bash
bash training/run_train.sh --quick && python training/eval_base_vs_ft.py --quick
```

Gold answers are computed by the tool layer, so no training label contains a
hallucinated figure. Full description in
[training/MODEL_CARD.md](training/MODEL_CARD.md).

The 15 public questions are held out entirely and used only for measurement.

## Results

| Configuration | Calibration score |
|---|---:|
| Fallback planner + deterministic composer (no models running) | **71.6%** |
| Qwen routing + base Nemotron | [fill in on cluster] |
| Qwen routing + fine-tuned Nemotron | [fill in on cluster] |

The first row is the floor the system cannot drop below if both model servers
fail. Reproduce with `python scripts/eval_public.py --no-judge`.

Tool-layer latency, measured locally:

| Operation | Time |
|---|---:|
| RBA and ASX metrics | < 10 ms |
| AFR anchored whole-word count (`\bunemployment\b`) | < 20 ms |
| AFR multi-word alternation, first call | ~2.8 s |
| AFR multi-word alternation, cached | < 10 ms |
| AFR index build (one-off) | ~80 s |

## Known limitations

- **Broad AFR alternations cost ~2.8 s on first call.** Patterns like
  `interest rates?|cash rate|rate cut` need per-document verification. Results are
  cached, and anchored single-word patterns are effectively free.
- **The fallback planner is keyword-based.** It covers common question shapes but
  misroutes multi-date event windows phrased without ISO dates. It exists for
  resilience and local development, not as the primary path.
- **RBA decision dates are effective dates.** The organizer notes one historical
  case where the judge expected the board-meeting date (2010-11-02) rather than
  the effective date (2010-11-03). The supplied file contains only effective
  dates, so that difference cannot be resolved from the data.
- **The RBA file extends to 2026** while AFR and ASX end in 2021. Queries touching
  the extension carry a note flagging it as a forward extension, and
  cross-dataset questions past 2021 return a stated limitation rather than a
  fabricated join.
- **Tabcorp carries a known price artifact.** It is never excluded automatically —
  questions must say so — but any result including it carries a caveat note.
- **Sentiment is the weakest category**, being the only judgement not computed
  from the data.
