# Execution Guide — Architecture, Serving, and Agent Design

## What you're building and why

The challenge is a **structured financial data Q&A system**, not a general-purpose chatbot. Questions
may require exact calculations, document retrieval, financial sentiment or direction classification,
cross-dataset reasoning, or a justified finding that the supplied evidence cannot support a claim.
The required architecture is:

1. The supplied **Qwen3.6-35B-A3B-FP8 `agent-brain`** plans the approach, selects tools, and emits tool calls with the correct arguments
2. The **agent runtime** validates and executes those calls as exact queries against the raw data, then returns structured results to Qwen
3. The **fine-tuned Nemotron domain model** receives the question and verified tool results and synthesises the final clean answer

Participants fine-tune Nemotron, not Qwen3.6-35B-A3B-FP8. Nemotron is not the primary tool-calling model in this
architecture. Qwen3.6-35B-A3B-FP8 requests tool calls; application code executes them; fine-tuned Nemotron performs
the final grounded financial-domain synthesis.

---

## Infrastructure layout

Each team receives a two-node GIGABYTE Atom cluster with one NVIDIA GB10 per node. The organizers
provide the actual hostnames and IP addresses. The role labels below are not machine names.

```
brain/agent node                     fine-tuning/model node
─────────────────────                ────────────────────────
• LiteLLM proxy   :4000              • Fine-tuned vLLM  :8001
  ├─ agent-brain (Qwen3.6-35B-A3B-FP8)            (Nemotron-8B + LoRA)
  └─ domain-ft → model node :8001
• Agent server    :5000
• Eval harness
```

- The **brain/agent node** runs Qwen3.6-35B-A3B-FP8 reasoning, LiteLLM, and the agent web server
- The **fine-tuning/model node** runs training and vLLM serving the fine-tuned LoRA adapter; no weight merge is needed

---

## Agent serving stack

### 1. Start the LiteLLM proxy on the brain/agent node

The proxy is a local OpenAI-compatible gateway that routes model names to actual endpoints.

```bash
litellm --config /path/to/litellm_config.yaml --port 4000
```

Key routes in the config:
```yaml
model_list:
  - model_name: agent-brain
    litellm_params:
      model: openai/Qwen/Qwen3.6-35B-A3B-FP8
      api_base: http://localhost:8000/v1
  - model_name: domain-ft
    litellm_params:
      model: openai/nemotron-8b-finance-lora
      api_base: http://<fine-tuning-node-ip>:8001/v1
```

Replace `<fine-tuning-node-ip>` with the address assigned to your cluster.

### 2. Start your fine-tuned model on the fine-tuning/model node

Use the supplied launch script from the host. `$MODELS_DIR` is the host-side model directory;
the script mounts it at `/models` inside the vLLM container.

```bash
cd ~/Cognitivo_Training/finagent-finetune
source ~/team.env

# Locate the adapter produced by your selected training run.
find "$MODELS_DIR/checkpoints" -type d -name hf_adapter

ADAPTER_CHECKPOINT="$MODELS_DIR/checkpoints/<your-run>/checkpoints/<checkpoint>/hf_adapter" \
bash scripts/04_export_and_serve.sh
```

### 3. Start the agent server on the brain/agent node

Implement the agent in your repository's `src/` directory, then start it on all network interfaces.
For a FastAPI implementation, a typical command is:

```bash
cd ~/your-team-repository
uvicorn your_python_module:app --host 0.0.0.0 --port 5000
```

Endpoints the eval harness calls:
- `GET /health` — must return 200
- `POST /query` — receives `{"question": "..."}`, returns `{"answer": "...", "steps": N, "tool_trace": [...]}`

---

## Agent architecture

```
POST /query
    │
    ▼
AgentState (question, messages, steps, tool_trace)
    │
    ▼
┌── reason() ────────────────────────────────────────────────┐
│   brain model (Qwen3.6-35B-A3B-FP8 through agent-brain)               │
│   system prompt = tool docs + reasoning rules              │
│   if model emits tool_calls → act()                       │
│   if no tool_calls → synthesize()                         │
└─────────────────────────────────────────────────────────────┘
         │ tool_calls                      │ no tool_calls
         ▼                                 ▼
    act() — dispatch to               synthesize()
    tool registry, return             domain model (Nemotron-8B)
    structured result                 writes clean final answer
         │                                 │
         └──────── loop back ──────────────┘
                  (max 10 iterations)
```

The **Qwen3.6-35B-A3B-FP8 brain** decides which tool to call and with what arguments. The application runtime
executes each requested call and returns its structured result to Qwen. The **fine-tuned Nemotron
domain model** synthesises the final answer only after the Qwen3.6-35B-A3B-FP8 reasoning loop is complete. This
separation means:

- The brain can be swapped without retraining
- The supplied Qwen3.6-35B-A3B-FP8 brain owns planning, tool selection, and tool-call generation
- Application code owns validation and execution of tool requests
- Nemotron does not need to learn tool routing; its fine-tuning focuses on reading structured evidence, Australian financial terminology, complete answer composition, correct numerical formatting, and avoiding unsupported claims

---

## Tool interface

All data access goes through one function: `query_data(dataset, metric, ...)`.

| Dataset | Key metrics |
|---|---|
| `rba` | `count`, `count_changes`, `count_increases`, `count_decreases`, `extremes`, `max_hold_streak`, `lookup_rate`, `list` |
| `asx` | `annual_return`, `rank_annual_returns`, `full_sample_return`, `volatility`, `correlation`, `max_drawdown` |
| `afr` | `count`, `count_by_month`, `share` — all require `pattern=` (Python regex) |

**Important AFR search rules:**
- Pattern is matched case-insensitively across HEADLINE + SUBHEAD + INTRO + TEXT combined
- Word boundaries required: use `\bword\b` not just `word`
- One match per article, regardless of how many times the term appears

---

## Environment variables

Set these before starting the agent:

```bash
BRAIN_MODEL=agent-brain           # LiteLLM model name for reasoning
DOMAIN_FT_MODEL=domain-ft         # LiteLLM model name for Nemotron synthesis
DOMAIN_PREDICT_MODE=llm           # required after the adapter is live; bootstrap defaults to mock
LITELLM_URL=http://localhost:4000/v1
LITELLM_KEY=sk-local-cluster
JUDGE_LITELLM_URL=http://localhost:4000/v1   # used by eval harness scorer
```

Keep `DOMAIN_FT_MODEL=domain-ft` and update the LiteLLM route once your fine-tuned model is served
on the assigned fine-tuning/model node. The bootstrap begins with `DOMAIN_PREDICT_MODE=mock` only to
support pre-training plumbing tests. Switch it to `llm` before evaluation so final synthesis uses
the fine-tuned Nemotron adapter.

---

## Why does fine-tuning matter here?

Qwen3.6-35B-A3B-FP8 handles tool routing, but the final answer is still produced by Nemotron. An unmodified base
Nemotron can omit requested components, misread structured results, add unsupported context, or use
inconsistent financial terminology and formatting. Fine-tuning on the domain synthesis examples
teaches Nemotron:

- How to read the structured JSON results produced by `query_data()` and retrieval tools
- How RBA, ASX, and AFR concepts relate without inventing values that are absent from the evidence
- How to preserve exact counts, dates, rates, returns, signs, and units
- How to include every requested answer component in concise prose
- How to state limitations when the supplied tool evidence is incomplete

The supplied Qwen3.6-35B-A3B-FP8 brain must still be given accurate tool schemas and metric documentation because
Qwen3.6-35B-A3B-FP8, not Nemotron, chooses the tools and arguments.

**Historical system result:** an early baseline pipeline scored 0% on 15 questions, while the
integrated pipeline reached about 74% on a larger 75-question run. That improvement reflects the
combined Qwen3.6-35B-A3B-FP8 routing, deterministic tools, prompts, and fine-tuned Nemotron synthesis. Teams must
provide a controlled base-Nemotron versus fine-tuned-Nemotron comparison for the official model
quality assessment.

---

## Running the evaluation harness

The harness reads team info directly from `submission.json` files — no separate teams config to edit.
It sends up to three questions concurrently to each team by default (`--workers 3`). Your FastAPI
service, tool layer, and model endpoints must handle at least three simultaneous requests safely.

```
p3_eval/submissions/
  alpha/submission.json        ← organizer's own agent
  team-01/submission.json      ← copied from each team's repo on eval day
  team-02/submission.json
```

Organizers may alternatively export `submissions.json` from the portal and pass that manifest
directly with `--submissions /path/to/submissions.json`. Both forms contain the same submission
contract; the folder layout remains the participant-independent default.

```bash
cd AI_Training_and_Hackathon/p3_eval

# Smoke test against public questions (one team):
python eval_harness.py \
  --questions ../../jsonl_evaluation/public_questions.jsonl \
  --team alpha

# Full eval (hidden questions, all teams):
python eval_harness.py

# Run two teams in parallel (separate terminals):
python eval_harness.py --team alpha       # terminal 1
python eval_harness.py --team team-01     # terminal 2
```

Reports land in `p3_eval/reports/` as per-team JSON and a combined hidden-question leaderboard
JSON. The scorer uses the `agent-brain` model as an LLM judge — it is independent of your agent's
model. This harness produces the hidden-question category score (40%); organizers separately assess
fine-tuned model quality (30%) and architecture/repository quality (30%) before publishing the
weighted final leaderboard.
