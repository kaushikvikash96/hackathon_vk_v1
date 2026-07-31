# Team Submission Guide

This guide defines what each team must submit and how the organizers will call the submitted agent during evaluation.

## Contents

- [Required Repository Structure](#required-repository-structure)
- [Final `submission.json`](#final-submissionjson)
- [Agent API Contract](#agent-api-contract)
- [Official Scoring](#official-scoring)
- [Hidden-Question Scoring - 40%](#hidden-question-scoring-40)
- [What the leaderboard shows](#what-the-leaderboard-shows)
- [Your own detailed report](#your-own-detailed-report)
- [README Requirements](#readme-requirements)
- [Training Evidence](#training-evidence)
- [Submission Checklist](#submission-checklist)

---

## Required Repository Structure

Each team repository must be fully public for the entire event — there is no private-repository or
collaborator-access option. Organizers evaluate strictly against your public GitHub URL, so it must
be reachable by the organizers before the submission deadline.

```text
TeamSubmission/
  README.md
  submission.json
  src/
    .gitkeep
  training/
    .gitkeep
  logs/
    .gitkeep
  Participant_Package/
    answer_template.json
    Challenge_Brief.md
    public_questions.jsonl
    questions_template.json
    Setup_Instructions.md
    submission-guide.md
    submission_template.json
    validate.json
    handout/
      01_training_guide.md
      02_execution_guide.md
      03_scoring_and_examples.md
```

| Path | Required | Purpose |
|---|---:|---|
| `submission.json` | Yes | Final team metadata, pinned repository commit, and live agent endpoint registration. |
| `README.md` | Yes | Project summary, architecture, run instructions, endpoint notes, and known limitations. |
| `src/` | Yes | Agent source code and data-query or retrieval tools. |
| `training/` | Yes | Fine-tuning scripts, preparation notes, configuration, logs, metrics, and model summary. |
| `logs/` | Yes | Useful non-sensitive training or agent-run logs. |
| `Participant_Package/` | Yes | Challenge references, sample request and response files, validation rules, and handouts. |

The supplied environment contains the common dependencies used during the event. Official scoring calls the registered agent endpoint rather than rebuilding every project live.

## Final `submission.json`

`submission.json` is the single source of truth for your team — it registers your team identity, your GitHub commit, and tells the evaluation harness where to reach your agent. The harness reads this file directly; there is no separate teams config to edit.

The file `Participant_Package/submission_template.json` is a reference example. Update the real
`submission.json` at the repository root; the evaluator does not use the template file as your registration.

```json
{
  "schema_version": "1.0",
  "team_id": "team-example",
  "team_name": "Example Team",
  "github_url": "https://github.com/example/team-agent",
  "commit_sha": "0123456789abcdef0123456789abcdef01234567",
  "agent": {
    "endpoint": "http://172.20.x.x:5000",
    "health_path": "/health",
    "query_path": "/query",
    "timeout_seconds": 300
  },
  "model": {
    "endpoint": "http://172.20.x.x:8001/v1",
    "model_name": "nemotron-8b-finance-lora"
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `team_id` | Yes | Short identifier used in report filenames — no spaces |
| `team_name` | Yes | Display name on the leaderboard |
| `github_url` | Yes | Public repo URL |
| `commit_sha` | Yes | Exact 40-char commit hash to be judged |
| `agent.endpoint` | Yes | Full URL where your server is reachable — use IP, not hostname |
| `agent.health_path` | Yes | Path the harness calls for the pre-eval health check (usually `/health`) |
| `agent.query_path` | Yes | Path the harness POSTs questions to (usually `/query`) |
| `agent.timeout_seconds` | Yes | Set to 300; used per request and capped by the organizer's `--timeout` value |
| `model.model_name` | Yes | Name or alias of the fine-tuned model used by the submitted solution |
| `model.endpoint` | Conditional | Reachable OpenAI-compatible endpoint when direct model testing is the agreed assessment method |

Fine-tuned model quality is an official scoring category. If you cannot expose the model endpoint,
agree on another technical assessment method with the organizers before the deadline and document
it in your README.

These fields have different owners during evaluation:

- `agent.endpoint`, `health_path`, `query_path`, and `timeout_seconds` are used directly by the
  hidden-question harness.
- `github_url` and `commit_sha` are used by organizers when cloning the public repository for the
  architecture and repository assessment. The hidden-question harness records them as metadata but
  does not clone the repository itself.
- `model.model_name` and `model.endpoint` support the separate fine-tuned-model assessment. They are
  recorded in organizer reports but are not used to answer hidden questions.

**Use your machine's IP address** (`ip addr` to find it), not `localhost`. The organizer harness runs on a different machine. Do not put credentials or API keys in this file.

## Agent API Contract

Your submitted agent server must live in `src/` and expose the two endpoints below. Teams may choose
their own internal module structure while following the required API and architecture contract in
[Challenge Brief → Required Model Roles](Challenge_Brief.md#required-model-roles) (Qwen plans and
calls tools, application code executes them, fine-tuned Nemotron synthesizes the final answer —
including the `DOMAIN_PREDICT_MODE` mock-to-llm switch required before official evaluation).

### `GET /health`

Returns HTTP 200 when your agent is ready. The harness checks this before starting the evaluation. If the health check fails, your team is skipped.

```json
{"status": "ok"}
```

### `POST /query`

Each request contains one question and your agent must return one JSON response. The harness uses
up to **three concurrent requests per team** by default, so `/query`, shared state, and model serving
must safely handle at least three simultaneous calls.

**Request body template** (`questions_template.json`):
```json
{
  "question": "From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?"
}
```

**Response template** (`answer_template.json`):
```json
{
  "answer": "41 of the 175 decision records changed the rate: 20 increases and 21 decreases.",
  "steps": 3,
  "tool_trace": [
    {
      "tool": "query_data",
      "args": {"dataset": "rba", "metric": "count_changes"},
      "result": "41 changes: 20 increases, 21 decreases"
    }
  ]
}
```

| Field | Required | Graded | Notes |
|---|---|---|---|
| `answer` | Yes | **Yes** | The only response field scored by the automated hidden-question judge |
| `steps` | No | No | Total tool calls plus synthesis steps; retained for private diagnostics |
| `tool_trace` | No | No | List of `{tool, args, result}`; retained for private diagnostics |

**Slow-response penalty:** see [Challenge Brief → Response-Time Rules](Challenge_Brief.md#response-time-rules) — responses over 60 seconds lose 20% of earned points; over 300 seconds scores zero. Design your agent to return within 60 seconds.

**Malformed or timed-out responses** score zero for that question.

## Official Scoring

```text
final_score =
    (fine_tuned_model_score * 0.30)
  + (architecture_repository_score * 0.30)
  + (hidden_question_score * 0.40)
```

The complete rubric — what each category assesses, the Response-Time Rules, and the health-check
gate — is in [Challenge Brief → Scoring](Challenge_Brief.md#scoring).

## Hidden-Question Scoring - 40%

Each question has one or more grading components, each with its own point value:

```json
{
  "grading": {
    "max_score": 10,
    "components": [
      { "component_id": "C01", "points": 5, "expected_fact": "..." },
      { "component_id": "C02", "points": 5, "expected_fact": "..." }
    ]
  }
}
```

The LLM judge checks each `expected_fact` independently against your answer and awards that
component's points on a YES/NO basis — **partial credit is possible**. Each evaluation case may also
declare a `grading.tolerance_note` (e.g. `+/-0.02` percentage points for calculated returns); public
calibration questions expose their tolerance notes, hidden questions use the same schema without
revealing the expected facts.

See [Challenge Brief → What a Good Answer vs a Bad Answer Looks
Like](Challenge_Brief.md#what-a-good-answer-vs-a-bad-answer-looks-like) for worked examples of
full-credit, zero-credit, and partial-credit responses.

## What the leaderboard shows

The public leaderboard shows only **Rank**, **Team**, and **Score** — nothing else. Latency, tool usage, step counts, and availability are recorded per team but not published on the public leaderboard. Teams do not see other teams' internals.

Only the weighted final Score determines rank. The hidden-question category is calculated as
`sum(earned_points) / sum(max_points) x 100%` after slow-response penalties, then contributes 40%
to the final score.

## Your own detailed report

After the eval run, each team receives a **detailed private report** covering only their own agent:

- Hidden-question score, tool rate, avg steps, avg latency, P95 latency, availability, slow penalty
- Per-question breakdown: earned/max, latency, tool usage, per-component YES/NO verdicts

The report **excludes hidden grading facts** — you see whether each component passed but not the expected fact the judge checked. Other teams' data is never included.

## README Requirements

The `README.md` must include:

- Team name and short project summary.
- Exact command used to run the agent.
- Agent endpoint paths and expected response shape.
- High-level architecture (per [Challenge Brief → Required Model Roles](Challenge_Brief.md#required-model-roles)): Qwen planning and tool-call generation, runtime tool execution, retrieval, and fine-tuned Nemotron answer synthesis.
- Training summary explaining what was fine-tuned, which preparation method was used, and where supporting evidence is stored.
- Base-versus-fine-tuned evaluation results and the method organizers should use to assess the final model.
- Known limitations and failure cases.

## Training Evidence

The `training/` folder must contain enough evidence for judges to understand and reproduce the team's fine-tuning work. Suitable contents include:

- Training or fine-tuning scripts.
- Data-preparation scripts or notebook exports.
- Configuration and hyperparameters.
- Training logs, metrics, or screenshots.
- A short model card or model summary.
- Held-out results or representative comparisons showing improvement over the supplied base model.

## Submission Checklist

Before submitting, confirm that:

- The repository is public — no private repos or collaborator-access exceptions — and organizers can clone it without credentials.
- `submission.json` is at the repository root with your final IP, port, and commit SHA filled in.
- `commit_sha` is the exact 40-character commit hash to be judged.
- `Participant_Package/answer_template.json` is present and follows the required response shape.
- `GET /health` returns 200 from the IP in `submission.json`.
- `POST /query` accepts `{"question": "..."}` and returns a JSON object with a non-empty `answer`; `steps` and `tool_trace` are optional.
- `/query` and the model-serving stack handle concurrent requests safely (see Agent API Contract).
- Most responses return within 60 seconds (responses over 60s incur a 20% point deduction).
- `README.md`, `src/`, `training/`, and `logs/` contain the required material.
- The complete Qwen-plans / runtime-executes / fine-tuned-Nemotron-synthesizes architecture is implemented, and `DOMAIN_PREDICT_MODE=llm` is enabled after the adapter is served (see [Challenge Brief → Required Model Roles](Challenge_Brief.md#required-model-roles)).
- `model.model_name` identifies the fine-tuned model and `model.endpoint` is reachable when direct testing is the agreed assessment method.
- The repository documents base-versus-fine-tuned results and the complete architecture.
- No credentials or organizer-only evaluation material are committed.
