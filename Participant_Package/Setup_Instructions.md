# Participant Setup Information

The Atom environment is already prepared for participants. The datasets, base Nemotron model, Python environment, and local model-serving services are supplied by the organizers.

Participants should not download replacement datasets or switch to unrestricted external services during scoring.

## Contents

- [Supplied Datasets](#supplied-datasets)
- [Supplied Model](#supplied-model)
- [Reference Configuration](#reference-configuration)
- [Before Submission](#before-submission)

---

## Supplied Datasets

The mock hackathon uses exactly three datasets:

| Dataset | Supplied folder | Intended use |
|---|---|---|
| RBA cash-rate decisions | `RBA-Rates-2010-2026` | Rate changes, hikes/cuts, dates, targets, and period comparisons |
| ASX company prices | `ASX-18-companies-2015-2021-Jasonl` | Returns, volume, rankings, drawdowns, baskets, and event windows |
| AFR news corpus | `AFR Jasonl` | Article retrieval, pattern counts, date aggregation, and news evidence |

Use structured parsing and deterministic calculations for RBA and ASX data. Use the supplied local full-text search, indexed search, or RAG service for AFR records. Cross-dataset answers must respect the overlapping date coverage and clearly identify missing coverage.

### Dataset Field Schemas

| Dataset | Fields |
|---|---|
| AFR | `HEADLINE, SUBHEAD, INTRO, TEXT, NEWSPAPER, PUBLICATIONDATE` |
| ASX | `ticker, date, open, high, low, close, volume` |
| RBA | `Effective Date, Change % points, Cash rate target%` (UTF-8 BOM encoding) |

### AFR Text Search

> **All AFR pattern counts must search across `HEADLINE`, `SUBHEAD`, `INTRO`, and `TEXT`
> combined.** Searching only the headline or only the body will produce different counts that will
> not match the reference answers. Use case-insensitive, once-per-record matching: a record counts
> once even if the pattern appears in multiple fields.

Whole-word searches must use word-boundary anchors, such as `\bNAB\b` rather than just `NAB`.
Short acronyms without boundaries will match substrings in unrelated words and significantly
inflate counts.

These points are non-negotiable for reproducibility — scores are computed by running the same tool
calls against the same data, so a different search scope or field set will not match the reference
answers.

## Supplied Model

Participants receive **Llama-3.1-Nemotron-Nano-8B-v1** as the base model to fine-tune or adapt in Atom. Teams should:

1. Prepare suitable domain training examples.
2. Fine-tune or adapt the supplied model.
3. Record the training configuration and data-preparation method.
4. Connect the resulting model to the agent through the supplied local model alias or endpoint.
5. Use data tools for dataset-derived facts rather than relying on model memory.

After training, update the configured model alias or environment setting. Keep endpoints and credentials in environment variables rather than hard-coding them.

### Fine-Tuning Reference Baseline

The configuration below is a confirmed working starting point.

> **Note:** These values are a reference baseline, not a required configuration. Teams are
> encouraged to experiment with the tunable parameters and justify their choices while staying
> within the available hardware, event time, and model-context constraints.

| Parameter | Reference starting value |
|---|---|
| NeMo container | `nvcr.io/nvidia/nemo:25.09` |
| LoRA rank | 32 |
| Sequence length | 512 (longer sequences may run out of memory on a single node) |
| Learning rate | `5e-5` recommended (`1e-4` causes a loss spike after warmup) |
| Training steps | 100 for a full run; the step 20 checkpoint already shows meaningful improvement |

## Reference Configuration

The supplied agent scaffold reads settings through `agent/config.py`:

| Variable | Purpose |
|---|---|
| `LITELLM_BASE_URL` | Local OpenAI-compatible LiteLLM endpoint |
| `LITELLM_KEY` | Event environment credential |
| `BRAIN_MODEL` | Agent reasoning model alias |
| `DOMAIN_FT_MODEL` | Fine-tuned Nemotron/domain model alias |
| `DOMAIN_PREDICT_MODE` | Switches domain-model behavior between `mock` (bootstrap default) and `llm`. Must be `llm` before official evaluation so the fine-tuned model is actually used — see [Challenge Brief → Required Model Roles](Challenge_Brief.md#required-model-roles). |
| `EMBED_MODEL` | Local embedding model alias, when used |
| `QDRANT_URL` | Optional local AFR retrieval endpoint |
| `QDRANT_COLLECTION` | Optional AFR collection name |
| `MAX_AGENT_STEPS` | Maximum agent tool iterations |

Route article-grounded sentiment questions through your fine-tuned domain model using the `DOMAIN_FT_MODEL` alias. The model should receive the retrieved AFR article text and the applicable RBA rate as context and return a sentiment classification (positive, negative, or mixed) and a likely market direction. Do not force the model to emit a made-up numeric return or price forecast.

### Model Serving Endpoints

| Service | Default endpoint | Notes |
|---|---|---|
| LiteLLM proxy | `http://localhost:4000` | Configured by organizers; use `LITELLM_BASE_URL`, `BRAIN_MODEL=agent-brain`, `DOMAIN_FT_MODEL=domain-ft`, and switch `DOMAIN_PREDICT_MODE` from `mock` to `llm` after the adapter is live. |
| Qwen reasoning brain (vLLM) | Port `8000` on the assigned brain/agent node | Served by organizers behind the `agent-brain` alias for planning and tool-call generation. |
| Fine-tuned Nemotron (vLLM) | Port `8001` on the assigned fine-tuning/model node | Team deploys after training behind the `domain-ft` alias for final synthesis. |

Each team receives a two-node GIGABYTE Atom cluster with one NVIDIA GB10 per node. The organizers
provide the actual hostnames and IP addresses. Any hostname or IP shown in a command must be replaced
with the value assigned to your cluster.

> **Keep all credentials and endpoint URLs in environment variables.** Do not hard-code them in
> source files. Source the organizer-provided `~/team.env` before starting your services. The
> evaluation harness calls the registered agent endpoint; it does not inject variables into the
> participant's running process.

## Before Submission

Confirm that:

- the agent can read all three approved datasets;
- the fine-tuned model is used during inference;
- article-grounded sentiment questions route retrieved AFR context and the applicable RBA rate through your fine-tuned domain model;
- one public question can pass through the complete agent pipeline;
- the response passes the JSON Schema in `validate.json`.

See [Submission Guide → Submission Checklist](submission-guide.md#submission-checklist) for the
full pre-submission checklist (repository, `submission.json`, endpoints, and API contract).

Ask an organizer if a supplied path, endpoint, model alias, or credential is unavailable. Do not silently replace a missing organizer service with an external one.
