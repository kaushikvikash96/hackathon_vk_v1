# Runbook — running the agent on the Atom cluster

Ordered operating instructions for event day, on the two-node GIGABYTE Atom
cluster (one NVIDIA GB10 per node). Node roles:

| Node | Runs | Ports |
|---|---|---|
| **Node 0** — brain/agent | LiteLLM proxy, Qwen `agent-brain`, this agent | 4000, 8000, 5000 |
| **Node 1** — fine-tuning/model | LoRA training, fine-tuned Nemotron via vLLM | 8001 |

Replace `<node0-lan-ip>`, `<node1-lan-ip>`, `<node1-user>` and
`<your-public-repo-url>` with the values assigned to your cluster. The
`10.0.1.x` addresses below are the direct cable link between the two nodes.

Architecture and configuration reference: [README.md](README.md).

## Contents

- [Phase 0 — Cluster bring-up](#phase-0--cluster-bring-up)
- [Phase 1 — Project onto node 0](#phase-1--project-onto-node-0)
- [Phase 2 — LiteLLM and the brain](#phase-2--litellm-and-the-brain)
- [Phase 3 — Serve the agent](#phase-3--serve-the-agent)
- [Phase 4 — Train the adapter](#phase-4--train-the-adapter-node-1)
- [Phase 5 — Wire in the fine-tuned model](#phase-5--wire-in-the-fine-tuned-model)
- [Phase 6 — Submit](#phase-6--submit)
- [What most often goes wrong](#what-most-often-goes-wrong)
- [Troubleshooting](#troubleshooting)

---

## Phase 0 — Cluster bring-up

Skip if the organizers have already provisioned the cluster.

On **each node**, set the direct-cable IP:

```bash
sudo -E bash ~/Desktop/scripts/setup_node.sh    # node 0 -> 10.0.1.10, node 1 -> 10.0.1.11
```

On **node 0 only**, load the Docker images, wire SSH to node 1, copy the models,
and write the LiteLLM config and `team.env`:

```bash
cd ~/Desktop/setup-folder/scripts
sudo -E bash bootstrap_cluster.sh               # prompts for team id, both IPs, both usernames
```

**Verify before moving on:**

```bash
nvidia-smi                                      # GPU visible
ssh <node1-user>@10.0.1.11 nvidia-smi           # passwordless SSH works
curl http://localhost:8000/health               # Qwen vLLM answering
cat ~/team.env                                  # team/IP/model variables exist
```

---

## Phase 1 — Project onto node 0

```bash
cd ~
git clone <your-public-repo-url> agent && cd agent
# offline alternative: unzip the project snapshot from USB

source ~/team.env
pip install -r requirements.txt                 # most packages are already present
```

Point at the supplied datasets — confirm the real path first:

```bash
ls ~/Downloads/"Jasonl format DataSets"
export DATA_ROOT="$HOME/Downloads/Jasonl format DataSets"
```

`DATA_ROOT` must contain three subdirectories named `RBA Rates/`, `ASX/` and
`AFR/`. The supplied folders may be named `RBA-Rates-2010-2026`,
`ASX-18-companies-2015-2021-Jasonl` and `AFR Jasonl` — either symlink them into
those names or assemble a directory that uses them:

```bash
mkdir -p ~/data && cd ~/data
ln -sfn "$HOME/Downloads/Jasonl format DataSets/RBA-Rates-2010-2026" "RBA Rates"
ln -sfn "$HOME/Downloads/Jasonl format DataSets/ASX-18-companies-2015-2021-Jasonl" ASX
ln -sfn "$HOME/Downloads/Jasonl format DataSets/AFR Jasonl" AFR
export DATA_ROOT="$HOME/data"
cd ~/agent
```

Build the search index once, then prove the data layer:

```bash
python scripts/build_afr_index.py               # ~80 seconds
python -m pytest tests/test_golden.py -q        # must report: 24 passed
python scripts/smoke_test.py --skip-models
```

> **Stop here if the 24 golden tests do not pass.** They check the published
> reference values for every public calibration question. Every graded number
> flows through that layer, so a failure means `DATA_ROOT` is wrong or a dataset
> is incomplete — no amount of prompting recovers it later.

---

## Phase 2 — LiteLLM and the brain

```bash
litellm --config ~/litellm/config.yaml --port 4000 > ~/litellm.log 2>&1 &

curl http://localhost:4000/v1/models            # agent-brain and domain-ft listed
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"agent-brain","messages":[{"role":"user","content":"reply ok"}],"max_tokens":5}'
```

Confirm the brain's context limit. The bootstrap serves Qwen with
`--max-model-len 4096`, which the agent's prompt and tool-result budgets are
built around:

```bash
curl -s http://localhost:8000/v1/models | grep -o 'max_model_len[^,]*'
```

If it is larger, nothing breaks — the agent simply has more headroom.

---

## Phase 3 — Serve the agent

Start in bootstrap `mock` mode; Phase 5 switches it to the fine-tuned model.

```bash
export LITELLM_BASE_URL=http://localhost:4000/v1
export BRAIN_MODEL=agent-brain
export DOMAIN_FT_MODEL=domain-ft
export DOMAIN_PREDICT_MODE=mock

tmux new-session -d -s agent \
  "cd ~/agent && python -m uvicorn src.app:app --host 0.0.0.0 --port 5000"
```

**Verify:**

```bash
ip addr                                         # note node 0's LAN IP, not 10.0.1.10
curl http://<node0-lan-ip>:5000/health
python scripts/smoke_test.py --endpoint http://<node0-lan-ip>:5000
python scripts/eval_public.py                   # baseline with Qwen routing
```

Qwen is planning for real now, so the smoke test should report a path of
`plan>act>review>synthesize`. If it reports `fallback_planner`, the agent cannot
reach LiteLLM — fix that before continuing, or the brain is not being used.

---

## Phase 4 — Train the adapter (node 1)

Generate the training data on node 0, which has the datasets and a reachable
Qwen for sentiment labelling:

```bash
python training/label_sentiment.py --count 300
python training/make_dataset.py                 # ~2,000 examples including sentiment
```

Copy the project and generated data to node 1. Node 1 does **not** need the
785 MB dataset folder — training reads only `training/data/*.jsonl`:

```bash
rsync -av --exclude artifacts --exclude '.git' ~/agent/ <node1-user>@10.0.1.11:~/agent/
```

Train inside tmux. `earlyoom` kills long runs on a disconnected session, and
nothing is checkpointed before the first interval:

```bash
ssh <node1-user>@10.0.1.11
cd ~/agent && source ~/team.env

bash training/run_train.sh --quick --dry-run    # confirm resolved settings
tmux new-session -s train "bash training/run_train.sh --quick"
tmux attach -t train                            # watch progress
```

| Profile | Command | Steps | Wall time |
|---|---|---:|---|
| Quick | `bash training/run_train.sh --quick` | 25 | 30–45 min |
| Full | `bash training/run_train.sh` | 100 | 2–3 hours |

Run the quick profile first so an adapter is serving early and the whole
pipeline can be measured end to end. Start the full run afterwards if time
allows; the two write separate checkpoints, so report whichever measures better.

Serve a checkpoint:

```bash
cd ~/Cognitivo_Training/finagent-finetune
find "$MODELS_DIR/checkpoints" -type d -name hf_adapter
ADAPTER_CHECKPOINT=<path>/hf_adapter bash scripts/04_export_and_serve.sh
curl http://localhost:8001/v1/models            # on node 1
```

---

## Phase 5 — Wire in the fine-tuned model

On node 0, confirm the `domain-ft` alias points at node 1 and restart the proxy:

```bash
grep -A3 domain-ft ~/litellm/config.yaml        # api_base must be http://10.0.1.11:8001/v1
pkill -f litellm && litellm --config ~/litellm/config.yaml --port 4000 > ~/litellm.log 2>&1 &

curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"domain-ft","messages":[{"role":"user","content":"reply ok"}],"max_tokens":5}'
```

Flip the mode and restart the agent. **This is the step that decides whether the
fine-tuned model is actually used during evaluation:**

```bash
tmux kill-session -t agent
export DOMAIN_PREDICT_MODE=llm
tmux new-session -d -s agent \
  "cd ~/agent && DOMAIN_PREDICT_MODE=llm python -m uvicorn src.app:app --host 0.0.0.0 --port 5000"

curl http://<node0-lan-ip>:5000/health          # domain_predict_mode must read "llm"
```

Measure and record:

```bash
python training/eval_base_vs_ft.py --quick --base-model nemotron-base --ft-model domain-ft
python scripts/eval_public.py --endpoint http://<node0-lan-ip>:5000
python scripts/smoke_test.py --endpoint http://<node0-lan-ip>:5000
```

Fill the measured numbers into [training/MODEL_CARD.md](training/MODEL_CARD.md)
and the Results table in [README.md](README.md).

---

## Phase 6 — Submit

```bash
ip addr                                         # node 0 and node 1 LAN IPs
```

Edit `submission.json`:

| Field | Value |
|---|---|
| `team_id`, `team_name` | your team |
| `github_url` | your public repository |
| `agent.endpoint` | `http://<node0-lan-ip>:5000` |
| `model.endpoint` | `http://<node1-lan-ip>:8001/v1` |
| `model.model_name` | the name vLLM actually serves |
| `commit_sha` | filled in below |

```bash
git add -A && git commit -m "Final submission" && git push
git rev-parse HEAD                              # paste into commit_sha
git add submission.json && git commit -m "Pin commit SHA" && git push
```

**Test `/health` from a machine outside the cluster before the deadline.** A
failed health check skips the team entirely and scores zero on the 40%
hidden-question category.

Final checklist:

- [ ] `GET /health` returns 200 from an external machine
- [ ] `/health` reports `"domain_predict_mode":"llm"`
- [ ] `POST /query` returns non-empty `answer` for three concurrent requests
- [ ] `commit_sha` is the exact 40-character hash of the pushed commit
- [ ] repository is public and clones without credentials
- [ ] no credentials in any committed file or log

---

## What most often goes wrong

1. **`localhost` or `10.0.1.10` in `submission.json`.** Those addresses are the
   direct cable between the nodes; the harness runs off-cluster and cannot route
   to them. Use node 0's LAN IP, and bind uvicorn to `0.0.0.0`.
2. **`DOMAIN_PREDICT_MODE` left on `mock`.** The agent still answers, but it is
   not using the fine-tuned model, which forfeits model-quality and architecture
   credit. Startup logs a warning while it is set, and `/health` reports the
   current mode.
3. **Port ambiguity.** The submission template and execution guide specify
   `:5000`; the portal setup page says the harness connects to port `8001` on
   the head node. Port 8001 is free on node 0 — it belongs to node 1 — so serve
   both until an organizer confirms which is authoritative:
   ```bash
   tmux new-session -d -s agent8001 \
     "cd ~/agent && DOMAIN_PREDICT_MODE=llm python -m uvicorn src.app:app --host 0.0.0.0 --port 8001"
   ```
4. **Training outside tmux.** Nothing is checkpointed before the first interval,
   so a killed session means starting from scratch.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Golden tests fail | `DATA_ROOT` wrong or a dataset folder incomplete. Check the three subdirectory names. |
| Smoke test shows `fallback_planner` | The agent cannot reach LiteLLM. Check `LITELLM_BASE_URL`, that the proxy is running on 4000, and `~/litellm.log`. |
| `AFR index - missing` | Run `python scripts/build_afr_index.py`. The agent also builds it on first start, which delays readiness by ~80 s. |
| Answers contain `<think>` text | The synthesis system prompt lost its `detailed thinking off` prefix. Check `src/prompts.py`; the agent also strips residual blocks. |
| Responses exceed 60 s | Inspect `logs/agent-*.jsonl` for `brain_turns` and per-tool timings. Lower `MAX_AGENT_STEPS` or `AGENT_DEADLINE_S`. |
| `/health` 200 locally, refused externally | uvicorn bound to `127.0.0.1`, or a firewall. Bind `--host 0.0.0.0`. |
| Training run dies before a checkpoint | Not in tmux, or `MAX_SEQ_LEN` above 512 causing OOM. Keep 512, drop to 256 if needed. |
| Loss spikes after warmup | Learning rate above `5e-5`. The configs pin `5e-5` for this reason. |

Per-request diagnostics — question, latency, brain turns, tool calls, synthesis
mode, and node path — are written to `logs/agent-<date>.jsonl`.
