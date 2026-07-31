# Fine-Tuning Guide — Timings, Hardware, and Configuration

## Hardware available

Each team receives a two-node GIGABYTE Atom cluster. The organizers assign the actual hostnames and
IP addresses, so this guide refers to nodes by role rather than by machine name.

| Cluster role | Accelerator | Memory | Suggested use |
|---|---|---|---|
| **Brain/agent node** | 1× NVIDIA GB10 | 128 GB unified | Qwen3.6-35B-A3B-FP8 `agent-brain` serving + agent runtime |
| **Fine-tuning/model node** | 1× NVIDIA GB10 | 128 GB unified | Nemotron fine-tuning and fine-tuned model serving |

The required responsibility split during evaluation is: **the brain/agent node serves the supplied
Qwen3.6-35B-A3B-FP8 reasoning and tool-calling model through `agent-brain`; the fine-tuning/model node serves
your fine-tuned Nemotron model for final answer synthesis**. Participants fine-tune Nemotron, not
Qwen3.6-35B-A3B-FP8. The application runtime executes the tool calls that Qwen3.6-35B-A3B-FP8 requests.

---

## Expected training times

| Script | Model | Steps | Hardware | Approx wall time |
|---|---|---|---|---|
| `02_smoke_test.sh` | Llama-3.2-1B | 50 | 1 node | ~30 seconds |
| `02_smoke_qwen35.sh` | Supplied Qwen3.6-35B-A3B-FP8 `agent-brain` | 20 | 1 node | ~5–10 min (model load ~3 min) |
| `07_train_8b_quicktest.sh` | Nemotron-8B | 100 | 1 GB10 node | **~2–3 hours** |
| `03_train_1node.sh` | Nemotron-8B | 100 | 1 node | **~2–3 hours** |
| `03_train_2node.sh` | Nemotron-8B | 100–500 | 2 nodes | faster per step; startup overhead ~5 min |

> **Key insight:** The step-20 checkpoint already shows meaningful improvement over the base model. You don't need to wait for 100 steps to start evaluating. Save time by testing at step 20 and deciding whether to continue.

---

## Confirmed baseline configuration

These values are a working starting point, not a requirement. Experiment where time allows.

```env
MODEL_PATH   = Llama-3.1-Nemotron-Nano-8B-v1
MAX_STEPS    = 100
BATCH_SIZE   = 2
GRAD_ACCUM   = 4          # effective batch = 8
LORA_RANK    = 32
LR           = 5e-5       # do NOT use 1e-4 — causes loss spike at warmup step 50
MAX_SEQ_LEN  = 512        # longer sequences OOM on single node
WARMUP_STEPS = 50
NEMO_IMAGE   = nvcr.io/nvidia/nemo:25.09   # must be 25.09+; 25.04 crashes on GB10
CHECKPOINT_EVERY = 20
```

**Confirmed result with baseline config:** +110% composite improvement vs base model on 50 test samples. Best checkpoint: step 20, val loss 0.098.

---

## Step-by-step training workflow

Run the supplied training commands from the fine-tuning workspace:

```bash
cd ~/Cognitivo_Training/finagent-finetune
source ~/team.env
```

### 1. Validate the pipeline first (~30 sec)
Always run the smoke test before starting a full training run. It confirms the NeMo container, GPU, data paths, and checkpoint saving are all working.

```bash
bash scripts/02_smoke_test.sh
```

### 2. Prepare your training data
```bash
python scripts/01_prepare_data.py \
  --afr_dir  "/home/cognitivo/Downloads/Jasonl format DataSets/AFR Jasonl" \
  --asx_dir  "/home/cognitivo/Downloads/Jasonl format DataSets/ASX-18-companies-2015-2021-Jasonl" \
  --rba_file "/home/cognitivo/Downloads/Jasonl format DataSets/RBA-Rates-2010-2026/RBA-rates.jsonl" \
  --out_dir  data/
```

Dataset sizes after preparation:
- `data/train.jsonl` — 48,000 samples
- `data/val.jsonl` — 6,000 samples
- `data/test.jsonl` — 6,000 samples
- `data/smoke/` — 500-sample subset for pipeline validation

### 3. Launch training inside tmux (so it survives disconnects)

**On the fine-tuning/model node (recommended — keeps the other node free for brain serving):**
```bash
# Run on the node assigned for fine-tuning:
tmux new-session -s train8b "bash scripts/07_train_8b_quicktest.sh"

# Monitor on that node or connect using its assigned hostname/IP:
tail -f /tmp/nemo_8b_test.log
```

**On either available cluster node:**
```bash
tmux new-session -s finetune "bash scripts/03_train_1node.sh"
tail -f /tmp/nemo_1node.log
```

### 4. Export and serve your adapter
```bash
cd ~/Cognitivo_Training/finagent-finetune
source ~/team.env

# Find the host-side adapter path produced by your selected run.
find "$MODELS_DIR/checkpoints" -type d -name hf_adapter

ADAPTER_CHECKPOINT="$MODELS_DIR/checkpoints/<your-run>/checkpoints/<checkpoint>/hf_adapter" \
bash scripts/04_export_and_serve.sh
```

This starts vLLM on port 8001 of the fine-tuning/model node with your LoRA adapter loaded. No weight merge required — vLLM loads the adapter at runtime.

### 5. Point your agent at the fine-tuned model
Keep `BRAIN_MODEL=agent-brain` so Qwen3.6-35B-A3B-FP8 performs planning and emits tool calls. Update
`DOMAIN_FT_MODEL=domain-ft` and configure that LiteLLM alias to point at port 8001 on your assigned
fine-tuning/model node. The runtime executes Qwen3.6-35B-A3B-FP8's requested calls, then sends the accumulated
verified results to Nemotron for final synthesis.

The bootstrap sets `DOMAIN_PREDICT_MODE=mock` for pre-training plumbing checks. Once the adapter is
served, export `DOMAIN_PREDICT_MODE=llm` before starting the submitted agent so the fine-tuned model
is actually used during evaluation.

---

## Known issues and workarounds

| Issue | Workaround |
|---|---|
| `nemo:25.04` crashes on GB10 | Always use `nvcr.io/nvidia/nemo:25.09` |
| `LR=1e-4` → loss spike at step 50 | Use `LR=5e-5` |
| `MAX_SEQ_LEN > 512` OOM on single node | Stick to 512; drop to 256 if still OOMing |
| Training killed by earlyoom | Always run inside `tmux` |
| No checkpoint before step 20 | If it crashes before step 20, restart from scratch — nothing is saved |
| 2-node `MASTER_ADDR` | Use the IP address assigned to the primary training node; do not assume a hostname or fixed IP |

---

## Time planning recommendation

| Phase | Time budget |
|---|---|
| Smoke test + data prep | 15–30 min |
| First training run (100 steps) | 2–3 hours |
| Checkpoint eval + iteration | 30–60 min per cycle |
| Final export + serve + agent integration | 30 min |

Plan for at least one full 3-hour training window. The step-20 checkpoint is worth evaluating early — if it's already performing well, you can spend remaining time on agent improvements instead of training longer.
