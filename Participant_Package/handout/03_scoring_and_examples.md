# Scoring, Examples, and What Good Looks Like

## Official score

The final hackathon score combines three independently assessed categories:

| Category | Weight |
|---|---:|
| Fine-tuned model quality | 30% |
| Architecture and repository quality | 30% |
| Hidden-question evaluation | 40% |

```text
final_score =
    (fine_tuned_model_score * 0.30)
  + (architecture_repository_score * 0.30)
  + (hidden_question_score * 0.40)
```

The examples below explain how the automated hidden-question category is graded. Fine-tuned model
quality and architecture/repository quality are assessed separately using the rubric in the
[Challenge Brief](../Challenge_Brief.md#scoring).

## Hidden-question scoring - 40%

Each question has a `grading` block with one or more **components**. Each component is a specific fact that must appear in the answer. Points are awarded per-component — **partial credit is possible and common**.

```json
{
  "grading": {
    "max_score": 10,
    "components": [
      { "component_id": "C01", "points": 5, "expected_fact": "BHP.AX was best at +22.17%" },
      { "component_id": "C02", "points": 5, "expected_fact": "AMP.AX was worst at -50.04%" }
    ],
    "tolerance_note": "Calculated returns allow +/-0.02 percentage points."
  }
}
```

The organizer's independent **LLM judge** (Qwen3.6-35B-A3B-FP8, accessed through the private `agent-brain` service,
not through your submitted agent) receives the question, your answer, and each expected fact one at
a time. It replies YES or NO. Equivalent formatting is accepted:
- `"1,234"` and `"1234"` are both accepted
- `"Jan 2024"` and `"2024-01"` are both accepted
- Minor rephrasing that preserves meaning is accepted
- Case-specific `grading.tolerance_note` rules are applied, including declared numeric tolerances

**What is not accepted:**
- A correct number buried in wrong context ("there are 41 records in total, 20 of which are holds")
- Hedging language that changes the answer ("approximately 41", "roughly 20")
- Refusal to answer or redirecting to a different question

## Slow-response penalty

If your agent takes longer than **60 seconds** to return a response, **20% of the earned points for that question are deducted after scoring**. This is separate from the timeout (300s) — your answer still arrives and is graded, but you lose 20% of whatever you earned.

| Response time | Effect |
|---|---|
| ≤ 60s | No penalty — full earned points |
| > 60s, ≤ 300s | 20% deduction on earned points for that question |
| > 300s (timeout) | Connection error — 0 points, same as no answer |

**Example:** Your answer earns 8/10, but took 83 seconds. Penalty = 8 × 0.20 = 1.6 pts deducted. Final = 6.4/10.

**Design implication:** Aim for ≤3 tool calls per question. If your agent loops more than 5 times or calls `list` on a large dataset, it will likely exceed 60 seconds on complex questions.

---

## Hidden-question score breakdown from real evaluation

All percentages in this section are automated hidden-question scores. They demonstrate model and
agent progress but are not the final weighted hackathon score.

### Our agent (alpha team) across training iterations

These are historical whole-system iterations, not a controlled model-only ablation. The current
required architecture uses Qwen3.6-35B-A3B-FP8 for planning and tool-call generation and fine-tuned Nemotron for
final synthesis.

| Run | Score | Tool use | Avg steps | Notes |
|---|---|---|---|---|
| First run — base model | 0.0% | 0% | 0 | No tool calls at all |
| After tool wiring | 0.0% | 100% | ~2 | Called tools but wrong metrics |
| After prompt tuning | 60.6% | 100% | ~2.5 | First meaningful results |
| After fine-tuning (step 20) | 74.5% | 100% | 3.59 | Best checkpoint |
| After fine-tuning (final) | 74.2% | 100% | 3.59 | Comparable to step 20 |

**Key takeaway:** the jump from 0% to about 74% came from the complete system: Qwen3.6-35B-A3B-FP8 routing,
deterministic tools, prompt improvements, and fine-tuned Nemotron synthesis. For the official 30%
model-quality score, teams must separately compare base Nemotron and fine-tuned Nemotron while
keeping the Qwen3.6-35B-A3B-FP8 routing and tool pipeline fixed.

### Best team-01 hidden-question benchmark run

| Difficulty | Score |
|---|---|
| Easy | 93.3% |
| Medium | 79.4% |
| Hard | 64.0% |
| **Overall** | **79.0%** |

Tool use: 100% | Avg steps: 4.65

---

## What zero looks like — and why

### Example 1: No tool use (base model, first run)

**Question:** From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?

**Answer (base model, 0% score):**
> "The user is asking for statistics about RBA cash rate decisions from the first record to the last. I need to find the RBA records in the team's corpora. I will use the `retrieve` tool to search for 'RBA cash rate decisions' or 'RBA board minutes' or similar. Then I will analyze the retrieved documents to count the total number of decisions that changed the rate, and how many were increases vs decreases…"

**Why it fails:**
- `retrieve` searches AFR news articles — it will never return structured RBA decision counts
- The answer never produces a number
- This is a broken planning loop: Qwen3.6-35B-A3B-FP8 should emit the required structured tool call before Nemotron is asked to synthesize an answer

### Example 2: Tool called but wrong metric

**Question:** What was the longest period the RBA held rates unchanged?

**Answer (early iteration, 0% score):**
> "Based on my search of the RBA data, the cash rate was held steady for an extended period during 2016-2019."

**Why it fails:**
- The agent didn't call `query_data(dataset="rba", metric="max_hold_streak")`
- It likely called `list` and tried to eyeball the output
- The answer gives no exact count of days, no start/end dates — zero components verified

### Example 3: Partial credit — right number, missing context

**Question:** What is the highest cash-rate target ever in the RBA dataset, and how many records show it?

**Answer (partial, 50% score):**
> "The highest cash-rate target in the RBA dataset is 4.75, which first appeared on the effective date 2010-11-03. This maximum rate appears across 11 effective-date records."

**Why it only gets 50%:** The grading had 3 components: the rate (4.75 ✓), the first effective date (2010-11-03 ✓), and the record count (11 ✓). But in the actual grading this run, C01 for "4.75" was awarded, C02 for "2010-11-03" was missing (the judge expected "2010-11-02"), and C03 for "11 records" was awarded. Date precision matters.

---

## What full marks looks like

### Easy question — 100% score

**Question:** What is the lowest cash-rate target in the RBA dataset, when did it first take effect, and how many decision records show that rate?

**Answer:**
> "The lowest cash-rate target in the RBA dataset was 0.1, which first took effect on 2020-11-04, and 16 decision records show that rate."

**Why it works:**
- One sentence, three facts, no hedging
- All three grading components satisfied: `0.1`, `2020-11-04`, `16`
- Concise — the judge doesn't need to find the answer buried in paragraphs

### Medium question — 100% score

**Question:** What was the longest stretch between two non-zero RBA rate changes?

**Answer:**
> "The longest stretch between two non-zero RBA rate changes was 1036 days, lasting from 2016-08-03 to 2019-06-05, during which the rate held at 1.5 before changing to 1.25."

**Why it works:**
- The agent called `query_data(dataset="rba", metric="max_hold_streak")` and got the exact result
- Every component is stated: duration (1036 days), start date (2016-08-03), end date (2019-06-05), rate during hold (1.5), rate after (1.25)
- No invented context, no hedging

### Hard question — 100% score

**Question:** How many hikes occurred during the 2022-2023 tightening cycle, and what was the cumulative increase?

**Answer:**
> "There were 13 hikes during the 4 May 2022 to 8 Nov 2023 tightening cycle, resulting in a cumulative increase of 4.25 percentage points. The target rate immediately before the first hike was 0.1 percent, and the final target reached on 8 Nov 2023 was 4.35 percent."

---

## Good architecture vs bad architecture

### Bad: single model, RAG-only

```
question → retrieve() → base LLM → answer
```

- Fails on any numerical question that requires exact counts
- AFR article search returns prose — you can't count rate decisions from news headlines
- The base model hallucinates numbers rather than admitting it needs structured data

### Bad: Nemotron used as the planner and tool caller

```
question → Nemotron → tool calls → Nemotron → answer
```

- It mixes the participant fine-tuning target with the orchestration role supplied by the organizers
- Tool selection becomes dependent on the Nemotron adapter instead of the stable Qwen `agent-brain`
- It does not follow the required separation between Qwen planning and Nemotron synthesis

### Bad: Qwen3.6-35B-A3B-FP8 routing with base-only synthesis

```
question → Qwen3.6-35B-A3B-FP8 → tools → base Nemotron → answer
```

- Qwen may select the correct tools, but the base Nemotron can still omit requested facts or misstate units
- Structured results may be copied incompletely or surrounded by unsupported financial claims
- This provides the correct control case for measuring the value of the fine-tuned Nemotron adapter

### Good: two-model agentic loop

```
question
    │
    ▼
brain (Qwen3.6-35B-A3B-FP8 agent-brain)  ← owns planning, tool selection, and tool-call generation
    │ tool_calls
    ▼
query_data / retrieve      ← exact structured results, no hallucination possible
    │ results
    ▼
brain loops until no more tool_calls needed
    │
    ▼
domain model (Nemotron-8B fine-tuned)   ← synthesises answer in correct format
    │
    ▼
clean answer
```

Why this works:
- The brain is big enough to reason across multiple tool results
- The application runtime validates and executes Qwen's requested calls against approved datasets
- The domain model was fine-tuned on the tool output format, so it knows how to read verified JSON results and write clean answers
- Deterministic tool use anchors dataset-derived facts; the hidden-question judge still scores the answer content, while the architecture review assesses how well the tools are designed and used

### Recommended implementation checklist

- [ ] `BRAIN_MODEL=agent-brain` sends planning and tool-call generation to the supplied Qwen3.6-35B-A3B-FP8 model
- [ ] The application runtime, not either model, validates and executes tool calls
- [ ] `DOMAIN_FT_MODEL` sends final synthesis to the team's fine-tuned Nemotron model
- [ ] `DOMAIN_PREDICT_MODE=llm` is set after the adapter is live; bootstrap `mock` mode is disabled
- [ ] The agent and model servers handle three concurrent `/query` requests without shared-state errors
- [ ] Dataset-derived numerical answers use at least one `query_data()` call — `retrieve()` alone does not produce exact calculations
- [ ] Hard questions may need 3-5 tool calls: look up the range, get the specific metric, verify the result
- [ ] AFR counts require `pattern=` with word boundaries (`\bterm\b`) — bare substring matches over-count
- [ ] The answer should state all grading components explicitly, not bury them in caveats
- [ ] For RBA date questions: use `lookup_rate(date_from=<date>)` to get the rate in effect ON or BEFORE a date, not the nearest date which could be in the future
- [ ] For ASX questions excluding Tabcorp: always pass `exclude_tickers=["TAH.AX"]`

---

## Hidden-question formula

```
hidden_question_score = sum(earned_points) / sum(max_points) x 100%
                        (after any slow-response penalty is applied)
```

The **public leaderboard shows only Rank, Team, and the weighted final Score**. Within the
hidden-question category, component-level fact verification determines earned points; tool-use and
step-count fields are private diagnostics and do not directly earn points.

After the run, each team receives a **private detailed report on their own agent only** — overall metrics (score, tool rate, avg/P95 latency, availability, slow penalty) plus a per-question breakdown with component YES/NO verdicts. Hidden grading facts are stripped from the report, and no other team's data is included. You will not see other teams' metrics, and they will not see yours.

**Availability is a gate.** If `GET /health` does not return 200 at the start of the run, the team is skipped entirely and no questions are graded. Test your endpoint from a different machine before submitting.

**Latency still affects your score indirectly** through the 60-second slow-response penalty — even though the leaderboard doesn't display it. Keep responses fast to avoid the 20% deduction.
