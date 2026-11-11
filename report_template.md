# Soup take-home: LoRA → merge → 4-bit. Ship or not?

**Danial Myrzatayev** · `<<FILL: date>>` · Qwen2.5-1.5B-Instruct · Russian · Colab T4 · Soup `<<FILL FROM RUN: soup version (outputs/soup_capabilities.json)>>`

> Template rule: every number comes from the run's files. `outputs/report_numbers.md` collects them. Keep failed or skipped
> steps in the report: say what broke, what was tried, what was learned. Target ≤ 2 pages; cut prose before cutting evidence.

## 1. Setup and choices (≈6 lines)
- **Model** Qwen2.5-1.5B-Instruct (Apache-2.0, 1.54B params, tied 151,936-token embeddings, bf16 checkpoint). T4 has no bf16, so it trains in **fp16**.
- **Data** `d0rj/alpaca-cleaned-ru`@`<<FILL FROM RUN: dataset_revision>>`: `<<FILL FROM RUN: n_raw>>` rows → filters/dedupe → **1,500 train / 200 held-out** (seed 42), leakage audit `<<FILL FROM RUN: leakage_audit>>`. Why: clean licence and format, a language I evaluate natively. Caveat: machine-translated alpaca.
- **LoRA** r 16, α 32, all 7 projections (`<<FILL FROM RUN: n LoRA params>>` = 1.2 %), lr 2e-4 cosine, 2 epochs, 2 × 8 batch, max_len 1,024 (longer rows dropped, not truncated), grad checkpointing.
- **Stack** Soup `train → merge → merge --save-format 4bit (bnb NF4) → ship`; independent verification `verify_trained.py` (transformers + peft only). Fallbacks used: `<<FILL FROM RUN: none / which step and why>>`.

## 2. Memory budget: estimate (before training) vs measured
| Stage | Estimate (GiB) | Measured torch peak alloc / reserved | nvidia-smi max (− baseline) | Gap |
|---|---|---|---|---|
| Train (peak row T=`<<FILL>>`, B=2) | `<<FILL FROM RUN: est alloc / est smi>>` | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` | `<<FILL FROM RUN: %>>` |
| Merge (Soup, CPU) | VRAM `<<FILL>>`, RAM `<<FILL>>` | `<<FILL FROM RUN>>` (RSS `<<FILL>>`) | `<<FILL FROM RUN>>` | |
| 4-bit quantization | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` | |
| Soup's own estimate (`soup profile`) | `<<FILL FROM RUN: total>>` | | | |

Breakdown at the training peak (estimate): weights fp16 `<<FILL>>` · LoRA w+grad+Adam fp32 `<<FILL>>` · checkpointed inputs `<<FILL>>` · **logits+loss `<<FILL>>`** · one-layer recompute `<<FILL>>` · CUDA context (measured) `<<FILL FROM RUN: outputs/cuda_context.json>>`.
**Gap explained:** `<<FILL: 3–4 lines, e.g. which assumption was wrong. Was the longest padded batch shorter than 1,024? Is the loss more frugal (or wasteful) than 14 B per vocab element (Soup v0.77.2 charges 12 B)? Allocator cache = reserved − allocated; context = nvidia-smi − reserved.>>`

## 3. Proof that the model actually trained
**Why the loss curve isn't proof:** it is measured on training batches (memorisation looks identical). It can fall from predicting constant template tokens. It does not show that the *saved → merged → quantized* artefact still carries the change. In fp16 the GradScaler can skip steps while loss is still logged.
**The check** (`verify_trained.py`): (A) the adapter file read directly: ‖lora_B‖ (zero at init), 196/196 modules mapped, file == loaded. (B) merged == fp16(base + α/r·B·A), with non-LoRA params bit-identical. (C) held-out answer-token NLL with a paired bootstrap CI. (D) KL / top-1 vs base, vs the unmerged adapter, and vs the 4-bit model, with the base↔base-4-bit KL as the quantization noise floor. (Q) the 4-bit codes. (E) greedy generations. Two **negative controls**: lr ≈ 0 (NC1) and "merge produced the base" (NC2).

| Artefact | Held-out NLL (95 % CI) | Δ vs matched base | KL vs base | top-1 vs base |
|---|---|---|---|---|
| base fp16 | `<<FILL FROM RUN>>` | n/a | n/a | n/a |
| merged fp16 | `<<FILL FROM RUN>>` | `<<FILL FROM RUN: base−merged, CI>>` | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` |
| 4-bit artefact | `<<FILL FROM RUN>>` | `<<FILL FROM RUN: base4−merged4, CI; retained %>>` | `<<FILL>>` | `<<FILL>>` |
| NC1 (lr 1e-9) | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` | `<<FILL FROM RUN>>` |

Weights: median ‖ΔW‖/‖W‖ `<<FILL FROM RUN: B1>>`, merge error vs reference `<<FILL FROM RUN: B2>>`, fp16 rounding loss of the update `<<FILL FROM RUN: B4>>`; 4-bit codes changed by training `<<FILL FROM RUN: Q2>>`; KL(merged‖4-bit) `<<FILL FROM RUN: D3>>`.
**Verdicts:** main `<<FILL FROM RUN: model_trained>>` · NC1 flagged untrained `<<FILL FROM RUN>>` · NC2 (merged := base) `<<FILL FROM RUN>>` · 4-bit carries training `<<FILL FROM RUN>>`. Raw output: `outputs/verify/verify_summary.txt`.
**What it does NOT detect:** correctness/helpfulness (NLL rewards copying the translated references); regressions outside the held-out distribution; paraphrased contamination; sampling behaviour; serving runtimes with another template or system prompt; seed variance (thresholds are heuristics fixed before the run).

## 4. Silent failures (full table: `silent_failures.md`)
| Failure | How checked | Caught by Soup? (data doctor / pre-flight / ship) | Evidence |
|---|---|---|---|
| `<<FILL: top ~8 rows that actually mattered in this run>>` | | `<<FILL FROM RUN: from logs/cmd + outputs/fault_injection.md>>` | |

## 5. Verdict: `<<FILL: SHIP / DON'T SHIP>>`
`<<FILL: one call, backed by §2–4. If SHIP: under which conditions (artefact = 4-bit NF4 file, generation config, known limits). If DON'T SHIP: exactly what must change first and which check would then have to pass.>>`

## 6. What surprised me / what still concerns me
`<<FILL: one paragraph, in your own words, from what you saw in the run.>>`

## 7. AI usage
Claude Code (Anthropic) drafted the kit before the run: the notebook, `verify_trained.py`, `memory_budget.py`, `kit_common.py`,
`silent_failures.md` and this template, after reading the task PDF and the Soup repository. It invented no results: every number
above was filled from the run's files. The scripts were syntax-checked and smoke-tested on CPU with a tiny random Qwen2 model
(not on a T4). **What I verified or changed myself** (fill honestly):
- `<<FILL: checked the memory formulas (activation coefficients, 10/14 B per logit element) against the measured peaks>>`
- `<<FILL: checked that thresholds (A2 1e-3, B1 1e-4, C1 0.02 nats, D2 0.01) separate the main run from both controls, and were not tuned after seeing results>>`
- `<<FILL: read generations.md myself (Russian quality, EOS behaviour)>>`
- `<<FILL: read the Soup logs to decide what Soup's checks actually caught (not the keyword pointer)>>`
- `<<FILL: suggestions rejected or redone, and where the tool was unreliable>>`
