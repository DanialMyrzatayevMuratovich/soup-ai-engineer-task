# Soup take-home — LoRA → merge → 4-bit. Ship or not?

**Danial Myrzatayev** · 11.11.2026 · Qwen2.5-1.5B-Instruct · Russian · Colab T4 · Soup 0.77.2 (Python 3.13, torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2)
Raw evidence: `results/` (timestamped logs, raw `nvidia-smi` every 1 s, all JSON outputs, executed notebook). **Verdict: DON'T SHIP** (§5).

## 1. Setup
- **Data:** `d0rj/alpaca-cleaned-ru`@05321f66 (CC-BY-4.0, Russian, which I evaluate natively): 51,760 rows → filters (code blocks, low Cyrillic, too-short, exact dups) → 48,740 → **1,500 train / 200 held-out**, seed 42. Leakage audit: 0 eval prompts or answers in train. Caveat: machine-translated alpaca.
- **Training (Soup):** LoRA r 16, α 32, all 7 projections (18.46 M params, 1.2 %), lr 2e-4 cosine, 2 epochs, 188 steps, fp16 (T4 has no bf16), max_len 1,024, grad checkpointing. Then `soup merge` (fp16) and `soup merge --save-format 4bit` (bnb NF4, double-quant, fp16 compute). Independent check: `verify_trained.py` (transformers + peft only, not Soup).

## 2. Memory budget — estimate before training vs measured
| Stage | Estimate (GiB) | Measured torch alloc / reserved | nvidia-smi max | Gap (alloc) |
|---|---|---|---|---|
| Train, longest row T = 994, B = 2 | alloc 7.25 · smi 8.09 | 6.72 / **14.38** | **14.52** of 15.0 | −7 % |
| Merge (Soup merges on CPU) | VRAM ~0 · RAM 3.0–5.9 | 0.07 / 0.07 · **RSS 6.60** | 0.17 | RAM +12 % over range |
| 4-bit quantization | 1.50 (model 1.06) | 1.10 / 1.14 | 1.26 | −27 % |
| Soup's own `soup profile` | 7.57 total (logits 3.73, weights 3.09, activations 0.46, optimizer 0.15) | | | |

Estimate breakdown at T = 994: fp16 weights 2.88 · LoRA weights+grads+Adam fp32 0.28 · **logits backward 3.94** (14 B per vocab element × 2 × 994 × 151,936) · checkpointed activations + one-layer recompute ≈ 0.15 · CUDA context measured 0.11.
**Gap explained:** allocated memory matched within 7 %: the peak really is set by logits/loss, not by the model. The surprise is *reserved*: PyTorch's caching allocator kept 14.4 GiB because variable-length batches fragment the cache, so `nvidia-smi` showed the T4 almost full (14.52 GiB) while live tensors were 6.7 GiB. Any budget based on `nvidia-smi` would wrongly say "no headroom"; one based only on allocated memory would miss that a longer batch can OOM through fragmentation. Merge is a host-RAM problem on Colab (6.6 GiB of 12), not a VRAM one.

## 3. Proof that the model actually trained
**Why the loss curve is not proof:** train loss fell 1.37 → 0.99, but it is measured on training batches. Here it overstates learning: on held-out data the gain is only 0.084 nats, and the train–eval gap of the tuned model grew by **0.165 nats** vs the base — that part is memorisation. A loss curve also says nothing about whether the *saved → merged → quantized* artefact still carries the change.

**The check** (`verify_trained.py`) compares base, adapter, merged, 4-bit, base-4-bit and **two negative controls**: NC1 = same pipeline with lr ≈ 1e-9; NC2 = "merge silently produced the base".

| Artefact | Held-out answer NLL (95 % CI) | Δ vs matched base (paired bootstrap) | KL vs base | top-1 vs base |
|---|---|---|---|---|
| base fp16 | 1.303 [1.256, 1.355] | – | – | – |
| merged fp16 | 1.220 [1.173, 1.270] | **+0.084 [+0.076, +0.092]** | 0.125 | 0.863 |
| merged 4-bit | 1.283 [1.234, 1.336] | +0.074 vs base-4-bit (88 % kept); **only +0.021 vs base fp16** | 0.141 vs base-4-bit | 0.851 |
| NC1 (lr ≈ 0) | 1.303 | −0.000 [−0.00001, +0.00001] | 0.0000006 | 0.9999 |

Weights: all 196 LoRA B matrices moved off zero (median ‖B‖ 0.30; NC1 2.6e-7); median ‖ΔW‖/‖W‖ 0.82 %; merged == fp16(base + 2·B·A) exactly (max error 0.0), non-LoRA params bit-identical; adapter vs merged KL 1e-5; 62 % of 4-bit codes changed by training. **Verdicts:** main run *trained = YES*; NC1 flagged untrained (4/4 control checks FAIL); NC2 flagged (B1, B2, C1, D1, D2 FAIL, exit 2); 4-bit carries training = YES.
**What it does NOT detect:** whether answers are correct or helpful (NLL rewards matching machine-translated references); regressions outside this distribution; paraphrased contamination; sampled-decoding behaviour; a serving stack with another template; seed variance (thresholds are heuristics fixed before the run).

## 4. Silent failures (full table: `silent_failures.md`; injected-fault evidence: `results/outputs/fault_injection.md`)
| Failure that finishes without an error | How I checked | Caught by Soup? | Evidence |
|---|---|---|---|
| Empty assistant answers in data | injected 20/200 | **No** — `data validate` "200/200 valid", doctor OK | fault log |
| Eval rows leaked into train | injected 50 | **No** (validate, stats, doctor) | fault log |
| Answers in the wrong language | injected English answers | **No** | fault log |
| Wrong role names (`human`/`gpt`) | injected 10 % | validate **No**; doctor **Yes** (MAJOR, exit 2) | fault log |
| Truncated long answers | injected | doctor flags only **MINOR** (exit 0); dry-run accepts | fault log |
| Quantization silently on | omitted `quantization` | dry-run prints `Quant: 4bit` — **no warning** | fault log |
| Weak LoRA defaults | omitted r/α | dry-run uses r 64, α 16 (scale 0.25) — **no warning** | fault log |
| lr = 0 / typo'd config key | injected | **Yes** — refused | fault log |
| Model that did not learn | NC1 | not checked by Soup; my check catches it | §3 |
| Merge that returns the base | NC2 | not checked by Soup; my check catches it | §3 |
| Memorisation instead of learning | train vs held-out NLL gap | **No** | §3 (+0.165 nats) |
| 4-bit erases most of the gain vs the fp16 base | NLL base fp16 vs merged-4bit | **No** — `soup ship` compares base vs tuned, not quantized vs fp16 | §3 |
| Generation loops / no EOS | 12 greedy probes, rep-penalty 1.0 | **No** | EOS rate: base 50 %, merged 67 %, 4-bit 58 %; one 4-bit probe loops to the 200-token cap (rep-4 0.64) |
| Tokenizer file changed | hash of `tokenizer.json` per artefact | **No** | hash differs from base in every artefact; template and EOS/pad identical — root cause not yet verified |
| `soup ship` disagrees with itself | live vs offline evidence | — | live (bf16): **DON'T SHIP**, task 0.25 vs 0.25; offline (my NLL metric): **SHIP**, 0.272 vs 0.277; no noise floor in either |

## 5. Verdict: **DON'T SHIP** the 4-bit artefact
Training is real and the merge is correct, but nothing shows users would get a better model: (1) Soup's live task check shows no win (0.25 → 0.25) and general suites dropped (mini_instruction 1.00 → 0.96, mini_mmlu 0.85 → 0.825) — under the 5-point threshold, but on tiny suites with no noise floor, so a regression can't be ruled out; (2) a quarter of the held-out gain is memorisation (§3); (3) the shipped artefact is the 4-bit one, and it beats the fp16 base by only 0.021 nats; (4) decoding is unvalidated (loops on a greedy probe; only 12 probes).
**What must change first:** 1 epoch or early stopping on held-out NLL (or more data) until the train–eval gap stops growing; `soup ship --noise-floor 5` with a larger Russian task set and a judge or human review; evaluate the 4-bit model with the generation config users actually get (sampling, repetition_penalty 1.1) on ≥ 50 probes with EOS/loop checks; compare against 8-bit or AWQ/GPTQ to keep more of the gain; diff the `tokenizer.json` change. Ship only if the 4-bit model beats the fp16 base outside the noise floor and passes the decoding checks.

## 6. What surprised me / what still concerns me
<!-- Danial: rewrite in your own words before sending -->
I expected the loss curve to be the weak point; I did not expect the toolchain to disagree with itself. `soup ship` said DON'T SHIP live and SHIP offline on the same models, and neither run measured a noise floor, so both verdicts sit inside noise. I was also surprised that a GPU "full" in `nvidia-smi` held less than half of that in live tensors. What still concerns me: Soup's data checks passed empty answers, leaked eval rows and wrong-language answers, and nothing in the pipeline compares the quantized artefact with the fp16 base users would otherwise get.

## 7. AI usage
Claude Code (Anthropic) drafted the kit before the run — notebook, `verify_trained.py`, `memory_budget.py`, `kit_common.py`, `silent_failures.md`, report template — after reading the task PDF and the Soup repository, smoke-tested the scripts on CPU, ran the notebook in my Colab, and drafted this report from the run's files. No numbers were invented: every value above comes from `results/`. **What I verified myself:** <!-- Danial: fill honestly, e.g. read generations.md; checked that thresholds were fixed before the run and separate the main run from both controls; read the Soup fault logs; checked memory formulas against measured peaks -->
