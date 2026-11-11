#!/usr/bin/env python3
"""verify_trained.py: did the LoRA fine-tune really change the model, and did the
change survive the merge and the 4-bit quantization?

Standalone CLI. Needs torch, transformers, peft, safetensors, numpy, huggingface_hub;
bitsandbytes (CUDA) for the 4-bit parts. Nothing here trusts the training loss.

It evaluates up to six "views" of the model on the SAME held-out rows with the SAME
chat template, in lockstep (one example at a time, every view):

  base         base model (fp16)                    PeftModel with adapters disabled
  adapter      base + LoRA adapter, NOT merged      PeftModel, adapter "main"
  control      base + negative-control adapter      PeftModel, adapter "control" (optional)
  merged       merged fp16 checkpoint               the artifact produced by the merge step
  merged_4bit  4-bit artifact (bitsandbytes NF4)    optional
  base_4bit    base quantized with the SAME 4-bit config (matched control for merged_4bit)

Check groups (explicit thresholds in THRESHOLDS):
  A  adapter file: lora_B norms (B is zero-initialised, so an untrained adapter has B ~ 0),
     number of LoRA modules vs expected, every module maps onto a base weight,
     file tensors == tensors PEFT actually loaded, predicted fp16 rounding loss of the update
  B  weights: merged == fp16(base + scale * B @ A) (independent reference computed here),
     non-LoRA parameters bit-identical, relative weight delta per module type
  C  held-out answer-token NLL / perplexity, paired bootstrap 95% CI
  D  next-token KL divergence + top-1 agreement on answer positions (teacher forcing)
  E  greedy generations side by side (EOS reached? Russian? repetition?), non-finite logits
  Q  4-bit code level: artifact == quantize(merged) and != quantize(base)
  T  tokenizer / chat template / generation_config consistency across artifacts

Outputs in --out-dir: verify_results.json (machine-readable), verify_summary.txt (readable),
per_example_nll.csv, generations.jsonl, generations.md, verify.log.

Example:
  python verify_trained.py --base Qwen/Qwen2.5-1.5B-Instruct \
      --adapter outputs/soup_run --merged outputs/merged --quant4 outputs/merged_4bit \
      --control-adapter outputs/nc_run --control-merged outputs/nc_merged \
      --eval-data data/eval.jsonl --train-data data/train.jsonl \
      --prompts data/probe_prompts.jsonl --out-dir outputs/verify
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Thresholds. Fixed BEFORE the run. They are heuristics, not calibrated across seeds;
# the negative control is what shows they separate "trained" from "not trained".
# --------------------------------------------------------------------------------------
THRESHOLDS = {
    # A: adapter file
    "A1_expected_modules": "num_hidden_layers x len(target_modules)",
    "A2_min_lora_B_fro": 1e-3,            # per-module ||B||_F. B starts at exactly 0.
    "A2_min_frac_modules_nonzero": 0.90,  # share of modules whose ||B||_F exceeds the above
    "A3_max_abs_diff_file_vs_loaded": 1e-6,
    # B: weights of the merged artifact
    "B1_min_median_rel_delta": 1e-4,      # median_module ||W_merged - W_base||_F / ||W_base||_F
    "B2_max_err_vs_reference": 0.05,      # ||W_merged - W_ref||_F / ||W_ref - W_base||_F
    "B3_max_abs_nontarget_diff": 0.0,     # non-LoRA params must be bit-identical
    "B4_warn_fp16_rounding_loss": 0.30,   # ||fp16(W+dW) - W - dW|| / ||dW||  (WARN only)
    # C: held-out NLL (nats per answer token)
    "C1_min_nll_gain": 0.02,              # NLL(base) - NLL(merged) >= this AND CI low > 0
    "C2_min_retained_gain_frac": 0.50,    # (NLL base4 - NLL merged4) / (NLL base - NLL merged)
    "C2_warn_quant_nll_increase": 0.10,   # NLL(merged4) - NLL(merged)  (WARN only)
    # D: distributions on answer positions (teacher forcing)
    "D1_max_kl_adapter_vs_merged": 1e-3,
    "D1_min_top1_adapter_vs_merged": 0.99,
    "D2_min_kl_merged_vs_base": 0.01,
    "D3_warn_max_kl_merged_vs_4bit": 0.10,
    "D3_warn_min_top1_merged_vs_4bit": 0.85,
    # E: generations
    "E1_min_eos_rate": 0.90,
    "E1_min_cyrillic_ratio": 0.60,
    "E1_max_rep4": 0.30,
    "E2_max_nonfinite_logits": 0,
    # Q: 4-bit codes
    "Q1_max_mismatch_vs_requant_merged": 1e-6,   # fraction of weights
    "Q2_min_frac_4bit_weights_changed": 1e-4,    # quantize(merged) != quantize(base)
}

NOT_DETECTED = [
    "Whether answers are correct, factual or helpful. NLL rewards the likelihood of the "
    "reference answers, which are machine-translated model outputs: a gain can come from "
    "copying their style/translation artefacts, not from substance.",
    "Regressions outside the held-out distribution (other tasks, other languages, safety, "
    "refusals, long context, multi-turn). Held-out rows come from the same dataset as train.",
    "Paraphrased train/eval contamination: the leakage check in data prep is lexical only.",
    "Behaviour under sampling (temperature/top-p) or other decoding settings: everything here "
    "is greedy or teacher-forced with repetition_penalty=1.0.",
    "A deployment that uses a different runtime, chat template or system prompt (GGUF/Ollama/"
    "vLLM). Train and eval share Qwen's default system prompt, so a template error made "
    "consistently in both places cancels out and is invisible here.",
    "Seed-to-seed variance: thresholds are fixed heuristics, not calibrated over several runs.",
    "Small harmful changes on rare inputs; the probe set is ~12 prompts.",
    "The 4-bit checks cover the bitsandbytes NF4 artifact only, not GGUF/AWQ/GPTQ exports.",
]

LORA_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FP16_MAX = 65504.0

_LOG_FH = None


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    line = f"{now()} [verify] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sha256_text(s) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def to_py(o):
    """Make numpy / torch scalars JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else str(v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


# --------------------------------------------------------------------------------------
# Chat formatting: ONE function used for eval here and (imported) by the kit's data prep
# and fallback trainer, so length stats, training labels and evaluation agree.
# --------------------------------------------------------------------------------------
class TemplateError(ValueError):
    pass


def render_prompt(tokenizer, messages_without_answer) -> str:
    return tokenizer.apply_chat_template(
        messages_without_answer, tokenize=False, add_generation_prompt=True
    )


def encode_example(tokenizer, messages, max_length=None):
    """Tokenise a chat whose LAST message is the assistant answer.

    Returns dict(input_ids, answer_idx, n_prompt_tokens, n_total, content_exact,
    prefix_tokens_match). answer_idx are positions of the answer tokens: the answer
    content plus the end-of-turn token (<|im_end|> for Qwen). Template header tokens and
    the newline after <|im_end|> are excluded. Positions are found by character offsets
    in ONE tokenisation of the full string, so BPE merges across the prompt/answer
    boundary cannot shift them.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise TemplateError("last message must be the assistant answer")
    answer = messages[-1]["content"]
    prompt = render_prompt(tokenizer, messages[:-1])
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise TemplateError("rendered conversation does not start with the rendered prompt")
    eos = tokenizer.eos_token or ""
    a0 = len(prompt)
    content_exact = full[a0:a0 + len(answer)] == answer
    if content_exact and eos and full[a0 + len(answer):].startswith(eos):
        a1 = a0 + len(answer) + len(eos)
    else:
        j = full.find(eos, a0) if eos else -1
        if j < 0:
            raise TemplateError("cannot locate the end-of-turn token after the answer")
        a1 = j + len(eos)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = enc["offset_mapping"]
    answer_idx = [i for i, (s, e) in enumerate(offs) if e > s and s >= a0 and e <= a1]
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    n_prompt = sum(1 for (s, e) in offs if e <= a0)
    out = {
        "input_ids": ids,
        "answer_idx": answer_idx,
        "n_prompt_tokens": n_prompt,
        "n_total": len(ids),
        "content_exact": bool(content_exact),
        "prefix_tokens_match": list(p_ids) == ids[: len(p_ids)],
        "truncated": False,
    }
    if max_length is not None and len(ids) > max_length:
        out["truncated"] = True
    return out


# --------------------------------------------------------------------------------------
# transformers v4/v5 compatibility helpers
# --------------------------------------------------------------------------------------
def dtype_kw(torch_dtype):
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": torch_dtype} if (major, minor) >= (4, 56) else {"torch_dtype": torch_dtype}


def resolve_local_dir(path_or_id: str) -> Path:
    p = Path(path_or_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            path_or_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
        )
    )


def safetensor_index(model_dir: Path) -> dict:
    """tensor name -> file path, for every *.safetensors in a checkpoint dir."""
    from safetensors import safe_open

    idx = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                idx[k] = f
    return idx


def read_json(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def module_type(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def lora_scale(adapter_cfg: dict) -> float:
    r = adapter_cfg.get("r")
    alpha = adapter_cfg.get("lora_alpha")
    if adapter_cfg.get("use_rslora"):
        return float(alpha) / math.sqrt(float(r))
    return float(alpha) / float(r)


# --------------------------------------------------------------------------------------
# A: static adapter checks (CPU, no PEFT involved: reads the safetensors file directly)
# --------------------------------------------------------------------------------------
def adapter_file_tensors(adapter_dir: Path) -> tuple[dict, dict]:
    from safetensors.torch import load_file

    f = adapter_dir / "adapter_model.safetensors"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found (pickled adapter_model.bin is not supported)")
    sd = load_file(str(f))
    mods = {}
    for k, v in sd.items():
        m = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", k)
        if not m:
            continue
        path = k[: m.start()]
        if path.startswith("base_model.model."):
            path = path[len("base_model.model."):]
        mods.setdefault(path, {})[m.group(1)] = v
    other = [k for k in sd if ".lora_A" not in k and ".lora_B" not in k]
    return mods, {"n_tensors": len(sd), "non_lora_tensors": other[:20], "n_non_lora": len(other)}


def check_adapter_static(adapter_dir, base_dir, label, base_cfg):
    import torch
    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    acfg = read_json(adapter_dir / "adapter_config.json") or {}
    scale = lora_scale(acfg) if acfg.get("r") else float("nan")
    mods, extra = adapter_file_tensors(adapter_dir)
    base_idx = safetensor_index(base_dir)
    L = int(base_cfg.get("num_hidden_layers", 0))
    tm = acfg.get("target_modules")
    tm_list = sorted(tm) if isinstance(tm, (list, tuple)) else ([tm] if tm else [])
    expected = L * len([t for t in tm_list if t in LORA_TYPES]) if tm_list else None
    rows, unmapped = [], []
    handles = {}
    t0 = time.time()
    for path in sorted(mods):
        A = mods[path].get("A")
        B = mods[path].get("B")
        if A is None or B is None:
            unmapped.append(path + " (missing A or B)")
            continue
        wkey = path + ".weight"
        rec = {
            "module": path,
            "type": module_type(path),
            "A_fro": float(A.float().norm()),
            "B_fro": float(B.float().norm()),
            "B_absmax": float(B.float().abs().max()),
            "B_exact_zero": bool(B.abs().max().item() == 0.0),
        }
        if wkey not in base_idx:
            unmapped.append(path)
        else:
            fpath = base_idx[wkey]
            if fpath not in handles:
                handles[fpath] = safe_open(str(fpath), framework="pt")
            W = handles[fpath].get_tensor(wkey)
            W16 = W.to(torch.float16).float()
            dW = scale * (B.float() @ A.float())
            w_n = float(W16.norm())
            d_n = float(dW.norm())
            rec["W_fro"] = w_n
            rec["dW_fro"] = d_n
            rec["rel_dW"] = d_n / w_n if w_n > 0 else float("nan")
            ideal16 = (W16 + dW).to(torch.float16).float()
            got = ideal16 - W16
            rec["fp16_rounding_loss"] = float((got - dW).norm()) / d_n if d_n > 0 else float("nan")
            rec["fp16_frac_elements_unchanged"] = float((ideal16 == W16).float().mean())
        rows.append(rec)
    handles.clear()

    def agg(key, sel=None):
        vals = [r[key] for r in rows if key in r and (sel is None or r["type"] == sel)]
        vals = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
        if not vals:
            return None
        return {"min": min(vals), "median": float(np.median(vals)), "max": max(vals), "n": len(vals)}

    per_type = {
        t: {k: agg(k, t) for k in ("B_fro", "rel_dW", "fp16_rounding_loss", "fp16_frac_elements_unchanged")}
        for t in sorted({r["type"] for r in rows})
    }
    thr = THRESHOLDS["A2_min_lora_B_fro"]
    frac_nonzero = (sum(1 for r in rows if r["B_fro"] > thr) / len(rows)) if rows else 0.0
    out = {
        "label": label,
        "adapter_dir": str(adapter_dir),
        "adapter_config": {k: acfg.get(k) for k in (
            "base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "target_modules",
            "use_rslora", "use_dora", "modules_to_save", "bias", "task_type", "rank_pattern",
            "alpha_pattern")},
        "scale": scale,
        "n_lora_modules": len(rows),
        "expected_modules": expected,
        "unmapped_modules": unmapped[:20],
        "n_unmapped": len(unmapped),
        "n_B_exact_zero": sum(1 for r in rows if r["B_exact_zero"]),
        "frac_modules_B_above_thr": frac_nonzero,
        "B_fro": agg("B_fro"),
        "rel_dW": agg("rel_dW"),
        "fp16_rounding_loss": agg("fp16_rounding_loss"),
        "fp16_frac_elements_unchanged": agg("fp16_frac_elements_unchanged"),
        "per_type": per_type,
        "file_info": extra,
        "seconds": round(time.time() - t0, 1),
    }
    return out, rows, mods


# --------------------------------------------------------------------------------------
# GPU / model helpers
# --------------------------------------------------------------------------------------
def cuda_mem(torch):
    if not torch.cuda.is_available():
        return {}
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
    }


def free(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class View:
    """One way of getting logits: a plain model, or a PeftModel with an adapter on/off."""

    def __init__(self, name, model, kind="plain", adapter=None):
        self.name, self.model, self.kind, self.adapter = name, model, kind, adapter

    def _ctx(self):
        import contextlib

        if self.kind == "peft_off":
            return self.model.disable_adapter()
        if self.kind == "peft_on":
            self.model.set_adapter(self.adapter)
        return contextlib.nullcontext()

    def forward(self, **kw):
        with self._ctx():
            return self.model(**kw)

    def generate(self, **kw):
        with self._ctx():
            return self.model.generate(**kw)


def kl_rows(logp_p, logp_q):
    """Per-position KL(p || q) from log-probabilities, safe for -inf entries."""
    import torch

    p = logp_p.exp()
    diff = torch.where(p > 0, logp_p - logp_q, torch.zeros_like(logp_p))
    return (p * diff).sum(-1)


def bootstrap_ci(per_ex_nll, per_ex_n, other=None, B=2000, seed=0):
    nll = np.asarray(per_ex_nll, dtype=np.float64)
    n = np.asarray(per_ex_n, dtype=np.float64)
    if len(n) == 0 or not np.all(np.isfinite(nll)):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(B, len(n)))
    tot = n[idx].sum(1)
    if other is None:
        stat = nll[idx].sum(1) / tot
        point = nll.sum() / n.sum()
    else:
        o = np.asarray(other, dtype=np.float64)
        if not np.all(np.isfinite(o)):
            return None
        stat = (nll[idx].sum(1) - o[idx].sum(1)) / tot
        point = (nll.sum() - o.sum()) / n.sum()
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "B": B}


def cyr_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)


def rep4(ids) -> float:
    if len(ids) < 8:
        return 0.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model HF id or local dir")
    ap.add_argument("--adapter", required=True, help="trained LoRA adapter dir")
    ap.add_argument("--merged", required=True, help="merged fp16 checkpoint (dir or HF id)")
    ap.add_argument("--quant4", default=None, help="4-bit bitsandbytes checkpoint dir")
    ap.add_argument("--control-adapter", default=None, help="negative-control adapter dir")
    ap.add_argument("--control-merged", default=None, help="negative-control merged dir (weight checks)")
    ap.add_argument("--eval-data", required=True, help="held-out JSONL with 'messages'")
    ap.add_argument("--train-data", default=None, help="train JSONL (generalisation gap)")
    ap.add_argument("--prompts", default=None, help="probe prompts JSONL (id, prompt, kind, reference)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--kl-max-positions", type=int, default=256, help="answer positions per example used for KL")
    ap.add_argument("--hidden-scan-examples", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--gen-batch-size", type=int, default=6)
    ap.add_argument("--skip-generations", action="store_true")
    ap.add_argument("--skip-base4bit", action="store_true", help="save ~1 GiB VRAM; loses the matched 4-bit control")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"],
                    help="auto = float16 on CUDA, float32 on CPU")
    return ap.parse_args(argv)


def main(argv=None):
    global _LOG_FH
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(out_dir / "verify.log", "a", encoding="utf-8")
    t_start = time.time()
    log(f"START argv={sys.argv[1:] if argv is None else argv}")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import peft
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover
        log(f"FATAL: peft import failed: {exc!r}")
        raise

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if dev != args.device:
        log("WARNING: CUDA not available, falling back to CPU (4-bit checks will be skipped)")
    if args.dtype == "auto":
        fdtype = torch.float16 if dev.startswith("cuda") else torch.float32
    else:
        fdtype = getattr(torch, args.dtype)
    device_map = {"": 0} if dev.startswith("cuda") else {"": "cpu"}
    torch.manual_seed(args.seed)

    R = {
        "started": now(),
        "args": vars(args),
        "thresholds": THRESHOLDS,
        "not_detected": NOT_DETECTED,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu",
        },
        "memory": {},
        "timing_s": {},
        "checks": [],
    }
    try:
        import bitsandbytes as bnb  # noqa: F401

        R["versions"]["bitsandbytes"] = bnb.__version__
        have_bnb = dev.startswith("cuda")
    except Exception as exc:
        R["versions"]["bitsandbytes"] = f"unavailable: {exc!r}"[:200]
        have_bnb = False

    checks = R["checks"]

    def add(cid, name, status, value, threshold, meaning):
        checks.append({"id": cid, "name": name, "status": status, "value": to_py(value),
                       "threshold": threshold, "meaning": meaning})
        log(f"CHECK {cid:<10} {status:<5} {name}: value={value} threshold={threshold}")

    def phase_mem(name):
        R["memory"][name] = cuda_mem(torch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------------- static (A)
    t0 = time.time()
    base_dir = resolve_local_dir(args.base)
    base_cfg = read_json(base_dir / "config.json") or {}
    R["base_config"] = {k: base_cfg.get(k) for k in (
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "vocab_size", "tie_word_embeddings", "torch_dtype", "dtype")}
    A_main, A_rows, file_mods = check_adapter_static(args.adapter, base_dir, "main", base_cfg)
    R["adapter_main"] = A_main
    with open(out_dir / "adapter_modules_main.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in A_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(A_rows)
    A_ctrl = None
    ctrl_file_mods = None
    if args.control_adapter:
        A_ctrl, A_ctrl_rows, ctrl_file_mods = check_adapter_static(args.control_adapter, base_dir, "control", base_cfg)
        R["adapter_control"] = A_ctrl
    R["timing_s"]["static_adapter"] = round(time.time() - t0, 1)

    def a_checks(A, suffix):
        exp = A["expected_modules"]
        ok1 = (exp is None or A["n_lora_modules"] == exp) and A["n_unmapped"] == 0 and A["n_lora_modules"] > 0
        add(f"A1{suffix}", "LoRA modules present and mapped onto base weights", "PASS" if ok1 else "FAIL",
            {"n": A["n_lora_modules"], "expected": exp, "unmapped": A["n_unmapped"]},
            "n == expected, unmapped == 0",
            "Catches target_modules that matched nothing/too little, or an adapter for a different base.")
        fr = A["frac_modules_B_above_thr"]
        add(f"A2{suffix}", "lora_B moved away from its zero init", "PASS" if fr >= THRESHOLDS["A2_min_frac_modules_nonzero"] else "FAIL",
            {"frac_modules": fr, "B_fro": A["B_fro"], "n_exact_zero": A["n_B_exact_zero"]},
            f">= {THRESHOLDS['A2_min_frac_modules_nonzero']} of modules with ||B||_F > {THRESHOLDS['A2_min_lora_B_fro']}",
            "B=0 at init, so an adapter that never received useful updates has B ~ 0 and does nothing.")
        loss = (A.get("fp16_rounding_loss") or {}).get("median")
        if loss is not None:
            add(f"B4{suffix}", "Predicted fp16 rounding loss of the update at merge",
                "WARN" if loss > THRESHOLDS["B4_warn_fp16_rounding_loss"] else "PASS",
                {"median": loss, "frac_elements_unchanged_median": (A.get("fp16_frac_elements_unchanged") or {}).get("median")},
                f"median <= {THRESHOLDS['B4_warn_fp16_rounding_loss']} (WARN only)",
                "Share of the LoRA update norm lost when base+delta is stored in fp16.")

    a_checks(A_main, "")
    if A_ctrl is not None:
        a_checks(A_ctrl, "_control")

    # ---------------------------------------------------------------- tokenizer + data
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    gcfg_base = read_json(base_dir / "generation_config.json") or {}
    e = gcfg_base.get("eos_token_id")
    for x in (e if isinstance(e, list) else [e]):
        if x is not None:
            eos_ids.add(int(x))
    R["eos_ids"] = sorted(eos_ids)

    def encode_rows(rows, tag):
        encs, skipped = [], {"template_error": 0, "too_long": 0, "no_answer_tokens": 0}
        prefix_mismatch = 0
        for i, r in enumerate(rows):
            try:
                en = encode_example(tok, r["messages"], max_length=args.max_length)
            except TemplateError:
                skipped["template_error"] += 1
                continue
            if en["truncated"]:
                skipped["too_long"] += 1
                continue
            if len(en["answer_idx"]) == 0 or en["answer_idx"][0] == 0:
                skipped["no_answer_tokens"] += 1
                continue
            prefix_mismatch += 0 if en["prefix_tokens_match"] else 1
            en["row"] = i
            en["id"] = r.get("id", i)
            encs.append(en)
        log(f"{tag}: encoded {len(encs)}/{len(rows)} rows, skipped={skipped}, prompt-prefix token mismatch={prefix_mismatch}")
        return encs, {"n_rows": len(rows), "n_used": len(encs), "skipped": skipped,
                      "prompt_prefix_token_mismatch": prefix_mismatch}

    eval_rows = load_jsonl(args.eval_data, args.max_eval)
    eval_enc, R["eval_encoding"] = encode_rows(eval_rows, "eval")
    train_enc = []
    if args.train_data:
        train_rows = load_jsonl(args.train_data, args.max_train)
        train_enc, R["train_encoding"] = encode_rows(train_rows, "train-subset")

    # ---------------------------------------------------------------- load models
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log("loading base (fp16) + adapter(s) as PeftModel")
    base_model = AutoModelForCausalLM.from_pretrained(str(base_dir), device_map=device_map, **dtype_kw(fdtype))
    base_model.eval()
    gen_cfg_original = {}
    try:
        gen_cfg_original["base"] = base_model.generation_config.to_dict()
    except Exception:
        pass
    pm = PeftModel.from_pretrained(base_model, args.adapter, adapter_name="main", is_trainable=False)
    if args.control_adapter:
        pm.load_adapter(args.control_adapter, adapter_name="control", is_trainable=False)
    pm.eval()
    phase_mem("load_base_peft")

    from peft.tuners.lora import LoraLayer

    inner = pm.base_model.model
    lora_layers = {n: m for n, m in inner.named_modules() if isinstance(m, LoraLayer)}

    # A3: file == loaded
    def a3(file_mods_, adapter_name, suffix):
        max_diff, missing_in_model, missing_in_file = 0.0, [], []
        for path, ab in file_mods_.items():
            m = lora_layers.get(path)
            if m is None or adapter_name not in m.lora_A:
                missing_in_model.append(path)
                continue
            for key, mod_dict in (("A", m.lora_A), ("B", m.lora_B)):
                t_loaded = mod_dict[adapter_name].weight.detach().float().cpu()
                t_file = ab[key].float()
                if t_loaded.shape != t_file.shape:
                    missing_in_model.append(path + f" (shape {tuple(t_file.shape)} vs {tuple(t_loaded.shape)})")
                    continue
                max_diff = max(max_diff, float((t_loaded - t_file).abs().max()))
        for path, m in lora_layers.items():
            if adapter_name in m.lora_A and path not in file_mods_:
                missing_in_file.append(path)
        ok = (not missing_in_model) and (not missing_in_file) and max_diff <= THRESHOLDS["A3_max_abs_diff_file_vs_loaded"]
        add(f"A3{suffix}", "Adapter tensors PEFT loaded == tensors in the file", "PASS" if ok else "FAIL",
            {"max_abs_diff": max_diff, "file_modules_not_in_model": len(missing_in_model),
             "model_modules_not_in_file": len(missing_in_file), "examples": (missing_in_model + missing_in_file)[:5]},
            f"0 missing, max diff <= {THRESHOLDS['A3_max_abs_diff_file_vs_loaded']}",
            "Catches an adapter that was saved but is silently not applied (key/name mismatch).")

    a3(file_mods, "main", "")
    if ctrl_file_mods is not None:
        a3(ctrl_file_mods, "control", "_control")

    log("loading merged checkpoint")
    merged_dir = resolve_local_dir(args.merged)
    merged = AutoModelForCausalLM.from_pretrained(str(merged_dir), device_map=device_map, **dtype_kw(fdtype))
    merged.eval()
    try:
        gen_cfg_original["merged"] = merged.generation_config.to_dict()
    except Exception:
        pass
    phase_mem("load_merged")

    # ---------------------------------------------------------------- B: weights
    def b_checks(target_model, adapter_name, suffix):
        tB = time.time()
        rows = []
        target_paths = set()
        for path, m in lora_layers.items():
            if adapter_name not in m.lora_A:
                continue
            target_paths.add(path)
            Wb = m.base_layer.weight.detach()
            A = m.lora_A[adapter_name].weight.detach().float()
            Bm = m.lora_B[adapter_name].weight.detach().float()
            s = float(m.scaling[adapter_name])
            dW = s * (Bm @ A)
            try:
                Wm = target_model.get_submodule(path).weight.detach()
            except AttributeError:
                rows.append({"module": path, "missing_in_merged": True})
                continue
            if Wm.shape != Wb.shape:
                rows.append({"module": path, "shape_mismatch": True})
                continue
            Wb32, Wm32 = Wb.float(), Wm.float()
            ref_single = (Wb32 + dW).to(Wb.dtype).float()            # one rounding
            ref_peft = (Wb + dW.to(Wb.dtype)).float()                 # PEFT-style: delta cast, then add
            d_ref_single = float((ref_single - Wb32).norm())
            d_ref_peft = float((ref_peft - Wb32).norm())
            e_single = float((Wm32 - ref_single).norm()) / max(d_ref_single, 1e-30)
            e_peft = float((Wm32 - ref_peft).norm()) / max(d_ref_peft, 1e-30)
            wb_n = float(Wb32.norm())
            rows.append({
                "module": path,
                "type": module_type(path),
                "rel_delta": float((Wm32 - Wb32).norm()) / wb_n if wb_n > 0 else float("nan"),
                "err_vs_reference": min(e_single, e_peft),
                "exact_match_frac": max(float((Wm32 == ref_single).float().mean()), float((Wm32 == ref_peft).float().mean())),
                "frac_equal_to_base": float((Wm32 == Wb32).float().mean()),
                "ref_delta_is_zero": d_ref_single == 0.0 and d_ref_peft == 0.0,
            })
        # non-LoRA parameters must be bit-identical
        nontarget_max, nontarget_bad, n_compared = 0.0, [], 0
        inner_params = dict(inner.named_parameters())
        for pname, p in target_model.named_parameters():
            mod_path, _, leaf = pname.rpartition(".")
            if mod_path in target_paths and leaf == "weight":
                continue
            cand = f"{mod_path}.base_layer.{leaf}" if mod_path in target_paths else pname
            q = inner_params.get(cand)
            if q is None or q.shape != p.shape:
                nontarget_bad.append(pname + " (no counterpart)")
                continue
            n_compared += 1
            d = float((p.detach().float() - q.detach().float()).abs().max())
            if d > nontarget_max:
                nontarget_max = d
            if d > THRESHOLDS["B3_max_abs_nontarget_diff"]:
                nontarget_bad.append(f"{pname} maxdiff={d:.3g}")
        good = [r for r in rows if "rel_delta" in r]
        med_rel = float(np.median([r["rel_delta"] for r in good])) if good else 0.0
        meaningful = [r for r in good if not r["ref_delta_is_zero"]]
        max_err = max((r["err_vs_reference"] for r in meaningful), default=0.0)
        per_type = {}
        for t in sorted({r["type"] for r in good}):
            v = [r["rel_delta"] for r in good if r["type"] == t]
            per_type[t] = {"median_rel_delta": float(np.median(v)), "max_rel_delta": float(max(v)), "n": len(v)}
        broken = [r["module"] for r in rows if "rel_delta" not in r]
        add(f"B1{suffix}", "Merged weights differ from base (median relative delta)",
            "PASS" if med_rel >= THRESHOLDS["B1_min_median_rel_delta"] else "FAIL",
            {"median_rel_delta": med_rel, "per_type": per_type, "frac_elements_equal_to_base_median":
             float(np.median([r["frac_equal_to_base"] for r in good])) if good else None},
            f">= {THRESHOLDS['B1_min_median_rel_delta']}",
            "Catches 'merge produced the base model' (adapter not applied, or update rounded away).")
        add(f"B2{suffix}", "Merged == fp16(base + scale*B@A), computed independently",
            "PASS" if (not broken and max_err <= THRESHOLDS["B2_max_err_vs_reference"]) else "FAIL",
            {"max_err_vs_reference": max_err, "n_modules": len(good), "n_meaningful": len(meaningful),
             "broken_modules": broken[:5],
             "min_exact_match_frac": min((r["exact_match_frac"] for r in good), default=None)},
            f"max <= {THRESHOLDS['B2_max_err_vs_reference']} (relative to the size of the update)",
            "Catches wrong scale (alpha/r), transposed/misassigned modules, merging the wrong adapter.")
        add(f"B3{suffix}", "Non-LoRA parameters untouched by the merge",
            "PASS" if (not nontarget_bad and n_compared > 0) else "FAIL",
            {"max_abs_diff": nontarget_max, "n_compared": n_compared, "offenders": nontarget_bad[:5]},
            f"max abs diff <= {THRESHOLDS['B3_max_abs_nontarget_diff']}",
            "Catches dtype round-trips (bf16<->fp16), modules_to_save surprises, resized embeddings.")
        R[f"weights{suffix}"] = {"per_type": per_type, "seconds": round(time.time() - tB, 1),
                                 "n_nontarget_compared": n_compared}
        with open(out_dir / f"merge_modules{suffix or '_main'}.csv", "w", newline="", encoding="utf-8") as fh:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    with torch.no_grad():
        b_checks(merged, "main", "")
    phase_mem("weights_main")

    # ---------------------------------------------------------------- 4-bit models
    q4 = None
    base4 = None
    qcfg = None
    if args.quant4 and have_bnb:
        try:
            q4_dir = Path(args.quant4)
            qcfg = (read_json(q4_dir / "config.json") or {}).get("quantization_config")
            log(f"loading 4-bit artifact {q4_dir} quantization_config={qcfg}")
            q4 = AutoModelForCausalLM.from_pretrained(str(q4_dir), device_map=device_map, **dtype_kw(torch.float16))
            q4.eval()
            try:
                gen_cfg_original["merged_4bit"] = q4.generation_config.to_dict()
            except Exception:
                pass
            phase_mem("load_4bit")
            if not args.skip_base4bit and qcfg:
                from transformers import BitsAndBytesConfig

                try:
                    bcfg = BitsAndBytesConfig.from_dict(dict(qcfg))
                except Exception as exc:
                    log(f"BitsAndBytesConfig.from_dict failed ({exc!r}); rebuilding from known keys")
                    cd = qcfg.get("bnb_4bit_compute_dtype", "float16")
                    bcfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
                        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
                        bnb_4bit_compute_dtype=getattr(torch, cd) if isinstance(cd, str) else torch.float16,
                    )
                log("quantizing the BASE with the same 4-bit config (matched control)")
                base4 = AutoModelForCausalLM.from_pretrained(str(base_dir), device_map=device_map,
                                                             quantization_config=bcfg, **dtype_kw(torch.float16))
                base4.eval()
                phase_mem("load_base4bit")
        except Exception as exc:  # a broken 4-bit load must not take the fp16 checks down with it
            log(f"4-bit load failed: {exc!r}")
            add("Q0", "4-bit artifact loads with bitsandbytes", "FAIL", repr(exc)[:300], "loads", "The 4-bit file could not be loaded")
            q4 = None
            base4 = None
    elif args.quant4:
        log("4-bit checks skipped: bitsandbytes/CUDA unavailable")

    # Q: code-level provenance of the 4-bit artifact
    if q4 is not None:
        try:
            import bitsandbytes.functional as BF

            tq = time.time()
            n_mod, mismatch_m, changed_mb, art_vs_b, details = 0, [], [], [], []
            for name, mod in q4.named_modules():
                w = getattr(mod, "weight", None)
                qs = getattr(w, "quant_state", None)
                if qs is None:
                    continue
                lm = lora_layers.get(name)
                Wb = (lm.base_layer.weight if lm is not None else inner.get_submodule(name).weight).detach()
                Wm = merged.get_submodule(name).weight.detach()
                art = BF.dequantize_4bit(w.data, quant_state=qs).float()
                kw = dict(blocksize=qs.blocksize, quant_type=qs.quant_type,
                          compress_statistics=bool(getattr(qs, "nested", False)))
                qm, sm = BF.quantize_4bit(Wm.to(torch.float16).contiguous(), **kw)
                qb, sb = BF.quantize_4bit(Wb.to(torch.float16).contiguous(), **kw)
                Qm = BF.dequantize_4bit(qm, quant_state=sm).float()
                Qb = BF.dequantize_4bit(qb, quant_state=sb).float()
                if art.shape != Qm.shape:
                    art = art.reshape(Qm.shape)
                mm = float((art != Qm).float().mean())
                ch = float((Qm != Qb).float().mean())
                ab = float((art != Qb).float().mean())
                mismatch_m.append(mm)
                changed_mb.append(ch)
                art_vs_b.append(ab)
                n_mod += 1
                if n_mod <= 3 or mm > THRESHOLDS["Q1_max_mismatch_vs_requant_merged"]:
                    details.append({"module": name, "mismatch_vs_requant_merged": mm,
                                    "frac_changed_merged_vs_base": ch, "artifact_vs_requant_base": ab})
                del art, Qm, Qb, qm, qb, sm, sb
            R["quant4_codes"] = {
                "n_modules": n_mod,
                "mismatch_vs_requant_merged": {"median": float(np.median(mismatch_m)) if mismatch_m else None,
                                               "max": max(mismatch_m) if mismatch_m else None},
                "frac_changed_merged_vs_base": {"median": float(np.median(changed_mb)) if changed_mb else None,
                                                "mean": float(np.mean(changed_mb)) if changed_mb else None},
                "artifact_vs_requant_base": {"median": float(np.median(art_vs_b)) if art_vs_b else None},
                "details": details[:10],
                "seconds": round(time.time() - tq, 1),
            }
            ok_q1 = n_mod > 0 and max(mismatch_m) <= THRESHOLDS["Q1_max_mismatch_vs_requant_merged"]
            add("Q1", "4-bit artifact == quantize(merged fp16)", "PASS" if ok_q1 else "WARN",
                R["quant4_codes"]["mismatch_vs_requant_merged"], f"max <= {THRESHOLDS['Q1_max_mismatch_vs_requant_merged']}",
                "Proves the 4-bit file was made from THIS merged model (not from the base or another merge). "
                "WARN, not FAIL: a different but legitimate quantization path can differ slightly.")
            fr = float(np.mean(changed_mb)) if changed_mb else 0.0
            add("Q2", "Fine-tune changed 4-bit codes (quantize(merged) != quantize(base))",
                "PASS" if fr >= THRESHOLDS["Q2_min_frac_4bit_weights_changed"] else "FAIL",
                {"mean_frac_weights_changed": fr}, f">= {THRESHOLDS['Q2_min_frac_4bit_weights_changed']}",
                "If ~0, NF4 rounding erased the update: the 4-bit artifact IS the 4-bit base.")
        except Exception as exc:
            log(f"Q checks failed: {exc!r}")
            add("Q1", "4-bit artifact == quantize(merged fp16)", "SKIP", repr(exc)[:200], "-", "bitsandbytes API error")
        phase_mem("quant_codes")

    # ---------------------------------------------------------------- views
    gen_views, tv = [], []
    views = [View("base", pm, "peft_off"), View("adapter", pm, "peft_on", "main")]
    if args.control_adapter:
        views.append(View("control", pm, "peft_on", "control"))
    views.append(View("merged", merged))
    if q4 is not None:
        views.append(View("merged_4bit", q4))
    if base4 is not None:
        views.append(View("base_4bit", base4))
    vnames = [v.name for v in views]
    pairs = [("merged", "base"), ("adapter", "merged"), ("merged", "merged_4bit"),
             ("base", "base_4bit"), ("merged_4bit", "base_4bit"), ("control", "base")]
    pairs = [p for p in pairs if p[0] in vnames and p[1] in vnames]
    R["views"] = vnames

    def run_pass(encs, view_list, pair_list, tag, hidden_scan=0):
        tP = time.time()
        res = {v.name: {"nll": [], "n": [], "nonfinite": 0, "max_abs_logit": 0.0} for v in view_list}
        kl = {f"{a}||{b}": {"kl_sum": 0.0, "agree": 0, "n": 0} for a, b in pair_list}
        hid = {v.name: [] for v in view_list if v.name in ("base", "merged")}
        need = {x for p in pair_list for x in p}
        for ei, en in enumerate(encs):
            ids = torch.tensor([en["input_ids"]], device=dev)
            am = torch.ones_like(ids)
            ans = torch.tensor(en["answer_idx"], device=dev)
            pos = ans - 1
            tgt = ids[0, ans]
            store = {}
            for v in view_list:
                want_h = ei < hidden_scan and v.name in hid
                with torch.inference_mode():
                    out = v.forward(input_ids=ids, attention_mask=am, use_cache=False, output_hidden_states=want_h)
                logits = out.logits
                mx, mn = float(logits.max()), float(logits.min())
                nf = 0 if (math.isfinite(mx) and math.isfinite(mn)) else int((~torch.isfinite(logits)).sum())
                res[v.name]["nonfinite"] += nf
                res[v.name]["max_abs_logit"] = max(res[v.name]["max_abs_logit"], abs(mx), abs(mn))
                if want_h:
                    hid[v.name].append([float(h.float().abs().max()) for h in out.hidden_states])
                sel = logits[0, pos].float()
                del out, logits
                logp = torch.log_softmax(sel, dim=-1)
                del sel
                tok_nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
                res[v.name]["nll"].append(float(tok_nll.sum()))
                res[v.name]["n"].append(int(tok_nll.numel()))
                if v.name in need:
                    store[v.name] = logp[: args.kl_max_positions].clone()
                del logp, tok_nll
            for a, b in pair_list:
                la, lb = store[a], store[b]
                k = kl_rows(la, lb)
                d = kl[f"{a}||{b}"]
                d["kl_sum"] += float(k.sum())
                d["agree"] += int((la.argmax(-1) == lb.argmax(-1)).sum())
                d["n"] += int(k.numel())
            del store
            if (ei + 1) % 25 == 0:
                log(f"{tag}: {ei + 1}/{len(encs)} examples")
        summ = {}
        for name, d in res.items():
            tot_n = sum(d["n"])
            tot = sum(d["nll"])
            mean = tot / tot_n if tot_n else float("nan")
            summ[name] = {
                "nll_per_token": mean,
                "ppl": math.exp(mean) if math.isfinite(mean) and mean < 50 else float("inf"),
                "n_tokens": tot_n,
                "n_examples": len(d["n"]),
                "ci95": (bootstrap_ci(d["nll"], d["n"], B=args.bootstrap, seed=args.seed) or {}).get("ci95"),
                "nonfinite_logits": d["nonfinite"],
                "max_abs_logit": d["max_abs_logit"],
            }
        kls = {k: {"kl_mean": v["kl_sum"] / v["n"] if v["n"] else None,
                   "top1_agreement": v["agree"] / v["n"] if v["n"] else None,
                   "n_positions": v["n"]} for k, v in kl.items()}
        hidden = {k: {"max_abs_per_layer": (np.max(np.array(v), axis=0).tolist() if v else None),
                      "global_max_abs": (float(np.max(v)) if v else None)} for k, v in hid.items()}
        return res, summ, kls, hidden, round(time.time() - tP, 1)

    log(f"eval pass over {len(eval_enc)} examples, views={vnames}, pairs={pairs}")
    raw, summ, kls, hidden, secs = run_pass(eval_enc, views, pairs, "eval", hidden_scan=args.hidden_scan_examples)
    R["timing_s"]["eval_pass"] = secs
    R["heldout"] = summ
    R["kl"] = kls
    R["hidden_state_scan"] = hidden
    phase_mem("eval_pass")

    # paired differences (positive = second model better, i.e. lower NLL)
    def paired(a, b):
        if a not in raw or b not in raw:
            return None
        return bootstrap_ci(raw[a]["nll"], raw[a]["n"], other=raw[b]["nll"], B=args.bootstrap, seed=args.seed)

    diffs = {}
    for a, b in [("base", "merged"), ("base", "adapter"), ("base", "control"), ("base_4bit", "merged_4bit"),
                 ("merged", "merged_4bit"), ("base", "merged_4bit"), ("base", "base_4bit")]:
        d = paired(a, b)
        if d is not None:
            diffs[f"{a}-{b}"] = d
    R["heldout_diffs"] = diffs
    if "merged" in raw:
        imp = [(bn / n - mn / n) for bn, mn, n in zip(raw["base"]["nll"], raw["merged"]["nll"], raw["merged"]["n"])]
        R["heldout_frac_examples_improved"] = float(np.mean([x > 0 for x in imp])) if imp else None

    with open(out_dir / "per_example_nll.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "id", "n_answer_tokens"] + [f"nll_sum_{n}" for n in raw])
        for i, en in enumerate(eval_enc):
            w.writerow([en["row"], en["id"], raw[next(iter(raw))]["n"][i]] + [raw[n]["nll"][i] for n in raw])

    # C checks
    d = diffs.get("base-merged")
    if d:
        ok = d["point"] >= THRESHOLDS["C1_min_nll_gain"] and d["ci95"][0] > 0
        add("C1", "Held-out answer NLL: base - merged (nats/token, paired bootstrap)", "PASS" if ok else "FAIL",
            d, f"point >= {THRESHOLDS['C1_min_nll_gain']} and CI95 low > 0",
            "The merged model assigns higher likelihood to unseen reference answers than the base.")
    dc = diffs.get("base-control")
    if dc:
        ok = dc["point"] >= THRESHOLDS["C1_min_nll_gain"] and dc["ci95"][0] > 0
        add("C1_control", "Held-out NLL: base - control (negative control, expected FAIL)", "PASS" if ok else "FAIL",
            dc, f"point >= {THRESHOLDS['C1_min_nll_gain']} and CI95 low > 0",
            "A negative control must not pass; if it does, the check cannot tell trained from untrained.")
    d4 = diffs.get("base_4bit-merged_4bit")
    if d4 and d:
        gain = d["point"]
        retained = d4["point"] / gain if gain > 0 else float("nan")
        ok = d4["ci95"][0] > 0 and math.isfinite(retained) and retained >= THRESHOLDS["C2_min_retained_gain_frac"]
        add("C2", "4-bit keeps the gain: NLL(base_4bit) - NLL(merged_4bit), retained fraction", "PASS" if ok else "FAIL",
            {"diff": d4, "retained_frac_of_fp16_gain": retained},
            f"CI95 low > 0 and retained >= {THRESHOLDS['C2_min_retained_gain_frac']}",
            "Compares like with like (both 4-bit), so quantization noise does not masquerade as training.")
    dq = diffs.get("merged-merged_4bit")
    if dq:
        inc = -dq["point"]
        add("C2b", "Quantization cost on held-out NLL: NLL(merged_4bit) - NLL(merged)",
            "WARN" if inc > THRESHOLDS["C2_warn_quant_nll_increase"] else "PASS",
            {"increase": inc, "ci95_of_merged_minus_4bit": dq["ci95"]},
            f"<= {THRESHOLDS['C2_warn_quant_nll_increase']} (WARN only)", "How much 4-bit costs on Russian held-out text.")

    # D checks
    k = kls.get("adapter||merged")
    if k and k["kl_mean"] is not None:
        ok = k["kl_mean"] <= THRESHOLDS["D1_max_kl_adapter_vs_merged"] and k["top1_agreement"] >= THRESHOLDS["D1_min_top1_adapter_vs_merged"]
        add("D1", "Merged behaves like base+adapter (KL, top-1)", "PASS" if ok else "FAIL", k,
            f"KL <= {THRESHOLDS['D1_max_kl_adapter_vs_merged']}, top1 >= {THRESHOLDS['D1_min_top1_adapter_vs_merged']}",
            "Functional proof that the merge preserved the adapter's effect.")
    k = kls.get("merged||base")
    if k and k["kl_mean"] is not None:
        floor = (kls.get("base||base_4bit") or {}).get("kl_mean")
        add("D2", "Merged changed next-token distributions vs base (KL)",
            "PASS" if k["kl_mean"] >= THRESHOLDS["D2_min_kl_merged_vs_base"] else "FAIL",
            {**k, "quant_noise_floor_kl_base_vs_base4": floor,
             "ratio_to_quant_floor": (k["kl_mean"] / floor) if floor else None},
            f"KL >= {THRESHOLDS['D2_min_kl_merged_vs_base']}",
            "An untrained adapter gives KL ~ 0. Ratio to the 4-bit noise floor shows if training is bigger than quant noise.")
    k = kls.get("control||base")
    if k and k["kl_mean"] is not None:
        add("D2_control", "Control changed distributions vs base (expected FAIL)",
            "PASS" if k["kl_mean"] >= THRESHOLDS["D2_min_kl_merged_vs_base"] else "FAIL", k,
            f"KL >= {THRESHOLDS['D2_min_kl_merged_vs_base']}", "Negative control must fail.")
    k = kls.get("merged||merged_4bit")
    if k and k["kl_mean"] is not None:
        ok = k["kl_mean"] <= THRESHOLDS["D3_warn_max_kl_merged_vs_4bit"] and k["top1_agreement"] >= THRESHOLDS["D3_warn_min_top1_merged_vs_4bit"]
        add("D3", "4-bit fidelity to merged fp16 (KL, top-1)", "PASS" if ok else "WARN", k,
            f"KL <= {THRESHOLDS['D3_warn_max_kl_merged_vs_4bit']}, top1 >= {THRESHOLDS['D3_warn_min_top1_merged_vs_4bit']} (WARN only)",
            "How far 4-bit drifts from the model that was evaluated in fp16.")

    # E2 non-finite
    nf = {n: s["nonfinite_logits"] for n, s in summ.items()}
    hmax = {n: h["global_max_abs"] for n, h in hidden.items()}
    add("E2", "No inf/NaN logits (fp16 overflow on T4)", "PASS" if max(nf.values()) <= THRESHOLDS["E2_max_nonfinite_logits"] else "FAIL",
        {"nonfinite_logits": nf, "hidden_state_max_abs": hmax, "fp16_max": FP16_MAX},
        "0 non-finite", "fp16 has max 65504; Qwen checkpoints are bf16. Overflow gives NaN/garbage silently.")

    # generalisation gap on a train subset
    if train_enc:
        tv = [v for v in views if v.name in ("base", "merged")]
        log(f"train-subset pass over {len(train_enc)} examples")
        traw, tsum, _, _, secs = run_pass(train_enc, tv, [], "train-subset")
        R["timing_s"]["train_pass"] = secs
        R["train_subset"] = tsum
        gap = {}
        for n in ("base", "merged"):
            if n in tsum and n in summ:
                gap[n] = summ[n]["nll_per_token"] - tsum[n]["nll_per_token"]
        R["generalisation_gap"] = {"eval_minus_train_nll": gap,
                                   "merged_minus_base_gap": (gap.get("merged", 0) - gap.get("base", 0)) if gap else None,
                                   "note": "base gap = distribution difference of the two subsets; extra gap of merged ~ memorisation"}
        phase_mem("train_pass")

    # ---------------------------------------------------------------- E: generations
    if args.prompts and not args.skip_generations:
        from transformers import GenerationConfig

        tg = time.time()
        prompts = load_jsonl(args.prompts)
        pad_id = tok.pad_token_id
        gcfg = GenerationConfig(do_sample=False, max_new_tokens=args.max_new_tokens, repetition_penalty=1.0,
                                eos_token_id=sorted(eos_ids), pad_token_id=pad_id,
                                temperature=None, top_p=None, top_k=None)
        R["generation_config_used"] = gcfg.to_dict()
        tok.padding_side = "left"
        gen_views = [v for v in views if v.name in ("base", "merged", "merged_4bit", "control")]
        outs = {}
        for v in gen_views:
            # make OUR config the model default too, so no model-shipped default
            # (Qwen ships repetition_penalty/temperature/top_p) is merged back in.
            for mdl in {id(v.model): v.model, id(getattr(v.model, "base_model", v.model)): getattr(v.model, "base_model", v.model)}.values():
                try:
                    mdl.generation_config = gcfg
                except Exception:
                    pass
            try:
                inner.generation_config = gcfg
            except Exception:
                pass
            res = []
            for i in range(0, len(prompts), args.gen_batch_size):
                chunk = prompts[i:i + args.gen_batch_size]
                texts = [render_prompt(tok, [{"role": "user", "content": p["prompt"]}]) for p in chunk]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                with torch.inference_mode():
                    o = v.generate(**enc, generation_config=gcfg)
                new = o[:, enc["input_ids"].shape[1]:].tolist()
                for p, ids_ in zip(chunk, new):
                    stop = next((j for j, t in enumerate(ids_) if t in eos_ids or t == pad_id), None)
                    body = ids_[:stop] if stop is not None else ids_
                    text = tok.decode(body, skip_special_tokens=True)
                    res.append({"id": p.get("id"), "kind": p.get("kind"), "new_tokens": len(body),
                                "stopped_eos": stop is not None, "cyrillic_ratio": round(cyr_ratio(text), 3),
                                "rep4": round(rep4(body), 3), "replacement_char": "�" in text, "text": text})
            outs[v.name] = res
            log(f"generated {len(res)} outputs for view={v.name}")
        tok.padding_side = "right"
        with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as fh:
            for i, p in enumerate(prompts):
                row = {"id": p.get("id"), "kind": p.get("kind"), "prompt": p["prompt"], "reference": p.get("reference")}
                for n, res in outs.items():
                    row[n] = res[i]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out_dir / "generations.md", "w", encoding="utf-8") as fh:
            fh.write("# Greedy generations, side by side\n\n")
            fh.write(f"Config: do_sample=False, repetition_penalty=1.0, max_new_tokens={args.max_new_tokens}, "
                     f"eos={sorted(eos_ids)}. Same chat template for every model.\n\n")
            for i, p in enumerate(prompts):
                fh.write(f"## {p.get('id')} ({p.get('kind')})\n\n**Prompt:** {p['prompt']}\n\n")
                if p.get("reference"):
                    fh.write(f"**Reference (dataset):** {p['reference'][:800]}\n\n")
                for n, res in outs.items():
                    r = res[i]
                    fh.write(f"**{n}** (tokens={r['new_tokens']}, eos={r['stopped_eos']}, cyr={r['cyrillic_ratio']}, "
                             f"rep4={r['rep4']}):\n\n```\n{r['text'][:1500]}\n```\n\n")
        gsum = {}
        for n, res in outs.items():
            ru = [r for r in res if r.get("kind") != "en_input"]
            gsum[n] = {
                "eos_rate": float(np.mean([r["stopped_eos"] for r in res])) if res else None,
                "median_new_tokens": float(np.median([r["new_tokens"] for r in res])) if res else None,
                "frac_cyrillic_ok": float(np.mean([r["cyrillic_ratio"] >= THRESHOLDS["E1_min_cyrillic_ratio"] for r in ru])) if ru else None,
                "max_rep4": max((r["rep4"] for r in res), default=None),
                "n": len(res),
            }
        if "merged" in outs and "base" in outs:
            same = sum(1 for a, b in zip(outs["merged"], outs["base"]) if a["text"] == b["text"])
            gsum["merged_identical_to_base"] = same
        if "merged_4bit" in outs and "merged" in outs:
            gsum["merged4_identical_to_merged"] = sum(1 for a, b in zip(outs["merged_4bit"], outs["merged"]) if a["text"] == b["text"])
        R["generations"] = gsum
        R["timing_s"]["generations"] = round(time.time() - tg, 1)
        for n in ("merged", "merged_4bit"):
            g = gsum.get(n)
            if not g:
                continue
            ok = (g["eos_rate"] >= THRESHOLDS["E1_min_eos_rate"]
                  and (g["frac_cyrillic_ok"] is None or g["frac_cyrillic_ok"] >= 0.9)
                  and g["max_rep4"] <= THRESHOLDS["E1_max_rep4"])
            add(f"E1_{n}", f"Generations of {n}: stop at EOS, Russian, not degenerate", "PASS" if ok else "WARN", g,
                f"eos_rate >= {THRESHOLDS['E1_min_eos_rate']}, >=90% prompts cyrillic >= {THRESHOLDS['E1_min_cyrillic_ratio']}, "
                f"rep4 <= {THRESHOLDS['E1_max_rep4']} (WARN only: read generations.md yourself)",
                "Teacher-forced metrics can look fine while free generation never stops or switches language.")
        if gsum.get("merged_identical_to_base") is not None:
            same = gsum["merged_identical_to_base"]
            add("E3", "Greedy outputs changed vs base on at least one probe", "PASS" if same < len(prompts) else "WARN",
                {"identical": same, "n": len(prompts)}, "identical < n", "All-identical outputs suggest nothing changed.")
        phase_mem("generations")

    # ---------------------------------------------------------------- T: artifacts metadata
    def tok_fingerprint(path):
        try:
            t = AutoTokenizer.from_pretrained(str(path))
        except Exception as exc:
            return {"error": repr(exc)[:200]}
        probe = "Привет! Как дела?<|im_end|>"
        return {"chat_template_sha": sha256_text(t.chat_template if isinstance(t.chat_template, str) else json.dumps(t.chat_template)),
                "eos": t.eos_token, "pad": t.pad_token, "bos": t.bos_token, "len": len(t),
                "probe_ids_sha": sha256_text(json.dumps(t(probe, add_special_tokens=True)["input_ids"]))}

    fps = {"base": tok_fingerprint(base_dir), "adapter": tok_fingerprint(args.adapter), "merged": tok_fingerprint(merged_dir)}
    if args.quant4:
        fps["merged_4bit"] = tok_fingerprint(args.quant4)
    R["tokenizer_fingerprints"] = fps
    ref = fps["base"]
    diffs_t = {n: [k for k in ("chat_template_sha", "eos", "pad", "len", "probe_ids_sha") if fp.get(k) != ref.get(k)]
               for n, fp in fps.items() if n != "base" and "error" not in fp}
    errs = {n: fp["error"] for n, fp in fps.items() if "error" in fp}
    hard = {n: d for n, d in diffs_t.items() if set(d) & {"chat_template_sha", "eos", "probe_ids_sha"}}
    add("T1", "Tokenizer + chat template identical across artifacts", "FAIL" if hard else ("WARN" if (errs or any(diffs_t.values())) else "PASS"),
        {"differences": diffs_t, "load_errors": errs}, "no differences",
        "A shipped artifact with another template/EOS/pad silently changes inference.")
    keys = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id", "pad_token_id", "bos_token_id")
    gdiff = {}
    for n, gcd in gen_cfg_original.items():
        gdiff[n] = {k: gcd.get(k) for k in keys}
    R["generation_config_shipped"] = gdiff
    vals = {json.dumps(v, sort_keys=True) for v in gdiff.values()}
    add("T2", "generation_config.json consistent across artifacts", "PASS" if len(vals) <= 1 else "WARN", gdiff,
        "identical", "Users get the artifact's defaults (Qwen ships sampling + repetition_penalty), not our greedy eval.")

    # ---------------------------------------------------------------- control merged weights
    if args.control_merged and args.control_adapter:
        log("control merged: weight checks (4-bit models released first)")
        q4 = base4 = None
        views = gen_views = tv = v = None
        free(torch)
        cm = AutoModelForCausalLM.from_pretrained(str(resolve_local_dir(args.control_merged)), device_map=device_map, **dtype_kw(fdtype))
        with torch.no_grad():
            b_checks(cm, "control", "_control")
        del cm
        free(torch)
        phase_mem("weights_control")

    # ---------------------------------------------------------------- verdicts
    st = {c["id"]: c["status"] for c in checks}
    core = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "D1", "D2", "E2"]
    present = [c for c in core if c in st]
    trained = all(st[c] == "PASS" for c in present) and {"A2", "B1", "C1", "D2"} <= set(present)
    verdict = {"model_trained": "YES" if trained else "NO",
               "failed_core_checks": [c for c in present if st[c] != "PASS"],
               "missing_core_checks": [c for c in core if c not in st]}
    if any(c.endswith("_control") for c in st):
        ctrl_core = [c for c in ("A2_control", "B1_control", "C1_control", "D2_control") if c in st]
        ctrl_flagged = all(st[c] == "FAIL" for c in ctrl_core) and len(ctrl_core) > 0
        verdict["negative_control_flagged_untrained"] = "YES" if ctrl_flagged else "NO (the check does NOT separate trained from untrained)"
        verdict["control_checks"] = {c: st[c] for c in ctrl_core}
    if "C2" in st or "Q2" in st:
        q_core = [c for c in ("Q2", "C2") if c in st]
        verdict["4bit_carries_training"] = "YES" if all(st[c] == "PASS" for c in q_core) else "NO"
        verdict["4bit_checks"] = {c: st.get(c) for c in ("Q1", "Q2", "C2", "C2b", "D3", "E1_merged_4bit") if c in st}
    R["verdict"] = verdict
    R["finished"] = now()
    R["timing_s"]["total"] = round(time.time() - t_start, 1)

    with open(out_dir / "verify_results.json", "w", encoding="utf-8") as fh:
        json.dump(to_py(R), fh, indent=2, ensure_ascii=False)

    lines = [f"verify_trained.py summary  ({R['finished']})", "=" * 78]
    lines.append(f"views: {', '.join(R['views'])}")
    lines.append("")
    lines.append("Held-out answer-token NLL (nats/token), 95% bootstrap CI over examples:")
    for n, s in summ.items():
        ci = "[" + ", ".join(f"{x:.4f}" for x in s["ci95"]) + "]" if s.get("ci95") else "n/a"
        lines.append(f"  {n:<12} nll={s['nll_per_token']:.4f} ppl={s['ppl']:.3f} ci95={ci} tokens={s['n_tokens']} nonfinite={s['nonfinite_logits']}")
    lines.append("Paired differences (positive = second model has lower NLL):")
    for k_, d_ in diffs.items():
        lines.append(f"  {k_:<26} {d_['point']:+.4f}  ci95=[{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}]")
    lines.append("KL(p||q) and top-1 agreement on answer positions:")
    for k_, d_ in kls.items():
        if d_["kl_mean"] is not None:
            lines.append(f"  {k_:<26} kl={d_['kl_mean']:.5f} top1={d_['top1_agreement']:.4f} positions={d_['n_positions']}")
    lines.append("")
    lines.append("Checks:")
    for c in checks:
        lines.append(f"  [{c['status']:<4}] {c['id']:<11} {c['name']}")
    lines.append("")
    lines.append("VERDICT: " + json.dumps(verdict, ensure_ascii=False))
    lines.append("")
    lines.append("What this check does NOT detect:")
    for s in NOT_DETECTED:
        lines.append(f"  - {s}")
    text = "\n".join(lines)
    (out_dir / "verify_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    log(f"DONE in {R['timing_s']['total']} s; results in {out_dir}")
    return 0 if trained else 2


if __name__ == "__main__":
    sys.exit(main())
