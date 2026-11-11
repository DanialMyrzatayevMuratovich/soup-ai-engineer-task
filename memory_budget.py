#!/usr/bin/env python3
"""memory_budget.py: VRAM/RAM budget for LoRA SFT -> merge -> 4-bit export, computed BEFORE
training from the model's config.json and the run parameters. A second mode compares
the estimate with measured peaks and prints the gap.

Estimate (run before training):
  python memory_budget.py --model Qwen/Qwen2.5-1.5B-Instruct --batch-size 2 --seq-len 1024 \
      --data-report data/data_report.json --lora-r 16 --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
      --out outputs/memory_budget.json

Compare (after the run):
  python memory_budget.py --compare outputs/measured_memory.json --estimate outputs/memory_budget.json

Units: GiB = 2**30 bytes (nvidia-smi reports MiB = 2**20 bytes). All formulas are printed
next to their numbers. The model is an assumption-laden estimate, and the comparison mode
exists to find out which assumptions were wrong.

Assumptions (state them, then measure them):
  * Frozen base stored in fp16 (2 B/param). T4 has no bf16, Qwen ships bf16 -> converted to fp16.
  * LoRA params are fp32 under fp16 AMP (GradScaler needs fp32 trainables): weights 4 B + grads 4 B
    + AdamW exp_avg 4 B + exp_avg_sq 4 B = 16 B per trainable param. No fp32 master copy of the base.
  * Gradient checkpointing per decoder layer (HF style): every layer's INPUT hidden state is kept
    (L x B x T x H x 2 B); one layer is recomputed at a time during backward.
  * Dynamic padding: a batch is padded to its longest row, so the training PEAK is set by the
    longest row in the data (its batch partner is padded up to it), not by the mean length.
  * Logits are the dominant activation for small models with big vocabularies (Qwen: 151,936).
    Forward: fp16 lm_head output (2 B) + fp32 upcast inside the loss (4 B) + log_softmax output saved
    for backward (4 B) = 10 B per (token x vocab). Backward adds grad of log-probs (4 B) and grad of
    the logits (4 B) while the saved log-probs and fp16 logits are alive -> ~14 B per element at peak.
    Loss is computed for all positions even though only answer tokens carry labels.
    (Soup v0.77.2's own estimator, utils/vram_estimate.py, charges a measured 12 B per element.)
  * CUDA context + kernels are invisible to torch.cuda.memory_allocated but visible in nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

GiB = 2**30
MiB = 2**20

# Public config of Qwen/Qwen2.5-1.5B-Instruct (used only if config.json cannot be read).
FALLBACK_QWEN25_15B = {
    "model_type": "qwen2", "hidden_size": 1536, "intermediate_size": 8960, "num_hidden_layers": 28,
    "num_attention_heads": 12, "num_key_value_heads": 2, "vocab_size": 151936,
    "tie_word_embeddings": True, "torch_dtype": "bfloat16",
}

LORA_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_config(model: str, config_json: str | None):
    if config_json:
        return json.loads(Path(config_json).read_text()), config_json
    p = Path(model)
    if (p / "config.json").exists():
        return json.loads((p / "config.json").read_text()), str(p / "config.json")
    try:
        from huggingface_hub import hf_hub_download

        f = hf_hub_download(model, "config.json")
        return json.loads(Path(f).read_text()), f
    except Exception as exc:  # offline
        print(f"[memory_budget] WARNING: cannot fetch config.json ({exc!r}); using built-in Qwen2.5-1.5B values",
              file=sys.stderr)
        return dict(FALLBACK_QWEN25_15B), "builtin-fallback"


def dims(cfg):
    H = int(cfg["hidden_size"])
    I = int(cfg["intermediate_size"])
    L = int(cfg["num_hidden_layers"])
    nh = int(cfg["num_attention_heads"])
    nkv = int(cfg.get("num_key_value_heads") or nh)
    hd = int(cfg.get("head_dim") or H // nh)
    V = int(cfg["vocab_size"])
    return H, I, L, nh, nkv, hd, V


def module_shapes(cfg):
    """(in_features, out_features) of each projection type."""
    H, I, L, nh, nkv, hd, V = dims(cfg)
    q_out, kv_out = nh * hd, nkv * hd
    return {"q_proj": (H, q_out), "k_proj": (H, kv_out), "v_proj": (H, kv_out), "o_proj": (q_out, H),
            "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H)}


def param_counts(cfg):
    H, I, L, nh, nkv, hd, V = dims(cfg)
    shapes = module_shapes(cfg)
    qkv_bias = cfg.get("model_type") in ("qwen2",) or bool(cfg.get("attention_bias", False))
    mlp_bias = bool(cfg.get("mlp_bias", False))
    lin_w = sum(i * o for i, o in shapes.values())
    bias = 0
    if qkv_bias:
        bias += shapes["q_proj"][1] + shapes["k_proj"][1] + shapes["v_proj"][1]
    if mlp_bias:
        bias += 2 * I + H
    norms = 2 * H
    per_layer = lin_w + bias + norms
    embed = V * H
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else V * H
    total = embed + L * per_layer + H + lm_head
    return {
        "embed": embed, "lm_head_untied": lm_head, "per_layer": per_layer,
        "per_layer_linear_weights": lin_w, "per_layer_bias": bias, "per_layer_norms": norms,
        "layers_total": L * per_layer, "final_norm": H, "total": total,
        "linear_weight_params_total": L * lin_w, "non_linear_params_total": total - L * lin_w,
        "qkv_bias": qkv_bias, "tied_embeddings": bool(cfg.get("tie_word_embeddings", False)),
    }


def lora_counts(cfg, r, targets):
    shapes = module_shapes(cfg)
    L = int(cfg["num_hidden_layers"])
    per_type = {}
    for t in targets:
        if t not in shapes:
            per_type[t] = {"note": "unknown module type: not counted"}
            continue
        i, o = shapes[t]
        per_type[t] = {"in": i, "out": o, "per_module": r * (i + o), "total": L * r * (i + o)}
    total = sum(v.get("total", 0) for v in per_type.values())
    return per_type, total, L * len([t for t in targets if t in shapes])


def activations(cfg, B, T, r, n_targets, dropout, ckpt, act_bytes=2):
    """Activation memory of the decoder stack (excluding logits), bytes."""
    H, I, L, nh, nkv, hd, V = dims(cfg)
    q_out, kv_out = nh * hd, nkv * hd
    BT = B * T
    a = act_bytes
    # LoRA: dropout output (saved by lora_A) + dropout mask (1 B) + lora_A output (fp32, r wide)
    in_widths = {"q_proj": H, "k_proj": H, "v_proj": H, "o_proj": q_out, "gate_proj": H, "up_proj": H, "down_proj": I}
    lora_in = sum(in_widths.values()) if n_targets else 0
    lora_bytes = BT * lora_in * (a + (1 if dropout > 0 else 0)) + BT * 7 * r * 4
    norms = 2 * BT * H * a                                  # two RMSNorm inputs
    rope = BT * (q_out + kv_out) * a                        # q, k before rotary
    attn = BT * (q_out + 2 * q_out + q_out) * a + B * nh * T * 4   # q, k/v (repeat_kv to nh heads), out, logsumexp
    mlp = 3 * BT * I * a                                    # gate_out, up_out, act(gate)
    layer_saved = lora_bytes + norms + rope + attn + mlp
    transient = 2 * (BT * I * a) + 2 * (BT * H * a)         # grads flowing through one layer
    ckpt_inputs = L * BT * H * a
    final_hidden = BT * H * a
    if ckpt:
        peak_stack = ckpt_inputs + layer_saved + transient + final_hidden
    else:
        peak_stack = L * layer_saved + transient + final_hidden
    return {
        "layer_saved": layer_saved, "layer_transient": transient, "ckpt_inputs": ckpt_inputs,
        "final_hidden": final_hidden, "peak_stack": peak_stack, "no_ckpt_stack": L * layer_saved + transient,
        "formula": ("ckpt: L*B*T*H*2 (layer inputs) + one layer recomputed: "
                    "B*T*[sum(LoRA in-widths)*(2+1) + 7*r*4 + 2H*2 + (q+kv)*2 + 4*q*2 + 3I*2] "
                    "+ transient B*T*(2I+2H)*2"),
    }


def logits_mem(cfg, B, T, fwd_bytes, bwd_bytes):
    V = int(cfg["vocab_size"])
    elems = B * T * V
    return {"elements": elems, "fwd_peak": elems * fwd_bytes, "bwd_peak": elems * bwd_bytes,
            "fp16_only": elems * 2,
            "formula": f"B*T*V = {B}*{T}*{V}; fwd {fwd_bytes} B/elem (fp16 out + fp32 upcast + log_softmax), "
                       f"bwd {bwd_bytes} B/elem (+grad logp + grad logits)"}


def gib(x):
    return x / GiB


def estimate(a):
    cfg, cfg_src = load_config(a.model, a.config_json)
    H, I, L, nh, nkv, hd, V = dims(cfg)
    targets = [t.strip() for t in a.target_modules.split(",") if t.strip()]
    pc = param_counts(cfg)
    lora_per_type, lora_total, n_lora_modules = lora_counts(cfg, a.lora_r, targets)

    weights = pc["total"] * a.weight_bytes
    lora_states = lora_total * (a.adapter_bytes + a.adapter_bytes + a.optim_state_bytes)
    ctx = a.cuda_context_gib * GiB

    # sequence-length scenarios
    rep = None
    if a.data_report and Path(a.data_report).exists():
        rep = json.loads(Path(a.data_report).read_text())
    scen = {"config_max_length": a.seq_len}
    if rep:
        st = (rep.get("token_stats") or {}).get("train_total") or {}
        if st.get("max"):
            scen["data_max_row"] = int(min(st["max"], a.seq_len))
        if st.get("p50"):
            scen["data_median_row"] = int(st["p50"])

    train = {}
    for name, T in scen.items():
        act = activations(cfg, a.batch_size, T, a.lora_r, len(targets), a.lora_dropout, a.grad_ckpt)
        lg = logits_mem(cfg, a.batch_size, T, a.logits_bytes_fwd, a.logits_bytes_bwd)
        at_loss = weights + lora_states + act["ckpt_inputs"] + act["final_hidden"] + lg["bwd_peak"]
        at_layer = weights + lora_states + act["peak_stack"]
        alloc = max(at_loss, at_layer)
        reserved = alloc * (1 + a.frag_frac)
        train[name] = {
            "seq_len": T, "batch_size": a.batch_size,
            "weights_gib": gib(weights), "lora_states_gib": gib(lora_states),
            "ckpt_inputs_gib": gib(act["ckpt_inputs"]), "layer_recompute_gib": gib(act["layer_saved"] + act["layer_transient"]),
            "logits_fwd_gib": gib(lg["fwd_peak"]), "logits_bwd_gib": gib(lg["bwd_peak"]),
            "stack_without_ckpt_gib": gib(act["no_ckpt_stack"]),
            "peak_at_loss_gib": gib(at_loss), "peak_at_layer_recompute_gib": gib(at_layer),
            "peak_component": "logits/loss" if at_loss >= at_layer else "layer recompute",
            "est_torch_allocated_gib": gib(alloc), "est_torch_reserved_gib": gib(reserved),
            "est_nvidia_smi_gib": gib(reserved + ctx),
            "fits_gpu": gib(reserved + ctx) <= a.gpu_gib,
            "logits_formula": lg["formula"],
        }

    # merge
    largest_mod = max(i * o for i, o in module_shapes(cfg).values())
    adapter_fp32 = lora_total * 4
    merge = {
        "soup_merge_device": "cpu (device_map='cpu' in Soup v0.75 and v0.77.2 source; confirm in run log)",
        "vram_cpu_merge_gib": 0.0,
        "ram_low_gib": gib(weights + adapter_fp32 + largest_mod * 4),
        "ram_high_gib": gib(2 * weights + adapter_fp32 + largest_mod * 4),
        "ram_formula": "fp16 base + fp32 adapter + one fp32 delta (largest module); high = + one extra copy while serialising",
        "vram_if_gpu_merge_gib": gib(weights + adapter_fp32 + largest_mod * 4) + a.cuda_context_gib,
        "disk_merged_fp16_gib": gib(weights),
    }

    # 4-bit (bitsandbytes NF4, quantize-on-load of the merged fp16 checkpoint)
    lin = pc["linear_weight_params_total"]
    bs = a.quant_blocksize
    absmax = (lin / bs * 1 + lin / (bs * 256) * 4) if a.double_quant else (lin / bs * 4)
    non_q = pc["non_linear_params_total"] * 2 + (pc["lm_head_untied"] * 2)
    q_final = lin * 0.5 + absmax + non_q
    largest_tensor = max(largest_mod, pc["embed"]) * 2
    quant = {
        "linear_params_quantized": lin,
        "nf4_weights_gib": gib(lin * 0.5), "absmax_gib": gib(absmax),
        "fp16_unquantized_gib": gib(non_q),
        "final_model_gib": gib(q_final),
        "est_peak_alloc_gib": gib(q_final + largest_tensor),
        "est_nvidia_smi_gib": gib((q_final + largest_tensor) * (1 + a.frag_frac)) + a.cuda_context_gib,
        "formula": (f"linear*0.5 B + absmax ({'double-quant: 1 B per block of ' + str(bs) + ' + fp32 per 256 blocks' if a.double_quant else 'fp32 per block'}) "
                    "+ fp16 embeddings/norms/biases (+ untied lm_head); peak = final + largest fp16 tensor in flight"),
        "soup_4bit_path": "Soup `merge --save-format 4bit`: CPU merge -> fp16 staged on disk -> reload with BitsAndBytesConfig on GPU -> save",
        "gguf_path": "llama.cpp convert + quantize run on CPU: VRAM ~0, RAM ~ fp16 weights + output",
    }

    # inference + kit verification pass
    T = a.seq_len
    kv = 2 * L * nkv * hd * T * 2
    infer = {
        "fp16_weights_gib": gib(weights),
        "kv_cache_per_1k_tokens_gib": gib(2 * L * nkv * hd * 1024 * 2),
        "teacher_forced_logits_gib": gib(T * V * 2 + 2 * T * V * 4),
        "fp16_generate_peak_gib": gib(weights + kv + V * 4) + a.cuda_context_gib,
        "4bit_generate_peak_gib": gib(q_final + kv + V * 4) + a.cuda_context_gib,
    }
    n_views = 6
    verify = {
        "models_gib": gib(weights * 2 + 2 * adapter_fp32 + 2 * q_final),
        "kl_storage_gib": gib(n_views * a.kl_positions * V * 4),
        "transient_gib": gib(T * V * 2 + 2 * T * V * 4),
    }
    verify["est_peak_alloc_gib"] = sum(verify.values())
    verify["note"] = "verify_trained.py keeps base(+2 adapters), merged, merged_4bit, base_4bit resident at once"

    out = {
        "inputs": vars(a), "config_source": cfg_src,
        "config": {k: cfg.get(k) for k in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                                           "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                                           "tie_word_embeddings", "torch_dtype")},
        "params": pc,
        "lora": {"per_type": lora_per_type, "total_params": lora_total, "n_modules": n_lora_modules,
                 "frac_of_base": lora_total / pc["total"], "fp32_weights_gib": gib(lora_total * 4),
                 "weights+grads+adam_gib": gib(lora_states),
                 "scale_alpha_over_r": a.lora_alpha / a.lora_r},
        "train": train, "merge": merge, "quantize_4bit": quant, "inference": infer, "verify": verify,
        "cuda_context_gib": a.cuda_context_gib, "frag_frac": a.frag_frac, "gpu_gib": a.gpu_gib,
    }
    return out


def fmt_table(est):
    pc = est["params"]
    lines = []
    lines.append(f"Model config from: {est['config_source']}")
    lines.append(f"Params total {pc['total']:,} (embed {pc['embed']:,}, layers {pc['layers_total']:,}, "
                 f"tied={pc['tied_embeddings']}, qkv_bias={pc['qkv_bias']})")
    lo = est["lora"]
    lines.append(f"LoRA r={est['inputs']['lora_r']} alpha={est['inputs']['lora_alpha']} scale={lo['scale_alpha_over_r']:.3g}: "
                 f"{lo['total_params']:,} params in {lo['n_modules']} modules ({100*lo['frac_of_base']:.2f}% of base)")
    for t, v in lo["per_type"].items():
        if "total" in v:
            lines.append(f"   {t:<10} in={v['in']:<5} out={v['out']:<5} r*(in+out)={v['per_module']:>8,}  x layers = {v['total']:>11,}")
    lines.append("")
    hdr = f"{'component (GiB)':<34}" + "".join(f"{k:>20}" for k in est["train"])
    lines.append("TRAINING  (" + ", ".join(f"{k}: T={v['seq_len']}" for k, v in est["train"].items()) +
                 f", B={next(iter(est['train'].values()))['batch_size']})")
    lines.append(hdr)
    for key, label in [("weights_gib", "base weights fp16"), ("lora_states_gib", "LoRA w+grad+Adam (fp32)"),
                       ("ckpt_inputs_gib", "checkpointed layer inputs"), ("layer_recompute_gib", "one layer recompute"),
                       ("logits_fwd_gib", "logits fwd (10 B/elem)"), ("logits_bwd_gib", "logits bwd (14 B/elem)"),
                       ("stack_without_ckpt_gib", "(stack if NO grad ckpt)"),
                       ("est_torch_allocated_gib", "= est. torch max_allocated"),
                       ("est_torch_reserved_gib", "  + allocator cache"),
                       ("est_nvidia_smi_gib", "  + CUDA context = nvidia-smi")]:
        lines.append(f"{label:<34}" + "".join(f"{v[key]:>20.2f}" for v in est["train"].values()))
    lines.append(f"{'peak set by':<34}" + "".join(f"{v['peak_component']:>20}" for v in est["train"].values()))
    lines.append(f"{'fits ' + str(est['gpu_gib']) + ' GiB card':<34}" + "".join(f"{str(v['fits_gpu']):>20}" for v in est["train"].values()))
    lines.append("")
    m = est["merge"]
    lines.append(f"MERGE     {m['soup_merge_device']}")
    lines.append(f"   VRAM (cpu merge) {m['vram_cpu_merge_gib']:.2f}   RAM {m['ram_low_gib']:.2f}-{m['ram_high_gib']:.2f}   "
                 f"VRAM if merged on GPU {m['vram_if_gpu_merge_gib']:.2f}   disk {m['disk_merged_fp16_gib']:.2f}")
    q = est["quantize_4bit"]
    lines.append(f"4-BIT     final model {q['final_model_gib']:.2f} (nf4 {q['nf4_weights_gib']:.2f} + absmax {q['absmax_gib']:.3f} "
                 f"+ fp16 rest {q['fp16_unquantized_gib']:.2f});  peak alloc {q['est_peak_alloc_gib']:.2f};  nvidia-smi {q['est_nvidia_smi_gib']:.2f}")
    i = est["inference"]
    lines.append(f"INFERENCE fp16 generate {i['fp16_generate_peak_gib']:.2f}, 4-bit generate {i['4bit_generate_peak_gib']:.2f}, "
                 f"KV/1k tokens {i['kv_cache_per_1k_tokens_gib']:.3f}")
    v = est["verify"]
    lines.append(f"VERIFY    models {v['models_gib']:.2f} + KL store {v['kl_storage_gib']:.2f} + transient {v['transient_gib']:.2f} "
                 f"= {v['est_peak_alloc_gib']:.2f} GiB alloc")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- compare
STAGE_MAP = {
    # measured stage -> (estimate section, key for allocated, key for nvidia-smi)
    "train": ("train", "est_torch_allocated_gib", "est_nvidia_smi_gib"),
    "merge": ("merge", "vram_cpu_merge_gib", None),
    "quantize_4bit": ("quantize_4bit", "est_peak_alloc_gib", "est_nvidia_smi_gib"),
    "verify": ("verify", "est_peak_alloc_gib", None),
}


def compare(measured_path, estimate_path, out_prefix):
    meas = json.loads(Path(measured_path).read_text())
    est = json.loads(Path(estimate_path).read_text())
    rows = []
    for stage, m in meas.items():
        if stage not in STAGE_MAP:
            continue
        sec, k_alloc, k_smi = STAGE_MAP[stage]
        e = est.get(sec, {})
        if sec == "train":
            scen = "data_max_row" if "data_max_row" in e else "config_max_length"
            e = e.get(scen, {})
            label = f"train ({scen}, T={e.get('seq_len')})"
        else:
            label = stage
        ea = e.get(k_alloc)
        es = e.get(k_smi) if k_smi else (ea + est.get("cuda_context_gib", 0) if ea is not None else None)
        ma, mr = m.get("torch_max_allocated_gib"), m.get("torch_max_reserved_gib")
        smi = m.get("nvsmi_max_used_gib")
        base = m.get("nvsmi_baseline_gib") or 0.0
        smi_delta = (smi - base) if smi is not None else None
        row = {"stage": label, "est_alloc_gib": ea, "est_nvsmi_gib": es, "meas_alloc_gib": ma, "meas_reserved_gib": mr,
               "meas_nvsmi_gib": smi, "meas_nvsmi_minus_baseline_gib": smi_delta, "rss_gib": m.get("ru_maxrss_gib"),
               "source": m.get("source")}
        if ea is not None and ma is not None:
            row["gap_alloc_gib"] = ma - ea
            row["gap_alloc_pct"] = 100 * (ma - ea) / ea if ea else None
        if ma is not None and mr is not None:
            row["allocator_cache_gib"] = mr - ma
        if mr is not None and smi_delta is not None:
            row["outside_torch_gib"] = smi_delta - mr
        hints = []
        if sec == "merge":
            lo, hi = e.get("ram_low_gib"), e.get("ram_high_gib")
            row["est_ram_gib"] = [lo, hi]
            rss = m.get("ru_maxrss_gib")
            if rss is not None and lo is not None:
                where = "inside" if lo <= rss <= hi else ("below" if rss < lo else "above")
                hints.append(f"peak RSS {rss:.2f} GiB is {where} the estimated RAM range {lo:.2f}-{hi:.2f} GiB "
                             "(merge cost is host RAM when Soup merges on CPU)")
        if row.get("gap_alloc_pct") is not None:
            g = row["gap_alloc_pct"]
            if g < -20:
                hints.append("measured << estimate: longest padded batch shorter than assumed, loss/logits implemented "
                             "more frugally (chunked/fused CE, logits_to_keep), or CUDA not used in this process")
            elif g > 20:
                hints.append("measured >> estimate: extra full-vocab copies (entropy/accuracy metrics on logits, "
                             "shift_logits .contiguous()), fp32 activations, eval/generation inside training, or no grad ckpt")
        if row.get("allocator_cache_gib", 0) and ma and row["allocator_cache_gib"] > 0.15 * ma:
            hints.append("reserved >> allocated: caching allocator kept freed blocks (fragmentation from variable-length batches)")
        if row.get("outside_torch_gib") is not None and row["outside_torch_gib"] > 0.25:
            hints.append("nvidia-smi above torch reserved: CUDA context, cuBLAS/cuDNN workspaces, other processes on the GPU")
        if ma in (None, 0) and smi_delta:
            hints.append("no torch peak from this process: GPU work happened in a child process or not at all; trust nvidia-smi")
        row["hints"] = hints
        rows.append(row)

    def f(x, w=9):
        return f"{x:>{w}.2f}" if isinstance(x, (int, float)) and x is not None else f"{'-':>{w}}"

    head = (f"{'stage':<34}{'est alloc':>10}{'meas alloc':>11}{'gap %':>8}{'reserved':>10}{'est smi':>9}"
            f"{'smi max':>9}{'smi-base':>9}{'RSS':>7}")
    lines = ["Memory: estimate vs measured (GiB)", head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['stage']:<34}{f(r['est_alloc_gib'],10)}{f(r['meas_alloc_gib'],11)}{f(r.get('gap_alloc_pct'),8)}"
                     f"{f(r['meas_reserved_gib'],10)}{f(r['est_nvsmi_gib'])}{f(r['meas_nvsmi_gib'])}"
                     f"{f(r['meas_nvsmi_minus_baseline_gib'])}{f(r['rss_gib'],7)}")
        for h in r["hints"]:
            lines.append(f"    - {h}")
    lines.append("")
    lines.append("nvidia-smi is sampled (default every 1 s) and can miss sub-second spikes; torch peaks are exact for the "
                 "process that recorded them. 'smi-base' subtracts memory already in use when the stage started.")
    text = "\n".join(lines)
    Path(out_prefix + ".json").write_text(json.dumps(rows, indent=2))
    md = ["| stage | est alloc | meas alloc | gap % | reserved | est nvidia-smi | nvidia-smi max | smi - baseline | RSS |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join([r["stage"]] + [f(r[k]).strip() for k in (
            "est_alloc_gib", "meas_alloc_gib")] + [f(r.get("gap_alloc_pct")).strip()] + [f(r[k]).strip() for k in (
            "meas_reserved_gib", "est_nvsmi_gib", "meas_nvsmi_gib", "meas_nvsmi_minus_baseline_gib", "rss_gib")]) + " |")
    Path(out_prefix + ".md").write_text("\n".join(md) + "\n\n" + "\n".join(
        f"- **{r['stage']}**: " + "; ".join(r["hints"]) for r in rows if r["hints"]) + "\n")
    print(text)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config-json", default=None, help="local config.json (skips the download)")
    ap.add_argument("--batch-size", type=int, default=2, help="per-device micro-batch")
    ap.add_argument("--grad-accum", type=int, default=8, help="informational: no effect on peak memory")
    ap.add_argument("--seq-len", type=int, default=1024, help="data.max_length (padding ceiling)")
    ap.add_argument("--data-report", default=None, help="data/data_report.json with token_stats")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=",".join(LORA_TYPES))
    ap.add_argument("--weight-bytes", type=float, default=2.0, help="frozen base bytes/param (fp16)")
    ap.add_argument("--adapter-bytes", type=float, default=4.0, help="trainable bytes/param (fp32 under fp16 AMP)")
    ap.add_argument("--optim-state-bytes", type=float, default=8.0, help="AdamW exp_avg + exp_avg_sq, fp32")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--logits-bytes-fwd", type=float, default=10.0)
    ap.add_argument("--logits-bytes-bwd", type=float, default=14.0)
    ap.add_argument("--cuda-context-gib", type=float, default=0.40, help="assumed; measure: nvidia-smi after torch.cuda.init()")
    ap.add_argument("--frag-frac", type=float, default=0.10, help="assumed reserved/allocated overhead")
    ap.add_argument("--gpu-gib", type=float, default=15360 / 1024, help="T4 reports 15360 MiB total")
    ap.add_argument("--quant-blocksize", type=int, default=64)
    ap.add_argument("--double-quant", dest="double_quant", action="store_true", default=True)
    ap.add_argument("--no-double-quant", dest="double_quant", action="store_false")
    ap.add_argument("--kl-positions", type=int, default=256, help="verify_trained.py --kl-max-positions")
    ap.add_argument("--out", default="outputs/memory_budget.json")
    ap.add_argument("--compare", default=None, help="measured JSON (compare mode)")
    ap.add_argument("--estimate", default=None, help="estimate JSON (compare mode)")
    ap.add_argument("--compare-out", default="outputs/memory_compare")
    a = ap.parse_args(argv)

    if a.compare:
        est_path = a.estimate or a.out
        compare(a.compare, est_path, a.compare_out)
        return 0

    est = estimate(a)
    text = fmt_table(est)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(est, indent=2))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"\nwritten: {a.out} and {Path(a.out).with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
