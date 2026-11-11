# Numbers for the report (auto-collected from run outputs)

Generated 2026-11-11T18:26:52.701+00:00 from files under /content/soup-task. Copy values into report_template.md; anything marked NOT AVAILABLE did not run or failed: say so in the report.

## Setup
- soup version: 0.77.2; train backend used: soup
- versions (verify): {'python': '3.13.16', 'torch': '2.14.1+cu130', 'transformers': '5.22.1', 'peft': '0.21.2', 'cuda': '13.0', 'device': 'Tesla T4', 'bitsandbytes': '0.50.2'}
- dataset d0rj/alpaca-cleaned-ru@05321f66bb70be21bd1ea2cb8fe3ca18693a688a: raw 51760, after filters 48740, train 1500, eval 200
- drop reasons: {'code_block': 1602, 'exact_dup_user': 30, 'output_too_short': 63, 'low_cyrillic': 960, 'exact_dup_output': 365}
- leakage audit: {'eval_user_in_train': 0, 'eval_answer_in_train': 0, 'train_eval_source_index_overlap': 0}
- train token length (total): {'n': 1500, 'min': 48, 'p50': 230.5, 'p90': 611.0, 'p95': 690.05, 'p99': 877.02, 'max': 994, 'mean': 288.54533333333336}

## Memory
- estimate train [config_max_length] T=1024: alloc 7.38 GiB, nvidia-smi 8.23 GiB, logits bwd 4.06 GiB, peak set by logits/loss
- estimate train [data_max_row] T=994: alloc 7.25 GiB, nvidia-smi 8.09 GiB, logits bwd 3.94 GiB, peak set by logits/loss
- estimate train [data_median_row] T=230: alloc 4.10 GiB, nvidia-smi 4.62 GiB, logits bwd 0.91 GiB, peak set by logits/loss
- estimate 4-bit model 1.06 GiB; merge RAM 3.00-5.87 GiB
- measured train (data_max_row, T=994): torch alloc 6.72, reserved 14.38, nvidia-smi max 14.52, gap -7%  hints: ['reserved >> allocated: caching allocator kept freed blocks (fragmentation from variable-length batches)']
- measured merge: torch alloc 0.07, reserved 0.07, nvidia-smi max 0.17, gap <<NOT AVAILABLE: missing in run outputs>>%  hints: ['peak RSS 6.60 GiB is above the estimated RAM range 3.00-5.87 GiB (merge cost is host RAM when Soup merges on CPU)']
- measured quantize_4bit: torch alloc 1.10, reserved 1.14, nvidia-smi max 1.26, gap -27%  hints: ['measured << estimate: longest padded batch shorter than assumed, loss/logits implemented more frugally (chunked/fused CE, logits_to_keep), or CUDA not used in this process']
- measured verify: torch alloc 6.97, reserved 7.32, nvidia-smi max 12.01, gap -33%  hints: ['measured << estimate: longest padded batch shorter than assumed, loss/logits implemented more frugally (chunked/fused CE, logits_to_keep), or CUDA not used in this process', 'nvidia-smi above torch reserved: CUDA context, cuBLAS/cuDNN workspaces, other processes on the GPU']

## Proof of training (verify_trained.py)
- NLL base: 1.3034 (ci95 [1.2560073880027665, 1.354986021297779]), ppl 3.682, nonfinite 0
- NLL adapter: 1.2195 (ci95 [1.1725063989956523, 1.2698175341833076]), ppl 3.385, nonfinite 0
- NLL control: 1.3034 (ci95 [1.2560132029372542, 1.354997370470925]), ppl 3.682, nonfinite 0
- NLL merged: 1.2195 (ci95 [1.17251141787456, 1.2698257713762586]), ppl 3.385, nonfinite 0
- NLL merged_4bit: 1.2827 (ci95 [1.2342258734190386, 1.335847252090558]), ppl 3.606, nonfinite 0
- NLL base_4bit: 1.3564 (ci95 [1.3072240965521378, 1.409842276957374]), ppl 3.882, nonfinite 0
- diff base-merged: +0.0840 ci95 [0.07603450159630983, 0.09203022787970658]
- diff base-adapter: +0.0839 ci95 [0.07604765955583798, 0.09201604266204369]
- diff base-control: -0.0000 ci95 [-1.1693901722848047e-05, 6.8885025895953635e-06]
- diff base_4bit-merged_4bit: +0.0737 ci95 [0.06605678173724142, 0.08154643871104529]
- diff merged-merged_4bit: -0.0632 ci95 [-0.06947352731500149, -0.057379821132412204]
- diff base-merged_4bit: +0.0207 ci95 [0.010392027370805705, 0.031596281139971344]
- diff base-base_4bit: -0.0530 ci95 [-0.05889312442475144, -0.04745420035192538]
- KL merged||base: 0.12477, top-1 0.8633
- KL adapter||merged: 0.00001, top-1 0.9985
- KL merged||merged_4bit: 0.07565, top-1 0.8906
- KL base||base_4bit: 0.07875, top-1 0.8864
- KL merged_4bit||base_4bit: 0.14064, top-1 0.8507
- KL control||base: 0.00000, top-1 0.9999
- adapter B norms: {'min': 0.10316880792379379, 'median': 0.29907986521720886, 'max': 0.8570602536201477, 'n': 196}; control: {'min': 8.772655490929537e-08, 'median': 2.5716214224758005e-07, 'max': 6.310819458121841e-07, 'n': 196}
- adapter rel dW: {'min': 0.004059462572393172, 'median': 0.008207520666608084, 'max': 0.012718859458886032, 'n': 196}; fp16 rounding loss: {'min': 0.016283835149435863, 'median': 0.025239163090733416, 'max': 0.05100785159357363, 'n': 196}
- 4-bit codes: {'median': 0.6480298638343811, 'mean': 0.6200011123199852}
- generalisation gap: {'eval_minus_train_nll': {'base': 0.011759514385839775, 'merged': 0.1765122022128871}, 'merged_minus_base_gap': 0.16475268782704733, 'note': 'base gap = distribution difference of the two subsets; extra gap of merged ~ memorisation'}
- generations: {'base': {'eos_rate': 0.5, 'median_new_tokens': 198.5, 'frac_cyrillic_ok': 1.0, 'max_rep4': 0.533, 'n': 12}, 'control': {'eos_rate': 0.5, 'median_new_tokens': 198.5, 'frac_cyrillic_ok': 1.0, 'max_rep4': 0.533, 'n': 12}, 'merged': {'eos_rate': 0.6666666666666666, 'median_new_tokens': 147.0, 'frac_cyrillic_ok': 1.0, 'max_rep4': 0.447, 'n': 12}, 'merged_4bit': {'eos_rate': 0.5833333333333334, 'median_new_tokens': 165.5, 'frac_cyrillic_ok': 1.0, 'max_rep4': 0.635, 'n': 12}, 'merged_identical_to_base': 0, 'merged4_identical_to_merged': 0}
- VERDICT main: {'model_trained': 'YES', 'failed_core_checks': [], 'missing_core_checks': [], 'negative_control_flagged_untrained': 'YES', 'control_checks': {'A2_control': 'FAIL', 'B1_control': 'FAIL', 'C1_control': 'FAIL', 'D2_control': 'FAIL'}, '4bit_carries_training': 'YES', '4bit_checks': {'Q1': 'PASS', 'Q2': 'PASS', 'C2': 'PASS', 'C2b': 'PASS', 'D3': 'PASS', 'E1_merged_4bit': 'WARN'}}
- VERDICT NC2 (merged := base): {'model_trained': 'NO', 'failed_core_checks': ['B1', 'B2', 'C1', 'D1', 'D2'], 'missing_core_checks': []}

## Checks table (main run)
- [PASS] A1 LoRA modules present and mapped onto base weights: {"n": 196, "expected": 196, "unmapped": 0}
- [PASS] A2 lora_B moved away from its zero init: {"frac_modules": 1.0, "B_fro": {"min": 0.10316880792379379, "median": 0.29907986521720886, "max": 0.8570602536201477, "n": 196}, "n_exact_zero": 0}
- [PASS] B4 Predicted fp16 rounding loss of the update at merge: {"median": 0.025239163090733416, "frac_elements_unchanged_median": 0.0281982421875}
- [PASS] A1_control LoRA modules present and mapped onto base weights: {"n": 196, "expected": 196, "unmapped": 0}
- [FAIL] A2_control lora_B moved away from its zero init: {"frac_modules": 0.0, "B_fro": {"min": 8.772655490929537e-08, "median": 2.5716214224758005e-07, "max": 6.310819458121841e-07, "n": 196}, "n_exact_zero": 0}
- [WARN] B4_control Predicted fp16 rounding loss of the update at merge: {"median": 1.0, "frac_elements_unchanged_median": 1.0}
- [PASS] A3 Adapter tensors PEFT loaded == tensors in the file: {"max_abs_diff": 0.0, "file_modules_not_in_model": 0, "model_modules_not_in_file": 0, "examples": []}
- [PASS] A3_control Adapter tensors PEFT loaded == tensors in the file: {"max_abs_diff": 0.0, "file_modules_not_in_model": 0, "model_modules_not_in_file": 0, "examples": []}
- [PASS] B1 Merged weights differ from base (median relative delta): {"median_rel_delta": 0.00821162525672935, "per_type": {"down_proj": {"median_rel_delta": 0.004333685133944776, "max_rel_delta": 0.005115222002878221, "n": 28}, "gate_proj": {"median_rel_delta": 0.008848481911546347, "max_rel_delta": 0.012723875579853806, "n": 28}, "k_proj": {"median_rel_delta": 0.00
- [PASS] B2 Merged == fp16(base + scale*B@A), computed independently: {"max_err_vs_reference": 0.0, "n_modules": 196, "n_meaningful": 196, "broken_modules": [], "min_exact_match_frac": 1.0}
- [PASS] B3 Non-LoRA parameters untouched by the merge: {"max_abs_diff": 0.0, "n_compared": 142, "offenders": []}
- [PASS] Q1 4-bit artifact == quantize(merged fp16): {"median": 0.0, "max": 0.0}
- [PASS] Q2 Fine-tune changed 4-bit codes (quantize(merged) != quantize(base)): {"mean_frac_weights_changed": 0.6200011123199852}
- [PASS] C1 Held-out answer NLL: base - merged (nats/token, paired bootstrap): {"point": 0.08395158760657379, "ci95": [0.07603450159630983, 0.09203022787970658], "B": 2000}
- [FAIL] C1_control Held-out NLL: base - control (negative control, expected FAIL): {"point": -2.7918773233215806e-06, "ci95": [-1.1693901722848047e-05, 6.8885025895953635e-06], "B": 2000}
- [PASS] C2 4-bit keeps the gain: NLL(base_4bit) - NLL(merged_4bit), retained fraction: {"diff": {"point": 0.0736957375461699, "ci95": [0.06605678173724142, 0.08154643871104529], "B": 2000}, "retained_frac_of_fp16_gain": 0.8778361392226869}
- [PASS] C2b Quantization cost on held-out NLL: NLL(merged_4bit) - NLL(merged): {"increase": 0.0632196417935203, "ci95_of_merged_minus_4bit": [-0.06947352731500149, -0.057379821132412204]}
- [PASS] D1 Merged behaves like base+adapter (KL, top-1): {"kl_mean": 1.3709689171292202e-05, "top1_agreement": 0.9984873441124071, "n_positions": 29749}
- [PASS] D2 Merged changed next-token distributions vs base (KL): {"kl_mean": 0.12477428773105947, "top1_agreement": 0.8632895223368853, "n_positions": 29749, "quant_noise_floor_kl_base_vs_base4": 0.0787541937464814, "ratio_to_quant_floor": 1.584351026850988}
- [FAIL] D2_control Control changed distributions vs base (expected FAIL): {"kl_mean": 6.441314671833152e-07, "top1_agreement": 0.9999327708494403, "n_positions": 29749}
- [PASS] D3 4-bit fidelity to merged fp16 (KL, top-1): {"kl_mean": 0.07565059963184355, "top1_agreement": 0.8905509428888366, "n_positions": 29749}
- [PASS] E2 No inf/NaN logits (fp16 overflow on T4): {"nonfinite_logits": {"base": 0, "adapter": 0, "control": 0, "merged": 0, "merged_4bit": 0, "base_4bit": 0}, "hidden_state_max_abs": {"base": 6652.0, "merged": 6064.0}, "fp16_max": 65504.0}
- [WARN] E1_merged Generations of merged: stop at EOS, Russian, not degenerate: {"eos_rate": 0.6666666666666666, "median_new_tokens": 147.0, "frac_cyrillic_ok": 1.0, "max_rep4": 0.447, "n": 12}
- [WARN] E1_merged_4bit Generations of merged_4bit: stop at EOS, Russian, not degenerate: {"eos_rate": 0.5833333333333334, "median_new_tokens": 165.5, "frac_cyrillic_ok": 1.0, "max_rep4": 0.635, "n": 12}
- [PASS] E3 Greedy outputs changed vs base on at least one probe: {"identical": 0, "n": 12}
- [PASS] T1 Tokenizer + chat template identical across artifacts: {"differences": {"adapter": [], "merged": [], "merged_4bit": []}, "load_errors": {}}
- [PASS] T2 generation_config.json consistent across artifacts: {"base": {"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "bos_token_id": 151643}, "merged": {"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1, "eos_token
- [FAIL] B1_control Merged weights differ from base (median relative delta): {"median_rel_delta": 0.0, "per_type": {"down_proj": {"median_rel_delta": 0.0, "max_rel_delta": 0.0, "n": 28}, "gate_proj": {"median_rel_delta": 0.0, "max_rel_delta": 0.0, "n": 28}, "k_proj": {"median_rel_delta": 0.0, "max_rel_delta": 0.0, "n": 28}, "o_proj": {"median_rel_delta": 0.0, "max_rel_delta"
- [PASS] B2_control Merged == fp16(base + scale*B@A), computed independently: {"max_err_vs_reference": 0.0, "n_modules": 196, "n_meaningful": 0, "broken_modules": [], "min_exact_match_frac": 1.0}
- [PASS] B3_control Non-LoRA parameters untouched by the merge: {"max_abs_diff": 0.0, "n_compared": 142, "offenders": []}

## Static artifact checks
- [OK] S-adapter-template adapter chat template == base: {"artifact": "cd8e9439f0570856", "base": "cd8e9439f0570856"}
- [DIFF] S-adapter-tokenizer adapter tokenizer.json == base: {"artifact": "3fd169731d2cbde9", "base": "c0382117ea329cdf"}
- [OK] S-adapter-eos-pad adapter eos/pad tokens == base: {"artifact": ["<|im_end|>", "<|endoftext|>"], "base": ["<|im_end|>", "<|endoftext|>"]}
- [OK] S-merged-template merged chat template == base: {"artifact": "cd8e9439f0570856", "base": "cd8e9439f0570856"}
- [DIFF] S-merged-tokenizer merged tokenizer.json == base: {"artifact": "3fd169731d2cbde9", "base": "c0382117ea329cdf"}
- [OK] S-merged-eos-pad merged eos/pad tokens == base: {"artifact": ["<|im_end|>", "<|endoftext|>"], "base": ["<|im_end|>", "<|endoftext|>"]}
- [OK] S-merged-genconfig merged generation_config.json == base (users get these defaults): {"artifact": {"bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "repetition_penalty": 1.1, "temperature": 0.7, "top_k": 20, "top_p": 0.8, "transformers_version": "5.22.1"}, "base": {"bos_token_id": 151643, "pad_token_id": 151643, "do_sample": true, 
- [OK] S-merged_4bit-template merged_4bit chat template == base: {"artifact": "cd8e9439f0570856", "base": "cd8e9439f0570856"}
- [DIFF] S-merged_4bit-tokenizer merged_4bit tokenizer.json == base: {"artifact": "3fd169731d2cbde9", "base": "c0382117ea329cdf"}
- [OK] S-merged_4bit-eos-pad merged_4bit eos/pad tokens == base: {"artifact": ["<|im_end|>", "<|endoftext|>"], "base": ["<|im_end|>", "<|endoftext|>"]}
- [OK] S-merged_4bit-genconfig merged_4bit generation_config.json == base (users get these defaults): {"artifact": {"bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "repetition_penalty": 1.1, "temperature": 0.7, "top_k": 20, "top_p": 0.8, "transformers_version": "5.22.1"}, "base": {"bos_token_id": 151643, "pad_token_id": 151643, "do_sample": true, 
- [OK] S-adapter-targets adapter target_modules == requested: {"adapter": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"], "requested": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
- [OK] S-adapter-rank adapter r/alpha == requested: {"r": 16, "alpha": 32}
- [OK] S-adapter-base adapter base_model_name_or_path == base we evaluate: "Qwen/Qwen2.5-1.5B-Instruct"
- [OK] S-adapter-extras no modules_to_save / DoRA surprises: {"modules_to_save": null, "use_dora": false}
- [OK] S-4bit-config 4-bit quantization_config: {"_load_in_4bit": true, "_load_in_8bit": false, "bnb_4bit_compute_dtype": "float16", "bnb_4bit_quant_storage": "uint8", "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": true, "llm_int8_enable_fp32_cpu_offload": false, "llm_int8_has_fp16_weight": false, "llm_int8_skip_modules": null, "llm_i
- [OK] S-merged-dtype merged checkpoint dtype is float16: "float16"
- [OK] S-train-steps optimizer steps ~= ceil(ceil(N/bs)/ga)*epochs, +-1 per epoch for floor/ceil (else rows were dropped): {"expected": 188, "logged_global_step": 188, "max_steps": 188}
- [OK] S-train-finite all logged loss/grad_norm finite: {"nonfinite_entries": 0, "n_logged": 37}
- training summary: {'source': 'outputs/soup_run/checkpoint-188/trainer_state.json', 'global_step': 188, 'max_steps': 188, 'n_logged': 37, 'first_loss': 1.3668246269226074, 'last_loss': 0.9926782608032226, 'min_loss': 0.9926782608032226, 'max_grad_norm': 0.7732017040252686, 'nonfinite_entries': 0, 'train_runtime': None, 'skipped_steps_fp16': None}

## soup ship
- live verdict file: {'decision': "DON'T SHIP", 'task_win': {'mode': 'metric', 'base': 0.25, 'tuned': 0.25, 'won': False}, 'benchmark_deltas': [{'name': 'mini_instruction', 'base': 1.0, 'tuned': 0.9583333333333334, 'delta': -0.041667, 'regressed': False}, {'name': 'mini_mmlu', 'base': 0.85, 'tuned': 0.825, 'delta': -0.025, 'regressed': False}], 'failed_rule': 'task_win', 'forgetting_threshold': 0.05, 'soup_version': '0.77.2', 'noise_floor': None, 'numerics': 'bfloat16'}
- offline (our NLL metric) verdict: {'decision': 'SHIP', 'task_win': {'mode': 'metric', 'base': 0.2716031032791081, 'tuned': 0.2772927389190789, 'won': True}, 'benchmark_deltas': [{'name': 'mini_instruction', 'base': 1.0, 'tuned': 0.9583333333333334, 'delta': -0.041667, 'regressed': False}, {'name': 'mini_mmlu', 'base': 0.85, 'tuned': 0.825, 'delta': -0.025, 'regressed': False}], 'failed_rule': None, 'forgetting_threshold': 0.05, 'soup_version': '0.77.2', 'noise_floor': None, 'numerics': None}
