| stage | est alloc | meas alloc | gap % | reserved | est nvidia-smi | nvidia-smi max | smi - baseline | RSS |
|---|---|---|---|---|---|---|---|---|
| train (data_max_row, T=994) | 7.25 | 6.72 | -7.38 | 14.38 | 8.09 | 14.52 | 14.52 | 4.26 |
| merge | 0.00 | 0.07 | - | 0.07 | 0.11 | 0.17 | 0.17 | 6.60 |
| quantize_4bit | 1.50 | 1.10 | -26.69 | 1.14 | 1.76 | 1.26 | 1.26 | 6.60 |
| verify | 10.34 | 6.97 | -32.58 | 7.32 | 10.45 | 12.01 | 12.01 | 4.49 |

- **train (data_max_row, T=994)**: reserved >> allocated: caching allocator kept freed blocks (fragmentation from variable-length batches)
- **merge**: peak RSS 6.60 GiB is above the estimated RAM range 3.00-5.87 GiB (merge cost is host RAM when Soup merges on CPU)
- **quantize_4bit**: measured << estimate: longest padded batch shorter than assumed, loss/logits implemented more frugally (chunked/fused CE, logits_to_keep), or CUDA not used in this process
- **verify**: measured << estimate: longest padded batch shorter than assumed, loss/logits implemented more frugally (chunked/fused CE, logits_to_keep), or CUDA not used in this process; nvidia-smi above torch reserved: CUDA context, cuBLAS/cuDNN workspaces, other processes on the GPU
