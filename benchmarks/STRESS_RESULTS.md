# PolarQuant Metal -- Long-Context Stress Test Results

**Date:** 2026-03-31 13:39:01  
**Config:** B=1, heads=32/8 (GQA 4:1), D=128, bits=3  
**Mode:** Decode (L_q=1)  
**Trials:** 10 (median), 3 warmup  
**Device:** Device(gpu, 0), Metal=True  

## Performance vs Context Length

| Context | FP16 (ms) | Naive (ms) | Fused (ms) | Fused/FP16 | Fused/Naive | Cos Sim | FP16 (MB) | PQ-3bit (MB) | Compression |
|--------:|----------:|-----------:|-----------:|----------:|------------:|--------:|----------:|-------------:|------------:|
| 4,096 | 0.76 | 2.40 | 1.34 | 0.57x | 1.79x | 0.965412 | 16.0 | 3.5 | 4.6x |
| 8,192 | 1.27 | 4.76 | 2.63 | 0.48x | 1.81x | 0.968337 | 32.0 | 7.0 | 4.6x |
| 12,288 | 1.85 | 7.29 | 4.43 | 0.42x | 1.65x | 0.967314 | 48.0 | 10.5 | 4.6x |
| 16,384 | 2.39 | 9.42 | 5.80 | 0.41x | 1.63x | 0.964822 | 64.0 | 14.0 | 4.6x |
| 24,576 | 3.47 | 14.66 | 8.70 | 0.40x | 1.68x | 0.966669 | 96.0 | 21.0 | 4.6x |
| 32,768 | 4.85 | 18.52 | 11.67 | 0.42x | 1.59x | 0.965726 | 128.0 | 28.0 | 4.6x |

## Per-Operation Breakdown (Fused Pipeline)

| Context | Q Rot (ms) | QK Kernel (ms) | Softmax (ms) | SV Kernel (ms) | Out Rot (ms) | Pipeline (ms) |
|--------:|-----------:|---------------:|-------------:|---------------:|-------------:|--------------:|
| 4,096 | 0.098 | 0.224 | 0.099 | 1.166 | 0.095 | 1.337 |
| 8,192 | 0.092 | 0.335 | 0.095 | 2.283 | 0.100 | 2.732 |
| 12,288 | 0.129 | 0.477 | 0.136 | 3.859 | 0.130 | 4.300 |
| 16,384 | 0.116 | 0.584 | 0.143 | 5.342 | 0.127 | 5.880 |
| 24,576 | 0.137 | 0.800 | 0.150 | 7.950 | 0.132 | 8.665 |
| 32,768 | 0.115 | 1.058 | 0.149 | 10.620 | 0.126 | 11.614 |

## Sparse V Threshold Impact (16K Context)

| Threshold | Time (ms) | Cos Sim vs FP16 | Notes |
|----------:|----------:|----------------:|:------|
| 0.0 (disabled) | 5.71 | 0.964822 | baseline |
| 0.01 | 1.88 | 0.306939 | 3.03x vs disabled |
| 0.05 | 1.88 | 0.000000 | 3.04x vs disabled |

## Analysis

- **Scaling (4,096 to 32,768):** context grows 8x, fused time grows 8.7x, FP16 time grows 6.3x
- **Best fused/naive speedup:** 1.81x at 8,192 context
- **Memory saved at 32,768:** 100.0 MB (4.6x compression)
- **Worst-case cosine similarity:** 0.964822
- **WARNING:** Cosine similarity dropped below 0.99 at long contexts -- may need investigation
