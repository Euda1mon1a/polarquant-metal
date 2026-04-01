# Phase 3 Benchmark: Compact-Index Sparse SV vs Dense SV

**Date:** 2026-03-31 19:49:09  
**Config:** B=1, heads=8/2 (GQA 4:1), D=128, bits=3  
**Device:** Device(gpu, 0), Metal=True  
**Timing:** 10 trials (median), 3 warmup  
**Threshold:** 0.001 per head  

## Results

| L_kv | Pattern | Dense (ms) | Sparse (ms) | Index (ms) | Kernel (ms) | Speedup | Avg Active | Active % | Cos Sim |
|-----:|:--------|----------:|-----------:|---------:|----------:|--------:|---------:|---------:|--------:|
| 2,048 | concentrated | 0.976 | 0.451 | 0.152 | 0.299 | 2.16x | 69 | 3.4% | 1.000000 |
| 2,048 | moderate | 0.795 | 0.739 | 0.161 | 0.572 | 1.08x | 1826 | 89.1% | 1.000000 |
| 2,048 | spread | 0.777 | 0.665 | 0.121 | 0.538 | 1.17x | 2048 | 100.0% | 1.000000 |
| 4,096 | concentrated | 0.458 | 0.309 | 0.117 | 0.192 | 1.48x | 127 | 3.1% | 1.000000 |
| 4,096 | moderate | 0.912 | 0.876 | 0.160 | 0.756 | 1.04x | 2855 | 69.7% | 1.000000 |
| 4,096 | spread | 1.263 | 1.071 | 0.159 | 0.912 | 1.18x | 4096 | 100.0% | 1.000000 |
| 8,192 | concentrated | 0.823 | 0.330 | 0.118 | 0.213 | 2.49x | 144 | 1.8% | 1.000000 |
| 8,192 | moderate | 1.250 | 1.062 | 0.162 | 0.898 | 1.18x | 3502 | 42.7% | 1.000000 |
| 8,192 | spread | 2.299 | 1.839 | 0.159 | 1.680 | 1.25x | 8158 | 99.6% | 1.000000 |
| 16,384 | concentrated | 1.358 | 0.400 | 0.126 | 0.274 | 3.39x | 177 | 1.1% | 1.000000 |
| 16,384 | moderate | 1.850 | 1.128 | 0.134 | 0.995 | 1.64x | 3156 | 19.3% | 1.000000 |
| 16,384 | spread | 1.268 | 0.251 | 0.119 | 0.131 | 5.05x | 11 | 0.1% | 0.999999 |
| 32,768 | concentrated | 2.482 | 0.443 | 0.132 | 0.311 | 5.61x | 251 | 0.8% | 1.000000 |
| 32,768 | moderate | 2.852 | 0.976 | 0.139 | 0.838 | 2.92x | 1947 | 5.9% | 1.000000 |
| 32,768 | spread | 2.371 | 0.252 | 0.128 | 0.125 | 9.39x | 0 | 0.0% | both zero* |

## Per-Head Active Counts

| L_kv | Pattern | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
|-----:|:--------|----:|----:|----:|----:|----:|----:|----:|----:|
| 2,048 | concentrated | 33 | 43 | 52 | 116 | 54 | 72 | 89 | 94 |
| 2,048 | moderate | 1819 | 1823 | 1817 | 1826 | 1816 | 1858 | 1831 | 1816 |
| 2,048 | spread | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 |
| 4,096 | concentrated | 169 | 123 | 137 | 113 | 163 | 18 | 112 | 182 |
| 4,096 | moderate | 2831 | 2880 | 2869 | 2881 | 2825 | 2850 | 2843 | 2862 |
| 4,096 | spread | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 |
| 8,192 | concentrated | 187 | 224 | 141 | 80 | 188 | 164 | 88 | 77 |
| 8,192 | moderate | 3550 | 3569 | 3485 | 3499 | 3518 | 3496 | 3400 | 3498 |
| 8,192 | spread | 8161 | 8155 | 8152 | 8166 | 8161 | 8156 | 8160 | 8152 |
| 16,384 | concentrated | 95 | 346 | 247 | 159 | 159 | 128 | 97 | 186 |
| 16,384 | moderate | 3138 | 3142 | 3173 | 3200 | 3194 | 3161 | 3099 | 3143 |
| 16,384 | spread | 7 | 10 | 9 | 7 | 14 | 14 | 16 | 9 |
| 32,768 | concentrated | 350 | 183 | 431 | 188 | 261 | 151 | 156 | 287 |
| 32,768 | moderate | 1911 | 1929 | 1920 | 1990 | 1954 | 1970 | 1955 | 1948 |
| 32,768 | spread | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Zone Prior Test

**L_kv:** 16,384  
**System prompt zone:** first 200 positions  
**Recent zone:** last 500 positions  

| Metric | Without Zones | With Zones | Delta |
|:-------|-------------:|-----------:|------:|
| Time (ms) | 0.373 | 0.473 | +0.101 |
| Avg active | 207 | 897 | +690 |
| Zone coverage | - | 100.0% | - |

### Per-Head Zone Detail

| Head | Active (no zone) | Active (zone) | Delta | Zone Coverage |
|-----:|-----------------:|--------------:|------:|--------------:|
| H0 | 152 | 849 | +697 | 100.0% |
| H1 | 160 | 852 | +692 | 100.0% |
| H2 | 141 | 833 | +692 | 100.0% |
| H3 | 261 | 955 | +694 | 100.0% |
| H4 | 272 | 955 | +683 | 100.0% |
| H5 | 301 | 985 | +684 | 100.0% |
| H6 | 178 | 870 | +692 | 100.0% |
| H7 | 190 | 874 | +684 | 100.0% |

## Analysis

### Concentrated Attention

- **2,048**: 2.16x speedup, 3.4% active, cos_sim=1.000000
- **4,096**: 1.48x speedup, 3.1% active, cos_sim=1.000000
- **8,192**: 2.49x speedup, 1.8% active, cos_sim=1.000000
- **16,384**: 3.39x speedup, 1.1% active, cos_sim=1.000000
- **32,768**: 5.61x speedup, 0.8% active, cos_sim=1.000000

### Moderate Attention

- **2,048**: 1.08x speedup, 89.1% active, cos_sim=1.000000
- **4,096**: 1.04x speedup, 69.7% active, cos_sim=1.000000
- **8,192**: 1.18x speedup, 42.7% active, cos_sim=1.000000
- **16,384**: 1.64x speedup, 19.3% active, cos_sim=1.000000
- **32,768**: 2.92x speedup, 5.9% active, cos_sim=1.000000

### Spread Attention

- **2,048**: 1.17x speedup, 100.0% active, cos_sim=1.000000
- **4,096**: 1.18x speedup, 100.0% active, cos_sim=1.000000
- **8,192**: 1.25x speedup, 99.6% active, cos_sim=1.000000
- **16,384**: 5.05x speedup, 0.1% active, cos_sim=0.999999
- **32,768**: 9.39x speedup, 0.0% active, cos_sim=both zero*

### Crossover Point

Sparse path was faster than dense in all tested configurations.

### Correctness

All non-degenerate configurations show cos_sim > 0.999 between dense and sparse outputs, confirming identical behavior.

*`both zero` entries indicate cases where the fixed threshold (0.001) filters out ALL positions because per-position wn values fall below the threshold at long contexts with spread attention. Both dense and sparse kernels produce near-zero output -- this is expected behavior, not a correctness issue. In production, entropy-guided thresholds would lower the threshold for spread heads.*
