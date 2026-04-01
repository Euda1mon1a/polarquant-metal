# Experiment 9: Erlang Queuing Model for Sparse V Workload Prediction

**Date:** 2026-03-31 18:03:56  
**Config:** B=1, heads=8/2 (GQA 4:1), D=128, bits=3  
**Context:** L_kv=16,384, L_q=1 (decode)  
**Device:** Device(gpu, 0), Metal=True  

## Hypothesis

The SV kernel iterates ALL L_kv positions to check `if (|wn_val| > threshold)` even though concentrated heads skip ~99% of positions. Attention weight distributions have predictable structure. From the top-k attention weights, we can estimate the total number of above-threshold positions using Erlang-style utilization analysis, without scanning all positions.

This enables:
1. Pre-computing expected workload per head
2. Choosing kernel dispatch strategy (skip / sparse / dense)
3. Skipping the kernel entirely when predicted workload ~ 0

## Prediction Models

### Top-k Exponential Tail
Sample top-k weights (partial sort, O(L_kv)). Fit exponential decay from w_1 (largest) to w_k (k-th). Extrapolate to find rank where weight drops below threshold.

### Erlang Utilization
Map to queuing theory: arrival_rate = mean attention weight, service_rate = threshold. Utilization rho = mean/threshold. For exponential distribution: P(w > t) = exp(-t/mean). Expected active = L_kv * P(w > threshold).

### Hybrid (Top-k + Erlang)
Use top-k to calibrate the tail decay rate, then use Erlang-style CDF extrapolation for the unseen tail.

## Distributions Tested

| Distribution | Mean Entropy | Description |
|:-------------|:-----------:|:------------|
| Concentrated | 0.1299 | ~50 positions with high weight (softmax temp=8) |
| Moderate | 0.5829 | ~500 positions with meaningful weight (softmax temp=3) |
| Spread | 0.9995 | Nearly uniform (softmax temp=0.1) |
| Power-law | 0.5279 | Zipf distribution (alpha=1.2), realistic for transformers |
| Bimodal | 0.5681 | Two Gaussian peaks at 1/3 and 2/3 of context |

## Prediction Accuracy

| Distribution | Threshold | Actual | Top-k | Err% | Erlang | Err% | Hybrid | Err% |
|:-------------|----------:|-------:|------:|----:|-------:|----:|-------:|----:|
| Concentrated | 0.0001 | 43 | 43 | 0.0% | 3183 | 7346.2% | 43 | 0.0% |
| Concentrated | 0.0005 | 22 | 22 | 0.0% | 5 | 79.5% | 22 | 0.0% |
| Concentrated | 0.0010 | 16 | 16 | 0.0% | 0 | 100.0% | 16 | 0.0% |
| Concentrated | 0.0050 | 8 | 8 | 0.0% | 0 | 100.0% | 8 | 0.0% |
| Concentrated | 0.0100 | 6 | 6 | 0.0% | 0 | 100.0% | 6 | 0.0% |
| Moderate | 0.0001 | 862 | 164 | 81.0% | 3183 | 269.4% | 165 | 80.9% |
| Moderate | 0.0005 | 252 | 125 | 50.5% | 5 | 98.2% | 126 | 50.1% |
| Moderate | 0.0010 | 136 | 106 | 22.0% | 0 | 100.0% | 107 | 21.4% |
| Moderate | 0.0050 | 24 | 24 | 0.0% | 0 | 100.0% | 24 | 0.0% |
| Moderate | 0.0100 | 11 | 11 | 0.0% | 0 | 100.0% | 11 | 0.0% |
| Spread | 0.0001 | 0 | 0 | 0.0% | 3183 | 318325.8% | 0 | 0.0% |
| Spread | 0.0005 | 0 | 0 | 0.0% | 5 | 453.6% | 0 | 0.0% |
| Spread | 0.0010 | 0 | 0 | 0.0% | 0 | 0.1% | 0 | 0.0% |
| Spread | 0.0050 | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| Spread | 0.0100 | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| Power-law | 0.0001 | 575 | 137 | 76.2% | 3183 | 453.6% | 138 | 76.1% |
| Power-law | 0.0005 | 150 | 108 | 28.1% | 5 | 97.0% | 109 | 27.5% |
| Power-law | 0.0010 | 84 | 84 | 0.0% | 0 | 100.0% | 84 | 0.0% |
| Power-law | 0.0050 | 22 | 22 | 0.0% | 0 | 100.0% | 22 | 0.0% |
| Power-law | 0.0100 | 12 | 12 | 0.0% | 0 | 100.0% | 12 | 0.0% |
| Bimodal | 0.0001 | 346 | 1197 | 245.9% | 3183 | 820.0% | 1198 | 246.1% |
| Bimodal | 0.0005 | 274 | 738 | 169.3% | 5 | 98.3% | 739 | 169.6% |
| Bimodal | 0.0010 | 234 | 540 | 130.8% | 0 | 100.0% | 541 | 131.3% |
| Bimodal | 0.0050 | 90 | 90 | 0.0% | 0 | 100.0% | 90 | 0.0% |
| Bimodal | 0.0100 | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |

## Top-k Sensitivity (threshold=0.001)

| Distribution | k=50 | k=100 | k=200 | k=500 |
|:-------------|-----:|------:|------:|------:|
| Concentrated | 0.0% | 0.0% | 0.0% | 0.0% |
| Moderate | 52.2% | 22.0% | 0.0% | 0.0% |
| Spread | 0.0% | 0.0% | 0.0% | 0.0% |
| Power-law | 33.8% | 0.0% | 0.0% | 0.0% |
| Bimodal | 395.9% | 130.8% | 16.0% | 0.0% |

## Dispatch Decision Accuracy

Dispatch policy: SKIP (<1%), SPARSE (<10%), DENSE (>50%)

| Distribution | Threshold | Oracle | Top-k | Match | Erlang | Match | Hybrid | Match |
|:-------------|----------:|:------:|:-----:|------:|:-----:|------:|:------:|------:|
| Concentrated | 0.0001 | skip | skip | 100% | sparse | 0% | skip | 100% |
| Concentrated | 0.0005 | skip | skip | 100% | skip | 100% | skip | 100% |
| Concentrated | 0.0010 | skip | skip | 100% | skip | 100% | skip | 100% |
| Concentrated | 0.0050 | skip | skip | 100% | skip | 100% | skip | 100% |
| Concentrated | 0.0100 | skip | skip | 100% | skip | 100% | skip | 100% |
| Moderate | 0.0001 | sparse | sparse | 50% | sparse | 100% | sparse | 62% |
| Moderate | 0.0005 | sparse | skip | 0% | skip | 0% | skip | 0% |
| Moderate | 0.0010 | skip | skip | 88% | skip | 88% | skip | 88% |
| Moderate | 0.0050 | skip | skip | 100% | skip | 100% | skip | 100% |
| Moderate | 0.0100 | skip | skip | 100% | skip | 100% | skip | 100% |
| Spread | 0.0001 | skip | skip | 100% | sparse | 0% | skip | 100% |
| Spread | 0.0005 | skip | skip | 100% | skip | 100% | skip | 100% |
| Spread | 0.0010 | skip | skip | 100% | skip | 100% | skip | 100% |
| Spread | 0.0050 | skip | skip | 100% | skip | 100% | skip | 100% |
| Spread | 0.0100 | skip | skip | 100% | skip | 100% | skip | 100% |
| Power-law | 0.0001 | sparse | skip | 0% | sparse | 100% | skip | 0% |
| Power-law | 0.0005 | skip | skip | 100% | skip | 100% | skip | 100% |
| Power-law | 0.0010 | skip | skip | 100% | skip | 100% | skip | 100% |
| Power-law | 0.0050 | skip | skip | 100% | skip | 100% | skip | 100% |
| Power-law | 0.0100 | skip | skip | 100% | skip | 100% | skip | 100% |
| Bimodal | 0.0001 | sparse | sparse | 100% | sparse | 100% | sparse | 100% |
| Bimodal | 0.0005 | sparse | sparse | 100% | skip | 0% | sparse | 100% |
| Bimodal | 0.0010 | sparse | sparse | 100% | skip | 0% | sparse | 100% |
| Bimodal | 0.0050 | skip | skip | 100% | skip | 100% | skip | 100% |
| Bimodal | 0.0100 | skip | skip | 100% | skip | 100% | skip | 100% |

## Prediction Cost

At L_kv=16,384, concentrated distribution, threshold=0.001:

| Method | Median (us) | Mean (us) | vs Scan |
|:-------|:-----------:|:---------:|:-------:|
| Full scan | 141.2 | 159.8 | 1.00x |
| Top-k (k=50) | 1814.2 | 1794.7 | 12.84x |
| Top-k (k=100) | 1752.0 | 1814.9 | 12.40x |
| Top-k (k=200) | 1663.1 | 1787.7 | 11.77x |
| Top-k (k=500) | 1788.6 | 1887.8 | 12.66x |
| Erlang utilization | 1097.6 | 1127.5 | 7.77x |
| Hybrid (k=100) | 1639.8 | 1744.5 | 11.61x |

## Context Length Scaling

| L_kv | Scan (us) | Top-k (us) | Erlang (us) | Scan/Top-k | Scan/Erlang |
|-----:|:---------:|:----------:|:-----------:|:---------:|:-----------:|
| 1,024 | 128.2 | 1006.7 | 945.8 | 0.13x | 0.14x |
| 4,096 | 117.5 | 1333.5 | 1006.5 | 0.09x | 0.12x |
| 8,192 | 145.5 | 1345.4 | 1049.5 | 0.11x | 0.14x |
| 16,384 | 107.8 | 1483.4 | 1035.8 | 0.07x | 0.10x |
| 32,768 | 180.6 | 1689.0 | 1104.8 | 0.11x | 0.16x |
| 65,536 | 133.5 | 2146.0 | 1041.0 | 0.06x | 0.13x |

## Summary

### Prediction Error (across all distributions and thresholds)

| Model | Mean Error | Median Error | Max Error | Cases <20% |
|:------|:---------:|:-----------:|:---------:|:---------:|
| Top-k (k=100) | 32.2% | 0.0% | 245.9% | 68% |
| Erlang utilization | 13165.7% | 100.0% | 318325.8% | 16% |
| Hybrid | 32.1% | 0.0% | 246.1% | 68% |

### Dispatch Decision Accuracy

| Model | Mean Accuracy | Min Accuracy |
|:------|:------------:|:------------:|
| Topk | 89.5% | 0% |
| Erlang | 79.5% | 0% |
| Hybrid | 90.0% | 0% |

## Analysis

**Best prediction model:** Hybrid (mean error 32.1%)

### Model Strengths and Weaknesses

- **Concentrated**: Best=Top-k (0.0% mean error)
- **Moderate**: Best=Hybrid (30.5% mean error)
- **Spread**: Best=Top-k (0.0% mean error)
- **Power-law**: Best=Hybrid (20.7% mean error)
- **Bimodal**: Best=Top-k (109.2% mean error)

### Cost-Effectiveness

Top-k (k=100) is **12.4x slower** than full scan. The overhead of prediction exceeds the cost of just running the threshold check.

Erlang utilization is **7.8x slower** than full scan.

### Scaling Behavior

Over 64x context length increase (1,024 -> 65,536):
- Full scan cost grows 1.0x
- Top-k cost grows 2.1x
- Both scale similarly

## Conclusion

### Verdict: PARTIAL

Workload prediction shows promise but is not reliable enough for production dispatch:

- 68% of predictions within 20% error (target: >70%)
- 90% dispatch accuracy (target: >80%)
- Prediction cost exceeds scan cost -- no cost benefit

The approach may be viable at longer context lengths where the scan becomes more expensive.

---
*Generated by exp9_erlang_workload.py on 2026-03-31 18:03:56*
