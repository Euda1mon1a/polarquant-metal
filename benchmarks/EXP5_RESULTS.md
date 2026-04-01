# Experiment 5: Hub Token Protection in Sparse V

**Date:** 2026-03-31 17:26:09  
**Config:** B=1, heads=8/2 (GQA 4:1), D=128, bits=3  
**Context:** L_kv=16,384, L_q=1 (decode)  
**Hub positions:** 50 (system prompt tokens)  
**Device:** Device(gpu, 0), Metal=True  

## Hypothesis

Some token positions are "hub tokens" -- critical across ALL heads simultaneously (system prompt tokens, key instructions, conversation anchors). Per-head entropy-guided thresholds (Phase 2a) might prune these hub positions on individual heads even though they carry disproportionate global importance. Hub protection ensures these positions always pass the sparse V threshold check.

## Strategy Comparison

### Hub Attention Pattern

Per-head entropy: H0=0.487 H1=0.482 H2=0.486 H3=0.480 H4=0.486 H5=0.487 H6=0.485 H7=0.487 

| Strategy | Time (ms) | Cos Sim | Skip % | Hub Skip % | Non-Hub Skip % | Speedup |
|:---------|----------:|--------:|-------:|----------:|---------------:|--------:|
| Baseline (t=0) | 4.52 | 0.983927 | 0.0% | 0.0% | 0.0% | 1.00x |
| Fixed (t=0.01) | 1.41 | 0.983319 | 99.1% | 87.0% | 99.8% | 3.22x |
| Entropy-guided | 1.34 | 0.983806 | 99.0% | 86.9% | 99.7% | 3.37x |
| Hub-protected | 2.05 | 0.976078 | 94.7% | 0.0% | 99.7% | 2.21x |

**Per-head cosine similarity vs FP16:**

| Head | Entropy | Threshold | Baseline | Fixed | Entropy | Hub-Protected |
|-----:|--------:|----------:|---------:|------:|--------:|--------------:|
| H0 | 0.4873 | 0.00532 | 0.983097 | 0.982624 | 0.982510 | 0.973053 |
| H1 | 0.4818 | 0.00545 | 0.984640 | 0.983768 | 0.984295 | 0.979157 |
| H2 | 0.4858 | 0.00535 | 0.985302 | 0.984981 | 0.985484 | 0.979374 |
| H3 | 0.4805 | 0.00549 | 0.979381 | 0.979602 | 0.979967 | 0.971900 |
| H4 | 0.4861 | 0.00535 | 0.982228 | 0.981135 | 0.981880 | 0.971518 |
| H5 | 0.4866 | 0.00533 | 0.981938 | 0.980613 | 0.981528 | 0.973717 |
| H6 | 0.4848 | 0.00538 | 0.987061 | 0.986756 | 0.987104 | 0.979250 |
| H7 | 0.4866 | 0.00534 | 0.986948 | 0.986131 | 0.986740 | 0.979502 |

### Varied Entropy + Hubs

Per-head entropy: H0=0.406 H1=0.414 H2=0.416 H3=0.407 H4=0.764 H5=0.763 H6=0.762 H7=0.763 

| Strategy | Time (ms) | Cos Sim | Skip % | Hub Skip % | Non-Hub Skip % | Speedup |
|:---------|----------:|--------:|-------:|----------:|---------------:|--------:|
| Baseline (t=0) | 4.73 | 0.983268 | 0.0% | 0.0% | 0.0% | 1.00x |
| Fixed (t=0.01) | 1.41 | 0.977380 | 98.8% | 88.0% | 99.3% | 3.36x |
| Entropy-guided | 1.74 | 0.983205 | 93.5% | 76.0% | 94.4% | 2.71x |
| Hub-protected | 2.18 | 0.979223 | 89.7% | 0.0% | 94.4% | 2.17x |

**Per-head cosine similarity vs FP16:**

| Head | Entropy | Threshold | Baseline | Fixed | Entropy | Hub-Protected |
|-----:|--------:|----------:|---------:|------:|--------:|--------------:|
| H0 | 0.4057 | 0.00720 | 0.979232 | 0.979037 | 0.979195 | 0.973967 |
| H1 | 0.4135 | 0.00704 | 0.983709 | 0.983974 | 0.983833 | 0.982287 |
| H2 | 0.4163 | 0.00698 | 0.981922 | 0.981846 | 0.981942 | 0.974891 |
| H3 | 0.4070 | 0.00717 | 0.986892 | 0.986580 | 0.986600 | 0.983766 |
| H4 | 0.7636 | 0.00067 | 0.982629 | 0.665294 | 0.982630 | 0.981647 |
| H5 | 0.7635 | 0.00067 | 0.978047 | 0.644917 | 0.978047 | 0.976296 |
| H6 | 0.7618 | 0.00068 | 0.985718 | 0.794070 | 0.985718 | 0.984674 |
| H7 | 0.7625 | 0.00068 | 0.982574 | 0.669100 | 0.982574 | 0.981519 |

## Hub Fraction Sweep

How much of the KV cache should be protected as hub tokens?

Baseline (no pruning) cos sim: 0.984223  
Entropy-guided (no hub prot) cos sim: 0.984035  

| Protected % | N Positions | Cos Sim | Delta vs Entropy | Skip Rate | Time (ms) |
|-----------:|:-----------:|--------:|----------------:|----------:|----------:|
| 1% | 163 | 0.983088 | -0.000948 | 99.0% | 1.71 |
| 2% | 327 | 0.982398 | -0.001638 | 99.0% | 1.75 |
| 5% | 819 | 0.979143 | -0.004892 | 99.0% | 1.93 |
| 10% | 1638 | 0.972897 | -0.011139 | 99.0% | 2.20 |

## Hub Stability Across Queries

Mean pairwise Jaccard overlap of hub masks across 5 queries: **0.0245**

Hub positions are **unstable** across queries. Hub identification would need per-query computation, making it less practical for production.

## Analysis

### Hub Attention Pattern

- Entropy-guided cos sim: 0.983806, hub skip: 86.9%
- Hub-protected cos sim: 0.976078, hub skip: 0.0% (by design)
- Quality delta: -0.007728
- Hub protection shows no quality benefit in this pattern

### Varied Entropy

- Entropy-guided cos sim: 0.983205, hub skip: 76.0%
- Hub-protected cos sim: 0.979223, hub skip: 0.0% (by design)
- Quality delta: -0.003983
- Hub protection shows no quality benefit in this pattern

## Conclusion

### Verdict: NEGATIVE

Hub token protection via wn boosting **degrades quality** rather than improving it.
Every test shows hub-protected cosine similarity *lower* than entropy-guided:

- Hub pattern: 0.976 vs 0.984 (delta = -0.008)
- Varied entropy: 0.979 vs 0.983 (delta = -0.004)
- Fraction sweep: quality drops monotonically as more positions are protected

**Root causes identified:**

1. **wn boosting introduces artifacts.** Setting `|wn| = threshold + epsilon` for hub
   positions injects synthetic signal that doesn't reflect the true attention pattern.
   The kernel treats these boosted values as real attention contributions, producing
   corrupted output.

2. **Hub positions are unstable across queries.** Jaccard overlap = 0.024 (essentially
   random). Hub-ness is a property of the *query*, not the KV cache. The same KV
   positions are NOT consistently important across different decode steps.

3. **True system prompt tokens are NOT recovered.** 0% precision/recall for the known
   hub positions (0:50). The mean-attention-across-heads metric doesn't identify
   structurally important positions; it just finds positions that happen to have
   slightly above-average weight in one random query.

4. **Entropy-guided thresholds already work well.** The Phase 2a adaptive approach
   achieves 0.983-0.984 cos sim with 93-99% skip rates. There is no quality gap
   that hub protection needs to fill.

**The AAPM scale-free analogy does not transfer:** In AAPM's scheduling graph,
hub nodes have *structural* importance from the network topology (degree
distribution). In attention, "hub-ness" is ephemeral and query-dependent -- there
is no stable topology to exploit.

### Next Steps (revised)

- **Do NOT integrate hub protection into production.** The wn boosting approach is
  harmful and the identification is unreliable.
- If revisiting this idea, explore position-aware approaches that use structural
  information (e.g., always protect positions 0:N for system prompt, based on
  known prompt template length) rather than attention-based identification.
- Test whether real model attention weights show more stable hub patterns than
  synthetic weights -- real models may have learned structural biases.
- Consider per-position threshold floors as an alternative: instead of boosting
  wn values, reduce the threshold for specific position ranges.
