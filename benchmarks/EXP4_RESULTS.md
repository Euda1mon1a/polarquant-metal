# Experiment 4: Spectral Concentration for Per-Head Adaptive Bit-Width

**Date:** 2026-03-31 14:41:00  
**Config:** B=1, heads=8/2 (GQA 4:1), D=128  
**Context:** L_kv=8,192, L_q=1 (decode)  
**Bit-widths tested:** 2, 3, 4  
**Quality threshold:** cos_sim >= 0.95  
**Device:** Device(gpu, 0), Metal=True  

## Hypothesis

Different attention heads have different spectral profiles. Heads with concentrated spectral energy (periodic patterns) are predictable and tolerate cheaper 2-bit quantization. Heads with diffuse spectral energy need 3-bit or more. By analyzing the power spectral density of attention weights, we can adaptively select bit-widths per head, saving memory without sacrificing quality.

## Part 1: Individual Pattern Analysis

| Pattern | Spectral Conc. | Top-5 Conc. | Rec. Bits | 2-bit CosSim | 3-bit CosSim | 4-bit CosSim |
|:--------|---------------:|------------:|----------:|-------------:|-------------:|-------------:|
| Periodic-64 | 0.0312 | 0.1562 | 4 | 0.946335 | 0.980584 | 0.996066 |
| Periodic-128 | 0.0156 | 0.0781 | 4 | 0.939924 | 0.980511 | 0.996292 |
| Sparse-20 | 0.0018 | 0.0078 | 4 | 0.943367 | 0.985965 | 0.995303 |
| Sparse-100 | 0.0020 | 0.0084 | 4 | 0.913927 | 0.976856 | 0.993583 |
| Diffuse | 0.0020 | 0.0089 | 4 | 0.939601 | 0.982537 | 0.994633 |

### Memory per Pattern (1 KV head, L_kv=8192)

| Pattern | 2-bit (KB) | 3-bit (KB) | 4-bit (KB) | FP16 (KB) |
|:--------|----------:|-----------:|-----------:|----------:|
| Periodic-64 | 272.0 | 432.0 | 528.0 | 2048.0 |
| Periodic-128 | 272.0 | 432.0 | 528.0 | 2048.0 |
| Sparse-20 | 272.0 | 432.0 | 528.0 | 2048.0 |
| Sparse-100 | 272.0 | 432.0 | 528.0 | 2048.0 |
| Diffuse | 272.0 | 432.0 | 528.0 | 2048.0 |

## Part 2: Realistic 8-Head Adaptive Bit-Width

| Head | Name | KV Head | Spectral Conc. | Rec. Bits | CosSim Rec. | CosSim 3-bit | Delta |
|-----:|:-----|--------:|---------------:|----------:|------------:|-------------:|------:|
| H0 | periodic-32 | KV0 | 0.0625 | 4 | 0.995690 | 0.982415 | +0.0133 |
| H1 | periodic-128 | KV0 | 0.0156 | 4 | 0.996292 | 0.980511 | +0.0158 |
| H2 | sparse-10 | KV0 | 0.0011 | 4 | 0.996001 | 0.981231 | +0.0148 |
| H3 | sparse-50 | KV0 | 0.0020 | 4 | 0.996388 | 0.980759 | +0.0156 |
| H4 | decay | KV1 | 0.1003 | 3 | 0.984121 | 0.984121 | +0.0000 |
| H5 | bursty | KV1 | 0.0105 | 4 | 0.995063 | 0.983650 | +0.0114 |
| H6 | uniform | KV1 | 0.0022 | 4 | 0.995612 | 0.981405 | +0.0142 |
| H7 | uniform+recency | KV1 | 0.6079 | 2 | 0.939257 | 0.981308 | -0.0421 |

## Part 3: Memory Savings (V Cache Only)

### Context length: 8,192

KV head bit assignments: {0: 4, 1: 4}

| Strategy | Memory | Saving |
|:---------|-------:|-------:|
| Uniform 3-bit | 0.84 MB | baseline |
| Adaptive (GQA-safe) | 1.03 MB | -22.2% |
| Adaptive (per-head optimistic) | 0.95 MB | -12.0% |

### Context length: 16,384

KV head bit assignments: {0: 4, 1: 4}

| Strategy | Memory | Saving |
|:---------|-------:|-------:|
| Uniform 3-bit | 1.69 MB | baseline |
| Adaptive (GQA-safe) | 2.06 MB | -22.2% |
| Adaptive (per-head optimistic) | 1.89 MB | -12.0% |

## Part 4: Threshold Validation

Spectral concentration thresholds: high > 0.3 (2-bit), medium > 0.1 (3-bit), low (4-bit)

| Pattern | Spectral Conc. | Recommended | 2-bit CosSim | 2-bit Safe? | Prediction |
|:--------|---------------:|:------------|-------------:|:------------|:-----------|
| Periodic-64 | 0.0312 | 4-bit | 0.946335 | No | Correct |
| Periodic-128 | 0.0156 | 4-bit | 0.939924 | No | Correct |
| Sparse-20 | 0.0018 | 4-bit | 0.943367 | No | Correct |
| Sparse-100 | 0.0020 | 4-bit | 0.913927 | No | Correct |
| Diffuse | 0.0020 | 4-bit | 0.939601 | No | Correct |

**Prediction accuracy: 100%**

## Part 5: TurboQuantKVCache Integration

`TurboQuantKVCache` already supports separate K/V bit-widths via the `bits_v` parameter:

```python
cache = TurboQuantKVCache(bits=3, bits_v=2)  # K=3-bit, V=2-bit
```

This means adaptive V bit-width is deployable now. The `bits_v` parameter is per-cache (per-layer), not per-head. For true per-head adaptive bits within a layer, the kernel would need per-head bit-width parameters. With GQA, per-KV-head is the practical granularity.

## Conclusion

### Key Findings

- Periodic heads show varied tolerance for 2-bit quantization. Spectral concentration alone may not be sufficient to predict 2-bit safety.
- **Diffuse heads need higher bit-widths** as expected. 2-bit quantization degrades quality for uniform attention patterns.
- Prediction accuracy: 100% across 5 test patterns.
- GQA constraints limit memory savings when query heads sharing a KV head have mixed patterns (the KV head must use the highest bit-width any sharing head needs).

### Verdict

**MIXED**: Spectral concentration partially predicts 2-bit safety but thresholds need refinement. Consider combining spectral analysis with entropy or kurtosis for a more robust predictor.
