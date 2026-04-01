# PolarQuant Metal — Test-Time Compute Scaling

## The Analogy to Context Efficiency

PolarQuant makes a fixed model *effectively smarter at long context* by changing how compute is spent during inference — not by changing weights. The parallel question: **can the same fixed model produce better answers by changing how compute is spent during the forward pass or generation loop?**

Yes. This is test-time compute (TTC) scaling, and PolarQuant's freed KV budget is what makes it locally viable.

---

## Direct Parallels

### Best-of-N Sampling

Generate N responses in parallel, score with a reward/verifier model, return the best. Quality scales logarithmically with N. This is what o1/o3 and DeepSeek-R1 do internally.

With 4.6× KV compression, N=4 costs roughly the memory budget that previously held N=1. On the 35B model this meaningfully improves reasoning accuracy on hard problems with no retraining.

### Speculative Decoding

A small draft model (e.g., Qwen3.5-3B) proposes k tokens at once; the large model (35B) verifies them all in a single parallel forward pass and accepts/rejects. Net effect: **2–4× faster generation at identical quality**. The draft model runs in the memory PolarQuant freed up.

### Chain-of-Thought as a Compute Allocator

Your sparse SV kernel is a *compute allocator* — it dynamically identifies where attention is needed and skips the rest. The quality-side equivalent is **adaptive CoT depth**: easy questions get shallow chains, hard questions get deep ones.

This is what Quiet-STaR (2024) formalized: learned "thinking tokens" inserted between forward passes that increase internal computation without appearing in output. The adaptive depth analog is exactly like entropy-guided sparse SV using shallow attention on spread heads and deep attention on concentrated ones — same principle, different axis.

---

## Highest-ROI Techniques for This Stack

### 1. Speculative Decoding (3B + 35B)

The freed KV memory from PolarQuant is the exact budget needed to run a small draft model alongside the 35B. Previously memory-constrained on M4 Pro at long context; with compression, it becomes comfortable.

**Expected gain:** 2–3× faster generation at identical quality. No new training required.

**Implementation:** Route 8080 requests through a speculative decoding wrapper that uses a resident 3B model as the draft. The 35B verifies in batched parallel passes.

### 2. Contrastive Decoding

Run the same prompt through the 35B and Phi-4-Mini simultaneously. At each token, amplify the 35B's predictions and subtract Phi-4-Mini's (weighted). What the large model believes that the small model doesn't is precisely its unique capability.

Shown to improve factuality and reduce hallucination without retraining. The two-model stack at 8080/8081 already has exactly the right architecture — this is a routing change, not a new model.

### 3. Self-Consistency / Majority Vote

Generate 5–10 reasoning chains independently, take the majority answer. Accuracy on multi-step reasoning scales dramatically with chain count.

With 4.6× KV compression, 5 parallel chains fit in roughly the memory budget that previously held 1. Particularly valuable for clinical reasoning at 8086 — multi-step diagnostic reasoning is exactly the workload this improves most.

---

## The Research Frontier

**Process Reward Models (PRMs)** — a small verifier trained to score *intermediate reasoning steps* (not just final answers), enabling tree search over thought chains (MCTS-style) with early pruning of bad branches. This is what separates o3-level reasoning from standard CoT.

No public Metal implementation exists. The compute pattern — tree of partial KV caches, branch pruning based on verifier scores — is structurally analogous to the sparse SV problem. The compact-index kernel is arguably the right primitive to build branch-pruning attention on top of.

**The path:** PolarQuant provides the memory and compute efficiency layer. The quality layer is test-time search over reasoning paths. The bottleneck for doing that locally is exactly the KV cache memory and attention compute this stack now has headroom for.

---

## Hardware-Specific Analysis

### M4 Pro Mac Mini (64GB)

**Effective KV capacity with PolarQuant:** For KV-cache-bound tasks, this 64GB machine stores as much context as a ~294GB machine running FP16 KV cache (64GB × 4.6× compression). In practice: ~28GB wired by model weights, ~36GB headroom. PolarQuant turns that 36GB of KV headroom into the equivalent of ~166GB of FP16 KV — more context depth, more concurrent long conversations, or room for a draft model that wouldn't otherwise fit.

**Throughput profile:** ~273 GB/s memory bandwidth, 81 tok/s on 35B (with Phase 3). Speculative decoding with Phi-4-Mini as draft would push effective throughput to ~150–180 tok/s — draft proposals verified in parallel batches by the 35B, which is bandwidth-bound not compute-bound.

**Current state (2026-03-31):** PolarQuant Phase 3 LIVE on Qwen3.5-35B. Phi-4-Mini running on 8081 (not yet PQ-integrated). MedGemma DISABLED. 8 services total.

| Technique | Verdict | Notes |
|---|---|---|
| Speculative decoding (Phi-4 draft + 35B verify) | ✅ Primary target | Both models already running. Routing change only. |
| Contrastive decoding (35B - Phi-4-Mini) | ✅ Trivial | Already running both models side by side |
| Best-of-N, N=4 | ✅ Sweet spot | PQ compression makes 4 parallel streams viable |
| Best-of-N, N=8 | ⚠️ Tight at long context | Viable at <8K context |
| PolarQuant on Phi-4 + PaddleOCR-VL | ✅ Next porting targets | Frees more KV budget across the stack |
| Re-enable MedGemma with PQ | ⚠️ Depends on memory budget | PQ could bring it back under the shed threshold |
| 72B model inference | ⚠️ Short context only | 12GB swap at 80B (tested 2026-03-29) |
| MCTS reasoning tree (8+ branches) | ❌ Memory ceiling | Too many partial KV caches for 64GB |
| PRM full pipeline (3 models) | ❌ Too tight | Not enough headroom with 8 services |

**Role:** Always-on throughput-optimized multi-model server. This is where the stack *runs*.

### M5 Max MacBook Pro (128GB) — not yet purchased

**Effective KV capacity with PolarQuant:** 128GB × 4.6× = ~589GB equivalent for KV-cache-bound tasks. ~100GB available after OS + model weights → ~460GB equivalent KV headroom. A 72B model's KV cache at 32K drops from ~1.2GB to ~260MB — comfortably fits alongside draft model + reward model simultaneously.

**Bandwidth profile:** ~700 GB/s estimated (extrapolating from M4 Max's 546 GB/s with generational scaling). Roughly 2.5× the Mini's bandwidth.

| Technique | Verdict | Notes |
|---|---|---|
| Speculative decoding | ✅ Trivial | Comfortable headroom |
| Best-of-N, N=16+ | ✅ Primary clinical target | Diagnostic reasoning accuracy plateaus ~N=16–32 |
| 72B model + speculative decoding | ✅ Comfortable | Qwen2.5-72B + 7B draft ~45GB combined |
| MCTS reasoning tree (8–16 branches) | ✅ Unique capability | Requires holding many partial KV caches — not viable on Mini |
| PRM full pipeline (3 models) | ✅ Designed for this | 7B draft + reward model + 35B verifier |
| Full PolarQuant stack (all phases) | ✅ Drop-in | Same code, just more headroom |

**Role:** Deep reasoning machine on demand. 8+ branch MCTS and full PRM pipelines require 128GB; not viable on the Mini. This is where the stack *thinks*.

**Constraint:** Thermal throttling under sustained load. Extended Best-of-N or MCTS runs should be plugged in. Not always-on — the Mini is the server.

### Division of Labor

The Mini runs the always-on 8-service stack with speculative decoding and Best-of-N up to N=8. The MBP handles deep reasoning tasks — tree search, PRM pipelines, 72B inference — where memory depth matters more than always-on availability.

PolarQuant is what makes both machines punch above their weight class for any KV-cache-bound task. At long context and multi-chain reasoning, everything is KV-cache-bound.
