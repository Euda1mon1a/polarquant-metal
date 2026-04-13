"""
Gemma 4 31B Dense — MBP M5 Max A/B Benchmark
4-bit vs 8-bit: speed, memory, quality samples.

Usage:
  python bench_gemma4_mbp.py              # both variants
  python bench_gemma4_mbp.py --variant 4bit
  python bench_gemma4_mbp.py --variant 8bit
"""

import argparse
import json
import os
import subprocess
import sys
import time

import mlx.core as mx

MODELS = {
    "4bit": "mlx-community/gemma-4-31b-it-4bit",
    "8bit": "mlx-community/gemma-4-31b-it-8bit",
}

SPEED_PROMPT = "Explain the concept of entropy in three sentences."

QUALITY_PROMPTS = [
    ("Conciseness", "In one sentence, what is the difference between TCP and UDP?"),
    ("Reasoning", "A farmer has 17 sheep. All but 9 die. How many sheep are left?"),
    ("Medical", "What is the mechanism of action of metformin?"),
    ("Code", "Write a Python one-liner to flatten a list of lists."),
    ("Routing", "What time is it right now?"),  # Should be FAST-class
]


def get_memory_gb():
    active = getattr(mx.metal, "get_active_memory", lambda: 0)()
    peak = getattr(mx.metal, "get_peak_memory", lambda: 0)()
    return active / (1024**3), peak / (1024**3)


def swap_used_mb():
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        # vm.swapusage: total = 2048.00M  used = 975.81M  free = ...
        used = float(out.split("used = ")[1].split("M")[0])
        return used
    except Exception:
        return -1


def format_prompt(processor, config, text):
    from mlx_vlm import apply_chat_template
    messages = [{"role": "user", "content": text}]
    return apply_chat_template(processor, config, messages)


def generate(model, processor, prompt, max_tokens):
    from mlx_vlm import generate as vlm_generate
    result = vlm_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    return result.text, result.generation_tps, result.peak_memory


def run_variant(label, model_id):
    from mlx_vlm import load

    print(f"\n{'='*60}")
    print(f"  {label}  —  {model_id}")
    print(f"{'='*60}")

    swap_before = swap_used_mb()
    print(f"Swap before load: {swap_before:.0f} MB")

    print("Loading model (downloading if needed)…")
    t0 = time.perf_counter()
    model, processor = load(model_id)
    load_s = time.perf_counter() - t0
    config = model.config

    mem_gb, peak_gb = get_memory_gb()
    swap_after = swap_used_mb()
    print(f"Loaded in {load_s:.1f}s  |  GPU active {mem_gb:.1f}GB  peak {peak_gb:.1f}GB")
    print(f"Swap after load: {swap_after:.0f} MB  (delta: {swap_after - swap_before:+.0f} MB)")

    # Warmup
    wp = format_prompt(processor, config, "Hello")
    generate(model, processor, wp, max_tokens=8)

    # Speed — 3 trials
    print("\n--- Speed (3 trials, 100 tokens) ---")
    sp = format_prompt(processor, config, SPEED_PROMPT)
    speeds = []
    for i in range(3):
        _, tps, _ = generate(model, processor, sp, max_tokens=100)
        speeds.append(tps)
        print(f"  Trial {i+1}: {tps:.1f} tok/s")
    avg = sum(speeds) / len(speeds)
    print(f"  Avg: {avg:.1f} tok/s")

    mem_gb2, peak_gb2 = get_memory_gb()
    swap_final = swap_used_mb()

    # Quality samples
    print("\n--- Quality samples ---")
    quality = []
    for tag, q in QUALITY_PROMPTS:
        p = format_prompt(processor, config, q)
        text, tps, _ = generate(model, processor, p, max_tokens=80)
        text = text.strip().replace("\n", " ")
        print(f"  [{tag}] {tps:.0f}t/s  Q: {q[:55]}")
        print(f"         A: {text[:120]}")
        quality.append({"tag": tag, "q": q, "a": text, "tps": round(tps, 1)})

    return {
        "variant": label,
        "model": model_id,
        "load_s": round(load_s, 1),
        "mem_active_gb": round(mem_gb2, 2),
        "mem_peak_gb": round(peak_gb2, 2),
        "swap_delta_mb": round(swap_after - swap_before, 1),
        "swap_final_mb": round(swap_final, 1),
        "avg_tok_s": round(avg, 1),
        "trials_tok_s": [round(s, 1) for s in speeds],
        "quality": quality,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["4bit", "8bit", "both"], default="both")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    variants = ["4bit", "8bit"] if args.variant == "both" else [args.variant]
    results = {}

    for v in variants:
        results[v] = run_variant(v.upper(), MODELS[v])
        mx.metal.clear_cache()
        print(f"\nCleared Metal cache after {v}.")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for v, r in results.items():
        print(f"  {v:4s}  {r['avg_tok_s']:5.1f} tok/s  "
              f"{r['mem_active_gb']:5.1f} GB  "
              f"swap Δ {r['swap_delta_mb']:+.0f} MB")

    out = args.output or os.path.join(
        os.path.dirname(__file__),
        f"GEMMA4_MBP_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
