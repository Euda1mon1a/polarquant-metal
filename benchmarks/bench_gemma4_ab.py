"""
A/B Benchmark: Gemma 4 (E4B + 26B MoE) vs current stack (Phi-4 + Qwen3.5).

Tests:
  1. Generation speed (tok/s) with and without PolarQuant
  2. Memory footprint (GPU wired bytes)
  3. Quality: router classification accuracy (E4B vs Phi-4)
  4. Quality: PA query responses (26B vs Qwen3.5)

Usage:
  python bench_gemma4_ab.py --model e4b          # E4B only
  python bench_gemma4_ab.py --model 26b          # 26B only
  python bench_gemma4_ab.py --model both         # Full A/B
  python bench_gemma4_ab.py --model e4b --no-pq  # Without PolarQuant
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

# --- Router test cases (subset of autoresearch-router 34/34 suite) ---
ROUTER_TESTS = [
    ("what time is it", "FAST"),
    ("set a timer for 10 minutes", "FAST"),
    ("turn off the living room lights", "FAST"),
    ("what's the weather", "FAST"),
    ("remind me to call mom at 3pm", "ESCALATE"),
    ("add milk to my shopping list", "FAST"),
    ("play some jazz music", "FAST"),
    ("how do I fix a leaky faucet", "ESCALATE"),
    ("what's the capital of France", "ESCALATE"),
    ("summarize the last email from John", "ESCALATE"),
    ("take a note about the meeting tomorrow", "ESCALATE"),
    ("what appointments do I have today", "ESCALATE"),
    ("translate hello to Spanish", "ESCALATE"),
    ("calculate 15% tip on $85", "FAST"),
    ("who won the Super Bowl last year", "ESCALATE"),
    ("create a calendar event for lunch Friday", "ESCALATE"),
    ("what's 72 divided by 8", "FAST"),
    ("read my latest text messages", "FAST"),
    ("explain quantum entanglement simply", "ESCALATE"),
    ("send a text to Sarah saying I'll be late", "FAST"),
]

PA_QUERIES = [
    "What's the current status of my solar production today?",
    "Compare my energy usage this week to last week.",
    "Are there any service alerts I should know about?",
    "Draft a brief professional email declining a meeting invitation.",
    "What 3D prints should I queue up this weekend based on my recent designs?",
    "Summarize the key points from my last meeting notes.",
    "What's the weather forecast for the next 3 days and should I adjust my outdoor plans?",
    "Help me debug why my text router might be misclassifying calendar queries.",
]


def format_prompt(processor, config, text, system=None):
    """Format prompt using the model's chat template."""
    from mlx_vlm import apply_chat_template

    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
    else:
        messages = text

    return apply_chat_template(processor, config, messages)


def run_generate(model, processor, prompt, max_tokens=100):
    """Run generation and return (text, tok/s, elapsed, peak_mem_gb)."""
    from mlx_vlm import generate

    result = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    return (
        result.text,
        result.generation_tps,
        result.generation_tokens / result.generation_tps if result.generation_tps > 0 else 0,
        result.peak_memory,
    )


def measure_memory():
    """Return current Metal memory usage in GB."""
    get_active = getattr(mx, 'get_active_memory', None) or mx.metal.get_active_memory
    get_peak = getattr(mx, 'get_peak_memory', None) or mx.metal.get_peak_memory
    return get_active() / (1024 ** 3), get_peak() / (1024 ** 3)


def run_router_test(model, processor, config, tests):
    """Run router classification test suite and return accuracy."""
    correct = 0
    results = []

    system_prompt = (
        "You are a message router. Classify each user message as either FAST "
        "(can be handled by a simple tool/shortcut without an LLM) or ESCALATE "
        "(needs LLM reasoning). Reply with ONLY the word FAST or ESCALATE."
    )

    for query, expected in tests:
        try:
            prompt = format_prompt(processor, config, query, system=system_prompt)
            text, tps, elapsed, _ = run_generate(
                model, processor, prompt, max_tokens=5
            )
            text_upper = text.strip().upper()
            if "FAST" in text_upper and "ESCALATE" not in text_upper:
                predicted = "FAST"
            elif "ESCALATE" in text_upper:
                predicted = "ESCALATE"
            else:
                predicted = f"UNCLEAR({text.strip()[:20]})"

            is_correct = predicted == expected
            correct += is_correct
            results.append({
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "raw": text.strip()[:30],
                "correct": is_correct,
                "tps": round(tps, 1),
            })
        except Exception as e:
            results.append({
                "query": query,
                "expected": expected,
                "predicted": "ERROR",
                "correct": False,
                "error": str(e),
            })

    accuracy = correct / len(tests) if tests else 0
    return accuracy, results


def run_pa_quality_test(model, processor, config, queries):
    """Run PA quality queries and return responses for manual review."""
    results = []
    for query in queries:
        try:
            prompt = format_prompt(processor, config, query)
            text, tps, elapsed, peak = run_generate(
                model, processor, prompt, max_tokens=200
            )
            results.append({
                "query": query,
                "response": text,
                "tok_s": round(tps, 1),
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            results.append({
                "query": query,
                "response": f"ERROR: {e}",
                "tok_s": 0,
                "elapsed": 0,
            })
    return results


def benchmark_model(model_id, use_polarquant=True, run_router=True, run_pa=True):
    """Full benchmark for a single model."""
    from mlx_vlm import load

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"PolarQuant: {'ON' if use_polarquant else 'OFF'}")
    print(f"{'='*60}")

    print("Loading model...")
    t0 = time.perf_counter()
    model, processor = load(model_id)
    load_time = time.perf_counter() - t0
    config = model.config
    print(f"Loaded in {load_time:.1f}s")

    mem_after_load, peak_after_load = measure_memory()
    print(f"Memory after load: {mem_after_load:.2f} GB (peak: {peak_after_load:.2f} GB)")

    if use_polarquant:
        from polarquant_metal.integration import make_fused_cache, _is_gemma4
        lang = model.language_model if hasattr(model, 'language_model') else model
        is_g4 = _is_gemma4(lang)
        print(f"Gemma4 detected: {is_g4}")
        caches = make_fused_cache(lang, bits=3, bits_v=2, boundary_layers=1)
        from collections import Counter
        cache_types = dict(Counter(type(c).__name__ for c in caches))
        print(f"Cache types: {cache_types}")

    # Warmup
    print("\nWarming up...")
    warmup_prompt = format_prompt(processor, config, "Hello")
    run_generate(model, processor, warmup_prompt, max_tokens=5)

    # Generation speed test
    print("\n--- Generation Speed ---")
    speeds = []
    test_prompt = format_prompt(
        processor, config, "Explain the concept of entropy in three sentences."
    )
    for trial in range(3):
        text, tps, elapsed, peak = run_generate(
            model, processor, test_prompt, max_tokens=100
        )
        speeds.append(tps)
        print(f"  Trial {trial+1}: {tps:.1f} tok/s ({elapsed:.2f}s)")

    avg_speed = sum(speeds) / len(speeds)
    mem_after_gen, peak_after_gen = measure_memory()

    result = {
        "model": model_id,
        "polarquant": use_polarquant,
        "load_time_s": round(load_time, 1),
        "memory_gb": round(mem_after_gen, 2),
        "peak_memory_gb": round(peak_after_gen, 2),
        "avg_tok_s": round(avg_speed, 1),
        "speeds": [round(s, 1) for s in speeds],
    }

    if run_router:
        print("\n--- Router Classification ---")
        accuracy, router_results = run_router_test(
            model, processor, config, ROUTER_TESTS
        )
        result["router_accuracy"] = round(accuracy, 3)
        result["router_correct"] = sum(1 for r in router_results if r["correct"])
        result["router_total"] = len(router_results)
        failures = [r for r in router_results if not r["correct"]]
        print(f"  Accuracy: {result['router_correct']}/{result['router_total']} ({accuracy:.0%})")
        if failures:
            print(f"  Failures:")
            for f in failures:
                print(f"    '{f['query']}': expected {f['expected']}, got {f['predicted']} (raw: {f.get('raw','')})")
        result["router_failures"] = failures

    if run_pa:
        print("\n--- PA Quality Queries ---")
        pa_results = run_pa_quality_test(model, processor, config, PA_QUERIES)
        result["pa_results"] = pa_results
        avg_pa_speed = sum(r["tok_s"] for r in pa_results) / len(pa_results)
        print(f"  Avg speed: {avg_pa_speed:.1f} tok/s across {len(pa_results)} queries")
        for r in pa_results:
            preview = r["response"][:80].replace("\n", " ")
            print(f"  [{r['tok_s']}t/s] {r['query'][:50]}... -> {preview}...")

    return result


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 A/B Benchmark")
    parser.add_argument("--model", choices=["e4b", "26b", "both"], default="e4b")
    parser.add_argument("--no-pq", action="store_true", help="Disable PolarQuant")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    results = {}

    if args.model in ("e4b", "both"):
        r = benchmark_model(
            "mlx-community/gemma-4-e4b-it-4bit",
            use_polarquant=not args.no_pq,
            run_router=True,
            run_pa=False,
        )
        results["gemma4_e4b"] = r

    if args.model in ("26b", "both"):
        model_26b = "arthurcollet/gemma-4-26B-A4B-it-mlx-mxfp8"
        r = benchmark_model(
            model_26b,
            use_polarquant=not args.no_pq,
            run_router=False,
            run_pa=True,
        )
        results["gemma4_26b"] = r

    out_path = args.output or os.path.join(
        os.path.dirname(__file__), f"GEMMA4_BENCH_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
