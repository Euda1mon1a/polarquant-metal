"""
Gemma 4 31B 4-bit (MBP, local) vs Qwen3.5-35B-A3B (Mini, API)
Both with PolarQuant KV compression.

Gemma 4: loaded via mlx-vlm, PQ applied via make_fused_cache(bits=3)
         Bottleneck is dense weight matmul — PQ reduces KV memory but
         speed gain is small (~0-3%) vs the 5% on Qwen3.5 MoE.
Qwen3.5: called via Mini's OpenAI-compatible API (port 8080).
         PQ already active on Mini (75.3 tok/s vs 71.4 baseline).

Usage:
  cd ~/workspace/polarquant-metal
  python benchmarks/bench_model_compare.py
  python benchmarks/bench_model_compare.py --gemma-only
  python benchmarks/bench_model_compare.py --qwen-only
  python benchmarks/bench_model_compare.py --mini-ip 192.168.4.202
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import textwrap

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GEMMA_MODEL = "mlx-community/gemma-4-31b-it-4bit"
QWEN_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
MINI_BASE_URL = "http://192.168.4.202:8080"

SYSTEM = (
    "You are a knowledgeable assistant working with a board-certified family medicine "
    "physician who also builds software systems. Provide direct, accurate answers. "
    "Do not add medical disclaimers, suggest consulting a doctor, or hedge unnecessarily. "
    "Be concise. Prefer bullet points over paragraphs where structure helps."
)

# ── Real DEVONthink content ───────────────────────────────────────────────────

HPE_EXCERPT = """
From 'Project Summary — Dr. Shapiro (HPE)' (DEVONthink/Medicine/HPE Fellowship):

The project with the strongest HPE fit is a fetal presentation ultrasound simulator designed
to make POCUS training accessible for military providers in deployed/austere settings and
civilian providers in rural Hawaii. Family medicine residents need procedural competency in
obstetric ultrasound, but access is limited by volume and geography.

Design: tracked ultrasound probe on a fetal phantom, position data → rendering engine →
physics-based acoustic simulation → scoring module (scan completeness, time-to-identification,
anatomical coverage).

What exists: architecture document, Python scaffold, README. No hardware purchased.

Also running: self-hosted AI assistant on Mac Mini — MedGemma, semantic search over 272
clinical documents, three-tier routing, zero data leaving the device. Possible thesis:
how does access to a privacy-preserving on-premise clinical AI affect trainee
information-seeking behavior, time-to-answer, and self-reported confidence?

FHE project: encrypted XGBoost inference, ~8s on Apple Silicon, published on PyPI.
Could enable multi-site HPE outcome research without exposing trainee data.

Self-assessment: 'I tend to design ambitious systems and then build the first 30%.'
AAPM (residency scheduler, 1260 commits) and weather model are the exceptions — both
finished because they had real users.
"""

N8N_EXCERPT = """
From 'Architecture Evolution' (DEVONthink/Engineering/AAPM Origin — n8n Era):

The n8n scheduling system evolved through 5 phases across 100 workflows:

Phase 1 (Jun 2025): Monolith — masterAssignmentsPairingProcessor, 93 nodes.
One filter node per rotation (~40 rotations). 12 Merge+Code pairs for pairing logic.

Phase 2 (Jul 2025): 'THIS IS THE ONE' — 33 nodes, 4 named phases in Code nodes:
PHASE 1: UNIVERSAL COVERAGE, PHASE 2: IMMUTABLE OVERRIDES,
PHASE 3: NEEDS-BASED OPTIMIZATION, PHASE 4: FORMAT FOR AIRTABLE POST.
Three DEBUG nodes left in production.

Phase 3 (Oct 2025): HTTP Nodes v7.0, 74 nodes. AI Agent (Claude/OpenAI switchable),
18 GitHub tool nodes used as key-value state store between sessions.

Phase 4 (Dec 2025): Clean 9-phase subworkflow architecture. Each phase 9-15 nodes.
Phase 0=Absence Loading, 1=Block Pairing, 2=Resident Association, 3=Faculty Assignment,
4=Call Scheduling, 7=Validation, 9=Excel Export.

Phase 5 (Apr 2026): Orchestrator with MCP Client tool — bridge to current AAPM stack.

Core brittleness: Airtable record IDs are unstable across sessions. The 50 AI Agent
Memory records in Airtable are mostly documentation of ID-staleness bugs.
Postgres migration eliminated this entirely.
"""

AIRTABLE_EXCERPT = """
From 'Schema Evolution' (DEVONthink/Engineering/AAPM Origin — Airtable Era):

May 2025 (18 tables) → Final (25 tables). Key additions:
- Unified Rotation Templates (merged PGY-1/2/3 tables into one with PGY field)
- Half-Day of the Week of Blocks (44 fields) — the architectural unlock
- Faculty Master Assignments (mirror of resident assignments)
- AI Agent Memory (50 records of LLM session state)
- Call Optimization Results (equity scoring, gap days, violation levels)

The half-day calendar scaffold was the unlock: once it existed, both resident and faculty
assignments could be matched deterministically by day × time × week.
Faculty scheduling lagged resident scheduling by ~6 weeks.

'Memory is for context; tables are for truth' — from AI Agent Memory record #105.
"""

# ── Test cases ────────────────────────────────────────────────────────────────

TESTS = [
    (
        "Medical — direct",
        "What is the mechanism of action of metformin in type 2 diabetes?",
        100,
    ),
    (
        "Medical — clinical nuance",
        "When would you choose metformin over an SGLT2 inhibitor as first-line in a new T2DM diagnosis?",
        120,
    ),
    (
        "Document synthesis — HPE",
        f"Given this document excerpt:\n\n{HPE_EXCERPT}\n\n"
        "What is the single most defensible MHPE thesis direction here, and what is the one thing "
        "that makes it succeed or fail?",
        200,
    ),
    (
        "Cross-document connection",
        f"Given these two document excerpts:\n\nDOC 1:\n{N8N_EXCERPT}\n\nDOC 2:\n{AIRTABLE_EXCERPT}\n\n"
        "What is the single deepest architectural lesson from the Airtable+n8n era, stated in one sentence? "
        "Then: does AAPM's current Postgres/FastAPI stack fully resolve it, or does any version of the "
        "same problem remain?",
        200,
    ),
    (
        "Self-assessment reflection",
        f"From this document:\n\n{HPE_EXCERPT}\n\n"
        "The author says 'I tend to design ambitious systems and then build the first 30%.' "
        "AAPM and the weather model are exceptions. What structural difference between those projects "
        "and the others explains why they got finished?",
        150,
    ),
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def wrap(text, width=90, indent="  "):
    lines = text.strip().split("\n")
    result = []
    for line in lines:
        if len(line) > width:
            result.extend(textwrap.wrap(line, width, subsequent_indent=indent))
        else:
            result.append(line)
    return "\n".join(result)


def probe_mini(base_url):
    """Return Mini model ID or None if unreachable."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return None


def response_tps(data, elapsed):
    """Prefer server-reported TPS; fall back to usage-derived wall-clock TPS."""
    polarquant_meta = data.get("x_polarquant", {})
    server_tps = polarquant_meta.get("generation_tps")
    if server_tps:
        return float(server_tps)

    usage = data.get("usage", {})
    gen_tokens = usage.get("completion_tokens", 0)
    if elapsed > 0 and gen_tokens > 0:
        return gen_tokens / elapsed
    return 0.0


def print_avg_line(label, results, context):
    valid_tps = [r["tps"] for r in (results or []) if r["tps"] > 0]
    if valid_tps:
        print(f"  {label:<15} {sum(valid_tps) / len(valid_tps):5.1f} tok/s  ({context})")
    elif results:
        print(f"  {label:<15} n/a          ({context}; no valid TPS samples)")


def cleanup_local_model(*objects):
    """Best-effort cleanup between large local model loads."""
    for obj in objects:
        del obj
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass


def strip_think_blocks(text):
    """Remove reasoning blocks from server responses when exposed by the API."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


# ── Gemma 4 runner ────────────────────────────────────────────────────────────

def run_gemma():
    from mlx_vlm import load, generate, apply_chat_template
    from polarquant_metal.integration import make_fused_cache

    print(f"\n{'='*70}")
    print(f"  GEMMA 4  —  {GEMMA_MODEL}")
    print(f"  Hardware: MBP M5 Max 128GB (local)")
    print(f"  PolarQuant: bits=3 (KV cache, fused Metal kernels)")
    print(f"  Note: Gemma Dense — PQ reduces KV memory; speed gain <3%")
    print(f"        (bottleneck is dense weight matmul, not KV bandwidth)")
    print(f"{'='*70}")

    print("Loading model...")
    t0 = time.perf_counter()
    model, processor = load(GEMMA_MODEL)
    config = model.config
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s\n")

    # Warmup (no PQ — just wakes GPU)
    messages = [{"role": "user", "content": "Hello"}]
    wp = apply_chat_template(processor, config, messages)
    generate(model, processor, wp, max_tokens=8, verbose=False)

    results = []
    for label, question, max_tokens in TESTS:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        prompt = apply_chat_template(processor, config, messages)

        # Fresh PQ cache per call
        pq_cache = make_fused_cache(model.language_model, bits=3)

        print(f"{'='*70}")
        print(f"[Gemma4] {label}")
        q_preview = question[:100].replace("\n", " ")
        print(f"Q: {q_preview}{'...' if len(question) > 100 else ''}\n")

        t0 = time.perf_counter()
        result = generate(
            model, processor, prompt,
            max_tokens=max_tokens,
            prompt_cache=pq_cache,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        text = strip_think_blocks(result.text)
        tps = result.generation_tps

        print(f"A ({tps:.0f} tok/s, {elapsed:.1f}s):")
        print(wrap(text))
        print()

        results.append({
            "model": "gemma4-4bit+PQ",
            "label": label,
            "tps": round(tps, 1),
            "elapsed": round(elapsed, 1),
            "response": text,
        })

    cleanup_local_model(model, processor)
    return results


# ── Qwen3.5 runner (Mini API) ─────────────────────────────────────────────────

def run_qwen(base_url, model_id):
    print(f"\n{'='*70}")
    print(f"  QWEN3.5  —  {model_id}")
    print(f"  Hardware: Mac Mini M4 Pro (via API {base_url})")
    print(f"  PolarQuant: active on Mini (75.3 tok/s vs 71.4 baseline)")
    print(f"  Note: Qwen3.5 MoE — PQ gives ~5% speedup (KV is the bottleneck)")
    print(f"{'='*70}\n")

    results = []
    for label, question, max_tokens in TESTS:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        print(f"{'='*70}")
        print(f"[Qwen3.5] {label}")
        q_preview = question[:100].replace("\n", " ")
        print(f"Q: {q_preview}{'...' if len(question) > 100 else ''}\n")

        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "model": "qwen35+PQ",
                "label": label,
                "tps": 0.0,
                "elapsed": 0.0,
                "response": f"ERROR: {e}",
            })
            continue

        elapsed = time.perf_counter() - t0
        choice = data["choices"][0]
        text = strip_think_blocks(choice["message"]["content"])

        tps = response_tps(data, elapsed)

        print(f"A ({tps:.0f} tok/s, {elapsed:.1f}s):")
        print(wrap(text))
        print()

        results.append({
            "model": "qwen35+PQ",
            "label": label,
            "tps": round(tps, 1),
            "elapsed": round(elapsed, 1),
            "response": text,
        })

    return results


def run_qwen_local():
    from mlx_vlm import load, generate, apply_chat_template
    from polarquant_metal.integration import make_fused_cache

    print(f"\n{'='*70}")
    print(f"  QWEN3.5  —  {QWEN_MODEL}")
    print(f"  Hardware: MBP M5 Max 128GB (local)")
    print(f"  PolarQuant: bits=3 (KV cache, fused Metal kernels)")
    print(f"  Note: Qwen3.5 MoE — PQ gives ~5% speedup (KV is the bottleneck)")
    print(f"{'='*70}")

    print("Loading model...")
    t0 = time.perf_counter()
    model, processor = load(QWEN_MODEL)
    config = model.config
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s\n")

    messages = [{"role": "user", "content": "Hello"}]
    wp = apply_chat_template(processor, config, messages)
    generate(model, processor, wp, max_tokens=8, verbose=False)

    results = []
    for label, question, max_tokens in TESTS:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        prompt = apply_chat_template(processor, config, messages)
        pq_cache = make_fused_cache(model.language_model, bits=3)

        print(f"{'='*70}")
        print(f"[Qwen3.5] {label}")
        q_preview = question[:100].replace("\n", " ")
        print(f"Q: {q_preview}{'...' if len(question) > 100 else ''}\n")

        t0 = time.perf_counter()
        result = generate(
            model, processor, prompt,
            max_tokens=max_tokens,
            prompt_cache=pq_cache,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        text = strip_think_blocks(result.text)
        tps = result.generation_tps

        print(f"A ({tps:.0f} tok/s, {elapsed:.1f}s):")
        print(wrap(text))
        print()

        results.append({
            "model": "qwen35+PQ-local",
            "label": label,
            "tps": round(tps, 1),
            "elapsed": round(elapsed, 1),
            "response": text,
        })

    cleanup_local_model(model, processor)
    return results


# ── Side-by-side summary ──────────────────────────────────────────────────────

def print_summary(gemma_results, qwen_results):
    print(f"\n{'='*70}")
    print("  SUMMARY  (quality review)")
    print(f"{'='*70}")
    print(f"  {'Test':<35s}  {'Gemma4+PQ':>12s}  {'Qwen3.5+PQ':>12s}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*12}")

    gemma_by_label = {r["label"]: r for r in (gemma_results or [])}
    qwen_by_label = {r["label"]: r for r in (qwen_results or [])}

    all_labels = [t[0] for t in TESTS]
    for label in all_labels:
        g = gemma_by_label.get(label)
        q = qwen_by_label.get(label)
        g_str = f"{g['tps']:.0f} tok/s" if g else "—"
        q_str = f"{q['tps']:.0f} tok/s" if q else "—"
        print(f"  {label:<35s}  {g_str:>12s}  {q_str:>12s}")

    print()
    print_avg_line("Gemma4+PQ avg:", gemma_results, "MBP M5 Max, local")
    print_avg_line("Qwen3.5+PQ avg:", qwen_results, "Mini M4 Pro, API")
    print()
    print("  Hardware context:")
    print("    Gemma4: dense 31B — activates all params per token, KV not bottleneck")
    print("    Qwen3.5: MoE ~3.5B active/token — KV bandwidth matters, PQ helps 5%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma-only", action="store_true")
    parser.add_argument("--qwen-only", action="store_true")
    parser.add_argument("--mini-ip", default="192.168.4.202")
    parser.add_argument("--mini-port", type=int, default=8080)
    parser.add_argument("--qwen-mode", choices=["api", "local"], default="api")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_url = f"http://{args.mini_ip}:{args.mini_port}"
    run_gemma_flag = not args.qwen_only
    run_qwen_flag = not args.gemma_only
    run_qwen_local_flag = run_qwen_flag and args.qwen_mode == "local"

    gemma_results = None
    qwen_results = None

    # Probe Mini before starting long Gemma load
    qwen_model_id = None
    if run_qwen_flag and not run_qwen_local_flag:
        print(f"Probing Mini at {base_url}...")
        qwen_model_id = probe_mini(base_url)
        if qwen_model_id:
            print(f"Mini online — model: {qwen_model_id}")
        else:
            print("Mini unreachable — Qwen section will be skipped.")
            run_qwen_flag = False

    if run_gemma_flag:
        gemma_results = run_gemma()

    if run_qwen_local_flag:
        qwen_results = run_qwen_local()
        qwen_model_id = QWEN_MODEL
    elif run_qwen_flag:
        qwen_results = run_qwen(base_url, qwen_model_id)

    if not gemma_results and not qwen_results:
        print("\nNo benchmark runs completed.")
        return 1

    print_summary(gemma_results, qwen_results)

    # Save results
    out = args.output or os.path.join(
        os.path.dirname(__file__),
        f"COMPARE_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out, "w") as f:
        json.dump(
            {
                "gemma4_pq": gemma_results,
                "qwen35_pq": qwen_results,
                "mini_url": base_url,
                "qwen_model_id": qwen_model_id,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Results → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
