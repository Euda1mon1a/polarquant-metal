"""
Gemma 4 31B 4-bit — DEVONthink content quality test.
Uses real documents from the user's DEVONthink vault as context.
System prompt suppresses medical disclaimer behavior.
"""

import time
import sys
import textwrap

MODEL = "mlx-community/gemma-4-31b-it-4bit"

# System prompt: direct, no hedging, physician context
SYSTEM = (
    "You are a knowledgeable assistant working with a board-certified family medicine "
    "physician who also builds software systems. Provide direct, accurate answers. "
    "Do not add medical disclaimers, suggest consulting a doctor, or hedge unnecessarily. "
    "Be concise. Prefer bullet points over paragraphs where structure helps."
)

# ── Real content pulled from DEVONthink ──────────────────────────────────────

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

# ── Test cases ───────────────────────────────────────────────────────────────

TESTS = [
    (
        "Medical — direct",
        "What is the mechanism of action of metformin in type 2 diabetes?",
        None,
        60,
    ),
    (
        "Medical — clinical nuance",
        "When would you choose metformin over an SGLT2 inhibitor as first-line in a new T2DM diagnosis?",
        None,
        120,
    ),
    (
        "Document synthesis — HPE",
        f"""Given this document excerpt:\n\n{HPE_EXCERPT}\n\n
What is the single most defensible MHPE thesis direction here, and what is the one thing
that makes it succeed or fail?""",
        None,
        200,
    ),
    (
        "Cross-document connection",
        f"""Given these two document excerpts:\n\nDOC 1:\n{N8N_EXCERPT}\n\nDOC 2:\n{AIRTABLE_EXCERPT}\n\n
What is the single deepest architectural lesson from the Airtable+n8n era, stated in one sentence?
Then: does AAPM's current Postgres/FastAPI stack fully resolve it, or does any version of the
same problem remain?""",
        None,
        200,
    ),
    (
        "Vision / DEVONthink use case",
        "I'm a physician building a DEVONthink database for clinical reference. "
        "What are the three highest-value things a locally-running 31B LLM could do "
        "with medical PDFs in DEVONthink that DEVONthink's built-in AI can't?",
        None,
        150,
    ),
    (
        "Self-assessment reflection",
        f"""From this document:\n\n{HPE_EXCERPT}\n\n
The author says 'I tend to design ambitious systems and then build the first 30%.'
AAPM and the weather model are exceptions. What structural difference between those projects
and the others explains why they got finished?""",
        None,
        150,
    ),
]


def wrap(text, width=90, indent="  "):
    lines = text.strip().split("\n")
    result = []
    for line in lines:
        if len(line) > width:
            result.extend(textwrap.wrap(line, width, subsequent_indent=indent))
        else:
            result.append(line)
    return "\n".join(result)


def main():
    from mlx_vlm import load, generate, apply_chat_template

    print(f"Loading {MODEL}...")
    t0 = time.perf_counter()
    model, processor = load(MODEL)
    config = model.config
    print(f"Loaded in {time.perf_counter()-t0:.1f}s\n")

    results = []

    for label, question, system_override, max_tokens in TESTS:
        sys_prompt = system_override or SYSTEM
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        prompt = apply_chat_template(processor, config, messages)

        print(f"{'='*70}")
        print(f"TEST: {label}")
        print(f"{'='*70}")
        q_preview = question[:120].replace("\n", " ")
        print(f"Q: {q_preview}{'...' if len(question) > 120 else ''}\n")

        t0 = time.perf_counter()
        result = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
        elapsed = time.perf_counter() - t0

        text = result.text.strip()
        tps = result.generation_tps

        print(f"A ({tps:.0f} tok/s, {elapsed:.1f}s):")
        print(wrap(text))
        print()

        results.append({
            "label": label,
            "tps": round(tps, 1),
            "elapsed": round(elapsed, 1),
            "response": text,
        })

    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['label']:40s} {r['tps']:4.0f} tok/s  {r['elapsed']:5.1f}s")


if __name__ == "__main__":
    main()
