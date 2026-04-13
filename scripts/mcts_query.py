#!/usr/bin/env python3
"""MCTS reasoning chain search via PolarQuant + speculative decoding.

Runs multi-step tree search over a language model, scoring branches with
either a draft model (Phase B) or a Process Reward Model (Phase C).

Usage:
    # Phase B: draft model scoring (no PRM needed)
    python scripts/mcts_query.py \\
        --query "A 52-year-old presents with sudden onset chest pain..." \\
        --branches 8 --depth 3

    # Phase C: PRM scoring
    python scripts/mcts_query.py \\
        --query "..." \\
        --prm Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B-4bit \\
        --branches 8 --depth 3

    # Best-of-N (depth 1 = flat search):
    python scripts/mcts_query.py --query "..." --branches 16 --depth 1
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")

MAIN_MODEL_DEFAULT = "mlx-community/Qwen2.5-72B-Instruct-4bit"
DRAFT_MODEL_DEFAULT = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def parse_args():
    p = argparse.ArgumentParser(description="MCTS reasoning tree search")
    p.add_argument("--query", required=True, help="Question or prompt to reason about")
    p.add_argument("--model", default=MAIN_MODEL_DEFAULT)
    p.add_argument("--draft-model", default=DRAFT_MODEL_DEFAULT)
    p.add_argument("--prm", default=None,
                   help="Optional PRM model ID (Phase C). Default: draft logprob scoring.")
    p.add_argument("--branches", type=int, default=8,
                   help="Branches per expansion")
    p.add_argument("--depth", type=int, default=3,
                   help="MCTS rounds (1 = Best-of-N)")
    p.add_argument("--max-step-tokens", type=int, default=150,
                   help="Max tokens per reasoning step")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--time-budget", type=float, default=None,
                   help="Optional wall-clock limit in seconds")
    p.add_argument("--show-all", action="store_true",
                   help="Print all branches, not just the best")
    return p.parse_args()


def main():
    args = parse_args()
    import mlx_lm
    from polarquant_metal.integration import make_fused_cache, patch_sdpa
    from polarquant_metal.tree_search import MCTSTree, draft_logprob_evaluator

    print(f"[mcts] Loading main model: {args.model}")
    patch_sdpa()
    model, tokenizer = mlx_lm.load(args.model)

    print(f"[mcts] Loading draft model: {args.draft_model}")
    draft_model, _ = mlx_lm.load(args.draft_model)

    # Build evaluator
    if args.prm:
        from polarquant_metal.prm import ProcessRewardModel
        print(f"[mcts] Loading PRM: {args.prm}")
        prm = ProcessRewardModel.load(args.prm)
        evaluator = prm.as_evaluator(question=args.query)
    else:
        evaluator = draft_logprob_evaluator(draft_model, tokenizer)

    # Prefill: encode system + user prompt into root cache, get seed token
    print(f"[mcts] Prefilling root cache...")
    root_caches = make_fused_cache(model, bits=args.bits, boundary_layers=2)

    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": "You are an expert reasoning assistant. Think step by step."},
        {"role": "user", "content": args.query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prefill: generate exactly 1 token to seed the tree
    # The cache will have prompt tokens processed; seed_token is the first response token
    seed_token = None
    for resp in mlx_lm.stream_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=1,
        sampler=make_sampler(temp=0.0),  # greedy for the seed
        prompt_cache=root_caches,
    ):
        seed_token = resp.token
        break

    print(f"[mcts] Seed token: {seed_token!r} = {tokenizer.decode([seed_token])!r}")

    # Run MCTS
    print(f"[mcts] Searching: {args.branches} branches × {args.depth} rounds")
    tree = MCTSTree(
        model=model,
        tokenizer=tokenizer,
        root_caches=root_caches,
        evaluator=evaluator,
        max_step_tokens=args.max_step_tokens,
        temperature=args.temperature,
        verbose=True,
    )

    tree.set_root_seed(seed_token)
    best = tree.search(
        prompt=args.query,
        n_branches=args.branches,
        depth=args.depth,
        time_budget=args.time_budget,
    )

    print("\n" + "=" * 60)
    print("BEST REASONING CHAIN")
    print("=" * 60)
    print(best.full_text)
    print(f"\n[score: {best.score/max(best.visits,1):.3f}, depth: {best.depth}]")

    if args.show_all:
        print("\n" + "=" * 60)
        print("ALL BRANCHES (root children, sorted by score)")
        print("=" * 60)
        children = sorted(
            tree.root.children,
            key=lambda n: n.score / max(n.visits, 1),
            reverse=True,
        )
        for i, node in enumerate(children):
            score = node.score / max(node.visits, 1)
            print(f"\n--- Branch {i+1} (score={score:.3f}) ---")
            print(node.full_text[:500] + ("..." if len(node.full_text) > 500 else ""))


if __name__ == "__main__":
    main()
