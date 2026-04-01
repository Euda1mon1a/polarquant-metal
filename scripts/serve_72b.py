#!/usr/bin/env python3
"""Launch the PolarQuant 72B server.

Default config:
    Main model:   mlx-community/Qwen2.5-72B-Instruct-4bit  (port 8082)
    Draft model:  mlx-community/Qwen2.5-7B-Instruct-4bit   (speculative, 4 tokens)
    PQ bits:      3
    Boundary:     2 layers FP16 at each end

Usage:
    python scripts/serve_72b.py
    python scripts/serve_72b.py --no-draft --port 8083
    python scripts/serve_72b.py --model mlx-community/Qwen2.5-72B-Instruct-8bit --bits 4
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure polarquant_metal is importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

MAIN_MODEL_DEFAULT = "mlx-community/Qwen2.5-72B-Instruct-4bit"
DRAFT_MODEL_DEFAULT = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def parse_args():
    p = argparse.ArgumentParser(description="PolarQuant 72B OpenAI-compatible server")
    p.add_argument("--model", default=MAIN_MODEL_DEFAULT,
                   help="Main model HF ID or local path")
    p.add_argument("--draft-model", default=DRAFT_MODEL_DEFAULT,
                   help="Draft model for speculative decoding")
    p.add_argument("--no-draft", action="store_true",
                   help="Disable speculative decoding (run main model only)")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default: loopback only)")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4],
                   help="PolarQuant bits for K/V cache")
    p.add_argument("--bits-v", type=int, default=None, choices=[2, 3, 4],
                   help="PolarQuant bits for V cache (default: same as --bits)")
    p.add_argument("--boundary-layers", type=int, default=2,
                   help="FP16 boundary layer count at each end")
    p.add_argument("--num-draft-tokens", type=int, default=4,
                   help="Tokens proposed per speculative step")
    return p.parse_args()


def main():
    args = parse_args()
    from polarquant_metal.serving.server import serve

    draft = None if args.no_draft else args.draft_model

    print(f"[serve_72b] Starting PolarQuant 72B server")
    print(f"  main model  : {args.model}")
    print(f"  draft model : {draft or 'none (disabled)'}")
    print(f"  port        : {args.port}")
    print(f"  bits K/V    : {args.bits}/{args.bits_v or args.bits}")
    print(f"  spec tokens : {args.num_draft_tokens if draft else 'n/a'}")

    serve(
        model_id=args.model,
        draft_model_id=draft,
        port=args.port,
        host=args.host,
        pq_bits=args.bits,
        boundary_layers=args.boundary_layers,
        num_draft_tokens=args.num_draft_tokens,
    )


if __name__ == "__main__":
    main()
