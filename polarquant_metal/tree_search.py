"""MCTS reasoning tree using PolarQuant KV cache forking.

Cache forking is O(1) — branches share packed prefix arrays and diverge lazily
on first update_and_fetch() via mx.concatenate, which allocates a new buffer.

Usage:
    model, tokenizer = mlx_lm.load("Qwen2.5-72B-Instruct-4bit")
    patch_sdpa()
    root_caches = make_fused_cache(model, bits=3)

    tree = MCTSTree(model, tokenizer, root_caches, evaluator=draft_logprob_evaluator)
    best = tree.search(prompt="...", n_branches=8, depth=3)
    print(best.text)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import mlx_lm

from .turboquant_cache import TurboQuantKVCache

# Default step-end markers for chain-of-thought parsing
_STEP_END_TOKENS = {"\n\n", "\n\nStep ", "\n\n---"}
_MAX_STEP_TOKENS = 150


def fork_layer_caches(caches: list[TurboQuantKVCache]) -> list[TurboQuantKVCache]:
    """Fork a full model's KV cache list for MCTS branching.

    MLX arrays are functional — new concatenations create new buffers.
    Sharing the prefix arrays is safe; branches diverge on first token append.

    Args:
        caches: List of TurboQuantKVCache (one per model layer).

    Returns:
        New list of TurboQuantKVCache instances sharing the same prefix data.
    """
    forked = []
    for src in caches:
        # Pass through non-TurboQuant caches (FP16 boundary layers, linear attn)
        if not isinstance(src, TurboQuantKVCache):
            forked.append(src)
            continue

        dst = TurboQuantKVCache(
            bits=src.turbo_bits,
            bits_v=src._bits_v,
            fused=src._fused,
            min_fused_context=src.min_fused_context,
            sparse_v_threshold=src.sparse_v_threshold,
            system_prompt_len=src.system_prompt_len,
            recent_zone_len=src.recent_zone_len,
        )
        if src.offset > 0:
            # Reference copy — safe because next append creates a new array
            dst.state = src.state
            dst.offset = src.offset
            dst._quantized = src._quantized
            # Carry entropy amortization state so forked branches don't cold-start
            if src._cached_thresholds is not None:
                dst._cached_thresholds = src._cached_thresholds
            dst._entropy_step_counter = src._entropy_step_counter
        forked.append(dst)
    return forked


@dataclass
class MCTSNode:
    """A node in the MCTS reasoning tree.

    Each node holds the full KV cache state at this point in the reasoning
    chain, plus the text and tokens generated to reach this node.

    seed_token is the last generated token at this node — it seeds the next
    generation step because mlx_lm.stream_generate requires at least one input
    token and the cache state does NOT include the last yielded token (that token
    was yielded but not yet processed into the cache).
    """
    caches: list                          # one per model layer
    tokens: list[int]                     # tokens generated at this node only
    text: str                             # text generated at this node only
    seed_token: Optional[int] = None      # last generated token (seeds children)
    score: float = 0.0                    # cumulative backpropagated value
    visits: int = 0
    parent: Optional[MCTSNode] = None
    children: list[MCTSNode] = field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False

    @property
    def full_text(self) -> str:
        """Concatenate text from root to this node."""
        parts = []
        node = self
        while node is not None:
            parts.append(node.text)
            node = node.parent
        return "".join(reversed(parts))

    def uct_score(self, c: float = 1.414) -> float:
        """UCT selection score. Unvisited nodes get +inf."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.score / self.visits
        return (self.score / self.visits
                + c * math.sqrt(math.log(self.parent.visits) / self.visits))


class MCTSTree:
    """MCTS reasoning tree over LLM generation chains.

    Phases:
        select   → UCT walk to a leaf
        expand   → fork caches, generate one step per branch
        evaluate → score each branch (draft logprob or PRM)
        backprop → propagate value to root
        best     → return highest-scoring leaf path
    """

    def __init__(
        self,
        model,
        tokenizer,
        root_caches: list,
        evaluator: Callable[[MCTSNode], float],
        uct_c: float = 1.414,
        step_end_tokens: set[str] = None,
        max_step_tokens: int = _MAX_STEP_TOKENS,
        temperature: float = 0.8,
        verbose: bool = False,
    ):
        """
        Args:
            model: The main language model (72B).
            tokenizer: Tokenizer matched to model.
            root_caches: KV cache list from make_fused_cache() after prefill.
            evaluator: Callable(node) -> float in [0, 1]. Called on each expanded node.
            uct_c: UCT exploration constant. Higher = more exploration.
            step_end_tokens: Set of strings that mark end of a reasoning step.
            max_step_tokens: Hard cap on tokens per step.
            temperature: Sampling temperature for branch generation.
            verbose: Print tree statistics after each expansion.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.uct_c = uct_c
        self.step_end_tokens = step_end_tokens or _STEP_END_TOKENS
        self.max_step_tokens = max_step_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.evaluator = evaluator

        self.root = MCTSNode(
            caches=root_caches,
            tokens=[],
            text="",
            visits=1,
        )

    # --- Core MCTS phases ---

    def select(self) -> MCTSNode:
        """Walk the tree from root to a leaf using UCT."""
        node = self.root
        while node.children and not node.is_terminal:
            node = max(node.children, key=lambda n: n.uct_score(self.uct_c))
        return node

    def expand(self, node: MCTSNode, n_branches: int) -> list[MCTSNode]:
        """Fork the cache N times and generate one reasoning step per branch.

        Args:
            node: Leaf node to expand.
            n_branches: Number of branches to create.

        Returns:
            List of new child MCTSNode instances.
        """
        if node.seed_token is None:
            raise RuntimeError(
                "MCTSNode.seed_token is None — call MCTSTree.set_root_seed() "
                "after prefilling the root cache to provide the first token."
            )
        children = []
        for _ in range(n_branches):
            branch_caches = fork_layer_caches(node.caches)
            step_tokens, step_text, last_tok = self._generate_step(
                branch_caches, node.seed_token
            )
            is_terminal = self._is_answer(step_text)
            child = MCTSNode(
                caches=branch_caches,
                tokens=step_tokens,
                text=step_text,
                seed_token=last_tok,
                parent=node,
                depth=node.depth + 1,
                is_terminal=is_terminal,
            )
            children.append(child)
        node.children = children
        return children

    def evaluate(self, nodes: list[MCTSNode]) -> list[float]:
        """Score each node using the provided evaluator.

        Returns:
            List of scores in [0, 1], one per node.
        """
        return [self.evaluator(n) for n in nodes]

    def backprop(self, node: MCTSNode, value: float) -> None:
        """Propagate a value up the tree, incrementing visit counts."""
        while node is not None:
            node.visits += 1
            node.score += value
            node = node.parent

    def best_path(self) -> MCTSNode:
        """Return the highest avg-value leaf."""
        best = self.root
        stack = list(self.root.children)
        while stack:
            node = stack.pop()
            if node.visits > 0 and node.score / node.visits > best.score / max(best.visits, 1):
                best = node
            stack.extend(node.children)
        return best

    def set_root_seed(self, seed_token: int):
        """Set the seed token on the root node after prefill.

        Args:
            seed_token: The first generated token from the prefill step.
                        This seeds the first expand() call.
        """
        self.root.seed_token = seed_token

    def search(
        self,
        prompt: str,
        n_branches: int = 8,
        depth: int = 3,
        time_budget: float = None,
    ) -> MCTSNode:
        """Run MCTS search for `depth` rounds.

        Args:
            prompt: Already encoded in the root_caches via prefill before calling this.
                    Pass the prompt text here for terminal detection only.
            n_branches: Branches per expansion.
            depth: Number of MCTS rounds (select → expand → evaluate → backprop).
            time_budget: Optional wall-clock limit in seconds.

        Returns:
            Best leaf node found.
        """
        t0 = time.monotonic()
        for round_i in range(depth):
            if time_budget and (time.monotonic() - t0) > time_budget:
                break
            leaf = self.select()
            if leaf.is_terminal:
                continue
            children = self.expand(leaf, n_branches)
            scores = self.evaluate(children)
            for child, score in zip(children, scores):
                self.backprop(child, score)
            if self.verbose:
                best = self.best_path()
                print(f"[MCTS round {round_i+1}/{depth}] "
                      f"expanded {n_branches} branches, "
                      f"best score: {best.score/max(best.visits,1):.3f}")
        return self.best_path()

    # --- Helpers ---

    def _generate_step(
        self,
        caches: list,
        seed_token: int,
    ) -> tuple[list[int], str, int]:
        """Generate tokens for one reasoning step.

        seed_token is the last token from the parent's generation. It hasn't
        been processed into the cache yet, so we pass it as the prompt —
        stream_generate will process it and generate continuations from it.

        Args:
            caches: KV cache list (modified in place by generation).
            seed_token: First token to feed; seeds the continuation.

        Returns:
            (tokens, text, last_token) for the generated step.
        """
        from mlx_lm.sample_utils import make_sampler

        tokens = []
        text_pieces = []
        last_token = seed_token

        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=[seed_token],
            max_tokens=self.max_step_tokens,
            sampler=make_sampler(temp=self.temperature),
            prompt_cache=caches,
        ):
            tokens.append(response.token)
            text_pieces.append(response.text)
            last_token = response.token
            if response.finish_reason:
                break
            accumulated = "".join(text_pieces)
            if any(marker in accumulated for marker in self.step_end_tokens):
                break
            if len(tokens) >= self.max_step_tokens:
                break

        return tokens, "".join(text_pieces), last_token

    def _is_answer(self, text: str) -> bool:
        """Detect if a step contains a final answer."""
        markers = ["final answer", "therefore", "in conclusion", "the answer is", "∴"]
        lower = text.lower()
        return any(m in lower for m in markers)


# --- Built-in evaluators ---

def draft_logprob_evaluator(draft_model, tokenizer) -> Callable[[MCTSNode], float]:
    """Score a reasoning step by the draft model's average token log-probability.

    Higher log-prob = more fluent/expected reasoning = higher score.
    Used in Phase B before the PRM is available.

    Args:
        draft_model: Small fast model (7B) for scoring.
        tokenizer: Shared tokenizer.

    Returns:
        Callable(node) -> float in [0, 1].
    """
    def _evaluate(node: MCTSNode) -> float:
        if not node.tokens:
            return 0.5
        text = node.full_text
        if not text.strip():
            return 0.5
        try:
            import mlx.nn as nn
            from mlx_lm.models.cache import make_prompt_cache

            all_tokens = tokenizer.encode(text)
            if len(all_tokens) < 2:
                return 0.5
            # Single forward pass over context; score the last node token
            input_ids = mx.array(all_tokens[:-1])[None]
            target_id = all_tokens[-1]
            tmp_cache = make_prompt_cache(draft_model)
            logits = draft_model(input_ids, cache=tmp_cache)
            last_logits = logits[0, -1, :]
            log_probs = nn.log_softmax(last_logits)
            score_lp = float(log_probs[target_id].item())
            # Map log-prob in [-10, 0] → [0, 1]
            return max(0.0, min(1.0, (score_lp + 10.0) / 10.0))
        except Exception:
            return 0.5

    return _evaluate
