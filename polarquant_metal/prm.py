"""Process Reward Model wrapper for MCTS step scoring.

Wraps a loaded PRM (e.g. Skywork-o1-Open-PRM-Qwen-2.5-7B) to score intermediate
reasoning steps. Plugs into MCTSTree(evaluator=...) as a callable.

Usage:
    prm = ProcessRewardModel.load("Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B-4bit")
    evaluator = prm.as_evaluator(question="What is the diagnosis?")
    tree = MCTSTree(model, tokenizer, caches, evaluator=evaluator)

Score semantics:
    1.0  = step is clearly correct / strongly advances toward answer
    0.5  = neutral / uncertain
    0.0  = step is clearly wrong

Implementation:
    Skywork-PRM returns per-token reward logits. We extract the logit at the last
    token position for the "good" (+) vs "bad" (-) reward tokens, apply softmax
    over those two positions, and return P(good) as the score.

    For models without an explicit reward token, we fall back to draft-model
    log-probability: higher fluency ≈ more plausible reasoning step.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import mlx.core as mx
import mlx_lm

logger = logging.getLogger(__name__)

_SKYWORK_GOOD_TOKEN = "+"
_SKYWORK_BAD_TOKEN = "-"


class ProcessRewardModel:
    """Wraps a reward model for step-level scoring in MCTS.

    Args:
        model: Loaded MLX language model with LM head.
        tokenizer: Tokenizer matched to the model.
        good_token: String for the positive reward token (Skywork: "+").
        bad_token: String for the negative reward token (Skywork: "-").
        use_logprob_fallback: Fall back to log-prob scoring if reward tokens
            are not found in the vocabulary.
    """

    def __init__(
        self,
        model,
        tokenizer,
        good_token: str = _SKYWORK_GOOD_TOKEN,
        bad_token: str = _SKYWORK_BAD_TOKEN,
        use_logprob_fallback: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.use_logprob_fallback = use_logprob_fallback
        self.good_token_id: Optional[int] = None
        self.bad_token_id: Optional[int] = None

        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        self.good_token_id = vocab.get(good_token)
        self.bad_token_id = vocab.get(bad_token)

        if self.good_token_id is None:
            if use_logprob_fallback:
                logger.warning(
                    f"PRM: reward token '{good_token}' not in vocab — "
                    "using log-probability fallback scoring."
                )
            else:
                raise ValueError(
                    f"Reward token '{good_token}' not in tokenizer vocab. "
                    "Set use_logprob_fallback=True or supply correct good_token."
                )

    @classmethod
    def load(
        cls,
        model_id: str,
        good_token: str = _SKYWORK_GOOD_TOKEN,
        bad_token: str = _SKYWORK_BAD_TOKEN,
        **kwargs,
    ) -> ProcessRewardModel:
        """Load a PRM from HuggingFace or local path.

        Args:
            model_id: HF model ID or local directory.
                Recommended: "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B-4bit"
        """
        logger.info(f"Loading PRM: {model_id}")
        model, tokenizer = mlx_lm.load(model_id)
        return cls(model, tokenizer, good_token=good_token, bad_token=bad_token, **kwargs)

    def score_step(self, question: str, reasoning_prefix: str) -> float:
        """Score a reasoning step given the original question and accumulated text.

        Args:
            question: The original question/prompt.
            reasoning_prefix: All reasoning text up to and including this step.

        Returns:
            float in [0, 1]. Higher = better reasoning step.
        """
        prompt = self._format_prompt(question, reasoning_prefix)
        tokens = self.tokenizer.encode(prompt)
        if self.good_token_id is not None:
            return self._score_via_reward_tokens(tokens)
        return self._score_via_logprob(tokens)

    def as_evaluator(self, question: str) -> Callable:
        """Return a callable compatible with MCTSTree(evaluator=...).

        Args:
            question: The question being reasoned about (fixed across the tree).

        Returns:
            Callable(MCTSNode) -> float.
        """
        def _fn(node) -> float:
            return self.score_step(question, node.full_text)
        return _fn

    # --- Internal ---

    def _format_prompt(self, question: str, reasoning: str) -> str:
        """Format question + reasoning for the PRM.

        Skywork-PRM convention: present the question + partial solution.
        The model returns a high probability for "+" at the end when the step
        is correct, and "-" when incorrect.
        """
        return (
            f"Question: {question}\n\n"
            f"Partial solution:\n{reasoning}\n\n"
            "Is this step correct? Answer with + or -:"
        )

    def _score_via_reward_tokens(self, tokens: list[int]) -> float:
        """Score by softmax(logit[good_token], logit[bad_token]) at last position."""
        try:
            input_arr = mx.array(tokens)[None]       # (1, L)
            logits = self.model(input_arr)           # (1, L, vocab_size)
            last = logits[0, -1, :]                  # (vocab_size,)
            good_logit = float(last[self.good_token_id].item())
            if self.bad_token_id is not None:
                bad_logit = float(last[self.bad_token_id].item())
                m = max(good_logit, bad_logit)
                exp_g = math.exp(good_logit - m)
                exp_b = math.exp(bad_logit - m)
                return exp_g / (exp_g + exp_b)
            # Single token: sigmoid
            return 1.0 / (1.0 + math.exp(-good_logit))
        except Exception as exc:
            logger.warning(f"PRM reward-token scoring failed ({exc}), falling back.")
            return self._score_via_logprob(tokens)

    def _score_via_logprob(self, tokens: list[int]) -> float:
        """Fallback: score via mean per-token log-probability (fluency proxy)."""
        if len(tokens) < 2:
            return 0.5
        total_lp = 0.0
        count = 0
        for resp in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=tokens[:-1],
            max_tokens=min(len(tokens) - 1, 64),
        ):
            if resp.logprobs is not None:
                lp = float(mx.max(resp.logprobs).item())
                total_lp += lp
                count += 1
            if resp.finish_reason:
                break
        if count == 0:
            return 0.5
        mean_lp = total_lp / count
        # Map [-10, 0] → [0, 1]
        return max(0.0, min(1.0, (mean_lp + 10.0) / 10.0))
