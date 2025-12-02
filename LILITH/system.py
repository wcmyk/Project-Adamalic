"""Multi-LLM coordinator that separates general and code-specialized angels."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

import torch

from .config import ModelConfig, TrainingConfig
from .data import CharacterTokenizer
from .model import GPTDecoder
from .train import train

Route = Literal["general", "code", "auto"]


@dataclass
class LLMProfile:
    """Container describing an instantiated LLM for a specific role."""

    name: str
    role: Literal["general", "code"]
    tokenizer: CharacterTokenizer
    model: GPTDecoder
    block_size: int


class AngelicMultiLLM:
    """Bootstrap and route between general-purpose and code-focused angels."""

    def __init__(self, general: LLMProfile, code: LLMProfile):
        self.general = general
        self.code = code

    @classmethod
    def bootstrap(
        cls,
        general_corpus: Iterable[str],
        code_corpus: Iterable[str],
        general_model: ModelConfig,
        code_model: ModelConfig,
        train_config: TrainingConfig,
        block_size: int = 128,
    ) -> "AngelicMultiLLM":
        general_tokenizer = CharacterTokenizer(general_corpus)
        code_tokenizer = CharacterTokenizer(code_corpus)

        general_model_cfg = replace(general_model, vocab_size=general_tokenizer.vocab_size)
        code_model_cfg = replace(code_model, vocab_size=code_tokenizer.vocab_size)

        general_model_inst = train(
            corpus=general_corpus,
            model_config=general_model_cfg,
            train_config=train_config,
            block_size=block_size,
        )
        code_model_inst = train(
            corpus=code_corpus,
            model_config=code_model_cfg,
            train_config=train_config,
            block_size=block_size,
        )

        general_profile = LLMProfile(
            name="General Angel",
            role="general",
            tokenizer=general_tokenizer,
            model=general_model_inst,
            block_size=block_size,
        )
        code_profile = LLMProfile(
            name="Coding Angel",
            role="code",
            tokenizer=code_tokenizer,
            model=code_model_inst,
            block_size=block_size,
        )
        return cls(general=general_profile, code=code_profile)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        route: Route = "auto",
    ) -> str:
        profile = self._select_profile(prompt, route)
        device = next(profile.model.parameters()).device
        profile.model.eval()
        prompt_ids = torch.tensor([profile.tokenizer.encode(prompt)], device=device)
        tokens = profile.model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        return profile.tokenizer.decode(tokens[0].tolist())

    def _select_profile(self, prompt: str, route: Route) -> LLMProfile:
        if route == "general":
            return self.general
        if route == "code":
            return self.code
        # auto-routing heuristics: favor the code specialist if the prompt contains common code tokens
        code_markers = ("def ", "class ", "{", "}", ";", "import ", "#include", "</", "console.", "fn ")
        if any(marker in prompt for marker in code_markers):
            return self.code
        return self.general


__all__ = ["AngelicMultiLLM", "LLMProfile", "Route"]
