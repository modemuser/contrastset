"""Adapters: the only way contrastset talks to a model.

Both the subject model and the generator model are reached through the same tiny
`Adapter` interface, so any backend that can turn a prompt into text works.
"""

from contrastset.adapters.anthropic import AnthropicAdapter
from contrastset.adapters.base import Adapter, AdapterError, Scorer, make_adapter
from contrastset.adapters.fake import FakeAdapter
from contrastset.adapters.hf import HFAdapter, Score
from contrastset.adapters.ollama import OllamaAdapter
from contrastset.adapters.openai import OpenAIAdapter

__all__ = ["Adapter", "AnthropicAdapter", "AdapterError", "FakeAdapter", "HFAdapter", "Score", "Scorer", "OllamaAdapter", "OpenAIAdapter", "make_adapter"]
