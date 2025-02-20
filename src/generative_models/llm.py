from __future__ import annotations

from abc import ABC, abstractmethod

from src.types.openai import ConversationHistory


class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_content(self, messages: ConversationHistory) -> str:
        pass

    @abstractmethod
    def __str__(self):
        pass