from src.generative_models.anthropic_model import AnthropicModel
from src.generative_models.google_model import GoogleModel
from src.generative_models.llm import LLM
from src.generative_models.local_model import LocalModel
from src.generative_models.openai_model import OpenAIModel


def get_generation_model(model_name: str) -> LLM:
    if model_name in ["gemini-1.0-pro-001", "gemini-2.0-flash-001", "gemini-2.5-pro"]:
        return GoogleModel(model_name)
    elif model_name in ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09"]:
        return OpenAIModel(model_name)
    elif model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"]:
        return AnthropicModel(model_name)
    else:
        return LocalModel(model_name)
