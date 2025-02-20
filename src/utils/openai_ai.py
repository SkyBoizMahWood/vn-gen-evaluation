import copy

from src.types.openai import ConversationHistory, OpenAIRole


def append_openai_message(
        message: str,
        role: OpenAIRole = "user",
        history: ConversationHistory = None) -> ConversationHistory:
    if history is None:
        history = []

    history = copy.deepcopy(history)

    history.append({
        "role": role,
        "content": message
    })

    return history
