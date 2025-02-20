from typing import Literal

from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam)

OpenAIRole = Literal["system", "assistant", "user"]
ConversationHistory = list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam |
                                ChatCompletionAssistantMessageParam]
