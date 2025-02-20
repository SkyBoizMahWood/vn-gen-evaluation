from time import sleep

from openai import (APIConnectionError, APIError, APITimeoutError, OpenAI,
                    RateLimitError)

from src.generative_models.llm import LLM
from src.types.openai import ConversationHistory


class OpenAIModel(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI()
    
    def generate_content(self, messages: ConversationHistory, stream: bool = False) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                seed=42,
                temperature=0,
                stream=stream,
            )
            response = ""
            if stream:
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()
            else:
                response = chat_completion.choices[0].message.content
            return response.strip()
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            print(f"OpenAI API error: {e}")
            sleep(3)
            return self.generate_content(messages)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
    
    def __str__(self):
        return f"OpenAIModel(model_name={self.model_name})"
