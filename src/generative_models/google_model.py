import os
from time import sleep

from google import genai
from google.genai import types
from google.genai.errors import APIError
from google.genai.types import HarmCategory, HarmBlockThreshold
from requests import ReadTimeout

from src.generative_models.llm import LLM
from src.types.openai import ConversationHistory
from src.utils.google_ai import map_openai_history_to_google_history

config=types.GenerateContentConfig(
    temperature=0.0,
    safety_settings=[
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.OFF
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.OFF
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.OFF
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.OFF
        ),
        types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=HarmBlockThreshold.OFF
        ),
    ]
)


class GoogleModel(LLM):
    def __init__(self, model_name: str):
        api_keys_str = os.environ.get("GOOGLE_API_KEY", "")
        self.api_keys = api_keys_str.split(",") if api_keys_str else []
        if not self.api_keys:
            raise ValueError("GOOGLE_API_KEY environment variable not set or empty")
        print(f"Google API keys: {self.api_keys}")
        self.current_key_index = 0
        super().__init__(model_name)
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))

    def _gemini(self, messages: ConversationHistory) -> str:
        last_message = messages[-1]
        if last_message["role"] in ["system", "assistant"]:
            raise ValueError(f"Last message role is not user: {last_message['role']}")
        history = map_openai_history_to_google_history(messages[:-1])
        chat = self.client.chats.create(model=self.model_name, history=history)
        chat_completion = chat.send_message(
            message=last_message["content"],
            config=config,
        )
        response = chat_completion.text
        return response.strip()

    def generate_content(self, messages: ConversationHistory) -> str:
        try:
            return self._gemini(messages)
        except (APIError) as e:
            print(f"Google API error: {e}")
            # Switch to next API key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))
            print(f"Switched to API key index {self.current_key_index}")
            sleep(3)
            return self.generate_content(messages)
        except ReadTimeout as e:
            print(f"Read Timeout error: {e}")
            sleep(3)
            return self.generate_content(messages)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
    
    def __str__(self):
        return f"GoogleModel(model_name={self.model_name})"
