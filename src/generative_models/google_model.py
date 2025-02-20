import os
from time import sleep

import google.generativeai as genai
from google.api_core.exceptions import (DeadlineExceeded, InternalServerError,
                                        ServiceUnavailable, TooManyRequests)
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from src.generative_models.llm import LLM
from src.types.openai import ConversationHistory
from src.utils.google_ai import map_openai_history_to_google_history

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}


class GoogleModel(LLM):
    def __init__(self, model_name: str):
        api_keys_str = os.environ.get("GOOGLE_API_KEY", "")
        self.api_keys = api_keys_str.split(",") if api_keys_str else []
        if not self.api_keys:
            raise ValueError("GOOGLE_API_KEY environment variable not set or empty")
        print(f"Google API keys: {self.api_keys}")
        self.current_key_index = 0
        genai.configure(api_key=self.api_keys[self.current_key_index])
        super().__init__(model_name)
        self.client = GenerativeModel(self.model_name)

    def _gemini(self, messages: ConversationHistory, stream: bool = False) -> str:
        last_message = messages[-1]
        if last_message["role"] in ["system", "assistant"]:
            raise ValueError(f"Last message role is not user: {last_message['role']}")
        history = map_openai_history_to_google_history(messages[:-1])
        chat = self.client.start_chat(history=history)
        chat_completion = chat.send_message(
            last_message["content"],
            generation_config=GenerationConfig(
                temperature=0,
            ),
            safety_settings=safety_settings,
            stream=stream,
        )
        response = ""
        if stream:
            for chunk in chat_completion:
                response += chunk.candidates[0].content.parts[0].text
                print(chunk.candidates[0].content.parts[0].text, end="", flush=True)
            print()
        else:
            response = chat_completion.text
        return response.strip()

    def generate_content(self, messages: ConversationHistory) -> str:
        try:
            return self._gemini(messages)
        except (ServiceUnavailable, InternalServerError, TooManyRequests, DeadlineExceeded) as e:
            print(f"Google API error: {e}")
            # Switch to next API key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            genai.configure(api_key=self.api_keys[self.current_key_index])
            self.client = GenerativeModel(self.model_name)
            print(f"Switched to API key index {self.current_key_index}")
            sleep(3)
            return self.generate_content(messages)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
    
    def __str__(self):
        return f"GoogleModel(model_name={self.model_name})"
