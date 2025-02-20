import os

import requests
import ujson
from loguru import logger

from src.types.dataclasses import StoryChunk


class StoryChunkRepository:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StoryChunkRepository, cls).__new__(cls)
            cls._instance._initialize()
            logger.info("StoryChunkRepository instance created")
        return cls._instance
    
    def _initialize(self):
        self.api_endpoint = os.getenv("DATA_API_ENDPOINT")

    def get(self, chunk_id: str) -> StoryChunk:
        response = requests.get(f"{self.api_endpoint}/api/v1/story-chunk/{chunk_id}")
        if response.status_code == 200:
            return ujson.loads(response.content)
        else:
            raise ValueError(f"Failed to get story chunk for chunk {chunk_id}")
