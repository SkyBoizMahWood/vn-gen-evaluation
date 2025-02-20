import os

import requests
import ujson
from loguru import logger

from src.types.dataclasses import StoryData


class StoryDataRepository(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StoryDataRepository, cls).__new__(cls)
            cls._instance._initialize()
            logger.info("StoryDataRepository instance created")
        return cls._instance

    def _initialize(self):
        self.api_endpoint = os.getenv("DATA_API_ENDPOINT")

    def get(self, story_id: str) -> StoryData:
        response = requests.get(f"{self.api_endpoint}/api/v1/story-data/{story_id}")
        if response.status_code == 200:
            return ujson.loads(response.content)
        else:
            raise ValueError(f"Failed to get story data for story {story_id}")
