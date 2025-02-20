import os

import requests
import ujson
from loguru import logger

from src.types.dataclasses import StoryBranch


class StoryBranchRepository:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StoryBranchRepository, cls).__new__(cls)
            cls._instance._initialize()
            logger.info("StoryBranchRepository instance created")
        return cls._instance
    
    def _initialize(self):
        self.api_endpoint = os.getenv("DATA_API_ENDPOINT")

    def list_branches_from(self, chunk_id: str) -> list[StoryBranch]:
        response = requests.get(f"{self.api_endpoint}/api/v1/story-branch/list/{chunk_id}")
        if response.status_code == 200:
            return ujson.loads(response.content)
        else:
            raise ValueError(f"Failed to get story branches for chunk {chunk_id}")
