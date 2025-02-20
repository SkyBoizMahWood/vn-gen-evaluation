from pathlib import Path

from src.config import OUTPUT_DIR_PATH
from src.models.criterion import Criterion


class AnalysisContext:
    def __init__(self, story_id: str, trial_id: str, criterion_objs: list[Criterion]):
        self.story_id = story_id
        self.trial_id = trial_id
        self.criterion_objs = criterion_objs

    @property
    def data_dir(self) -> Path:
        return OUTPUT_DIR_PATH / self.story_id / self.trial_id

    def __str__(self):
        return f"AnalysisContext(story_data={self.story_id}, trial_id={self.trial_id})"
    