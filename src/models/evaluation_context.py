import re
from pathlib import Path

import ujson
from loguru import logger
from ujson import JSONDecodeError

from src.config import OUTPUT_DIR_PATH
from src.generative_models.llm import LLM
from src.models.criterion import Criterion


class EvaluationContext:
    def __init__(self, story_id: str, trial_id: str, generative_model: LLM, criterion_objs: list[Criterion]):
        self.story_id = story_id
        self.trial_id = trial_id
        self.generative_model = generative_model
        self.criterion_objs = criterion_objs
        self.current_chunk_id: str = None

    @property
    def output_dir(self) -> Path:
        output_dir = OUTPUT_DIR_PATH / self.story_id / self.trial_id
        if self.current_chunk_id is not None:
            output_dir = output_dir / self.current_chunk_id
        return output_dir
    
    def is_data_already_evaluated(self, criterion: Criterion):
        return (self.output_dir / f"{criterion.name}.json").exists()

    def save_raw_output_to_file(self, raw_output: str, story: str, criterion: Criterion):
        try:
            parsed_output = raw_output
            if "```json" in parsed_output:
                parsed_output = re.search(r"```json(.*)```", parsed_output, re.DOTALL).group(1).strip()
            parsed_output = re.search(r"\{.*}", parsed_output, re.DOTALL).group(0).strip()
            parsed_output = ujson.loads(parsed_output)
        except JSONDecodeError as e:
            logger.error(f"Failed to parse output: {e}")
            parsed_output = {"error": str(e)}

        json_obj = {
            "id": self.story_id,
            "story": story,
            "trial_id": self.trial_id,
            "model": self.generative_model.model_name,
            "criterion": criterion.criterion,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / f"{criterion.name}.json", "w") as file:
            ujson.dump(json_obj, file, indent=2)
            logger.info(f"Saved raw output to {str(self.output_dir / f'{criterion.name}.json')}")

    def __str__(self):
        return f"EvaluationContext(story_data={self.story_id}, trial_id={self.trial_id}, generative_model={self.generative_model.model_name})"
    