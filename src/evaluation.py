from loguru import logger

from src.models.evaluation_context import EvaluationContext
from src.prompts import (story_chunk_evaluation_prompt,
                         story_data_evaluation_prompt)
from src.repositories.story_branch import StoryBranchRepository
from src.repositories.story_chunk import StoryChunkRepository
from src.types.dataclasses import StoryData
from src.utils.dataclass import get_narrative_text, get_story_text
from src.utils.openai_ai import append_openai_message


def run_evaluation(context: EvaluationContext, story_data: StoryData):
    logger.info(f"Evaluating story {context.story_id}")
    evaluate_story_data(context, story_data)
    evaluate_story_chunks(context, story_data)
    logger.info(f"Finished evaluation for story {context.story_id}")


def evaluate_story_data(context: EvaluationContext, story_data: StoryData):
    for criterion in context.criterion_objs:
        logger.info(f"Evaluating {criterion.name}")
        if context.is_data_already_evaluated(criterion):
            logger.info(f"Skipping {criterion.name} as it is already evaluated")
            continue
        prompt = story_data_evaluation_prompt(story_data, criterion)
        response = context.generative_model.generate_content(append_openai_message(prompt))
        context.save_raw_output_to_file(response, get_story_text(story_data), criterion)


def evaluate_story_chunks(context: EvaluationContext, story_data: StoryData):
    frontiers = [story_data["start_chunk_id"]]
    while frontiers:
        chunk_id = frontiers.pop(0)
        evaluate_story_chunk(context, story_data, chunk_id)
        new_branches = StoryBranchRepository().list_branches_from(chunk_id)
        frontiers.extend([branch["target_chunk_id"] for branch in new_branches])
        logger.info(f"Data added to frontiers: +{len(new_branches)}, Total: {len(frontiers)}")


def evaluate_story_chunk(context: EvaluationContext, story_data: StoryData, chunk_id: str):
    logger.info(f"Evaluating chunk {chunk_id}")
    context.current_chunk_id = chunk_id
    story_chunk = StoryChunkRepository().get(chunk_id)
    for criterion in context.criterion_objs:
        if context.is_data_already_evaluated(criterion):
            logger.info(f"Skipping {criterion.name} as it is already evaluated")
            continue
        logger.info(f"Evaluating {criterion.name}")
        prompt = story_chunk_evaluation_prompt(story_chunk, story_data, criterion)
        response = context.generative_model.generate_content(append_openai_message(prompt))
        context.save_raw_output_to_file(response, get_narrative_text(story_chunk), criterion)
    logger.info(f"Finished evaluating chunk {story_chunk['id']}")
