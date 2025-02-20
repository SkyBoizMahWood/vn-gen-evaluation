from itertools import product
from pathlib import Path

import numpy as np
import ujson

from src.models.analysis_context import AnalysisContext
from src.models.criterion import Criterion


def run_analysis(context: AnalysisContext) -> dict:
    criterion_scores = {criterion.name: [] for criterion in context.criterion_objs}
    evaluation_obj = {criterion.name: {} for criterion in context.criterion_objs}
    # Evaluate main story data
    for criterion in context.criterion_objs:
        score_avg = calculate_criterion_scores(context.data_dir, criterion)
        if score_avg > 0:
            criterion_scores[criterion.name].append(score_avg)

    # Evaluate chunks
    chunk_dir_list = [c for c in context.data_dir.iterdir() if c.is_dir()]
    for story_chunk_path, criterion in product(chunk_dir_list, context.criterion_objs):
        score_avg = calculate_criterion_scores(story_chunk_path, criterion)
        if score_avg > 0:
            criterion_scores[criterion.name].append(score_avg)
            
    # Calculate average and standard deviation
    for criterion in context.criterion_objs:
        evaluation_obj[criterion.name]["mean"] = np.mean(criterion_scores[criterion.name])
        evaluation_obj[criterion.name]["sd"] = np.std(criterion_scores[criterion.name])
    return evaluation_obj


def calculate_criterion_scores(file_path: Path, criterion: Criterion) -> float:
    with open(file_path / f"{criterion.name}.json", 'r') as file:
        data = ujson.load(file)
    if "error" in data["parsed_output"]:
        return 0
    factor_scores = data["parsed_output"][criterion.name]
    score_avg, _ = calc_mean_sd([factor["score"] for factor in factor_scores])
    return score_avg


def calc_mean_sd(criterion_scores: list[float]) -> tuple[float, float]:
    if len(criterion_scores) == 0:
        return 0, 0
    return np.mean(criterion_scores), np.std(criterion_scores)
