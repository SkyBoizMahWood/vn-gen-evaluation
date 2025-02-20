from src.models.criterion import Criterion
from src.types.dataclasses import StoryChunk, StoryData
from src.utils.dataclass import (get_narrative_text, get_story_text,
                                 get_theme_text)


def story_chunk_evaluation_prompt(chunk: StoryChunk, story: StoryData, criterion: Criterion) -> str:
    concepts = f"\n\nConcepts:\n{get_theme_text(story)}" if criterion.name == "inspiration" else ""

    return f"""Evaluate the following visual novel game story according to the specified criteria and assign a score with a total of 10 for each criterion, where 10 is the best and 0 is the worst. Provide reasons for your scores. Make sure to output it in a MarkDown code block, i.e., between ```json and ```.

Output format:
```json
{{
"chunk_id": "{chunk['id']}",
"{criterion.name}": [{{ "factor_name": <factor>, "score": <int score out of 10>, "reason": <reason for the given score> }}]
}}
```

Criterion description:
{criterion.criterion}{concepts}

Story:
{get_narrative_text(chunk)}"""


def story_data_evaluation_prompt(story: StoryData, criterion: Criterion) -> str:
    concepts = f"\n\nConcepts:\n{get_theme_text(story)}" if criterion.name == "inspiration" else ""

    return f"""Evaluate the following visual novel game story according to the specified criteria and assign a score with a total of 10 for each criterion, where 10 is the best and 0 is the worst. Provide reasons for your scores. Make sure to output it in a MarkDown code block, i.e., between ```json and ```.

Output format:
```json
{{
"story_id": "{story['id']}",
"{criterion.name}": [{{ "factor_name": <factor>, "score": <int score out of 10>, "reason": <reason for the given score> }}]
}}
```

Criterion description:
{criterion.criterion}{concepts}

Story:
{get_story_text(story)}"""