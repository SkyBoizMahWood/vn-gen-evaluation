from typing import Optional, TypedDict


class StoryNarrative(TypedDict):
    id: int
    speaker: str
    speaker_id: int
    scene_title: str
    scene_id: int
    text: str


class StoryBranch(TypedDict):
    source_chunk_id: str
    target_chunk_id: str
    choice: Optional[dict]


class StoryChunk(TypedDict):
    id: str
    story_id: str
    chapter: int
    story_so_far: str
    story: list[StoryNarrative]
    num_opportunities: int
    history: str


class StoryData(TypedDict):
    id: str
    title: str
    genre: str
    themes: list[str]
    main_scenes: list[dict]
    main_characters: list[dict]
    synopsis: str
    chapter_synopses: list[dict]
    beginning: str
    endings: list[dict]
    generated_by: str
    approach: str
    start_chunk_id: str
