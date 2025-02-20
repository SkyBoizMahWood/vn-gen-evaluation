from src.types.dataclasses import StoryChunk, StoryData


def get_theme_text(story: StoryData):
    theme_text = ", ".join([theme for theme in story["themes"]])
    return theme_text


def get_story_text(story: StoryData):
    ending_text = "\n".join([f"{i+1}. {ending['ending']}" for i, ending in enumerate(story["endings"])])
    story_text = f"Synopsis:\n{story['synopsis']}\nEndings:\n{ending_text}"
    return story_text


def get_narrative_text(chunk: StoryChunk):
    narrative_text = '\n'.join([f"{narrative['speaker']}: {narrative['text']}" for narrative in chunk["story"]])
    return narrative_text
