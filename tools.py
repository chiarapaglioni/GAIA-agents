from typing import List
from tools.describe_image_tool import DescribeImageTool
from tools.openai_speech_to_text_tool import OpenAISpeechToTextTool
from tools.read_file_tool import ReadFileTool
from tools.youtube_transcription_tool import YouTubeTranscriptionTool
from tools.table_extractor_tool import TableExtractorTool

from smolagents import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    Tool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

def get_tools() -> List[Tool]:
    """
    Returns a list of available tools for the agent.

    Returns:
        List[Tool]: List of initialized tool instances.
    """
    tools = [
        DuckDuckGoSearchTool(),
        PythonInterpreterTool(),
        WikipediaSearchTool(),
        VisitWebpageTool(),
        DescribeImageTool(),
        OpenAISpeechToTextTool(),
        ReadFileTool(),
        YouTubeTranscriptionTool(),
        TableExtractorTool(),
    ]
    return tools