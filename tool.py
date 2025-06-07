from typing import List
from smolagents import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    Tool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

from tools.tools import (
    vision_tool,
    youtube_frames_to_images,
    ask_youtube_video, 
    read_text_file,
    file_from_url,
    transcribe_youtube, 
    audio_to_text, 
    extract_text_via_ocr, 
    summarize_csv_data,
    summarize_excel_data,
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
        vision_tool,
        youtube_frames_to_images,
        ask_youtube_video, 
        read_text_file,
        file_from_url,
        transcribe_youtube, 
        audio_to_text, 
        extract_text_via_ocr, 
        summarize_csv_data,
        summarize_excel_data,
    ]
    return tools