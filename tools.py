from typing import List

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
    ]
    return tools