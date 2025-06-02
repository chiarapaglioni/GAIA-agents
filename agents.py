from typing import Any, List, Optional

from smolagents import CodeAgent

from utils.logger import get_logger

logger = get_logger(__name__)


class Agent:
    """
    Agent class that wraps a CodeAgent and provides a callable interface for answering questions.

    Args:
        model (Any): The language model to use.
        tools (Optional[List[Any]]): List of tools to provide to the agent.
        prompt (Optional[str]): Custom prompt template for the agent.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Any]] = None,
        prompt: Optional[str] = None,
    ):
        logger.info("Initializing Agent")
        self.model = model
        self.tools = tools
        self.imports = [
            "pandas",
            "numpy",
            "os",
            "requests",
            "tempfile",
            "datetime",
            "json",
            "time",
            "re",
            "openpyxl",
            "pathlib",
            "sys",
        ]
        self.agent = CodeAgent(
            model=self.model,
            tools=self.tools,
            add_base_tools=True,
            additional_authorized_imports=self.imports,
        )
        self.prompt = prompt or (
            """
            You are an advanced AI assistant specialized in solving complex, real-world tasks that require multi-step reasoning, factual accuracy, and use of external tools.
        
            Follow these principles:
            - Be precise and concise. The final answer must strictly match the required format with no extra commentary.
            - Use tools intelligently. If a question involves external information, structured data, images, or audio, call the appropriate tool to retrieve or process it.
            - Reason step-by-step. Think through the solution logically and plan your actions carefully before answering.
            - Validate information. Always verify facts when possible instead of guessing.
            - Use code if needed. For calculations, parsing, or transformations, generate Python code and execute it. But be careful, some questions contains time-consuming tasks, so you should be careful with the code you run. Better analyze the question and think about the best way to solve it.
            - Don't forget to use `final_answer` to give the final answer.
            - Use name of file ONLY FROM "FILE:" section. THIS IF ALWAYS A FILE.

            IMPORTANT: When giving the final answer, output only the direct required result without any extra text like "Final Answer:" or explanations. YOU MUST RESPOND IN THE EXACT FORMAT AS THE QUESTION.

            QUESTION: {question}

            FILE: {context}

            ANSWER:
            """
        )
        logger.info("Agent initialized")

    def __call__(self, question: str, file_path: Optional[str] = None) -> str:
        """
        Run the agent to answer a question, optionally using a file as context.

        Args:
            question (str): The question to answer.
            file_path (Optional[str]): Path to a file to use as context (if any).

        Returns:
            str: The agent's answer as a string.
        """
        answer = self.agent.run(
            self.prompt.format(question=question, context=file_path)
        )
        answer = str(answer).strip("'").strip('"').strip()
        return answer