from typing import Any, List, Optional

from smolagents import CodeAgent

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class Agent:
    """
    Agent class that wraps a CodeAgent and provides a callable interface for answering questions.

    Args:
        model (Any): The language model to use.
        tools (Optional[List[Any]]): List of tools to provide to the agent.
        prompt (Optional[str]): Custom prompt template for the agent.
        verbose (bool): Whether to print debug information.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Any]] = None,
        prompt: Optional[str] = None,
        verbose: bool = False
    ):
        logger.info("Initializing Agent")
        self.model = model
        self.tools = tools
        self.verbose = verbose
        self.imports = [
            "pandas", "numpy", "os", "requests", "tempfile",
            "datetime", "json", "time", "re", "openpyxl",
            "pathlib", "sys"
        ]
        
        self.agent = CodeAgent(
            model=self.model,
            tools=self.tools,
            add_base_tools=True,
            additional_authorized_imports=self.imports,
        )
        
        self.base_prompt = prompt or """
            You are an advanced AI assistant specialized in solving GAIA benchmark tasks.
            Follow these rules strictly:
            1. Be precise - return ONLY the exact answer requested
            2. Use tools when needed (especially for file analysis)
            3. For reversed text questions, answer in normal text
            4. Never include explanations or reasoning in the final answer
            5. Always return the result — do not just print it

            {context}

            Remember: GAIA requires exact answer matching. Just provide the factual answer.
            """
        logger.info("Agent initialized")

    def __call__(self, question: str, files: List[str] = None) -> str:
        """Main interface that logs inputs/outputs and handles timing."""
        if self.verbose:
            print(f"Agent received question: {question[:50]}... with files: {files}")
        
        result = self.answer_question(question, files[0] if files else None)        
        return result

    def answer_question(self, question: str, task_file_path: Optional[str] = None) -> str:
        """
        Process a GAIA benchmark question with optional file context.
        
        Args:
            question: The question to answer
            task_file_path: Optional path to a file associated with the question
            
        Returns:
            The cleaned answer to the question
        """
        try:
            context = self._build_context(question, task_file_path)
            full_prompt = self.base_prompt.format(context=context)
            
            if self.verbose:
                print("Generated prompt:", full_prompt[:200] + "...")
            
            answer = self.agent.run(full_prompt)
            return self._clean_answer(str(answer))
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"ERROR: {str(e)}"

    def _build_context(self, question: str, file_path: Optional[str]) -> str:
        """Constructs the context section based on question and file."""
        context_lines = [f"QUESTION: {question}"]
        
        if file_path:
            context_lines.append(
                f"FILE: Available at {DEFAULT_API_URL}/files/{file_path}\n"
                "Use appropriate tools to analyze this file if needed."
            )
        
        # Handle reversed text questions
        if self._is_reversed_text(question):
            context_lines.append(
                f"NOTE: This question contains reversed text. "
                f"Original: {question}\nReversed: {question[::-1]}"
            )
        
        return "\n".join(context_lines)

    def _is_reversed_text(self, text: str) -> bool:
        """Detects if text appears to be reversed."""
        return text.startswith(".") or ".rewsna eht sa" in text

    def _clean_answer(self, answer: str) -> str:
        """Cleans the raw answer to match GAIA requirements."""
        # Remove common prefixes/suffixes
        for prefix in ["Final Answer:", "Answer:", "=>"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
        
        # Remove quotes and whitespace
        answer = answer.strip(" '\"")
        
        # Special handling for reversed answers
        if self._is_reversed_text(answer):
            return answer[::-1]
        
        return answer