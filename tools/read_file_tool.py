from smolagents import Tool

class ReadFileTool(Tool):
    """
    Tool to read a file and return its content.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        str: Content of the file or error message.
    """

    name = "read_file"
    description = "Reads a file and returns its content"
    inputs = {
        "file_path": {"type": "string", "description": "Path to the file to read"},
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        try:
            with open(file_path, "r") as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
