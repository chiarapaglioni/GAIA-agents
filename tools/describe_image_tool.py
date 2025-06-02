import base64
import os
from openai import OpenAI
from smolagents import Tool

client = OpenAI()


class DescribeImageTool(Tool):
    """
    Tool to analyze and describe any image using GPT-4 Vision API.

    Args:
        image_path (str): Path to the image file.
        description_type (str): Type of description to generate. Options:
            - "general": General description of the image
            - "detailed": Detailed analysis of the image
            - "chess": Analysis of a chess position
            - "text": Extract and describe text from the image
            - "custom": Custom description based on user prompt

    Returns:
        str: Description of the image based on the requested type.
    """

    name = "describe_image"
    description = "Analyzes and describes images using GPT-4 Vision API"
    inputs = {
        "image_path": {"type": "string", "description": "Path to the image file"},
        "description_type": {
            "type": "string",
            "description": "Type of description to generate (general, detailed, chess, text, custom)",
            "nullable": True,
        },
        "custom_prompt": {
            "type": "string",
            "description": "Custom prompt for description (only used when description_type is 'custom')",
            "nullable": True,
        },
    }
    output_type = "string"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_prompt(self, description_type: str, custom_prompt: str = None) -> str:
        """Get appropriate prompt based on description type."""
        prompts = {
            "general": "Provide a general description of this image. Focus on the main subjects, colors, and overall scene.",
            "detailed": """Analyze this image in detail. Include:
            1. Main subjects and their relationships
            2. Colors, lighting, and composition
            3. Any text or symbols present
            4. Context or possible meaning
            5. Notable details or interesting elements""",
            "chess": """Analyze this chess position and provide a detailed description including:
            1. List of pieces on the board for both white and black
            2. Whose turn it is to move
            3. Basic evaluation of the position
            4. Any immediate tactical opportunities or threats
            5. Suggested next moves with brief explanations""",
            "text": "Extract and describe any text present in this image. If there are multiple pieces of text, organize them clearly.",
        }
        return (
            custom_prompt
            if description_type == "custom"
            else prompts.get(description_type, prompts["general"])
        )

    def forward(
        self,
        image_path: str,
        description_type: str = "general",
        custom_prompt: str = None,
    ) -> str:
        try:
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            # Encode the image
            base64_image = self.encode_image(image_path)

            # Get appropriate prompt
            prompt = self.get_prompt(description_type, custom_prompt)

            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error analyzing image: {str(e)}"
