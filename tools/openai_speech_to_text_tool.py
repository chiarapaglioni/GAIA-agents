import os
import whisper
from smolagents import Tool


class OpenAISpeechToTextTool(Tool):
    """
    Tool to convert speech to text using OpenAI's Whisper model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcribed text from the audio file.
    """

    name = "transcribe_audio"
    description = "Transcribes audio to text and returns the text"
    inputs = {
        "audio_path": {"type": "string", "description": "Path to the audio file"},
    }
    output_type = "string"

    def forward(self, audio_path: str) -> str:
        try:
            model = whisper.load_model("small")

            if not os.path.exists(audio_path):
                return f"Error: Audio file not found at {audio_path}"

            result = model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
