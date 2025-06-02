from smolagents import Tool
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptionTool(Tool):
    """
    Tool to fetch the transcript of a YouTube video given its URL.

    Args:
        video_url (str): YouTube video URL.

    Returns:
        str: Transcript of the video as a single string.
    """

    name = "youtube_transcription"
    description = "Fetches the transcript of a YouTube video given its URL"
    inputs = {
        "video_url": {"type": "string", "description": "YouTube video URL"},
    }
    output_type = "string"

    def forward(self, video_url: str) -> str:
        video_id = video_url.strip().split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
