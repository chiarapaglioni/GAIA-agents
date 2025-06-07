import os
import numpy
import tempfile
import requests
import whisper
import imageio
import yt_dlp

from PIL import Image
from typing import List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from smolagents import tool, LiteLLMModel
import google.generativeai as genai
from pytesseract import image_to_string

load_dotenv()

#  Vision Tool 
@tool
def vision_tool(prompt: str, image_list: List[Image.Image]) -> str:
    """
    Analyzes one or more images using a multimodal model.
    Args:
        prompt (str): The user question or task.
        image_list (List[PIL.Image.Image]): A list of image objects.
    Returns:
        str: Model's response to the prompt about the images.
    """
    model_id = "gemini/gemini-1.5-flash"
    model = LiteLLMModel(model_id=model_id, api_key=os.getenv("GEMINI_API"), temperature=0.2)
    
    payload = [{"type": "text", "text": prompt}] + [{"type": "image", "image": img} for img in image_list]
    return model([{"role": "user", "content": payload}]).content


#  YouTube Frame Sampler 
@tool
def youtube_frames_to_images(url: str, every_n_seconds: int = 5) -> List[Image.Image]:
    """
    Downloads a YouTube video and extracts frames at regular intervals.

    Args:
        url (str): The URL of the YouTube video to process.
        every_n_seconds (int): The time interval in seconds between extracted frames.

    Returns:
        List[Image.Image]: A list of sampled frames as PIL images.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_cfg = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": os.path.join(temp_dir, "yt_video.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "force_ipv4": True
        }
        with yt_dlp.YoutubeDL(ydl_cfg) as ydl:
            ydl.extract_info(url, download=True)

        video_file = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.mp4')), None)
        reader = imageio.get_reader(video_file)
        fps = reader.get_meta_data().get("fps", 30)
        interval = int(fps * every_n_seconds)

        return [Image.fromarray(frame) for i, frame in enumerate(reader) if i % interval == 0]


#  YouTube QA via File URI 
@tool
def ask_youtube_video(url: str, question: str) -> str:
    """
    Sends a YouTube video to a multimodal model and asks a question about it.

    Args:
        url (str): The URI of the video file (already uploaded and hosted).
        question (str): The natural language question to ask about the video.

    Returns:
        str: The model's answer to the question.
    """
    try:
        client = genai.Client(api_key=os.getenv('GEMINI_API'))
        model = 'models/gemini-1.5-flash-latest'  # Update to latest model naming
        response = client.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [
                    {"text": question},
                    {"file_data": {"file_uri": url}}
                ]}
            ]
        )
        return response.text
    except Exception as e:
        return f"Error asking {model} about video: {str(e)}"


#  File Reading Tool 
@tool
def read_text_file(file_path: str) -> str:
    """
    Reads plain text content from a file.

    Args:
        file_path (str): The full path to the text file.

    Returns:
        str: The contents of the file, or an error message.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


#  File Downloader 
@tool
def file_from_url(url: str, save_as: Optional[str] = None) -> str:
    """
    Downloads a file from a URL and saves it locally.

    Args:
        url (str): The URL of the file to download.
        save_as (Optional[str]): Optional filename to save the file as.

    Returns:
        str: The local file path or an error message.
    """
    try:
        if not save_as:
            parsed = urlparse(url)
            save_as = os.path.basename(parsed.path) or f"file_{os.urandom(4).hex()}"

        file_path = os.path.join(tempfile.gettempdir(), save_as)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return f"File saved to {file_path}"
    except Exception as e:
        return f"Download failed: {e}"


#  Audio Transcription (YouTube) 
@tool
def transcribe_youtube(yt_url: str) -> str:
    """
    Transcribes the audio from a YouTube video using Whisper.

    Args:
        yt_url (str): The URL of the YouTube video.

    Returns:
        str: The transcribed text of the video.
    """
    model = whisper.load_model("small")

    with tempfile.TemporaryDirectory() as tempdir:
        ydl_opts = {
            "format": "bestaudio",
            "outtmpl": os.path.join(tempdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }],
            "quiet": True,
            "force_ipv4": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(yt_url, download=True)

        wav_file = next((os.path.join(tempdir, f) for f in os.listdir(tempdir) if f.endswith(".wav")), None)
        return model.transcribe(wav_file)['text']


#  Audio File Transcriber 
@tool
def audio_to_text(audio_path: str) -> str:
    """
    Transcribes an uploaded audio file into text using Whisper.

    Args:
        audio_path (str): The local file path to the audio file.

    Returns:
        str: The transcribed text or an error message.
    """
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        return f"Failed to transcribe: {e}"


#  OCR 
@tool
def extract_text_via_ocr(image_path: str) -> str:
    """
    Extracts text from an image using Optical Character Recognition (OCR).

    Args:
        image_path (str): The local path to the image file.

    Returns:
        str: The extracted text or an error message.
    """
    try:
        img = Image.open(image_path)
        return image_to_string(img)
    except Exception as e:
        return f"OCR failed: {e}"


#  CSV Analyzer 
@tool
def summarize_csv_data(path: str, query: str = "") -> str:
    """
    Provides a summary of the contents of a CSV file.

    Args:
        path (str): The file path to the CSV file.
        query (str): Optional query to run on the data.

    Returns:
        str: Summary statistics and column details or an error message.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return f"Loaded CSV with {len(df)} rows. Columns: {list(df.columns)}\n\n{df.describe()}"
    except Exception as e:
        return f"CSV error: {e}"


#  Excel Analyzer 
@tool
def summarize_excel_data(path: str, query: str = "") -> str:
    """
    Provides a summary of the contents of an Excel file.

    Args:
        path (str): The file path to the Excel file (.xls or .xlsx).
        query (str): Optional query to run on the data.

    Returns:
        str: Summary statistics and column details or an error message.
    """
    try:
        import pandas as pd
        df = pd.read_excel(path)
        return f"Excel file with {len(df)} rows. Columns: {list(df.columns)}\n\n{df.describe()}"
    except Exception as e:
        return f"Excel error: {e}"
