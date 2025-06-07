---
title: Agent GAIA
emoji: 🏆
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Benchmark Agent

This project is an AI agent built for the GAIA benchmark as part of the Hugging Face Agents course. It combines different LLM models and multimodal tools to reason over text, audio, images and video to solve complex tasks.


## Tools

The agent includes a variety of tools for handling diverse input types:

- **Vision Tool:** Analyze images using Gemini Vision.
- **YouTube Frame Extractor:** Sample video frames from YouTube at regular intervals.
- **YouTube QA Tool:** Ask questions about video content using Gemini via file URI.
- **OCR Tool:** Extract text from images using Tesseract.
- **Audio Transcriber:** Transcribe audio files and YouTube videos using Whisper.
- **File Tools:** Read plain text, download files from URLs, and summarize CSV or Excel files.

These tools are defined using the `@tool` decorator from the `smolagents` library, making them callable by the agent during task execution.

## Models Used

- `Gemini 2.5 Flash` (via Google's Generative AI API)
- **Whisper** for speech-to-text transcription
- **Hugging Face Transformers** (optional local model support)
- **LiteLLM** as a unified interface for calling external language models

## Installation

1. Install all required dependencies using

```bash
pip install -r requirements.txt
```

2. Convfigure environment with API_KEYS

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
echo "HF_TOKEN=your_hf_token" >> .env
```

3. Run the app

```bash
python app.py
```