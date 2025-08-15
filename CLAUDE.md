# LLM Video Editor

## Project Overview
An intelligent, prompt-driven video router/editor that processes video content and outputs platform-ready cuts for YouTube (16:9) and Instagram Reels (9:16).

## Architecture
- **Ingest**: Process single files or directories, probe media, cache audio/frames
- **Understand**: ASR transcription, optional OCR, scene detection
- **Plan**: LLM-driven EDL generation based on user prompts
- **Edit**: Apply cuts, reframe for aspect ratios, add captions, normalize audio
- **Render**: Export platform-specific formats with quality checks

## Technology Stack
- **Language**: Python 3.11
- **Framework**: LangGraph for agent orchestration
- **ASR**: faster-whisper or WhisperX
- **Scene Detection**: PySceneDetect or TransNetV2
- **LLM**: Llama 3.1/3.2, Qwen2.5-VL, or Mixtral-8x7B
- **Video Processing**: MoviePy + FFmpeg with NVENC acceleration
- **Export**: OpenTimelineIO (OTIO) for professional interchange

## Environment Setup with uv (Recommended)
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: brew install uv

# Setup project environment
./setup_environment.sh

# Manual setup alternative
uv python install 3.11  # Install Python 3.11
uv sync                  # Create environment and install dependencies
uv add --optional pro    # Install pro polish features (YOLO, Demucs, etc.)
```

## Legacy conda setup (Alternative)
```bash
# Create conda environment
conda create -n llm-video-editor python=3.11
conda activate llm-video-editor

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# With uv (recommended)
uv run llm-video-router --in ./videos --prompt "60s reel: top 3 takeaways, captions, upbeat" --target reels

# Direct execution (if activated)
llm-video-router --in ./videos --prompt "60s reel: top 3 takeaways, captions, upbeat" --target reels
```

## Development Commands
- `uv run pytest tests/` - Run tests
- `uv run llm-video-router --help` - Show CLI help
- `uv add package-name` - Add new dependency
- `uv sync` - Sync dependencies with pyproject.toml
- `uv python list` - List available Python versions

## Dependencies
See requirements.txt for full list of dependencies including:
- langgraph
- faster-whisper
- scenedetect
- moviepy
- opentimelineio
- torch
- transformers