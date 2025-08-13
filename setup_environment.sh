#!/bin/bash

# LLM Video Editor Environment Setup Script
set -e

echo "🎬 LLM Video Editor Environment Setup"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

echo "Testing installation..."
python -c "
import llm_video_editor
from llm_video_editor.presets import get_available_platforms
print('✅ Package installed successfully')
print('Available platforms:', get_available_platforms())
"

echo "Running tests..."
python -m pytest tests/ -v

echo "Checking FFmpeg availability..."
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg is installed"
    ffmpeg -version | head -n 1
else
    echo "❌ FFmpeg not found. Install with:"
    echo "   macOS: brew install ffmpeg"
    echo "   Ubuntu: sudo apt install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/"
fi

echo "Checking Ollama availability..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    ollama version
else
    echo "❌ Ollama not found. Install from: https://ollama.ai/"
    echo "   After installation, pull a model: ollama pull llama3.2"
fi

echo ""
echo "🚀 Setup Summary:"
echo "   ✅ Python environment: Ready"
echo "   ✅ Core dependencies: Installed" 
echo "   ✅ Tests: $(python -m pytest tests/ --tb=no -q | grep -c passed || echo 0) passing"
echo "   $(if command -v ffmpeg &> /dev/null; then echo '✅'; else echo '❌'; fi) FFmpeg: $(if command -v ffmpeg &> /dev/null; then echo 'Available'; else echo 'Not installed'; fi)"
echo "   $(if command -v ollama &> /dev/null; then echo '✅'; else echo '❌'; fi) Ollama: $(if command -v ollama &> /dev/null; then echo 'Available'; else echo 'Not installed'; fi)"
echo ""
echo "Next steps:"
echo "1. Install FFmpeg for video processing"
echo "2. Install Ollama for local LLM processing"  
echo "3. Test with sample video files"
echo ""
echo "Usage: llm-video-router --help"