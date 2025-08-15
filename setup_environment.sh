#!/bin/bash

# LLM Video Editor Environment Setup Script with uv
set -e

echo "🎬 LLM Video Editor Environment Setup with uv"
echo "=============================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing uv..."
    echo "Visit: https://docs.astral.sh/uv/getting-started/installation/"
    echo ""
    echo "Quick install options:"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Windows: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    echo "  Homebrew: brew install uv"
    echo "  pipx: pipx install uv"
    exit 1
fi

echo "✅ uv found: $(uv --version)"

# Check Python version requirement
echo "Checking Python 3.11 availability..."
if ! uv python list | grep -q "3.11"; then
    echo "Installing Python 3.11 with uv..."
    uv python install 3.11
fi

echo "Creating uv project environment..."
uv sync

echo "Installing core dependencies..."
uv add -e .

echo "Installing pro polish features (optional)..."
if uv add --optional pro 2>/dev/null; then
    echo "✅ Pro features installed successfully"
else
    echo "⚠️ Some pro features may require additional system dependencies"
    echo "   YOLO: May need CUDA for GPU acceleration"
    echo "   Demucs: Requires substantial RAM for large models"
fi

echo "Testing installation..."
uv run python -c "
import llm_video_editor
from llm_video_editor.presets import get_available_platforms
print('✅ Package installed successfully')
print('Available platforms:', get_available_platforms())
"

echo "Running tests..."
uv run pytest tests/ -v

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
echo "   ✅ uv environment: Ready"
echo "   ✅ Python 3.11: Installed"
echo "   ✅ Core dependencies: Installed" 
echo "   ✅ Tests: $(uv run pytest tests/ --tb=no -q | grep -c passed || echo 0) passing"
echo "   $(if command -v ffmpeg &> /dev/null; then echo '✅'; else echo '❌'; fi) FFmpeg: $(if command -v ffmpeg &> /dev/null; then echo 'Available'; else echo 'Not installed'; fi)"
echo "   $(if command -v ollama &> /dev/null; then echo '✅'; else echo '❌'; fi) Ollama: $(if command -v ollama &> /dev/null; then echo 'Available'; else echo 'Not installed'; fi)"
echo ""
echo "📚 uv Commands:"
echo "   uv run llm-video-router --help    # Run CLI tool"
echo "   uv run pytest                     # Run tests"
echo "   uv add package-name               # Add dependency"
echo "   uv sync                           # Sync dependencies"
echo "   uv python install 3.12           # Install Python version"
echo ""
echo "Next steps:"
echo "1. Install FFmpeg for video processing"
echo "2. Install Ollama for local LLM processing"  
echo "3. Test with sample video files: uv run llm-video-router --help"
echo ""
echo "Documentation: https://docs.astral.sh/uv/"