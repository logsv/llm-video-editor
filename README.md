# LLM Video Editor

An intelligent, prompt-driven video router/editor that processes video content and outputs platform-ready cuts for YouTube (16:9) and Instagram Reels (9:16). **Now fully functional with local LLM support using Ollama!** 🎉

## 🎬 Features

- **Intelligent Analysis**: Automatic speech recognition (ASR) with word-level timestamps using Whisper
- **Scene Detection**: Smart boundary detection for optimal cuts using PySceneDetect
- **Local LLM Planning**: AI-driven Edit Decision List (EDL) generation using Ollama (no API keys required!)
- **Platform Optimization**: Pre-configured settings for YouTube (16:9), Instagram Reels (9:16), TikTok
- **Professional Video Rendering**: GPU-accelerated video processing with FFmpeg
- **Complete Offline Processing**: All processing runs locally - no external dependencies
- **Smart Reframing**: Automatic aspect ratio conversion (16:9 to 9:16) with intelligent cropping
- **Generative B-roll**: AI-generated video content using Open-Sora v2 (text→video) and Stable Video Diffusion 1.1 (image→video)
- **Conversational CLI**: Natural-language interface for interactive video editing with prompt clarification

## 🎬 Workflow Demo

![LLM Video Editor Workflow](https://github.com/user-attachments/assets/workflow-demo.gif)

*Complete workflow demonstration: from prompt input to rendered 5-second Instagram Reel with quality validation*

## 🚀 Quick Start

### Installation

1. **Install uv** (Modern Python package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or: brew install uv
   ```

2. **Setup Project Environment**
   ```bash
   # Clone and setup (automated)
   ./setup_environment.sh
   
   # Or manual setup
   uv python install 3.11  # Install Python 3.11
   uv sync                  # Create environment and install dependencies
   uv add --optional pro    # Install pro polish features (YOLO, Demucs, etc.)
   ```

3. **Install Ollama** (for local LLM support)
   ```bash
   # macOS
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama3.1
   ```

4. **Install FFmpeg**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu
   sudo apt update && sudo apt install ffmpeg
   
   # Windows: Download from https://ffmpeg.org/
   ```

### Basic Usage

**Using the Python API (Recommended):**
```python
from llm_video_editor.core.ollama_planner import OllamaVideoPlanner
from llm_video_editor.core.media_probe import MediaProbe
from llm_video_editor.core.asr import ASRProcessor
from llm_video_editor.core.scene_detection import SceneDetector
from llm_video_editor.core.renderer import VideoRenderer

# Initialize components
planner = OllamaVideoPlanner(model_name="llama3.1")
renderer = VideoRenderer(use_gpu=True)
probe = MediaProbe()

# Process video
media_info = probe.probe_file("input.mp4")
edl = planner.generate_edl(
    prompt="Create a 30s Instagram Reel highlighting the best moments",
    media_info=media_info,
    target_platform="reels"
)

# Render final video
results = renderer.render_edl(edl, "input.mp4", "output/")
print(f"Final video: {results['final_video']}")
```

**Interactive CLI:**
```bash
# Start interactive mode (prompts for inputs)
uv run llm-video-router --interactive

# Or run without arguments for automatic interactive mode
uv run llm-video-router
```

**Command Line:**
```bash
# With uv (recommended)
uv run llm-video-router --use-ollama --ollama-model gpt-oss:latest --in ./videos --prompt "60s reel: top 3 takeaways, captions, upbeat" --target reels

# Direct execution (if environment activated)
llm-video-router --use-ollama --ollama-model llama3.1 --in ./videos --prompt "10min compilation: best moments" --target youtube
```

## 📋 Requirements

### System Dependencies
- **uv** (Modern Python package manager - recommended)
- **Ollama** (for local LLM inference)
- **FFmpeg** (with VideoToolbox/NVENC support for GPU acceleration)
- **Python 3.11+**

### Python Dependencies (Automatically Installed)
- **LangGraph** for workflow orchestration
- **faster-whisper** for speech recognition with local Whisper models
- **PySceneDetect** for intelligent scene boundary detection  
- **OpenTimelineIO** for professional video interchange
- **langchain-community** for Ollama LLM integration
- **requests** for API communication

### Tested Configurations
- ✅ **macOS**: Apple Silicon with VideoToolbox GPU acceleration
- ✅ **Linux**: NVIDIA GPUs with NVENC acceleration
- ⚠️ **Windows**: CPU-only (GPU acceleration coming soon)

## 🏗️ Architecture

```
Input Video → Media Probe → ASR + Scene Detection → Ollama LLM Planning → Video Rendering → Platform-Ready Output
```

### Workflow Steps

1. **Media Probe**: Extract video metadata, resolution, duration, and codec information
2. **ASR Processing**: Transcribe audio with word-level timestamps using local Whisper models
3. **Scene Detection**: Identify shot boundaries and content changes using PySceneDetect
4. **LLM Planning**: Generate Edit Decision List (EDL) using local Ollama models based on user prompt
5. **Video Rendering**: Apply cuts, reframing, and effects with GPU-accelerated FFmpeg
6. **Output Generation**: Create platform-optimized videos with subtitles and metadata

### Key Improvements ✨
- **100% Local Processing**: No API keys or internet required after initial setup
- **GPU Acceleration**: Hardware-accelerated video encoding (VideoToolbox/NVENC)
- **Smart Reframing**: Intelligent 16:9 to 9:16 conversion with content-aware cropping
- **Production Ready**: Generates actual video files, not just planning documents

## 🎯 Platform Presets

| Platform | Resolution | Aspect Ratio | Max Duration | Typical Duration |
|----------|-----------|--------------|--------------|------------------|
| YouTube  | 1920x1080 | 16:9        | 1 hour       | 10 minutes       |
| Reels    | 1080x1920 | 9:16        | 90s          | 30s              |
| TikTok   | 1080x1920 | 9:16        | 3 minutes    | 60s              |

## 🎨 Generative B-roll

### Text→Video (Open-Sora v2)
Set `OPEN_SORA_CMD` to your local infer script, or `OPEN_SORA_HOST` to a running server.

### Image→Video (SVD 1.1)
Install `diffusers`, `torch`, `accelerate`, accept the model license on Hugging Face.

```bash
# Example usage in interactive CLI
uv run llm-video-router --interactive
# Follow the prompts to input video file, editing prompt, and target platform
```

## 🔧 CLI Commands

### Main Processing
```bash
llm-video-router [OPTIONS]

Options:
  -i, --input PATH                Input video file or directory
  -p, --prompt TEXT               Editing prompt describing what you want to create
  -t, --target [youtube|reels|tiktok]  Target platform for the video
  --interactive                   Run in interactive mode (prompts for inputs)
  -o, --output PATH               Output directory for generated files
  --asr-model [tiny|base|small|medium|large-v3]  Whisper model size for speech recognition
  --scene-threshold FLOAT         Scene detection threshold (lower = more sensitive)
  --language TEXT                 Language code for ASR (e.g., en, es, fr). Auto-detect if not specified
  --planner-model TEXT            LLM model for planning (gpt-4, gpt-3.5-turbo, etc.)
  --use-ollama                    Use local Ollama instead of OpenAI
  --ollama-model TEXT             Ollama model name (llama3.2, codellama, mistral, etc.)
  --ollama-url TEXT               Ollama server URL
  --dry-run                       Show what would be done without actually processing
  -v, --verbose                   Enable verbose output
  --config PATH                   Configuration file path (JSON)
```

### Interactive Mode
```bash
# Start interactive mode - prompts for all inputs
uv run llm-video-router --interactive

# Or run without any arguments (automatically enters interactive mode)
uv run llm-video-router
```

## 🐍 Programmatic Usage

```python
from llm_video_editor.core.workflow import create_workflow

# Create workflow
workflow = create_workflow(
    asr_model="large-v3",
    scene_threshold=27.0,
    planner_model="gpt-4"
)

# Process video
result = workflow.run({
    "path": "video.mp4",
    "prompt": "Create a 30s highlights reel with captions",
    "target": "reels",
    "output_dir": "output"
})

print(f"Status: {result['status']}")
print(f"Generated {result['edl_summary']['clips_count']} clips")
```

## ⚙️ Configuration

Create a `config.json` file:

```json
{
  "asr_model": "large-v3",
  "scene_threshold": 27.0,
  "planner_model": "gpt-4",
  "language": "en",
  "gpu_acceleration": true,
  "platforms": {
    "reels": {
      "max_duration": 60,
      "quality": "medium"
    }
  }
}
```

## 📁 Output Structure

```
output/
├── video_name/
│   ├── edit_decision_list.json    # Main EDL file
│   ├── captions.srt              # Subtitle file  
│   ├── scenes.json               # Scene boundaries
│   ├── validation_report.json    # Quality checks
│   └── thumbnails/               # Scene thumbnails
└── processing_results.json       # Batch results
```

## 🎨 Example Prompts

- `"60s reel: top 3 takeaways, captions, upbeat"`
- `"5min YouTube highlights: best moments, lower thirds"`
- `"30s product demo: step-by-step, call-to-action"`
- `"2min tutorial summary: key points, engaging intro"`
- `"45s testimonial compilation: emotional moments"`

## 🔍 Quality Assurance

The system performs automatic validation:

- ✅ Duration constraints per platform
- ✅ Aspect ratio compatibility  
- ✅ Scene boundary alignment
- ✅ Speech coverage analysis
- ✅ Audio loudness standards (EBU-R128)

## 🧪 Testing

### Quick Test
```python
# Test the complete workflow
python test_ollama_workflow.py
```

Expected output:
- ✅ Media probing: Video metadata extraction
- ✅ Scene detection: Boundary identification
- ✅ ASR processing: Audio transcription
- ✅ Ollama LLM: EDL generation
- ✅ Video rendering: Platform-ready output

### Component Tests
```python
# Test individual components
python test_full_rendering.py    # Video rendering pipeline
python test_concat_fix.py       # Video concatenation
```

## 🛠️ Development

### Setup Development Environment
```bash
git clone <repository>
cd llm-video-editor

# Using uv (recommended)
uv python install 3.11
uv sync
uv add --optional pro

# Install Ollama and pull model
ollama serve &
ollama pull llama3.1
```

### Run Tests
```bash
uv run pytest tests/            # Unit tests  
uv run python test_ollama_workflow.py  # End-to-end test
```

### Code Quality
```bash
black llm_video_editor/
flake8 llm_video_editor/
mypy llm_video_editor/
```

## 📚 Advanced Features

### Custom LLM Models
```python
from langchain_openai import ChatOpenAI
from llm_video_editor.core.planner import VideoPlanner

# Use custom model
custom_llm = ChatOpenAI(model_name="gpt-3.5-turbo")
planner = VideoPlanner(llm=custom_llm)
```

### GPU Acceleration
```bash
# Enable NVENC for faster encoding
llm-video-router -i video.mp4 -p "quick edit" --gpu-accel
```

### Batch Processing
```bash
# Process entire directory
llm-video-router -i ./videos/ -p "social media cuts" -t reels
```

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # System package manager
   apt-get install ffmpeg  # Ubuntu
   brew install ffmpeg     # macOS
   # Windows: Download from https://ffmpeg.org/
   ```

2. **CUDA/GPU issues**
   ```bash
   # Check NVENC support
   ffmpeg -encoders | grep nvenc
   
   # Fallback to CPU
   llm-video-router --no-gpu-accel
   ```

3. **Memory issues with large models**
   ```bash
   # Use smaller Whisper model
   uv run llm-video-router --asr-model medium
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- 📖 Documentation: [docs/](docs/)
- 🐛 Bug Reports: [GitHub Issues](../../issues)
- 💡 Feature Requests: [GitHub Issues](../../issues)
- 💬 Discussions: [GitHub Discussions](../../discussions)

---

**Built with ❤️ using LangGraph, Whisper, and FFmpeg**