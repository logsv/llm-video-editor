# LLM Video Editor

An intelligent, prompt-driven video router/editor that processes video content and outputs platform-ready cuts for YouTube (16:9) and Instagram Reels (9:16).

## 🎬 Features

- **Intelligent Analysis**: Automatic speech recognition (ASR) with word-level timestamps
- **Scene Detection**: Smart boundary detection for optimal cuts
- **LLM Planning**: AI-driven Edit Decision List (EDL) generation from natural language prompts
- **Platform Optimization**: Pre-configured settings for YouTube, Instagram Reels, TikTok
- **Professional Output**: Exports EDL, SRT captions, and validation reports
- **GPU Acceleration**: NVENC hardware encoding support for faster rendering

## 🚀 Quick Start

### Installation

1. **Create Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate llm-video-editor
   ```

2. **Install Package**
   ```bash
   pip install -e .
   ```

### Basic Usage

```bash
# Create a 60s Instagram Reel
llm-video-router -i video.mp4 -p "60s reel: top 3 takeaways, captions" -t reels

# Process multiple videos for YouTube
llm-video-router -i ./videos -p "10min compilation: best moments" -t youtube

# Custom configuration
llm-video-router -i video.mp4 -p "Resumen en 30s" -t reels --language es --config config.json
```

## 📋 Requirements

### System Dependencies
- **FFmpeg** (with NVENC support for GPU acceleration)
- **Python 3.11+**
- **CUDA** (optional, for GPU acceleration)

### Python Dependencies
- LangGraph for workflow orchestration
- faster-whisper for speech recognition
- PySceneDetect for scene boundary detection
- OpenTimelineIO for professional interchange
- OpenAI/Transformers for LLM planning

## 🏗️ Architecture

```
Input Video → Probe → ASR + Scene Detection → LLM Planning → Validation → Output
```

### Workflow Steps

1. **Probe**: Extract media metadata and basic analysis
2. **ASR**: Transcribe with word-level timestamps using Whisper
3. **Scene Detection**: Identify shot boundaries using content analysis
4. **Planning**: Generate Edit Decision List using LLM based on user prompt
5. **Validation**: Quality checks and constraint validation
6. **Export**: EDL, captions, thumbnails, and validation reports

## 🎯 Platform Presets

| Platform | Resolution | Aspect Ratio | Max Duration | Typical Duration |
|----------|-----------|--------------|--------------|------------------|
| YouTube  | 1920x1080 | 16:9        | 1 hour       | 10 minutes       |
| Reels    | 1080x1920 | 9:16        | 90s          | 30s              |
| TikTok   | 1080x1920 | 9:16        | 3 minutes    | 60s              |

## 🔧 CLI Commands

### Main Processing
```bash
llm-video-router [OPTIONS]

Options:
  -i, --input PATH       Input video file or directory [required]
  -p, --prompt TEXT      Editing prompt [required]
  -t, --target CHOICE    Target platform (youtube|reels|tiktok)
  -o, --output PATH      Output directory
  --asr-model CHOICE     Whisper model size
  --scene-threshold FLOAT Scene detection threshold
  --language TEXT        Language code for ASR
  --dry-run              Preview without processing
  --config PATH          Configuration file
```

### Utility Commands
```bash
# Analyze video file
llm-video-router info video.mp4

# Detect scenes
llm-video-router scenes video.mp4 --thumbnails

# Transcribe audio
llm-video-router transcribe video.mp4 --format srt

# List platforms
llm-video-router platforms
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

## 🛠️ Development

### Setup Development Environment
```bash
git clone <repository>
cd llm-video-editor
conda env create -f environment.yml
conda activate llm-video-editor
pip install -e .
```

### Run Tests
```bash
python -m pytest tests/
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
   # Install via conda
   conda install ffmpeg
   
   # Or system package manager
   apt-get install ffmpeg  # Ubuntu
   brew install ffmpeg     # macOS
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
   llm-video-router --asr-model medium
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