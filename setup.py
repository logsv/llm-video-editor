from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-video-editor",
    version="0.1.0",
    author="LLM Video Editor Team",
    description="An intelligent, prompt-driven video router/editor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "langgraph>=0.0.40",
        "faster-whisper>=0.10.0",
        "scenedetect>=0.6.2",
        "moviepy>=1.0.3",
        "opentimelineio>=0.15.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "click>=8.0.0",
        "tqdm>=4.64.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "llm-video-router=llm_video_editor.cli:main",
        ],
    },
)