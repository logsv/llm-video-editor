"""
Command-line interface for LLM Video Editor.
"""
import os
import sys
from pathlib import Path
from typing import Optional
import json

import click
from tqdm import tqdm

from ..core.workflow import create_workflow
from ..presets.platform_presets import get_platform_specs
from ..utils.file_utils import find_video_files, validate_input_path


@click.command()
@click.option(
    '--input', '-i',
    required=False,
    type=click.Path(exists=True),
    help='Input video file or directory containing videos'
)
@click.option(
    '--prompt', '-p',
    required=False,
    type=str,
    help='Editing prompt describing what you want to create'
)
@click.option(
    '--target', '-t',
    default=None,
    type=click.Choice(['youtube', 'reels', 'tiktok'], case_sensitive=False),
    help='Target platform for the video'
)
@click.option(
    '--interactive',
    is_flag=True,
    default=False,
    help='Run in interactive mode (prompts for inputs)'
)
@click.option(
    '--output', '-o',
    default='output',
    type=click.Path(),
    help='Output directory for generated files'
)
@click.option(
    '--asr-model',
    default='large-v3',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v3'], case_sensitive=False),
    help='Whisper model size for speech recognition'
)
@click.option(
    '--scene-threshold',
    default=27.0,
    type=float,
    help='Scene detection threshold (lower = more sensitive)'
)
@click.option(
    '--language',
    type=str,
    help='Language code for ASR (e.g., en, es, fr). Auto-detect if not specified'
)
@click.option(
    '--planner-model',
    default='gpt-4',
    type=str,
    help='LLM model for planning (gpt-4, gpt-3.5-turbo, etc.)'
)
@click.option(
    '--use-ollama',
    is_flag=True,
    help='Use local Ollama instead of OpenAI'
)
@click.option(
    '--ollama-model',
    default='llama3.2',
    type=str,
    help='Ollama model name (llama3.2, codellama, mistral, etc.)'
)
@click.option(
    '--ollama-url',
    default='http://localhost:11434',
    type=str,
    help='Ollama server URL'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be done without actually processing'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Configuration file path (JSON)'
)
def main(
    input: Optional[str],
    prompt: Optional[str],
    target: Optional[str],
    interactive: bool,
    output: str,
    asr_model: str,
    scene_threshold: float,
    language: Optional[str],
    planner_model: str,
    use_ollama: bool,
    ollama_model: str,
    ollama_url: str,
    dry_run: bool,
    verbose: bool,
    config: Optional[str]
):
    """
    LLM Video Router/Editor - Generate platform-ready video cuts using AI.
    
    Examples:
    
      # Interactive mode (prompts for inputs)
      llm-video-router
      
      # Direct mode with arguments
      llm-video-router -i video.mp4 -p "60s reel: top 3 takeaways, captions" -t reels
      
      # Process multiple videos in a directory for YouTube
      llm-video-router -i ./videos -p "10min compilation: best moments" -t youtube
      
      # Use specific language and model
      llm-video-router -i spanish_video.mp4 -p "Resumen en 30s" -t reels --language es
    """
    try:
        # Interactive mode - start with conversational flow
        if not input or not prompt or not target or interactive:
            return interactive_workflow(
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                asr_model=asr_model,
                scene_threshold=scene_threshold,
                language=language,
                planner_model=planner_model,
                output=output,
                dry_run=dry_run,
                verbose=verbose,
                config=config
            )
        
        # Load configuration if provided
        config_data = {}
        if config:
            with open(config, 'r') as f:
                config_data = json.load(f)
        
        # Setup logging
        if verbose:
            import logging
            logging.basicConfig(level=logging.INFO)
        
        # Set default target if still None
        if target is None:
            target = 'youtube'
        
        # Validate and prepare inputs
        input_path = Path(input)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find video files
        video_files = find_video_files(input_path)
        if not video_files:
            click.echo(f"❌ No video files found in: {input_path}")
            sys.exit(1)
        
        click.echo(f"🎬 LLM Video Editor")
        click.echo(f"📁 Input: {input_path}")
        click.echo(f"📝 Prompt: {prompt}")
        click.echo(f"🎯 Target: {target.upper()}")
        click.echo(f"📂 Output: {output_path}")
        click.echo(f"🎥 Found {len(video_files)} video file(s)")
        
        # Show platform specs
        platform_specs = get_platform_specs(target)
        click.echo(f"📋 Platform specs: {platform_specs['resolution']} @ {platform_specs['aspect_ratio']}, max {platform_specs['max_duration']}s")
        
        if dry_run:
            click.echo("\\n🔍 DRY RUN - No files will be processed")
            for video_file in video_files:
                click.echo(f"  Would process: {video_file}")
            return
        
        # Initialize workflow
        if use_ollama:
            click.echo(f"\\n🤖 Initializing workflow with Ollama model: {ollama_model}...")
            actual_planner_model = ollama_model
        else:
            click.echo(f"\\n🤖 Initializing workflow with OpenAI model: {planner_model}...")
            actual_planner_model = planner_model
            
        workflow = create_workflow(
            asr_model=asr_model,
            scene_threshold=scene_threshold,
            planner_model=actual_planner_model,
            use_ollama=use_ollama,
            ollama_base_url=ollama_url
        )
        
        # Process each video file
        results = []
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for video_file in video_files:
                pbar.set_description(f"Processing {Path(video_file).name}")
                
                # Create output subdirectory for this video
                video_output_dir = output_path / Path(video_file).stem
                video_output_dir.mkdir(exist_ok=True)
                
                # Prepare workflow inputs
                workflow_inputs = {
                    "path": str(video_file),
                    "prompt": prompt,
                    "target": target.lower(),
                    "output_dir": str(video_output_dir),
                    **config_data  # Merge config data
                }
                
                if language:
                    workflow_inputs["language"] = language
                
                try:
                    # Run workflow
                    result = workflow.run(workflow_inputs)
                    results.append({
                        "file": str(video_file),
                        "status": result["status"],
                        "output_dir": str(video_output_dir),
                        "result": result
                    })
                    
                    if result["status"] == "completed":
                        click.echo(f"✅ Completed: {Path(video_file).name}")
                        if "edl_summary" in result:
                            edl = result["edl_summary"]
                            click.echo(f"   📊 {edl['clips_count']} clips, {edl['target_duration']:.1f}s target duration")
                    else:
                        click.echo(f"❌ Failed: {Path(video_file).name}")
                        if "error" in result.get("metadata", {}):
                            click.echo(f"   Error: {result['metadata']['error']}")
                
                except Exception as e:
                    click.echo(f"❌ Error processing {Path(video_file).name}: {str(e)}")
                    results.append({
                        "file": str(video_file),
                        "status": "error",
                        "error": str(e)
                    })
                
                pbar.update(1)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "completed")
        failed = len(results) - successful
        
        click.echo(f"\\n📊 Summary:")
        click.echo(f"   ✅ Successful: {successful}")
        click.echo(f"   ❌ Failed: {failed}")
        
        # Save results summary
        results_file = output_path / "processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"   📄 Results saved to: {results_file}")
        
        # Show next steps
        click.echo(f"\\n🎉 Processing complete!")
        click.echo(f"   📁 Check output directory: {output_path}")
        click.echo(f"   📋 Edit Decision Lists, captions, and validation reports are available")
        click.echo(f"   🎬 Use the EDL files with your video editor or our rendering tools")
        
    except KeyboardInterrupt:
        click.echo("\\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\\n💥 Fatal error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.group()
def cli():
    """LLM Video Editor CLI Tools."""
    pass


@cli.command()
@click.argument('video_file', type=click.Path(exists=True))
def info(video_file: str):
    """Display information about a video file."""
    from ..core.media_probe import MediaProbe
    
    try:
        click.echo(f"🔍 Analyzing: {video_file}")
        media_info = MediaProbe.probe_file(video_file)
        
        click.echo(f"\\n📊 Media Information:")
        click.echo(f"   Duration: {media_info.duration:.2f} seconds")
        click.echo(f"   Resolution: {media_info.width}x{media_info.height}")
        click.echo(f"   Aspect Ratio: {media_info.aspect_ratio}")
        click.echo(f"   Frame Rate: {media_info.fps:.2f} fps")
        click.echo(f"   Video Codec: {media_info.video_codec}")
        click.echo(f"   Has Audio: {'Yes' if media_info.has_audio else 'No'}")
        if media_info.has_audio:
            click.echo(f"   Audio Codec: {media_info.audio_codec}")
            click.echo(f"   Sample Rate: {media_info.audio_sample_rate} Hz")
        click.echo(f"   Bitrate: {media_info.bitrate:,} bps")
        
    except Exception as e:
        click.echo(f"❌ Error analyzing video: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='scenes.json', help='Output file for scene list')
@click.option('--threshold', default=27.0, help='Scene detection threshold')
@click.option('--thumbnails', is_flag=True, help='Generate scene thumbnails')
def scenes(video_file: str, output: str, threshold: float, thumbnails: bool):
    """Detect and export scenes from a video file."""
    from ..core.scene_detection import SceneDetector
    
    try:
        click.echo(f"🎬 Detecting scenes in: {video_file}")
        detector = SceneDetector(threshold=threshold)
        scenes_list = detector.detect_scenes(video_file)
        
        click.echo(f"✅ Detected {len(scenes_list)} scenes")
        
        # Export scene list
        detector.export_scene_list(scenes_list, output, format="json")
        click.echo(f"📄 Scene list saved to: {output}")
        
        # Generate thumbnails if requested
        if thumbnails:
            thumbnail_dir = Path(output).stem + "_thumbnails"
            thumbnail_paths = detector.get_scene_thumbnails(video_file, scenes_list, thumbnail_dir)
            click.echo(f"🖼️  Generated {len(thumbnail_paths)} thumbnails in: {thumbnail_dir}")
        
        # Show scene summary
        total_duration = sum(scene.duration for scene in scenes_list)
        avg_duration = total_duration / len(scenes_list) if scenes_list else 0
        
        click.echo(f"\\n📊 Scene Summary:")
        click.echo(f"   Total scenes: {len(scenes_list)}")
        click.echo(f"   Average duration: {avg_duration:.2f}s")
        if scenes_list:
            click.echo(f"   Shortest scene: {min(s.duration for s in scenes_list):.2f}s")
            click.echo(f"   Longest scene: {max(s.duration for s in scenes_list):.2f}s")
        else:
            click.echo(f"   No scenes detected")
        
    except Exception as e:
        click.echo(f"❌ Error detecting scenes: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='captions.srt', help='Output file for captions')
@click.option('--model', default='large-v3', help='Whisper model size')
@click.option('--language', help='Language code (e.g., en, es, fr)')
@click.option('--format', default='srt', type=click.Choice(['srt', 'vtt']), help='Output format')
def transcribe(video_file: str, output: str, model: str, language: str, format: str):
    """Transcribe a video file and export captions."""
    from ..core.asr import ASRProcessor
    
    try:
        click.echo(f"🎤 Transcribing: {video_file}")
        asr = ASRProcessor(model_size=model)
        
        segments = asr.transcribe_file(
            video_file,
            language=language,
            vad_filter=True,
            word_timestamps=True
        )
        
        # Export captions
        if format == 'srt':
            asr.export_srt(segments, output)
        else:
            asr.export_vtt(segments, output)
        
        click.echo(f"✅ Transcription complete")
        click.echo(f"📄 Captions saved to: {output}")
        
        # Show transcription summary
        total_duration = sum(seg.end - seg.start for seg in segments)
        
        click.echo(f"\\n📊 Transcription Summary:")
        click.echo(f"   Segments: {len(segments)}")
        click.echo(f"   Speech duration: {total_duration:.1f}s")
        click.echo(f"   Average segment length: {total_duration/len(segments):.1f}s")
        
    except Exception as e:
        click.echo(f"❌ Error transcribing video: {str(e)}")
        sys.exit(1)


@cli.command()
def platforms():
    """List available platform presets and their specifications."""
    platforms = ['youtube', 'reels', 'tiktok']
    
    click.echo("📱 Available Platform Presets:")
    for platform in platforms:
        specs = get_platform_specs(platform)
        click.echo(f"\\n{platform.upper()}:")
        click.echo(f"   Resolution: {specs['resolution']}")
        click.echo(f"   Aspect Ratio: {specs['aspect_ratio']}")
        click.echo(f"   Max Duration: {specs['max_duration']}s")
        click.echo(f"   Typical Duration: {specs['typical_duration']}s")


@cli.command()
def chat():
    """Start conversational LLM Video Router with generative B-roll."""
    from .chat import chat as chat_function
    
    # Call the chat function directly
    chat_function()


def interactive_workflow(**kwargs):
    """Interactive conversational workflow for video editing."""
    click.echo("🎬 Welcome to LLM Video Editor!")
    click.echo("=" * 50)
    
    while True:
        # Start with a prompt-first approach
        click.echo("\n💭 What would you like to create?")
        click.echo("Describe your vision (e.g., '60s reel highlighting key moments', 'TikTok video from this image')")
        
        user_prompt = click.prompt("✏️  Your creative prompt").strip()
        if not user_prompt:
            click.echo("❌ Please enter a prompt to continue.")
            continue
        
        # Ask about content type
        click.echo(f"\n📝 Great! For: '{user_prompt}'")
        click.echo("🎯 What type of content do you want to work with?")
        click.echo("1. 📹 Edit existing video file")
        click.echo("2. 🖼️  Generate video from image")
        click.echo("3. ✨ Generate video from text only") 
        click.echo("4. 🎵 Audio to video (with visualizations)")
        click.echo("5. ❓ Help me decide")
        
        while True:
            try:
                content_choice = click.prompt("Enter choice (1-5)", type=int)
                if content_choice in [1, 2, 3, 4, 5]:
                    break
                else:
                    click.echo("❌ Please enter a number between 1-5")
            except:
                click.echo("❌ Please enter a valid number")
        
        # Handle different content types
        if content_choice == 1:
            # Video editing workflow
            result = handle_video_editing_workflow(user_prompt, **kwargs)
        elif content_choice == 2:
            # Image to video workflow
            result = handle_image_to_video_workflow(user_prompt, **kwargs)
        elif content_choice == 3:
            # Text to video workflow
            result = handle_text_to_video_workflow(user_prompt, **kwargs)
        elif content_choice == 4:
            # Audio to video workflow
            result = handle_audio_to_video_workflow(user_prompt, **kwargs)
        elif content_choice == 5:
            # Help decide
            result = handle_help_decide_workflow(user_prompt, **kwargs)
        
        # Ask if user wants to continue
        click.echo(f"\n🎉 Workflow completed!")
        if result:
            click.echo(f"📁 Check your results in: {result.get('output_dir', 'output/')}")
        
        click.echo("\n🔄 What would you like to do next?")
        click.echo("1. 🔄 Create another video with new prompt")
        click.echo("2. 🔧 Modify the last result")
        click.echo("3. 🚪 Exit")
        
        while True:
            try:
                next_choice = click.prompt("Enter choice (1-3)", type=int)
                if next_choice in [1, 2, 3]:
                    break
                else:
                    click.echo("❌ Please enter 1, 2, or 3")
            except:
                click.echo("❌ Please enter a valid number")
        
        if next_choice == 1:
            continue  # Loop back to start
        elif next_choice == 2:
            click.echo("🔧 Modification feature coming soon!")
            continue
        else:
            click.echo("👋 Thanks for using LLM Video Editor!")
            break


def handle_video_editing_workflow(user_prompt: str, **kwargs):
    """Handle video file editing workflow."""
    click.echo(f"\n📹 Video Editing Workflow")
    click.echo("=" * 30)
    
    # Get video file
    while True:
        input_prompt = click.prompt("📁 Enter video file or directory path")
        input_path = Path(input_prompt)
        if input_path.exists():
            break
        else:
            click.echo(f"❌ Path does not exist: {input_path}")
    
    # Get target platform
    target = get_target_platform()
    
    # Process with existing workflow
    return process_video_workflow(
        input=str(input_path),
        prompt=user_prompt,
        target=target,
        **kwargs
    )


def handle_image_to_video_workflow(user_prompt: str, **kwargs):
    """Handle image to video generation workflow."""
    click.echo(f"\n🖼️  Image to Video Workflow")
    click.echo("=" * 30)
    
    # Get image file
    while True:
        image_path = click.prompt("🖼️  Enter image file path")
        image_file = Path(image_path)
        if image_file.exists() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            break
        else:
            click.echo(f"❌ Please provide a valid image file (jpg, png, etc.): {image_path}")
    
    # Get target platform
    target = get_target_platform()
    
    # Get duration
    click.echo("\n⏱️  Video duration:")
    while True:
        try:
            duration = click.prompt("Duration in seconds (2-10)", type=int, default=4)
            if 2 <= duration <= 10:
                break
            else:
                click.echo("❌ Duration must be between 2-10 seconds")
        except:
            click.echo("❌ Please enter a valid number")
    
    # Setup output directory
    output_path = Path(kwargs.get('output', 'output'))
    video_output_dir = output_path / f"i2v_{Path(image_path).stem}"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        click.echo(f"\n🎬 Generating {duration}s video from image...")
        click.echo(f"📝 Prompt: {user_prompt}")
        click.echo(f"🎯 Target: {target.upper()}")
        click.echo(f"🖼️  Source: {image_file.name}")
        
        from ..generative.broll import BrollGenerator, BrollRequest
        
        # Create B-roll generator
        generator = BrollGenerator(workdir=video_output_dir)
        
        # Create request
        request = BrollRequest(
            mode="i2v",
            prompt=user_prompt,
            seconds=duration,
            fps=24,
            resolution="576x1024" if target in ['reels', 'tiktok'] else "1024x576",
            image_path=Path(image_path)
        )
        
        # Generate video
        generated_video = generator.generate(request)
        
        # Apply platform-specific formatting if needed
        final_video_path = video_output_dir / f"{Path(image_path).stem}_{target}.mp4"
        
        # For now, just copy the generated video
        import shutil
        shutil.copy2(generated_video, final_video_path)
        
        click.echo(f"✅ Image-to-video generation completed!")
        click.echo(f"📹 Generated video: {final_video_path}")
        
        return {
            "status": "completed",
            "output_dir": str(video_output_dir),
            "generated_video": str(final_video_path),
            "source_image": str(image_path)
        }
        
    except ImportError as e:
        click.echo(f"❌ Missing dependencies for image-to-video generation:")
        click.echo(f"   Please install: pip install diffusers torch pillow")
        click.echo(f"   Error: {str(e)}")
        return {"status": "failed", "error": "Missing dependencies"}
        
    except Exception as e:
        click.echo(f"❌ Image-to-video generation failed: {str(e)}")
        return {"status": "failed", "error": str(e)}


def handle_text_to_video_workflow(user_prompt: str, **kwargs):
    """Handle text to video generation workflow."""
    click.echo(f"\n✨ Text to Video Workflow")
    click.echo("=" * 30)
    
    # Get additional context
    click.echo("📝 You can provide additional style or context:")
    click.echo("Examples: 'cinematic style', 'animated cartoon', 'realistic photography'")
    style_prompt = click.prompt("🎨 Style/context (or press Enter to skip)", default="")
    
    # Combine prompts
    if style_prompt.strip():
        full_prompt = f"{user_prompt}, {style_prompt}"
    else:
        full_prompt = user_prompt
    
    # Get target platform
    target = get_target_platform()
    
    # Get duration
    click.echo("\n⏱️  Video duration:")
    while True:
        try:
            duration = click.prompt("Duration in seconds (2-8)", type=int, default=4)
            if 2 <= duration <= 8:
                break
            else:
                click.echo("❌ Duration must be between 2-8 seconds")
        except:
            click.echo("❌ Please enter a valid number")
    
    # Setup output directory
    output_path = Path(kwargs.get('output', 'output'))
    video_output_dir = output_path / f"t2v_{full_prompt[:20].replace(' ', '_')}"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        click.echo(f"\n🎬 Generating {duration}s video from text...")
        click.echo(f"📝 Full prompt: {full_prompt}")
        click.echo(f"🎯 Target: {target.upper()}")
        
        from ..generative.broll import BrollGenerator, BrollRequest
        
        # Create B-roll generator
        generator = BrollGenerator(workdir=video_output_dir)
        
        # Create request
        request = BrollRequest(
            mode="t2v",
            prompt=full_prompt,
            seconds=duration,
            fps=24,
            resolution="576x1024" if target in ['reels', 'tiktok'] else "1024x576"
        )
        
        # Generate video
        generated_video = generator.generate(request)
        
        # Apply platform-specific formatting if needed
        final_video_path = video_output_dir / f"generated_{target}.mp4"
        
        # For now, just copy the generated video
        import shutil
        shutil.copy2(generated_video, final_video_path)
        
        click.echo(f"✅ Text-to-video generation completed!")
        click.echo(f"📹 Generated video: {final_video_path}")
        
        return {
            "status": "completed",
            "output_dir": str(video_output_dir),
            "generated_video": str(final_video_path),
            "prompt": full_prompt
        }
        
    except RuntimeError as e:
        if "OPEN_SORA" in str(e):
            click.echo(f"❌ Text-to-video requires Open-Sora setup:")
            click.echo(f"   Please set OPEN_SORA_CMD or OPEN_SORA_HOST environment variables")
            click.echo(f"   See documentation for Open-Sora installation")
            click.echo(f"   Error: {str(e)}")
            return {"status": "failed", "error": "Open-Sora not configured"}
        else:
            click.echo(f"❌ Text-to-video generation failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
            
    except Exception as e:
        click.echo(f"❌ Text-to-video generation failed: {str(e)}")
        return {"status": "failed", "error": str(e)}


def handle_audio_to_video_workflow(user_prompt: str, **kwargs):
    """Handle audio to video workflow."""
    click.echo(f"\n🎵 Audio to Video Workflow")
    click.echo("=" * 30)
    
    # Get audio file
    while True:
        audio_path = click.prompt("🎵 Enter audio file path")
        if Path(audio_path).exists():
            break
        else:
            click.echo(f"❌ Audio file does not exist: {audio_path}")
    
    target = get_target_platform()
    
    click.echo("🎬 Generating video with audio visualizations...")
    click.echo("⚠️  Audio-to-video workflow is in development!")
    click.echo("🔧 For now, please use a video file or check back later.")
    
    return {"status": "pending", "output_dir": "output"}


def handle_help_decide_workflow(user_prompt: str, **kwargs):
    """Help user decide on content type based on their prompt."""
    click.echo(f"\n❓ Let me help you decide!")
    click.echo("=" * 30)
    
    # Simple keyword analysis to suggest workflow
    prompt_lower = user_prompt.lower()
    
    click.echo(f"📝 Based on your prompt: '{user_prompt}'")
    
    # Analyze the prompt for keywords
    edit_keywords = ['edit', 'cut', 'highlight', 'reel', 'compilation', 'trim', 'clip', 'extract', 'shorten']
    image_keywords = ['image', 'photo', 'picture', 'from image', 'animate image', 'bring photo to life']
    text_keywords = ['story', 'script', 'create from scratch', 'generate video', 'make video about', 'visualize']
    
    edit_score = sum(1 for word in edit_keywords if word in prompt_lower)
    image_score = sum(1 for word in image_keywords if word in prompt_lower)
    text_score = sum(1 for word in text_keywords if word in prompt_lower)
    
    if image_score > 0 and image_score >= max(edit_score, text_score):
        click.echo("💡 Recommendation: **Image to Video** (Option 2)")
        click.echo("   Your prompt suggests animating an image or photo.")
        suggested = 2
    elif text_score > 0 and text_score >= max(edit_score, image_score):
        click.echo("💡 Recommendation: **Text to Video** (Option 3)")
        click.echo("   Your prompt suggests creating content from a text description.")
        suggested = 3
    elif edit_score > 0:
        click.echo("💡 Recommendation: **Video Editing** (Option 1)")
        click.echo("   Your prompt suggests editing existing video content.")
        suggested = 1
    else:
        # Default recommendation based on prompt complexity
        if len(user_prompt.split()) > 10:
            click.echo("💡 Recommendation: **Text to Video** (Option 3)")
            click.echo("   Your detailed description would work well for text-to-video generation.")
            suggested = 3
        else:
            click.echo("💡 Recommendation: **Video Editing** (Option 1)")  
            click.echo("   Most users start with editing existing video content.")
            suggested = 1
    
    use_suggestion = click.confirm(f"Use recommended option {suggested}?", default=True)
    
    if use_suggestion:
        if suggested == 1:
            return handle_video_editing_workflow(user_prompt, **kwargs)
        elif suggested == 2:
            return handle_image_to_video_workflow(user_prompt, **kwargs)
        elif suggested == 3:
            return handle_text_to_video_workflow(user_prompt, **kwargs)
    else:
        click.echo("👍 No problem! Let's go back to the main menu.")
        return {"status": "cancelled"}


def get_target_platform():
    """Get target platform from user."""
    click.echo("\n🎯 Select target platform:")
    click.echo("1. YouTube (16:9, landscape)")
    click.echo("2. Instagram Reels (9:16, portrait)")
    click.echo("3. TikTok (9:16, portrait)")
    
    while True:
        try:
            choice = click.prompt("Enter choice (1-3)", type=int)
            if choice == 1:
                return 'youtube'
            elif choice == 2:
                return 'reels'
            elif choice == 3:
                return 'tiktok'
            else:
                click.echo("❌ Please enter 1, 2, or 3")
        except:
            click.echo("❌ Please enter a valid number")


def process_video_workflow(input: str, prompt: str, target: str, **kwargs):
    """Process video using the existing workflow."""
    from ..core.workflow import create_workflow
    from ..utils.file_utils import find_video_files
    
    # Load configuration if provided
    config_data = {}
    if kwargs.get('config'):
        with open(kwargs['config'], 'r') as f:
            config_data = json.load(f)
    
    # Setup logging
    if kwargs.get('verbose'):
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Set default target if still None
    if target is None:
        target = 'youtube'
    
    # Validate and prepare inputs
    input_path = Path(input)
    output_path = Path(kwargs.get('output', 'output'))
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = find_video_files(input_path)
    if not video_files:
        click.echo(f"❌ No video files found in: {input_path}")
        return {"status": "failed", "error": "No video files found"}
    
    click.echo(f"🎬 Processing {len(video_files)} video file(s)...")
    click.echo(f"📝 Prompt: {prompt}")
    click.echo(f"🎯 Target: {target.upper()}")
    
    if kwargs.get('dry_run'):
        click.echo("\\n🔍 DRY RUN - No files will be processed")
        for video_file in video_files:
            click.echo(f"  Would process: {video_file}")
        return {"status": "dry_run", "output_dir": str(output_path)}
    
    # Initialize workflow
    if kwargs.get('use_ollama'):
        click.echo(f"\\n🤖 Using Ollama model: {kwargs.get('ollama_model')}...")
        actual_planner_model = kwargs.get('ollama_model')
    else:
        click.echo(f"\\n🤖 Using OpenAI model: {kwargs.get('planner_model')}...")
        actual_planner_model = kwargs.get('planner_model')
        
    workflow = create_workflow(
        asr_model=kwargs.get('asr_model', 'large-v3'),
        scene_threshold=kwargs.get('scene_threshold', 27.0),
        planner_model=actual_planner_model,
        use_ollama=kwargs.get('use_ollama', False),
        ollama_base_url=kwargs.get('ollama_url', 'http://localhost:11434')
    )
    
    # Process video file (just process the first one for now)
    video_file = video_files[0]
    video_output_dir = output_path / Path(video_file).stem
    video_output_dir.mkdir(exist_ok=True)
    
    # Prepare workflow inputs
    workflow_inputs = {
        "path": str(video_file),
        "prompt": prompt,
        "target": target.lower(),
        "output_dir": str(video_output_dir),
        **config_data
    }
    
    if kwargs.get('language'):
        workflow_inputs["language"] = kwargs['language']
    
    try:
        # Run workflow
        result = workflow.run(workflow_inputs)
        
        if result["status"] == "completed":
            click.echo(f"✅ Video processing completed!")
            return {
                "status": "completed",
                "output_dir": str(video_output_dir),
                "result": result
            }
        else:
            click.echo(f"❌ Video processing failed")
            return {
                "status": "failed",
                "error": result.get("error", "Unknown error")
            }
    
    except Exception as e:
        click.echo(f"❌ Error processing video: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == '__main__':
    # Register commands
    cli.add_command(main, 'process')
    cli()