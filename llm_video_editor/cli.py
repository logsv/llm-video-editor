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

from .core.workflow import create_workflow
from .presets.platform_presets import get_platform_specs
from .utils.file_utils import find_video_files, validate_input_path


@click.command()
@click.option(
    '--input', '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input video file or directory containing videos'
)
@click.option(
    '--prompt', '-p',
    required=True,
    type=str,
    help='Editing prompt describing what you want to create'
)
@click.option(
    '--target', '-t',
    default='youtube',
    type=click.Choice(['youtube', 'reels', 'tiktok'], case_sensitive=False),
    help='Target platform for the video'
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
    input: str,
    prompt: str,
    target: str,
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
    
      # Create a 60s Instagram Reel from a video
      llm-video-router -i video.mp4 -p "60s reel: top 3 takeaways, captions" -t reels
      
      # Process multiple videos in a directory for YouTube
      llm-video-router -i ./videos -p "10min compilation: best moments" -t youtube
      
      # Use specific language and model
      llm-video-router -i spanish_video.mp4 -p "Resumen en 30s" -t reels --language es
    """
    try:
        # Load configuration if provided
        config_data = {}
        if config:
            with open(config, 'r') as f:
                config_data = json.load(f)
        
        # Setup logging
        if verbose:
            import logging
            logging.basicConfig(level=logging.INFO)
        
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
    from .core.media_probe import MediaProbe
    
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
    from .core.scene_detection import SceneDetector
    
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
        click.echo(f"   Shortest scene: {min(s.duration for s in scenes_list):.2f}s")
        click.echo(f"   Longest scene: {max(s.duration for s in scenes_list):.2f}s")
        
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
    from .core.asr import ASRProcessor
    
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


if __name__ == '__main__':
    # Register commands
    cli.add_command(main, 'process')
    cli()