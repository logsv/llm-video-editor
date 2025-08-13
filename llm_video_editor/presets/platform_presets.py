"""
Platform-specific presets and specifications for video export.
"""
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PlatformSpec:
    """Platform specification container."""
    name: str
    max_duration: int  # seconds
    aspect_ratio: str
    resolution: str
    typical_duration: int  # seconds
    subtitle_style: str
    audio_format: str
    video_codec: str
    quality_preset: str


# Platform specifications
PLATFORM_SPECS = {
    "youtube": PlatformSpec(
        name="YouTube",
        max_duration=3600,  # 1 hour
        aspect_ratio="16:9",
        resolution="1920x1080",
        typical_duration=600,  # 10 minutes
        subtitle_style="lower_third",
        audio_format="aac",
        video_codec="h264",
        quality_preset="high"
    ),
    
    "reels": PlatformSpec(
        name="Instagram Reels",
        max_duration=90,
        aspect_ratio="9:16",
        resolution="1080x1920",
        typical_duration=30,
        subtitle_style="center_overlay",
        audio_format="aac",
        video_codec="h264",
        quality_preset="medium"
    ),
    
    "tiktok": PlatformSpec(
        name="TikTok",
        max_duration=180,  # 3 minutes
        aspect_ratio="9:16",
        resolution="1080x1920",
        typical_duration=60,
        subtitle_style="center_overlay",
        audio_format="aac",
        video_codec="h264",
        quality_preset="medium"
    ),
    
    "twitter": PlatformSpec(
        name="Twitter",
        max_duration=140,
        aspect_ratio="16:9",
        resolution="1280x720",
        typical_duration=30,
        subtitle_style="lower_third",
        audio_format="aac",
        video_codec="h264",
        quality_preset="medium"
    ),
    
    "linkedin": PlatformSpec(
        name="LinkedIn",
        max_duration=600,  # 10 minutes
        aspect_ratio="16:9",
        resolution="1920x1080",
        typical_duration=120,  # 2 minutes
        subtitle_style="lower_third",
        audio_format="aac",
        video_codec="h264",
        quality_preset="high"
    )
}


def get_platform_specs(platform: str) -> Dict[str, Any]:
    """
    Get platform specifications.
    
    Args:
        platform: Platform name (youtube, reels, tiktok, etc.)
        
    Returns:
        Dictionary with platform specifications
    """
    platform_lower = platform.lower()
    if platform_lower not in PLATFORM_SPECS:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(PLATFORM_SPECS.keys())}")
    
    spec = PLATFORM_SPECS[platform_lower]
    return {
        "name": spec.name,
        "max_duration": spec.max_duration,
        "aspect_ratio": spec.aspect_ratio,
        "resolution": spec.resolution,
        "typical_duration": spec.typical_duration,
        "subtitle_style": spec.subtitle_style,
        "audio_format": spec.audio_format,
        "video_codec": spec.video_codec,
        "quality_preset": spec.quality_preset
    }


def get_available_platforms() -> List[str]:
    """Get list of available platform names."""
    return list(PLATFORM_SPECS.keys())


def get_ffmpeg_preset(platform: str, gpu_acceleration: bool = True) -> Dict[str, Any]:
    """
    Get FFmpeg encoding preset for platform.
    
    Args:
        platform: Target platform
        gpu_acceleration: Whether to use GPU acceleration (NVENC)
        
    Returns:
        Dictionary with FFmpeg parameters
    """
    spec = PLATFORM_SPECS[platform.lower()]
    width, height = map(int, spec.resolution.split('x'))
    
    # Base preset
    preset = {
        "width": width,
        "height": height,
        "fps": 30,
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
        "audio_bitrate": "128k"
    }
    
    # Video codec and quality settings
    if gpu_acceleration:
        preset.update({
            "video_codec": "h264_nvenc",
            "preset": "p4",  # NVENC preset
            "profile": "main",
            "level": "4.1"
        })
    else:
        preset.update({
            "video_codec": "libx264",
            "preset": "medium",  # x264 preset
            "profile": "main",
            "level": "4.1"
        })
    
    # Quality-specific settings
    if spec.quality_preset == "high":
        preset.update({
            "crf": 18 if not gpu_acceleration else None,
            "bitrate": "10M",
            "maxrate": "12M",
            "bufsize": "20M"
        })
    elif spec.quality_preset == "medium":
        preset.update({
            "crf": 23 if not gpu_acceleration else None,
            "bitrate": "5M",
            "maxrate": "6M",
            "bufsize": "10M"
        })
    else:  # low quality
        preset.update({
            "crf": 28 if not gpu_acceleration else None,
            "bitrate": "2M",
            "maxrate": "3M",
            "bufsize": "5M"
        })
    
    # Platform-specific optimizations
    if platform.lower() in ["reels", "tiktok"]:
        # Mobile-optimized settings
        preset["movflags"] = "+faststart"
        preset["tune"] = "film"
    elif platform.lower() == "youtube":
        # YouTube-optimized settings
        preset["movflags"] = "+faststart"
        preset["tune"] = "film"
    
    return preset


def get_subtitle_style(platform: str) -> Dict[str, Any]:
    """
    Get subtitle styling for platform.
    
    Args:
        platform: Target platform
        
    Returns:
        Dictionary with subtitle style parameters
    """
    spec = PLATFORM_SPECS[platform.lower()]
    
    if spec.subtitle_style == "center_overlay":
        return {
            "position": "center",
            "font_size": 48,
            "font_family": "Arial Bold",
            "font_color": "white",
            "background_color": "rgba(0,0,0,0.7)",
            "border_width": 2,
            "border_color": "black",
            "margin_v": 100,  # Vertical margin from bottom
            "alignment": "center"
        }
    elif spec.subtitle_style == "lower_third":
        return {
            "position": "bottom",
            "font_size": 36,
            "font_family": "Arial",
            "font_color": "white",
            "background_color": "rgba(0,0,0,0.8)",
            "border_width": 1,
            "border_color": "black",
            "margin_v": 50,
            "alignment": "left"
        }
    else:
        # Default style
        return {
            "position": "bottom",
            "font_size": 32,
            "font_family": "Arial",
            "font_color": "white",
            "background_color": "rgba(0,0,0,0.5)",
            "border_width": 1,
            "border_color": "black",
            "margin_v": 75,
            "alignment": "center"
        }


def validate_platform_constraints(
    platform: str,
    duration: float,
    width: int,
    height: int
) -> Dict[str, Any]:
    """
    Validate content against platform constraints.
    
    Args:
        platform: Target platform
        duration: Video duration in seconds
        width: Video width in pixels
        height: Video height in pixels
        
    Returns:
        Dictionary with validation results
    """
    spec = PLATFORM_SPECS[platform.lower()]
    target_width, target_height = map(int, spec.resolution.split('x'))
    
    # Calculate aspect ratios
    current_aspect = width / height
    target_aspect_parts = spec.aspect_ratio.split(':')
    target_aspect = int(target_aspect_parts[0]) / int(target_aspect_parts[1])
    
    validation = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check duration
    if duration > spec.max_duration:
        validation["errors"].append(
            f"Duration {duration:.1f}s exceeds platform maximum of {spec.max_duration}s"
        )
        validation["valid"] = False
    
    # Check aspect ratio (allow some tolerance)
    aspect_tolerance = 0.1
    if abs(current_aspect - target_aspect) > aspect_tolerance:
        validation["warnings"].append(
            f"Aspect ratio {current_aspect:.2f} differs from platform standard {target_aspect:.2f}"
        )
    
    # Check resolution
    if width != target_width or height != target_height:
        validation["warnings"].append(
            f"Resolution {width}x{height} differs from platform standard {spec.resolution}"
        )
    
    # Platform-specific checks
    if platform.lower() in ["reels", "tiktok"]:
        if duration < 5:
            validation["warnings"].append("Very short duration may not perform well on this platform")
        if duration > spec.typical_duration * 2:
            validation["warnings"].append("Long duration may reduce engagement on this platform")
    
    return validation