"""
Platform presets and specifications.
"""

from .platform_presets import (
    get_platform_specs,
    get_available_platforms,
    get_ffmpeg_preset,
    get_subtitle_style,
    validate_platform_constraints
)

__all__ = [
    'get_platform_specs',
    'get_available_platforms', 
    'get_ffmpeg_preset',
    'get_subtitle_style',
    'validate_platform_constraints'
]