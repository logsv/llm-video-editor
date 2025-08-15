"""
File utilities for video processing.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import mimetypes


# Supported video file extensions
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
    '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts'
}

# Supported audio file extensions  
AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.aac', '.flac', '.ogg', '.wma', '.m4a'
}


def find_video_files(path: Path) -> List[str]:
    """
    Find all video files in a directory or return single file if path is a file.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    if path.is_file():
        if is_video_file(str(path)):
            video_files.append(str(path))
    elif path.is_dir():
        for file_path in path.rglob('*'):
            if file_path.is_file() and is_video_file(str(file_path)):
                video_files.append(str(file_path))
    
    return sorted(video_files)


def is_video_file(filepath: str) -> bool:
    """
    Check if file is a supported video file.
    
    Args:
        filepath: Path to file
        
    Returns:
        True if file is a video file
    """
    extension = Path(filepath).suffix.lower()
    return extension in VIDEO_EXTENSIONS


def is_audio_file(filepath: str) -> bool:
    """
    Check if file is a supported audio file.
    
    Args:
        filepath: Path to file
        
    Returns:
        True if file is an audio file
    """
    extension = Path(filepath).suffix.lower()
    return extension in AUDIO_EXTENSIONS


def validate_input_path(path: str) -> Dict[str, Any]:
    """
    Validate that input path exists and contains video files.
    
    Args:
        path: Input path
        
    Returns:
        Dictionary with validation results
    """
    path_obj = Path(path)
    
    result = {
        "path": path,
        "exists": path_obj.exists(),
        "valid": False,
        "is_video": False,
        "is_dir": False,
        "error": None
    }
    
    if not path_obj.exists():
        result["error"] = "File not found"
        return result
    
    if path_obj.is_file():
        result["is_video"] = is_video_file(path)
        if result["is_video"]:
            result["valid"] = True
        else:
            result["error"] = "not a supported video format"
    elif path_obj.is_dir():
        result["is_dir"] = True
        video_files = find_video_files(path_obj)
        if len(video_files) > 0:
            result["valid"] = True
            result["video_count"] = len(video_files)
        else:
            result["error"] = "No video files found in directory"
    
    return result


def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get basic file information.
    
    Args:
        filepath: Path to file
        
    Returns:
        Dictionary with file information
    """
    path = Path(filepath)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    return {
        'name': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified_time': stat.st_mtime,
        'is_video': is_video_file(filepath),
        'is_audio': is_audio_file(filepath)
    }


def create_output_filename(
    input_file: str,
    suffix: str = "",
    extension: str = ".mp4",
    output_dir: Optional[str] = None
) -> str:
    """
    Create output filename based on input file.
    
    Args:
        input_file: Input file path
        suffix: Suffix to add to filename
        extension: File extension for output
        output_dir: Output directory (uses input dir if None)
        
    Returns:
        Output file path
    """
    input_path = Path(input_file)
    
    # Create output filename
    output_name = input_path.stem
    if suffix:
        output_name += f"_{suffix}"
    output_name += extension
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir) / output_name
    else:
        output_path = input_path.parent / output_name
    
    return str(output_path)


def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def cleanup_temp_files(temp_files: List[str]) -> None:
    """
    Clean up temporary files.
    
    Args:
        temp_files: List of temporary file paths to delete
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception:
            pass  # Ignore cleanup errors


def get_safe_filename(filename: str) -> str:
    """
    Convert filename to safe version for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    if filename is None:
        return ""
    
    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    
    # Limit length
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
    
    return safe_name


def estimate_processing_time(
    input_files: List[str],
    target_duration: float = None
) -> Dict[str, float]:
    """
    Estimate processing time for video files.
    
    Args:
        input_files: List of input video files
        target_duration: Target duration for processed videos
        
    Returns:
        Dictionary with time estimates
    """
    from ..core.media_probe import MediaProbe
    
    total_input_duration = 0
    total_output_duration = 0
    
    for file_path in input_files:
        try:
            media_info = MediaProbe.probe_file(file_path)
            total_input_duration += media_info.duration
            
            if target_duration:
                total_output_duration += min(target_duration, media_info.duration)
            else:
                total_output_duration += media_info.duration
                
        except Exception:
            # Assume average duration if probe fails
            total_input_duration += 300  # 5 minutes
            total_output_duration += target_duration or 300
    
    # Rough estimates (in seconds)
    # Processing is typically 0.1-0.5x real-time depending on operations
    asr_time = total_input_duration * 0.1  # ASR is fast
    scene_detection_time = total_input_duration * 0.05  # Very fast
    rendering_time = total_output_duration * 2.0  # Rendering is slower
    
    total_estimate = asr_time + scene_detection_time + rendering_time
    
    return {
        'asr_seconds': asr_time,
        'scene_detection_seconds': scene_detection_time,
        'rendering_seconds': rendering_time,
        'total_seconds': total_estimate,
        'total_minutes': total_estimate / 60,
        'input_duration': total_input_duration,
        'output_duration': total_output_duration
    }


def check_disk_space(output_dir: str, estimated_size_mb: float) -> bool:
    """
    Check if there's enough disk space for output.
    
    Args:
        output_dir: Output directory path
        estimated_size_mb: Estimated output size in MB
        
    Returns:
        True if there's enough space
    """
    try:
        import shutil
        free_bytes = shutil.disk_usage(output_dir).free
        free_mb = free_bytes / (1024 * 1024)
        
        # Add 20% buffer
        required_mb = estimated_size_mb * 1.2
        
        return free_mb >= required_mb
        
    except Exception:
        return True  # Assume OK if we can't check


def organize_output_files(
    output_dir: str,
    create_subdirs: bool = True
) -> Dict[str, str]:
    """
    Organize output directory structure.
    
    Args:
        output_dir: Base output directory
        create_subdirs: Whether to create subdirectories
        
    Returns:
        Dictionary with subdirectory paths
    """
    base_path = Path(output_dir)
    
    subdirs = {
        'videos': 'videos',
        'audio': 'audio', 
        'subtitles': 'subtitles',
        'thumbnails': 'thumbnails',
        'edl': 'edl',
        'reports': 'reports',
        'temp': 'temp'
    }
    
    if create_subdirs:
        for subdir in subdirs.values():
            (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    return {k: str(base_path / v) for k, v in subdirs.items()}


def get_project_structure() -> Dict[str, Any]:
    """
    Get project structure information.
    
    Returns:
        Dictionary with project structure information
    """
    # Get the project root (assuming this file is in llm_video_editor/utils/)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    
    structure = {}
    
    # Check for main project directories
    main_dirs = ['llm_video_editor', 'tests', 'examples', 'docs']
    for dir_name in main_dirs:
        dir_path = project_root / dir_name
        structure[dir_name] = {
            'exists': dir_path.exists(),
            'path': str(dir_path),
            'is_dir': dir_path.is_dir() if dir_path.exists() else False
        }
        
        if dir_path.exists() and dir_path.is_dir():
            # Count files in directory
            try:
                files = list(dir_path.rglob('*'))
                py_files = [f for f in files if f.suffix == '.py']
                structure[dir_name]['total_files'] = len(files)
                structure[dir_name]['python_files'] = len(py_files)
            except:
                structure[dir_name]['total_files'] = 0
                structure[dir_name]['python_files'] = 0
    
    # Add project root info as a dict to match test expectations
    structure['project_info'] = {
        'project_root': str(project_root),
        'has_pyproject_toml': (project_root / 'pyproject.toml').exists(),
        'has_requirements_txt': (project_root / 'requirements.txt').exists()
    }
    
    return structure