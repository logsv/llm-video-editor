"""
Scene detection module using PySceneDetect for identifying shot boundaries.
"""
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import tempfile
import os

from scenedetect import detect, ContentDetector, ThresholdDetector
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager


@dataclass
class Scene:
    """Container for scene information."""
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int


class SceneDetector:
    """Scene detection using PySceneDetect."""
    
    def __init__(self, threshold: float = 27.0, min_scene_len: float = 1.0):
        """
        Initialize scene detector.
        
        Args:
            threshold: Threshold for content detection (lower = more sensitive)
            min_scene_len: Minimum scene length in seconds
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    def detect_scenes(self, video_path: str, method: str = "content") -> List[Scene]:
        """
        Detect scenes in video file.
        
        Args:
            video_path: Path to video file
            method: Detection method ("content" or "threshold")
            
        Returns:
            List of Scene objects with timing information
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Detecting scenes in: {video_path}")
        
        if method == "content":
            detector = ContentDetector(threshold=self.threshold)
        elif method == "threshold":
            detector = ThresholdDetector(threshold=self.threshold)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Detect scenes
        scene_list = detect(video_path, detector, show_progress=True)
        
        if not scene_list:
            print("No scenes detected, treating entire video as single scene")
            # Get video duration for single scene
            from .media_probe import MediaProbe
            media_info = MediaProbe.probe_file(video_path)
            scene_list = [(0.0, media_info.duration)]
        
        scenes = []
        for i, (start_time, end_time) in enumerate(scene_list):
            start_seconds = start_time.get_seconds() if hasattr(start_time, 'get_seconds') else start_time
            end_seconds = end_time.get_seconds() if hasattr(end_time, 'get_seconds') else end_time
            
            # Skip scenes that are too short
            duration = end_seconds - start_seconds
            if duration < self.min_scene_len:
                continue
            
            scenes.append(Scene(
                start_time=start_seconds,
                end_time=end_seconds,
                duration=duration,
                start_frame=int(start_seconds * 30),  # Approximate frame numbers
                end_frame=int(end_seconds * 30)
            ))
        
        # Ensure we have at least one scene - use entire video as single scene if no scenes detected
        if not scenes:
            print("No valid scenes detected after filtering, using entire video as single scene")
            # Get video duration for fallback scene
            from .media_probe import MediaProbe
            media_info = MediaProbe.probe_file(video_path)
            scenes.append(Scene(
                start_time=0.0,
                end_time=media_info.duration,
                duration=media_info.duration,
                start_frame=0,
                end_frame=int(media_info.duration * 30)
            ))
        
        print(f"Detected {len(scenes)} scenes")
        return scenes
    
    def detect_with_stats(self, video_path: str) -> Tuple[List[Scene], dict]:
        """
        Detect scenes and return additional statistics.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (scenes list, stats dictionary)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create a video manager and scene manager
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        
        # Add content detector
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        # Start video manager and perform scene detection
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        
        # Convert to Scene objects
        scenes = []
        for start_time, end_time in scene_list:
            start_seconds = start_time.get_seconds()
            end_seconds = end_time.get_seconds()
            duration = end_seconds - start_seconds
            
            if duration >= self.min_scene_len:
                scenes.append(Scene(
                    start_time=start_seconds,
                    end_time=end_seconds,
                    duration=duration,
                    start_frame=start_time.get_frames(),
                    end_frame=end_time.get_frames()
                ))
        
        # Gather statistics
        stats = {
            'total_scenes': len(scenes),
            'avg_scene_length': sum(s.duration for s in scenes) / len(scenes) if scenes else 0,
            'shortest_scene': min(s.duration for s in scenes) if scenes else 0,
            'longest_scene': max(s.duration for s in scenes) if scenes else 0,
            'fps': video_manager.get_framerate()
        }
        
        video_manager.release()
        return scenes, stats
    
    def get_scene_thumbnails(
        self,
        video_path: str,
        scenes: List[Scene],
        output_dir: str = "scene_thumbnails"
    ) -> List[str]:
        """
        Generate thumbnail images for detected scenes.
        
        Args:
            video_path: Path to video file
            scenes: List of detected scenes
            output_dir: Directory to save thumbnails
            
        Returns:
            List of paths to thumbnail images
        """
        Path(output_dir).mkdir(exist_ok=True)
        thumbnail_paths = []
        
        import subprocess
        
        for i, scene in enumerate(scenes):
            # Take thumbnail from middle of scene
            timestamp = scene.start_time + (scene.duration / 2)
            output_path = f"{output_dir}/scene_{i:03d}_{timestamp:.2f}s.jpg"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(timestamp),
                '-vframes', '1',
                '-q:v', '2',  # High quality
                '-vf', 'scale=320:240',  # Thumbnail size
                output_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                thumbnail_paths.append(output_path)
            except subprocess.CalledProcessError:
                continue
        
        return thumbnail_paths
    
    def export_scene_list(self, scenes: List[Scene], output_path: str, format: str = "csv") -> str:
        """
        Export scene list to file.
        
        Args:
            scenes: List of Scene objects
            output_path: Output file path
            format: Export format ("csv", "json", "txt")
            
        Returns:
            Path to exported file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scene', 'Start_Time', 'End_Time', 'Duration', 'Start_Frame', 'End_Frame'])
                for i, scene in enumerate(scenes):
                    writer.writerow([
                        i + 1,
                        f"{scene.start_time:.3f}",
                        f"{scene.end_time:.3f}",
                        f"{scene.duration:.3f}",
                        scene.start_frame,
                        scene.end_frame
                    ])
        
        elif format == "json":
            import json
            scene_data = [
                {
                    "scene_id": i + 1,
                    "start_time": scene.start_time,
                    "end_time": scene.end_time,
                    "duration": scene.duration,
                    "start_frame": scene.start_frame,
                    "end_frame": scene.end_frame
                }
                for i, scene in enumerate(scenes)
            ]
            with open(output_path, 'w') as f:
                json.dump(scene_data, f, indent=2)
        
        elif format == "txt":
            with open(output_path, 'w') as f:
                f.write("Scene Detection Results\\n")
                f.write("=" * 50 + "\\n")
                for i, scene in enumerate(scenes):
                    f.write(f"Scene {i+1:3d}: {scene.start_time:8.3f}s - {scene.end_time:8.3f}s "
                           f"({scene.duration:6.3f}s)\\n")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path
    
    def merge_short_scenes(self, scenes: List[Scene], min_duration: float = 2.0) -> List[Scene]:
        """
        Merge scenes that are shorter than minimum duration with adjacent scenes.
        
        Args:
            scenes: List of Scene objects
            min_duration: Minimum scene duration in seconds
            
        Returns:
            List of merged Scene objects
        """
        if not scenes:
            return scenes
        
        merged_scenes = []
        current_scene = scenes[0]
        
        for i in range(1, len(scenes)):
            next_scene = scenes[i]
            
            # If current scene is too short, merge with next
            if current_scene.duration < min_duration:
                current_scene = Scene(
                    start_time=current_scene.start_time,
                    end_time=next_scene.end_time,
                    duration=next_scene.end_time - current_scene.start_time,
                    start_frame=current_scene.start_frame,
                    end_frame=next_scene.end_frame
                )
            else:
                merged_scenes.append(current_scene)
                current_scene = next_scene
        
        # Add the last scene
        merged_scenes.append(current_scene)
        
        return merged_scenes