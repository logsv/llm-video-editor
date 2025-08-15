"""
Smart reframing module using YOLO object detection and ByteTrack for intelligent cropping.
Replaces static center crop with content-aware aspect ratio conversion.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
import subprocess

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from yolox.tracker.byte_tracker import BYTETracker  # type: ignore
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False


@dataclass
class DetectionResult:
    """Container for object detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    class_id: int


@dataclass
class TrackedObject:
    """Container for tracked object information."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    frame_number: int


@dataclass
class CropRegion:
    """Container for crop region with timing information."""
    start_frame: int
    end_frame: int
    x: int
    y: int
    width: int
    height: int
    confidence: float


class SmartReframer:
    """Smart reframing using YOLO detection and ByteTrack tracking."""
    
    def __init__(self, model_size: str = "yolo11n.pt", use_gpu: bool = True):
        """
        Initialize smart reframer.
        
        Args:
            model_size: YOLO model size (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_size = model_size
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.tracker = None
        
        # Person detection focus (class_id=0 in COCO)
        self.target_classes = [0]  # Person class
        self.min_confidence = 0.5
        
    def _load_model(self):
        """Lazy load YOLO model."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        if self.model is None:
            print(f"Loading YOLO model: {self.model_size}")
            self.model = YOLO(self.model_size)
            self.model.to(self.device)
    
    def _init_tracker(self):
        """Initialize ByteTracker."""
        if not BYTETRACK_AVAILABLE:
            # Fallback to simple tracking using IoU
            return None
        
        # ByteTracker configuration
        class TrackerArgs:
            def __init__(self):
                self.track_thresh = 0.5
                self.track_buffer = 30
                self.match_thresh = 0.8
                self.mot20 = False
        
        args = TrackerArgs()
        return BYTETracker(frame_rate=30, args=args)
    
    def analyze_video_for_reframing(
        self,
        video_path: str,
        target_aspect: float = 9/16,  # Default to 9:16 for reels
        sample_interval: int = 30  # Analyze every 30th frame
    ) -> List[CropRegion]:
        """
        Analyze video and determine optimal crop regions for reframing.
        
        Args:
            video_path: Path to input video
            target_aspect: Target aspect ratio (width/height)
            sample_interval: Frame sampling interval for analysis
            
        Returns:
            List of crop regions with timing information
        """
        self._load_model()
        self.tracker = self._init_tracker()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Analyzing video for smart reframing:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  Frames: {total_frames} @ {fps:.1f}fps")
        print(f"  Target aspect: {target_aspect:.3f}")
        
        # Calculate target dimensions
        target_width, target_height = self._calculate_target_dimensions(
            frame_width, frame_height, target_aspect
        )
        
        tracked_objects = []
        frame_number = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames for analysis
            if frame_number % sample_interval == 0:
                # Detect objects in frame
                detections = self._detect_objects(frame)
                
                # Track objects if ByteTrack available
                if self.tracker and detections:
                    tracked = self._update_tracker(detections, frame_number)
                    tracked_objects.extend(tracked)
                else:
                    # Simple detection without tracking
                    for det in detections:
                        tracked_objects.append(TrackedObject(
                            track_id=frame_number,
                            bbox=det['bbox'],
                            confidence=det['confidence'],
                            class_id=det['class_id'],
                            class_name=det['class_name'],
                            frame_number=frame_number
                        ))
            
            frame_number += 1
        
        cap.release()
        
        # Generate crop regions from tracked objects
        crop_regions = self._generate_crop_regions(
            tracked_objects, target_width, target_height, frame_width, frame_height, fps
        )
        
        print(f"Generated {len(crop_regions)} crop regions")
        return crop_regions
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using YOLO."""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter by target classes and confidence
                    if class_id in self.target_classes and confidence >= self.min_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.model.names[class_id]
                        })
        
        return detections
    
    def _update_tracker(self, detections: List[Dict[str, Any]], frame_number: int) -> List[TrackedObject]:
        """Update object tracker with new detections."""
        if not self.tracker:
            return []
        
        # Convert detections to tracker format
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            dets.append([x1, y1, x2, y2, det['confidence']])
        
        if not dets:
            return []
        
        dets = np.array(dets)
        tracked_objects = []
        
        try:
            online_targets = self.tracker.update(dets, (1080, 1920), (1080, 1920))
            
            for track in online_targets:
                bbox = track.tlbr.astype(int)
                track_id = track.track_id
                confidence = track.score
                
                tracked_objects.append(TrackedObject(
                    track_id=track_id,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    confidence=confidence,
                    class_id=0,  # Person class
                    class_name="person",
                    frame_number=frame_number
                ))
        except Exception as e:
            print(f"Tracking error: {e}")
            # Fallback to detections without tracking
            for det in detections:
                tracked_objects.append(TrackedObject(
                    track_id=frame_number,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    frame_number=frame_number
                ))
        
        return tracked_objects
    
    def _calculate_target_dimensions(
        self, 
        frame_width: int, 
        frame_height: int, 
        target_aspect: float
    ) -> Tuple[int, int]:
        """Calculate target crop dimensions maintaining aspect ratio."""
        current_aspect = frame_width / frame_height
        
        if target_aspect < current_aspect:
            # Target is taller, crop width
            target_width = int(frame_height * target_aspect)
            target_height = frame_height
        else:
            # Target is wider, crop height
            target_width = frame_width
            target_height = int(frame_width / target_aspect)
        
        return target_width, target_height
    
    def _generate_crop_regions(
        self,
        tracked_objects: List[TrackedObject],
        target_width: int,
        target_height: int,
        frame_width: int,
        frame_height: int,
        fps: float
    ) -> List[CropRegion]:
        """Generate smooth crop regions from tracked objects."""
        if not tracked_objects:
            # Fallback to center crop
            x = (frame_width - target_width) // 2
            y = (frame_height - target_height) // 2
            return [CropRegion(0, 999999, x, y, target_width, target_height, 1.0)]
        
        # Group objects by track_id and time
        tracks = {}
        for obj in tracked_objects:
            if obj.track_id not in tracks:
                tracks[obj.track_id] = []
            tracks[obj.track_id].append(obj)
        
        # Find the most prominent track (highest average confidence)
        best_track = None
        best_score = 0
        
        for track_id, objects in tracks.items():
            avg_confidence = sum(obj.confidence for obj in objects) / len(objects)
            if avg_confidence > best_score:
                best_score = avg_confidence
                best_track = objects
        
        if not best_track:
            # Fallback to center crop
            x = (frame_width - target_width) // 2
            y = (frame_height - target_height) // 2
            return [CropRegion(0, 999999, x, y, target_width, target_height, 1.0)]
        
        # Sort by frame number
        best_track.sort(key=lambda x: x.frame_number)
        
        # Generate crop regions with smoothing
        crop_regions = []
        window_size = 5  # Smoothing window
        
        for i, obj in enumerate(best_track):
            # Calculate center of bounding box
            x1, y1, x2, y2 = obj.bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate crop position centered on object
            crop_x = max(0, min(center_x - target_width // 2, frame_width - target_width))
            crop_y = max(0, min(center_y - target_height // 2, frame_height - target_height))
            
            # Smooth with neighboring frames
            if i > 0 and len(crop_regions) > 0:
                prev_region = crop_regions[-1]
                # Simple smoothing
                crop_x = int(0.7 * crop_x + 0.3 * prev_region.x)
                crop_y = int(0.7 * crop_y + 0.3 * prev_region.y)
            
            start_frame = obj.frame_number
            end_frame = best_track[i + 1].frame_number if i + 1 < len(best_track) else obj.frame_number + int(fps)
            
            crop_regions.append(CropRegion(
                start_frame=start_frame,
                end_frame=end_frame,
                x=crop_x,
                y=crop_y,
                width=target_width,
                height=target_height,
                confidence=obj.confidence
            ))
        
        return self._merge_similar_regions(crop_regions)
    
    def _merge_similar_regions(self, regions: List[CropRegion]) -> List[CropRegion]:
        """Merge similar adjacent crop regions to reduce cuts."""
        if len(regions) <= 1:
            return regions
        
        merged = [regions[0]]
        threshold = 50  # pixels
        
        for region in regions[1:]:
            last_region = merged[-1]
            
            # Check if regions are similar
            dx = abs(region.x - last_region.x)
            dy = abs(region.y - last_region.y)
            
            if dx < threshold and dy < threshold:
                # Merge regions
                merged[-1] = CropRegion(
                    start_frame=last_region.start_frame,
                    end_frame=region.end_frame,
                    x=(last_region.x + region.x) // 2,
                    y=(last_region.y + region.y) // 2,
                    width=region.width,
                    height=region.height,
                    confidence=max(last_region.confidence, region.confidence)
                )
            else:
                merged.append(region)
        
        return merged
    
    def apply_smart_reframing(
        self,
        input_video: str,
        output_video: str,
        crop_regions: List[CropRegion],
        fps: float = 30.0
    ) -> bool:
        """
        Apply smart reframing to video using calculated crop regions.
        
        Args:
            input_video: Path to input video
            output_video: Path to output video
            crop_regions: List of crop regions to apply
            fps: Video frame rate
            
        Returns:
            True if successful
        """
        if not crop_regions:
            return False
        
        # Create FFmpeg filter for dynamic cropping
        filter_parts = []
        
        for i, region in enumerate(crop_regions):
            start_time = region.start_frame / fps
            end_time = region.end_frame / fps
            
            # Create crop filter for this time segment
            crop_filter = f"crop={region.width}:{region.height}:{region.x}:{region.y}"
            
            if i == 0:
                filter_parts.append(f"[0:v]{crop_filter}[crop{i}]")
            
            # Add time-based selection
            if i == len(crop_regions) - 1:
                # Last segment
                filter_parts.append(f"[crop{i}]")
            else:
                next_start = crop_regions[i + 1].start_frame / fps
                filter_parts.append(f"[crop{i}]trim=start={start_time}:end={next_start}[seg{i}]")
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-filter_complex', ';'.join(filter_parts),
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'copy',
            output_video
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and Path(output_video).exists()
        except Exception as e:
            print(f"Error applying smart reframing: {e}")
            return False


def create_smart_reframer(model_size: str = "yolo11n.pt", use_gpu: bool = True) -> SmartReframer:
    """
    Create a smart reframer instance.
    
    Args:
        model_size: YOLO model size
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        SmartReframer instance
    """
    return SmartReframer(model_size=model_size, use_gpu=use_gpu)


def fallback_to_autoflip_style(
    input_video: str,
    output_video: str,
    target_aspect: float = 9/16
) -> bool:
    """
    Fallback to AutoFlip-style center crop when YOLO is not available.
    
    Args:
        input_video: Input video path
        output_video: Output video path
        target_aspect: Target aspect ratio
        
    Returns:
        True if successful
    """
    # Get video dimensions
    probe_cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        input_video
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Calculate crop dimensions
        current_aspect = width / height
        
        if target_aspect < current_aspect:
            # Crop width (going from wide to tall)
            crop_width = int(height * target_aspect)
            crop_height = height
            crop_x = (width - crop_width) // 2
            crop_y = 0
        else:
            # Crop height (going from tall to wide)
            crop_width = width
            crop_height = int(width / target_aspect)
            crop_x = 0
            crop_y = (height - crop_height) // 2
        
        # Apply crop using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y}',
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'copy',
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and Path(output_video).exists()
        
    except Exception as e:
        print(f"Fallback reframing error: {e}")
        return False