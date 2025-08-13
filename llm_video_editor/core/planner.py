"""
LLM-based planning module for generating Edit Decision Lists (EDL) from prompts.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate


@dataclass
class EditOperation:
    """Single edit operation in an EDL."""
    type: str  # "cut", "reframe", "subtitle", "audio_adjust", etc.
    params: Dict[str, Any]


@dataclass
class EDLClip:
    """Single clip entry in Edit Decision List."""
    clip_id: str
    source: str
    start_time: float
    end_time: float
    duration: float
    operations: List[EditOperation]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EditDecisionList:
    """Complete Edit Decision List for a video project."""
    target_platform: str  # "youtube", "reels", "tiktok"
    target_duration: float
    clips: List[EDLClip]
    global_operations: List[EditOperation]
    metadata: Dict[str, Any]


class VideoPlanner:
    """LLM-based video planning and EDL generation."""
    
    def __init__(self, llm: Optional[LLM] = None, model_name: str = "gpt-4"):
        """
        Initialize video planner.
        
        Args:
            llm: Language model instance. If None, creates ChatOpenAI instance
            model_name: Model name for default LLM
        """
        if llm is None:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        else:
            self.llm = llm
    
    def generate_edl(
        self,
        prompt: str,
        transcript_segments: List[Dict[str, Any]],
        scenes: List[Dict[str, Any]],
        media_info: Dict[str, Any],
        target_platform: str = "youtube"
    ) -> EditDecisionList:
        """
        Generate Edit Decision List based on user prompt and media analysis.
        
        Args:
            prompt: User's editing prompt/request
            transcript_segments: List of transcript segments with timing
            scenes: List of detected scenes with timing
            media_info: Media file information
            target_platform: Target platform ("youtube", "reels", "tiktok")
            
        Returns:
            EditDecisionList object
        """
        # Prepare context for LLM
        context = self._prepare_context(
            transcript_segments, scenes, media_info, target_platform
        )
        
        # Generate planning prompt
        planning_prompt = self._create_planning_prompt(prompt, context, target_platform)
        
        # Get LLM response
        messages = [
            SystemMessage(content=self._get_system_message()),
            HumanMessage(content=planning_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse and validate EDL
        edl_data = self._parse_edl_response(response.content)
        edl = self._create_edl_object(edl_data, target_platform, media_info)
        
        return edl
    
    def _prepare_context(
        self,
        transcript_segments: List[Dict[str, Any]],
        scenes: List[Dict[str, Any]],
        media_info: Dict[str, Any],
        target_platform: str
    ) -> Dict[str, Any]:
        """Prepare context information for LLM planning."""
        return {
            "media_info": {
                "duration": media_info.get("duration", 0),
                "fps": media_info.get("fps", 30),
                "width": media_info.get("width", 1920),
                "height": media_info.get("height", 1080),
                "aspect_ratio": media_info.get("aspect_ratio", "16:9")
            },
            "transcript": transcript_segments,
            "scenes": scenes,
            "platform_specs": self._get_platform_specs(target_platform)
        }
    
    def _get_platform_specs(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific specifications."""
        specs = {
            "youtube": {
                "max_duration": 3600,  # 1 hour
                "aspect_ratio": "16:9",
                "resolution": "1920x1080",
                "typical_duration": 600,  # 10 minutes
                "subtitle_style": "lower_third"
            },
            "reels": {
                "max_duration": 90,
                "aspect_ratio": "9:16",
                "resolution": "1080x1920",
                "typical_duration": 30,
                "subtitle_style": "center_overlay"
            },
            "tiktok": {
                "max_duration": 180,  # 3 minutes
                "aspect_ratio": "9:16",
                "resolution": "1080x1920",
                "typical_duration": 60,
                "subtitle_style": "center_overlay"
            }
        }
        return specs.get(platform, specs["youtube"])
    
    def _get_system_message(self) -> str:
        """Get system message for LLM."""
        return """You are a professional video editor AI. Your task is to analyze video content and create detailed Edit Decision Lists (EDL) based on user prompts.

Key responsibilities:
1. Analyze transcript and scene information
2. Select the most engaging and relevant content segments
3. Create smooth transitions and cuts at scene boundaries when possible
4. Optimize for the target platform's specifications and audience
5. Include appropriate reframing, subtitles, and audio adjustments

Always respond with valid JSON following the specified EDL schema."""
    
    def _create_planning_prompt(self, user_prompt: str, context: Dict[str, Any], target_platform: str) -> str:
        """Create detailed planning prompt for LLM."""
        platform_specs = context["platform_specs"]
        
        prompt = f"""Create an Edit Decision List for the following request:

USER REQUEST: {user_prompt}

TARGET PLATFORM: {target_platform}
Platform specifications:
- Max duration: {platform_specs['max_duration']}s
- Aspect ratio: {platform_specs['aspect_ratio']}
- Resolution: {platform_specs['resolution']}
- Typical duration: {platform_specs['typical_duration']}s

MEDIA INFORMATION:
- Source duration: {context['media_info']['duration']:.1f}s
- Current aspect ratio: {context['media_info']['aspect_ratio']}
- Resolution: {context['media_info']['width']}x{context['media_info']['height']}
- FPS: {context['media_info']['fps']}

TRANSCRIPT SEGMENTS:
{json.dumps(context['transcript'][:10], indent=2)}  # Show first 10 segments

SCENE BOUNDARIES:
{json.dumps(context['scenes'][:10], indent=2)}  # Show first 10 scenes

RULES:
1. Select segments with high engagement potential (clear speech, action, key points)
2. Keep cuts on or near scene boundaries when possible
3. Ensure smooth flow and narrative coherence
4. For vertical formats (9:16), plan reframing to focus on speakers/action
5. Include subtitle overlays for key dialogue
6. Normalize audio levels
7. Stay within target duration limits

REQUIRED JSON SCHEMA:
{{
  "target_duration": <float>,
  "clips": [
    {{
      "clip_id": "<string>",
      "source": "<source_file>",
      "start_time": <float>,
      "end_time": <float>,
      "operations": [
        {{
          "type": "reframe",
          "params": {{"target_aspect": "9:16", "focus": "speaker"}}
        }},
        {{
          "type": "subtitle",
          "params": {{"text": "...", "start": <float>, "end": <float>, "style": "center_overlay"}}
        }},
        {{
          "type": "audio_adjust",
          "params": {{"normalize": true, "target_lufs": -16}}
        }}
      ]
    }}
  ],
  "global_operations": [
    {{
      "type": "platform_export",
      "params": {{"format": "{target_platform}"}}
    }}
  ]
}}

Generate the EDL now:"""
        
        return prompt
    
    def _parse_edl_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into EDL dictionary."""
        try:
            # Extract JSON from response (in case there's additional text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in response")
            
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse EDL response: {e}\\nResponse: {response}")
    
    def _create_edl_object(self, edl_data: Dict[str, Any], target_platform: str, media_info: Dict[str, Any]) -> EditDecisionList:
        """Convert parsed EDL data to EditDecisionList object."""
        clips = []
        
        for i, clip_data in enumerate(edl_data.get("clips", [])):
            operations = []
            for op_data in clip_data.get("operations", []):
                operations.append(EditOperation(
                    type=op_data["type"],
                    params=op_data.get("params", {})
                ))
            
            clips.append(EDLClip(
                clip_id=clip_data.get("clip_id", f"clip_{i:03d}"),
                source=clip_data.get("source", getattr(media_info, "filepath", "unknown")),
                start_time=clip_data["start_time"],
                end_time=clip_data["end_time"],
                duration=clip_data["end_time"] - clip_data["start_time"],
                operations=operations,
                metadata=clip_data.get("metadata", {})
            ))
        
        global_operations = []
        for op_data in edl_data.get("global_operations", []):
            global_operations.append(EditOperation(
                type=op_data["type"],
                params=op_data.get("params", {})
            ))
        
        return EditDecisionList(
            target_platform=target_platform,
            target_duration=edl_data.get("target_duration", sum(clip.duration for clip in clips)),
            clips=clips,
            global_operations=global_operations,
            metadata={
                "generated_from": "llm_planner",
                "source_duration": media_info.get("duration", 0),
                "compression_ratio": edl_data.get("target_duration", 0) / media_info.get("duration", 1)
            }
        )
    
    def export_edl(self, edl: EditDecisionList, output_path: str, format: str = "json") -> str:
        """
        Export EDL to file.
        
        Args:
            edl: EditDecisionList object
            output_path: Output file path
            format: Export format ("json", "csv", "fcpxml")
            
        Returns:
            Path to exported file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(asdict(edl), f, indent=2, default=str)
        
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Clip_ID', 'Source', 'Start_Time', 'End_Time', 'Duration', 'Operations'])
                for clip in edl.clips:
                    operations_str = "; ".join([f"{op.type}({op.params})" for op in clip.operations])
                    writer.writerow([
                        clip.clip_id,
                        clip.source,
                        f"{clip.start_time:.3f}",
                        f"{clip.end_time:.3f}",
                        f"{clip.duration:.3f}",
                        operations_str
                    ])
        
        elif format == "fcpxml":
            # Basic FCPXML export (simplified)
            self._export_fcpxml(edl, output_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path
    
    def _export_fcpxml(self, edl: EditDecisionList, output_path: str) -> None:
        """Export EDL as Final Cut Pro XML (simplified version)."""
        from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
        from xml.dom import minidom
        
        # Create basic FCPXML structure
        fcpxml = Element("fcpxml", version="1.8")
        resources = SubElement(fcpxml, "resources")
        
        # Add resources (source files)
        sources = set(clip.source for clip in edl.clips)
        for i, source in enumerate(sources):
            asset = SubElement(resources, "asset", 
                             id=f"r{i+1}", 
                             name=Path(source).stem,
                             src=source)
        
        # Create project
        library = SubElement(fcpxml, "library")
        event = SubElement(library, "event", name="LLM Video Edit")
        project = SubElement(event, "project", name="Generated Edit")
        sequence = SubElement(project, "sequence", 
                            duration=f"{int(edl.target_duration * 30)}s",  # Assuming 30fps
                            format="r1")
        spine = SubElement(sequence, "spine")
        
        # Add clips to spine
        for clip in edl.clips:
            source_idx = list(sources).index(clip.source) + 1
            clip_element = SubElement(spine, "clip", 
                                    name=clip.clip_id,
                                    duration=f"{int(clip.duration * 30)}s",
                                    start=f"{int(clip.start_time * 30)}s")
            
        # Write to file
        rough_string = tostring(fcpxml, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        
        with open(output_path, 'w') as f:
            f.write(reparsed.toprettyxml(indent="  "))