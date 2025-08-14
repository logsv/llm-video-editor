"""
Ollama-based planning module for local LLM inference.
"""
import json
import requests
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .planner import VideoPlanner, EditDecisionList, EDLClip, EditOperation


class OllamaLLM:
    """Simple Ollama API wrapper for local LLM inference."""
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Ollama model name (e.g., llama3.2, codellama, mistral)
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
    
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Invoke the Ollama model with messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response dictionary with 'content' key
        """
        # Combine messages into a single prompt for Ollama
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if hasattr(message, 'content'):
                content = message.content
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(content)
        
        prompt = "\\n\\n".join(prompt_parts)
        
        # Make request to Ollama
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=120  # 2 minutes timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return type('Response', (), {'content': result.get('response', '')})()
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid response from Ollama: {e}")


class OllamaVideoPlanner(VideoPlanner):
    """Video planner using local Ollama LLM."""
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama video planner.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
        """
        ollama_llm = OllamaLLM(model_name=model_name, base_url=base_url)
        super().__init__(llm=ollama_llm)
        self.model_name = model_name
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.llm.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                return any(self.model_name in name for name in model_names)
            return False
        except:
            return False
    
    def _get_system_message(self) -> str:
        """Get system message optimized for local LLMs."""
        return """You are a professional video editor AI. Create detailed Edit Decision Lists (EDL) from user prompts.

TASK: Analyze video content and generate JSON EDL for the target platform.

KEY RESPONSIBILITIES:
1. Select engaging, relevant content segments
2. Ensure smooth transitions at scene boundaries  
3. Optimize for platform specs (YouTube 16:9, Reels 9:16, etc.)
4. Include reframing, subtitles, and audio adjustments

RESPONSE FORMAT: Valid JSON only, following the exact schema provided.

Be precise, concise, and focus on the most impactful content segments."""

    def generate_edl(
        self,
        prompt: str,
        transcript_segments: List[Dict[str, Any]],
        scenes: List[Dict[str, Any]], 
        media_info,  # Can be Dict or MediaInfo object
        target_platform: str = "youtube"
    ) -> EditDecisionList:
        """
        Generate EDL using Ollama with optimized prompting for local models.
        """
        # Prepare simplified context for better local model performance
        context = self._prepare_simplified_context(
            transcript_segments, scenes, media_info, target_platform
        )
        
        # Create optimized planning prompt for local models
        planning_prompt = self._create_local_planning_prompt(prompt, context, target_platform)
        
        # Create messages
        messages = [
            {"role": "system", "content": self._get_system_message()},
            {"role": "user", "content": planning_prompt}
        ]
        
        # Get response from Ollama
        response = self.llm.invoke(messages)
        
        # Parse and validate EDL
        edl_data = self._parse_edl_response_with_fallback(response.content)
        edl = self._create_edl_object(edl_data, target_platform, media_info)
        
        return edl
    
    def _prepare_simplified_context(
        self,
        transcript_segments: List[Dict[str, Any]],
        scenes: List[Dict[str, Any]],
        media_info: Dict[str, Any],
        target_platform: str
    ) -> Dict[str, Any]:
        """Prepare simplified context for better local model performance."""
        # Limit data to most relevant parts
        limited_transcript = transcript_segments[:5]  # First 5 segments
        limited_scenes = scenes[:8]  # First 8 scenes
        
        # Convert Scene objects to dicts if needed
        scenes_dict = []
        for scene in limited_scenes:
            if hasattr(scene, 'start_time'):  # Scene object
                scenes_dict.append({
                    "start_time": scene.start_time,
                    "end_time": scene.end_time,
                    "duration": scene.duration
                })
            else:  # Already a dict
                scenes_dict.append(scene)
        
        # Handle both dict and object media_info formats
        if hasattr(media_info, 'duration'):
            # MediaInfo object
            duration = media_info.duration
            aspect_ratio = f"{media_info.width}:{media_info.height}"
        else:
            # Dict format
            duration = media_info.get('duration', 0)
            width = media_info.get('width', 1920)
            height = media_info.get('height', 1080)
            aspect_ratio = f"{width}:{height}"
        
        return {
            "media_info": {
                "duration": duration,
                "aspect_ratio": aspect_ratio
            },
            "transcript_sample": limited_transcript,
            "scenes_sample": scenes_dict,
            "platform_specs": self._get_platform_specs(target_platform)
        }
    
    def _create_local_planning_prompt(self, user_prompt: str, context: Dict[str, Any], target_platform: str) -> str:
        """Create optimized prompt for local models."""
        platform_specs = context["platform_specs"]
        
        prompt = f"""Create an Edit Decision List for: "{user_prompt}"

TARGET: {target_platform.upper()}
- Max duration: {platform_specs['max_duration']}s  
- Aspect ratio: {platform_specs['aspect_ratio']}
- Resolution: {platform_specs['resolution']}

SOURCE MEDIA:
- Duration: {context['media_info']['duration']:.1f}s
- Current ratio: {context['media_info']['aspect_ratio']}

TRANSCRIPT SAMPLE (first segments):
{json.dumps(context['transcript_sample'], indent=1)}

SCENE SAMPLE (first scenes):
{json.dumps(context['scenes_sample'], indent=1)}

GENERATE JSON EDL:
{{
  "target_duration": <seconds>,
  "clips": [
    {{
      "clip_id": "clip_001",
      "source": "<source_file>", 
      "start_time": <start_seconds>,
      "end_time": <end_seconds>,
      "operations": [
        {{"type": "reframe", "params": {{"target_aspect": "{platform_specs['aspect_ratio']}", "focus": "speaker"}}}},
        {{"type": "subtitle", "params": {{"text": "...", "start": <start>, "end": <end>, "style": "{platform_specs['subtitle_style']}"}}}},
        {{"type": "audio_adjust", "params": {{"normalize": true, "target_lufs": -16}}}}
      ]
    }}
  ],
  "global_operations": [
    {{"type": "platform_export", "params": {{"format": "{target_platform}"}}}}
  ]
}}

Generate JSON now:"""
        
        return prompt
    
    def _parse_edl_response_with_fallback(self, response: str) -> Dict[str, Any]:
        """
        Parse EDL response with multiple fallback strategies.
        Handles markdown blocks, plain text, and malformed JSON.
        """
        # Strategy 1: Try base parser first (handles basic JSON extraction)
        try:
            return self._parse_edl_response(response)
        except (json.JSONDecodeError, ValueError) as base_error:
            print(f"Base parser failed: {base_error}")
        
        # Strategy 2: Extract from markdown code blocks
        try:
            return self._extract_json_from_markdown(response)
        except (json.JSONDecodeError, ValueError) as md_error:
            print(f"Markdown parser failed: {md_error}")
        
        # Strategy 3: Multi-step parsing with text analysis
        try:
            return self._parse_text_to_json(response)
        except Exception as text_error:
            print(f"Text-to-JSON parser failed: {text_error}")
        
        # Strategy 4: Create basic fallback EDL
        return self._create_fallback_edl(response)
    
    def _extract_json_from_markdown(self, response: str) -> Dict[str, Any]:
        """Extract JSON from markdown code blocks."""
        # Look for ```json or ```JSON blocks
        json_pattern = r'```(?:json|JSON)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            # Try each match, with comment removal if needed
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    # Try removing comments and parsing again
                    try:
                        cleaned_match = self._remove_json_comments(match)
                        return json.loads(cleaned_match)
                    except json.JSONDecodeError:
                        continue
        
        # Look for any JSON-like structure in code blocks
        code_block_pattern = r'```[^`]*(\{.*?\})[^`]*```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                # Try removing comments and parsing again
                try:
                    cleaned_match = self._remove_json_comments(match)
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    continue
        
        raise ValueError("No valid JSON found in markdown blocks")
    
    def _remove_json_comments(self, json_str: str) -> str:
        """Remove // comments from JSON string while preserving string content."""
        lines = json_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Find // but not in strings
            comment_pos = -1
            in_string = False
            escaped = False
            
            for i, char in enumerate(line):
                if escaped:
                    escaped = False
                    continue
                    
                if char == '\\':
                    escaped = True
                    continue
                    
                if char == '"' and not escaped:
                    in_string = not in_string
                    continue
                    
                if not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    comment_pos = i
                    break
            
            if comment_pos >= 0:
                cleaned_lines.append(line[:comment_pos].rstrip())
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _parse_text_to_json(self, response: str) -> Dict[str, Any]:
        """
        Parse natural language response into structured JSON.
        Creates EDL from text descriptions and step-by-step instructions.
        """
        # Extract key information using regex patterns
        duration_pattern = r'(?:duration|length|time):\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?'
        clip_pattern = r'clip[^:]*:\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)'
        
        # Find target duration
        duration_match = re.search(duration_pattern, response, re.IGNORECASE)
        target_duration = float(duration_match.group(1)) if duration_match else 60.0
        
        # Find clip time ranges
        clip_matches = re.findall(clip_pattern, response, re.IGNORECASE)
        
        clips = []
        for i, (start, end) in enumerate(clip_matches[:5]):  # Limit to 5 clips
            clips.append({
                "clip_id": f"clip_{i+1:03d}",
                "source": "input_video",
                "start_time": float(start),
                "end_time": float(end),
                "operations": [
                    {"type": "reframe", "params": {"target_aspect": "9:16", "focus": "center"}},
                    {"type": "audio_adjust", "params": {"normalize": True, "target_lufs": -16}}
                ]
            })
        
        # If no clips found, create from step-by-step instructions
        if not clips:
            clips = self._extract_steps_to_clips(response, target_duration)
        
        return {
            "target_duration": target_duration,
            "clips": clips,
            "global_operations": [
                {"type": "platform_export", "params": {"format": "reels"}}
            ]
        }
    
    def _extract_steps_to_clips(self, response: str, target_duration: float) -> List[Dict[str, Any]]:
        """Extract steps from text and convert to clips."""
        # Look for step patterns
        step_patterns = [
            r'(?:step|phase|part)\s*\d+[:\.]?\s*([^\n\r]+)',
            r'\d+[\.\)]\s*([^\n\r]+)',
            r'(?:first|second|third|then|next|finally)[:\.]?\s*([^\n\r]+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                steps.extend(matches[:3])  # Limit to 3 steps
                break
        
        # Convert steps to time segments
        clips = []
        if steps:
            segment_duration = target_duration / len(steps)
            for i, step in enumerate(steps):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, target_duration)
                
                clips.append({
                    "clip_id": f"step_{i+1:03d}",
                    "source": "input_video",
                    "start_time": start_time,
                    "end_time": end_time,
                    "operations": [
                        {"type": "reframe", "params": {"target_aspect": "9:16", "focus": "center"}},
                        {"type": "subtitle", "params": {
                            "text": step.strip()[:50], 
                            "start": start_time, 
                            "end": end_time,
                            "style": "center_overlay"
                        }},
                        {"type": "audio_adjust", "params": {"normalize": True, "target_lufs": -16}}
                    ]
                })
        else:
            # Create single default clip
            clips = [{
                "clip_id": "default_001",
                "source": "input_video", 
                "start_time": 0.0,
                "end_time": min(target_duration, 30.0),
                "operations": [
                    {"type": "reframe", "params": {"target_aspect": "9:16", "focus": "center"}},
                    {"type": "audio_adjust", "params": {"normalize": True, "target_lufs": -16}}
                ]
            }]
        
        return clips
    
    def _create_fallback_edl(self, response: str) -> Dict[str, Any]:
        """Create a basic fallback EDL when all parsing fails."""
        print(f"Creating fallback EDL for response: {response[:200]}...")
        
        return {
            "target_duration": 30.0,
            "clips": [{
                "clip_id": "fallback_001",
                "source": "input_video",
                "start_time": 0.0,
                "end_time": 30.0,
                "operations": [
                    {"type": "reframe", "params": {"target_aspect": "9:16", "focus": "center"}},
                    {"type": "subtitle", "params": {
                        "text": "Generated content", 
                        "start": 0.0, 
                        "end": 30.0,
                        "style": "center_overlay"
                    }},
                    {"type": "audio_adjust", "params": {"normalize": True, "target_lufs": -16}}
                ]
            }],
            "global_operations": [
                {"type": "platform_export", "params": {"format": "reels"}}
            ],
            "metadata": {
                "fallback_used": True,
                "original_response": response[:500]
            }
        }


def create_ollama_planner(model_name: str = "llama3.2", base_url: str = "http://localhost:11434") -> OllamaVideoPlanner:
    """
    Create an Ollama-based video planner.
    
    Args:
        model_name: Ollama model name
        base_url: Ollama server URL
        
    Returns:
        OllamaVideoPlanner instance
    """
    return OllamaVideoPlanner(model_name=model_name, base_url=base_url)