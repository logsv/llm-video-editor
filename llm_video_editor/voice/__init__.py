"""AI voice-over: local text-to-speech engines, text preparation and cached synthesis.

The agent writes the lines and decides where (and whether) narration helps; this
package only turns text into clean audio and reports its exact length. Nothing is
sent anywhere: the engines run on this machine.
"""

from .engines import ENGINES, Engine, EngineUnavailable, VoiceInfo, get_engine, list_voices
from .synth import Line, synthesize

__all__ = ["ENGINES", "Engine", "EngineUnavailable", "Line", "VoiceInfo", "get_engine",
           "list_voices", "synthesize"]
