"""HyperFrames motion graphics (HTML + GSAP compositions rendered to transparent WebM).

``toolchain`` installs and runs the pinned CLI; ``compose`` turns built-in templates
or custom HyperFrames projects into cached overlay files for the renderer.
"""

from .compose import GraphicError, render_graphic, templates
from .toolchain import HYPERFRAMES_VERSION, HyperFramesUnavailable, doctor, ensure

__all__ = ["HYPERFRAMES_VERSION", "GraphicError", "HyperFramesUnavailable", "doctor", "ensure",
           "render_graphic", "templates"]
