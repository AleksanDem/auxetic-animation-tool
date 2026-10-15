"""
Ядро физических движков и экспорта генератора анимаций PHD_0826.
"""

from .styles import PALETTES, ASPECT_RATIOS
from .base_engine import BaseAnimationEngine
from .engine_uniaxial import UniaxialEngine
from .engine_bending import BendingEngine
from .engine_indentation import IndentationEngine
from .exporter import export_to_gif, export_to_mp4, check_ffmpeg_available

__all__ = [
    "PALETTES",
    "ASPECT_RATIOS",
    "BaseAnimationEngine",
    "UniaxialEngine",
    "BendingEngine",
    "IndentationEngine",
    "export_to_gif",
    "export_to_mp4",
    "check_ffmpeg_available",
]
