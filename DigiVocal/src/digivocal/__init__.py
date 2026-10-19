"""DigiVocal: Mandarin articulatory-synthesis research MVP."""

from .gesture_planner import PlannerConfig, plan_gestures
from .synthesizer import synthesize_timeline
from .voice_profile import VoiceProfile

__all__ = [
    "PlannerConfig",
    "VoiceProfile",
    "plan_gestures",
    "synthesize_timeline",
]

__version__ = "0.1.0"
