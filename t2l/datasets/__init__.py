"""Dataset adapters, independent of the inference models."""

from .gtsinger import load_segment
from .schema import AudioSegment, Boundary, LyricUnit, Phoneme

__all__ = ["AudioSegment", "Boundary", "LyricUnit", "Phoneme", "load_segment"]
