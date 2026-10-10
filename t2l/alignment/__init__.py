"""Coarse candidate API; importing this package does not load inference models."""
from .candidates import BoundaryCandidate
from .coarse import coarse_align
from .text import describe_lyrics

__all__ = ["BoundaryCandidate", "coarse_align", "describe_lyrics"]
