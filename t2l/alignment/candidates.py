"""Serializable coarse predictions; no field represents calibrated accuracy."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundaryCandidate:
    candidate_id: str
    word_index: int
    line_index: int
    unit_in_line: int
    word_occurrence: int
    line_occurrence: int
    text: str
    language: str
    source: str
    start_seconds: Optional[float]
    end_seconds: Optional[float]
    raw_frames: Dict[str, Any]
    score: Optional[float]
    score_kind: str
    original_sample_rate: int
    original_num_samples: int
    start_sample: Optional[int]
    end_sample: Optional[int]
    phonemes: List[Any]
    search_window: Optional[Dict[str, Any]]
    confidence: Optional[float] = None
    confidence_status: str = "uncalibrated"
    status: str = "candidate"
    decoder: str = "legacy"
    constraints: Dict[str, Any] = field(default_factory=dict)
    fallback: List[str] = field(default_factory=list)
    text_metadata: Dict[str, Any] = field(default_factory=dict)
    sample_coordinate_kind: str = (
        "Nominal frame time rounded to the original sample grid; physical onset is uncalibrated"
    )

    def to_dict(self):
        return asdict(self)
