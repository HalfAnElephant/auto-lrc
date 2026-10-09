"""Source annotations and review candidates; these are not human ground truth."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


ANNOTATION_STATES = ("present", "absent", "unknown", "not_applicable")
FIRST_PHONEME_CATEGORIES = (
    "vowel", "plosive", "fricative", "affricate", "nasal", "approximant",
    "aspirate", "unknown", "not_applicable",
)


@dataclass
class Phoneme:
    label: str
    start_seconds: float
    end_seconds: float
    start_sample: int
    end_sample: int


@dataclass
class Boundary:
    candidate_seconds: float
    candidate_sample: int
    source: str = "textgrid:first_phone"
    human_truth_sample: Optional[int] = None
    human_truth_seconds: Optional[float] = None
    confidence: Optional[float] = None
    review_status: str = "pending"


@dataclass
class LyricUnit:
    text: str
    kind: str
    start_seconds: float
    end_seconds: float
    start_sample: int
    end_sample: int
    phonemes: List[Phoneme]
    boundary: Optional[Boundary]
    raw_annotations: Optional[Dict[str, Any]]
    source_json_index: Optional[int]
    review_status: str = "pending"
    first_sounding_phoneme: Optional[Phoneme] = None
    granularity: Dict[str, Any] = field(default_factory=dict)
    classifications: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    score_mapping: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSegment:
    segment_id: str
    song_id: str
    language: str
    singer: str
    technique: str
    group: str
    audio_path: str
    sample_rate: int
    num_samples: int
    units: List[LyricUnit]
    sources: Dict[str, Dict[str, str]]
    raw_json: List[Dict[str, Any]]
    textgrid_tiers: Dict[str, List[Dict[str, Any]]]
    score_notes: List[Dict[str, Any]]
    dataset_revision: Optional[str] = None
    source_recording_offset_samples: Optional[int] = None
    validation_issues: List[Dict[str, str]] = field(default_factory=list)
    review_status: str = "pending"
    timing_provenance: str = (
        "TextGrid numerical times parsed by praatio; original lexical tokens are "
        "not retained in these fields. Source files remain unchanged. Samples use "
        "nearest integer, half up, on this WAV's timeline."
    )
    score_alignment: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "gtsinger-source-v2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
