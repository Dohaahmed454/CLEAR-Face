"""Canonical CLEAR-Face emotion label space and dataset-label normalization.

The Milestone 1 label-mapping plan fixes a unified 7-class space shared by the
main dataset (AffectNet) and the external dataset (RAF-DB). AffectNet's eighth
class ("contempt") has no RAF-DB counterpart and is intentionally dropped.

This module is the single source of truth for class order -> index, used by the
manifest builder, the FER models, and every downstream script.
"""

# Fixed class order. The index of a class in this list is its integer label.
CLEAR_FACE_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLEAR_FACE_CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Alpha-only, lowercase aliases -> canonical class (or None to drop).
_TEXT_ALIASES = {
    "angry": "angry", "anger": "angry", "mad": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear", "afraid": "fear", "scared": "fear",
    "happy": "happy", "happiness": "happy", "joy": "happy", "smile": "happy",
    "neutral": "neutral", "neutrality": "neutral", "calm": "neutral",
    "sad": "sad", "sadness": "sad", "unhappy": "sad",
    "surprise": "surprise", "surprised": "surprise", "surprize": "surprise",
    # Explicitly dropped / non-target labels.
    "contempt": None, "contemptuous": None,
    "none": None, "uncertain": None, "nonface": None, "noface": None,
}

# AffectNet's canonical integer index -> class (some Kaggle mirrors use numeric
# folder names). 7 (contempt) is dropped.
AFFECTNET_INDEX = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "angry",
    7: None,
}


def normalize_label(raw):
    """Map a raw dataset/folder label to a canonical CLEAR-Face class.

    Returns the canonical class name, or ``None`` if the label is unknown or
    intentionally excluded (e.g. AffectNet "contempt").
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "" or s == "nan":
        return None
    # Pure-integer labels -> AffectNet index map.
    if s.isdigit():
        return AFFECTNET_INDEX.get(int(s))
    if s in _TEXT_ALIASES:
        return _TEXT_ALIASES[s]
    # Fallback: strip non-alphabetic characters and retry.
    alpha = "".join(ch for ch in s if ch.isalpha())
    if alpha in _TEXT_ALIASES:
        return _TEXT_ALIASES[alpha]
    # Last resort: exact canonical name.
    return alpha if alpha in CLASS_TO_IDX else None


def is_valid_label(raw):
    """True if ``raw`` maps to a canonical CLEAR-Face class."""
    return normalize_label(raw) in CLASS_TO_IDX


def label_map_dict():
    """Return the {class_name: index} map saved alongside model checkpoints."""
    return dict(CLASS_TO_IDX)
