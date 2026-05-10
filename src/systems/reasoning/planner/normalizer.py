from __future__ import annotations

import re

from .ontology import ATTRIBUTES, ROOM_CLASSES

OBJECT_ALIASES = {
    "purple_box_cart": (
        "purple box cart",
        "purple box",
        "box cart",
        "storage box",
        "storage box cart",
    ),
    "tv": ("tv", "television"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed",),
    "chair": ("chair",),
    "refrigerator": ("refrigerator", "fridge"),
    "door": ("door",),
    "bus": ("bus",),
}

ATTRIBUTE_ALIASES = {
    "power_state": (
        "power state",
        "power",
        "off",
        "on",
    ),
    "open_state": (
        "open state",
        "open",
        "closed",
    ),
}

ROOM_ALIASES = {
    "living_room": ("living room",),
    "bedroom": ("bedroom",),
    "kitchen": ("kitchen",),
}

FIND_KEYWORDS = ("find", "locate", "look for", "찾아")
NAVIGATE_KEYWORDS = ("navigate", "go to", "move to", "approach", "head to", "이동", "가줘", "다가가")
INSPECT_KEYWORDS = ("inspect", "check", "status", "is ")
_FREEFORM_KO_TARGET_PATTERNS = (
    r"(?P<target>.+?)(?:을|를)?\s*(?:찾아|찾아서|찾고|찾아줘|찾아와)",
    r"(?P<target>.+?)(?:으로|로|에)?\s*(?:이동|가줘|가 줘|다가가)",
)
_FREEFORM_EN_TARGET_PATTERNS = (
    r"(?:find|locate|look for)\s+(?:the\s+|a\s+|an\s+)?(?P<target>.+)",
    r"(?:navigate to|go to|move to|approach|head to)\s+(?:the\s+|a\s+|an\s+)?(?P<target>.+)",
)
_TRAILING_TARGET_NOISE = (
    "and come back",
    "come back",
    "return",
    "돌아와",
    "다녀와",
    "보고 와",
    "찾아서",
    "찾아",
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_object_name(value: str | None) -> str | None:
    text = normalize_text(value or "")
    if not text:
        return None
    for suffix in _TRAILING_TARGET_NOISE:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"[\"'`.,!?;:()\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" _-/")
    if not text:
        return None
    return text.replace(" ", "_")


def _merge_alias_maps(
    base_aliases: dict[str, tuple[str, ...]],
    extra_aliases: dict[str, tuple[str, ...]] | None,
) -> dict[str, tuple[str, ...]]:
    if not extra_aliases:
        return dict(base_aliases)
    merged: dict[str, tuple[str, ...]] = {}
    for canonical in set(base_aliases) | set(extra_aliases):
        aliases = [*base_aliases.get(canonical, ()), *extra_aliases.get(canonical, ())]
        merged[canonical] = tuple(dict.fromkeys(normalize_text(alias) for alias in aliases if alias))
    return merged


def _find_alias_hits(text: str, alias_map: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = normalize_text(text)
    hits: list[str] = []
    for canonical, aliases in alias_map.items():
        if any(alias in normalized for alias in aliases):
            hits.append(canonical)
    return hits


def detect_object_class(
    text: str,
    extra_aliases: dict[str, tuple[str, ...]] | None = None,
) -> str | None:
    hits = _find_alias_hits(text, _merge_alias_maps(OBJECT_ALIASES, extra_aliases))
    if hits:
        return hits[0]
    return detect_freeform_object_name(text)


def detect_freeform_object_name(text: str) -> str | None:
    normalized = normalize_text(text)
    for pattern in (*_FREEFORM_KO_TARGET_PATTERNS, *_FREEFORM_EN_TARGET_PATTERNS):
        match = re.search(pattern, normalized)
        if match is None:
            continue
        candidate = normalize_object_name(match.group("target"))
        if candidate:
            return candidate
    return None


def detect_attribute(
    text: str,
    object_class: str | None = None,
    extra_aliases: dict[str, tuple[str, ...]] | None = None,
) -> str | None:
    hits = _find_alias_hits(text, _merge_alias_maps(ATTRIBUTE_ALIASES, extra_aliases))
    if object_class is None:
        return hits[0] if hits else None
    allowed = set(ATTRIBUTES.get(object_class, ()))
    for hit in hits:
        if hit in allowed:
            return hit
    if len(allowed) == 1 and any(keyword in normalize_text(text) for keyword in INSPECT_KEYWORDS):
        return next(iter(allowed))
    return None


def detect_room_hints(
    text: str,
    known_rooms: list[str] | None = None,
    extra_aliases: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    hits = _find_alias_hits(text, _merge_alias_maps(ROOM_ALIASES, extra_aliases))
    if known_rooms is None:
        return hits
    allowed = set(known_rooms)
    return [room for room in hits if room in allowed]


def detect_instance_hint(text: str) -> str | None:
    normalized = normalize_text(text)
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    match = re.search(r"(\d+)\s*(?:st|nd|rd|th)?", normalized)
    if match:
        return match.group(1)
    return None


def infer_intent(text: str, object_class: str | None, attribute: str | None) -> str:
    normalized = normalize_text(text)
    if object_class and any(token in normalized for token in FIND_KEYWORDS):
        return "find_object"
    if object_class and attribute is None and any(token in normalized for token in NAVIGATE_KEYWORDS):
        return "navigate_to_object"
    if object_class and attribute is not None:
        return "inspect_attribute"
    if object_class and any(token in normalized for token in INSPECT_KEYWORDS):
        return "inspect_attribute"
    return "unsupported"


def detect_desired_check(text: str, attribute: str | None) -> str:
    normalized = normalize_text(text)
    if attribute == "power_state":
        if "off" in normalized:
            return "is_off"
        if "on" in normalized:
            return "is_on"
        return "inspect"
    if attribute == "open_state":
        if "closed" in normalized:
            return "is_closed"
        if "open" in normalized:
            return "is_open"
        return "inspect"
    return "inspect"


def supports_object_name(value: str | None) -> bool:
    return normalize_object_name(value) is not None


def supports_room_name(value: str | None) -> bool:
    return value is None or value in ROOM_CLASSES
