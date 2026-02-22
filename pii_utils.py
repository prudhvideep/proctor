"""
PII detection, redaction, and rehydration utilities.

PIIVault stores original ↔ placeholder mappings in memory.
Regex-based PII detection serves as a fast fallback when the
FunctionGemma classify_input tool call is not available.
"""

import re
import uuid
from dataclasses import dataclass, field


# --- Regex patterns for common PII types ---

PII_PATTERNS = {
    "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "PHONE": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "CREDIT_CARD": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    "DOB": re.compile(r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b'),
    "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
}


@dataclass
class PIIEntity:
    """A single detected PII entity."""
    entity_type: str   # e.g. "SSN", "EMAIL", "NAME"
    value: str         # the original text
    start: int         # character offset in original text
    end: int           # character offset end
    placeholder: str = ""  # assigned during redaction


@dataclass
class PIIVault:
    """In-memory vault mapping placeholders back to original PII values."""
    mappings: dict = field(default_factory=dict)  # placeholder → original value
    entities: list = field(default_factory=list)   # list of PIIEntity

    def add(self, entity_type: str, value: str, start: int, end: int) -> str:
        """Register a PII entity and return its placeholder."""
        # Generate a unique, type-tagged placeholder
        short_id = uuid.uuid4().hex[:6]
        placeholder = f"[{entity_type}_{short_id}]"

        entity = PIIEntity(
            entity_type=entity_type,
            value=value,
            start=start,
            end=end,
            placeholder=placeholder,
        )
        self.entities.append(entity)
        self.mappings[placeholder] = value
        return placeholder

    def clear(self):
        """Reset the vault."""
        self.mappings.clear()
        self.entities.clear()


# Context-based name detection: "for John Doe", "patient Prudhvi", "Mr. Smith", etc.
NAME_CONTEXT_RE = re.compile(
    r'\b(?:for|patient|Mr\.?|Mrs\.?|Ms\.?|Dr\.?|to|from|contact|named?|name\s+is)\s+'
    r'((?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*)',
)


def detect_pii_regex(text: str) -> list[PIIEntity]:
    """Detect PII using regex patterns + name heuristics. Returns list of PIIEntity."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            found.append(PIIEntity(
                entity_type=pii_type,
                value=match.group(),
                start=match.start(),
                end=match.end(),
            ))

    # Name detection via context clues
    for match in NAME_CONTEXT_RE.finditer(text):
        name = match.group(1)
        start = match.start(1)
        end = match.end(1)
        # Skip if overlapping with an already-detected entity
        overlaps = any(not (end <= e.start or start >= e.end) for e in found)
        if not overlaps:
            found.append(PIIEntity(
                entity_type="NAME",
                value=name,
                start=start,
                end=end,
            ))

    # Sort by position (earliest first)
    found.sort(key=lambda e: e.start)
    return found


def redact(text: str, entities: list[PIIEntity] = None, vault: PIIVault = None) -> tuple[str, PIIVault]:
    """
    Redact PII from text.

    If entities is None, runs regex detection automatically.
    Returns (redacted_text, vault).
    """
    if vault is None:
        vault = PIIVault()

    if entities is None:
        entities = detect_pii_regex(text)

    if not entities:
        return text, vault

    # Replace from end to start so offsets stay valid
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
    redacted = text
    for entity in sorted_entities:
        placeholder = vault.add(entity.entity_type, entity.value, entity.start, entity.end)
        redacted = redacted[:entity.start] + placeholder + redacted[entity.end:]

    return redacted, vault


def rehydrate(text: str, vault: PIIVault) -> str:
    """Swap placeholders back to original PII values."""
    result = text
    for placeholder, original in vault.mappings.items():
        result = result.replace(placeholder, original)
    return result


def has_pii_signals(text: str) -> bool:
    """Quick check if text likely contains PII (fast, no full scan)."""
    if '@' in text:  # email
        return True
    if re.search(r'\d{3}[-.\s]?\d{2}[-.\s]?\d{4}', text):  # SSN-like
        return True
    if re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text):  # phone-like
        return True
    if re.search(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', text):  # credit card
        return True
    if NAME_CONTEXT_RE.search(text):  # name after context word
        return True
    return False
