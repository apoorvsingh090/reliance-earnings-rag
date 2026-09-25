"""Validation: refuse to silently accept broken downloads.

Checks: file exists, non-trivial size, and magic bytes match the expected
kind (PDF starts with %PDF; the NSE iXBRL filings are HTML starting with
'<' / DOCTYPE). Per-file minimum sizes are enforced from the document spec
so a block page or truncated response can never enter the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    size: int
    kind: str  # "pdf" | "html"


def validate_file(path: Path, *, min_bytes: int, expected_kind: str) -> ValidationResult:
    if not path.exists():
        raise ValidationError(f"missing file: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ValidationError(f"{path.name}: only {size} bytes (< {min_bytes} minimum)")
    with path.open("rb") as f:
        head = f.read(2048)
    if expected_kind == "pdf":
        if not head.startswith(b"%PDF"):
            raise ValidationError(f"{path.name}: not a PDF (magic bytes mismatch)")
        return ValidationResult(path, size, "pdf")
    if expected_kind == "html":
        stripped = head.lstrip()
        if not (stripped.startswith(b"<") or b"<html" in head.lower()):
            raise ValidationError(f"{path.name}: not HTML (magic bytes mismatch)")
        return ValidationResult(path, size, "html")
    raise ValidationError(f"unknown expected_kind: {expected_kind}")
