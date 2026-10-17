from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).lower()
    return " ".join(TOKEN_RE.findall(value))


def tokenize(value: str) -> list[str]:
    return normalize_text(value).split()
