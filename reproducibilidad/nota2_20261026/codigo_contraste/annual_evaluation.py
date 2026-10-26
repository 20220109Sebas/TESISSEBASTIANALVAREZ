"""Controles del contraste anual, sin alterar los algoritmos congelados.

Los snapshots se recorren registro a registro. Los embeddings se consultan
mediante mmap y se convierten a floats Python solo al utilizar cada vector.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from .validation import sha256, timestamp


class JSONStream:
    def __init__(self, handle, chunk_size=65536, maximum_value_chars=16 * 1024 * 1024):
        self.handle, self.chunk_size = handle, chunk_size
        self.maximum = maximum_value_chars
        self.buffer, self.eof = "", False
        self.decoder = json.JSONDecoder()

    def more(self):
        chunk = self.handle.read(self.chunk_size)
        self.eof = not chunk
        self.buffer += chunk
        if len(self.buffer) > self.maximum:
            raise ValueError("Un valor JSON supera el límite de lectura incremental")

    def whitespace(self):
        while True:
            self.buffer = self.buffer.lstrip()
            if self.buffer or self.eof:
                return
            self.more()

    def character(self, expected):
        self.whitespace()
        if not self.buffer.startswith(expected):
            raise ValueError(f"Estructura JSON inesperada, se esperaba {expected!r}")
        self.buffer = self.buffer[len(expected):]

    def peek(self):
        self.whitespace()
        return self.buffer[:1]

    def value(self):
        self.whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                # No aceptar un número o literal aún incompleto al final del bloque.
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                if self.buffer[end:end + 1] not in {"", ":", ",", "}", "]", " ", "\n", "\r", "\t"}:
                    raise ValueError("Separador JSON inválido")
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("JSON incompleto o inválido") from exc
                self.more()


def iter_snapshot_records(path: Path, chunk_size=65536):
    """Lee todo el objeto OCDS, verifica el cierre y no carga el array completo."""
    with path.open(encoding="utf-8") as handle:
        stream = JSONStream(handle, chunk_size)
        stream.character("{")
        keys = set()
        while stream.peek() != "}":
            key = stream.value()
            if not isinstance(key, str) or key in keys:
                raise ValueError("Clave raíz JSON inválida o duplicada")
            keys.add(key)
            stream.character(":")
            if key == "records":
                stream.character("[")
                if stream.peek() != "]":
                    while True:
                        record = stream.value()
                        if not isinstance(record, dict):
                            raise ValueError("El registro OCDS debe ser un objeto")
                        yield record
                        if stream.peek() == "]":
                            break
                        stream.character(",")
                stream.character("]")
            else:
                stream.value()
            if stream.peek() == "}":
                break
            stream.character(",")
            if stream.peek() == "}":
                raise ValueError("Coma final no permitida")
        stream.character("}")
        if "records" not in keys or stream.peek():
            raise ValueError("Faltan records o hay contenido después del paquete")


def classify_positive(row: dict, profiles: set[str], availability: dict, cutoff: date) -> tuple[str, str]:
    """La etiqueta cambia, nunca la población candidata ni los perfiles."""
    if row["supplier_id"] not in profiles:
        return "profile_not_retained", ""
    source = availability.get(row["opportunity_id"])
    if source is None:
        return "outside_frozen_candidate_ids", ""
    if source["status"] != "verified":
        return "candidate_not_retained", ""
    try:
        observed_day = date.fromisoformat(row["award_date"][:10])
    except (ValueError, TypeError):
        return "invalid_outcome_date", ""
    published = timestamp(source["text_published_at"])
    if published is None:
        raise ValueError("Candidato auditado sin fecha nativa de publicación")
    native_day = published.astimezone(ZoneInfo("America/Lima")).date()
    if observed_day < cutoff:
        return "outcome_before_cutoff", ""
    if observed_day < native_day:
        return "outcome_before_text_publication", ""
    return "verified_positive", "same_day_order_unresolved" if observed_day == native_day else "later_day"


class VectorRows(Sequence):
    def __init__(self, matrix, rows):
        self.matrix, self.rows = matrix, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return VectorRows(self.matrix, self.rows[index])
        return self.matrix[self.rows[index]].tolist()


class MemmapEmbedder:
    """Mismos floats y orden que PrecomputedEmbedder, sin expandir toda la matriz."""
    def __init__(self, root: Path):
        import numpy as np

        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if sha256(root / "embeddings.npy") != self.metadata["embeddings_sha256"]:
            raise ValueError("Cambió la matriz de embeddings")
        self.matrix = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if self.matrix.shape != (self.metadata["vector_count"], self.metadata["embedding_dimension"]):
            raise ValueError("Dimensiones de caché inconsistentes")
        entries = json.loads((root / "index.json").read_text(encoding="utf-8"))
        self.rows = {}
        for entry in entries:
            digest, row = entry["text_sha256"], int(entry["row"])
            if not 0 <= row < len(self.matrix) or digest in self.rows and self.rows[digest] != row:
                raise ValueError("Índice ambiguo o fuera de la matriz")
            self.rows[digest] = row
        self.model_name = self.metadata["model_name"]

    def encode(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest not in self.rows:
                raise KeyError("Texto ausente del caché anual")
            rows.append(self.rows[digest])
        return VectorRows(self.matrix, rows)
