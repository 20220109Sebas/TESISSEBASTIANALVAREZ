"""Opt-in maximum similarity to an audited historical award; no model loading."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SIGNALS = ("bm25", "semantic", "category", "region", "amount")
MODELS = ("labse", "bm25_labse", "hybrid_full")


class OrderedVectors:
    """Candidate order mapped into a read-only matrix without a whole-matrix copy."""
    ndim = 2

    def __init__(self, matrix, rows):
        self.matrix, self.rows = matrix, np.asarray(rows, dtype=np.int64)
        if self.rows.ndim != 1 or np.any(self.rows < 0) or np.any(self.rows >= len(matrix)):
            raise ValueError("Invalid candidate rows")
        self.shape = (len(rows), matrix.shape[1])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return self.matrix[self.rows[key]]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def award_text(award):
    return " ".join(v for v in (award.title, award.description, award.technical_text) if v).strip()


def checked(path, expected):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected:
        raise ValueError(f"Changed input: {path}")


class SourceVectors:
    """Read only blocks referenced by the frozen source index."""

    def __init__(self, cache, store):
        self.cache, self.store = Path(cache), Path(store)
        metadata = json.loads((self.cache / "metadata.json").read_text())
        snapshot = self.cache / "source_store_index.json"
        checked(snapshot, metadata["source_store_index_snapshot_sha256"])
        self.index = json.loads(snapshot.read_text())
        self.blocks = {}
        for field in ("model_name", "model_revision", "profile_strategy", "opportunity_strategy"):
            if self.index["fingerprint"][field] != metadata[field]:
                raise ValueError("Source index uses a different model or encoding strategy")

    def get(self, text):
        digest = text_hash(text)
        entry = self.index["texts"][digest]
        name = entry["block"]
        if Path(name).name != name:
            raise ValueError("Unsafe source block path")
        if name not in self.blocks:
            spec = self.index["blocks"][name]
            checked(self.store / name, spec["sha256"])
            block = np.load(self.store / name, mmap_mode="r", allow_pickle=False)
            if list(block.shape) != spec["shape"] or block.dtype != np.float32:
                raise ValueError("Invalid source block")
            self.blocks[name] = block
        block = self.blocks[name]
        row = entry["row"]
        if not isinstance(row, int) or not 0 <= row < len(block):
            raise ValueError("Source row outside block")
        vector = np.asarray(block[row], dtype=np.float64)
        if not np.isfinite(vector).all() or not math.isclose(float(np.linalg.norm(vector)), 1., abs_tol=1e-5):
            raise ValueError("Invalid source vector")
        return vector, {"text_sha256": digest, "block": name, "row": row,
                        "block_sha256": self.index["blocks"][name]["sha256"]}


def maximum_similarity(history_vectors, candidate_vectors, block_size=2048):
    """Float64 dot products, deterministic first-history tie, bounded candidate blocks."""
    history = np.asarray(history_vectors, dtype=np.float64)
    if (history.ndim != 2 or not len(history) or candidate_vectors.ndim != 2
            or history.shape[1] != candidate_vectors.shape[1] or block_size <= 0
            or not np.isfinite(history).all()):
        raise ValueError("Invalid vectors or block size")
    if not np.allclose(np.linalg.norm(history, axis=1), 1., rtol=0., atol=1e-5):
        raise ValueError("Historical embeddings must be L2 normalized")
    raw = np.empty(len(candidate_vectors), dtype=np.float64)
    support = np.empty(len(candidate_vectors), dtype=np.int64)
    for start in range(0, len(candidate_vectors), block_size):
        candidate = np.asarray(candidate_vectors[start:start + block_size], dtype=np.float64)
        if not np.isfinite(candidate).all():
            raise ValueError("Nonfinite candidate embedding")
        if not np.allclose(np.linalg.norm(candidate, axis=1), 1., rtol=0., atol=1e-5):
            raise ValueError("Candidate embeddings must be L2 normalized")
        scores = candidate @ history.T
        winners = np.argmax(scores, axis=1)
        raw[start:start + len(candidate)] = scores[np.arange(len(candidate)), winners]
        support[start:start + len(candidate)] = winners
    if not len(raw):
        raise ValueError("Empty candidate population")
    low, high = float(raw.min()), float(raw.max())
    normalized = np.full(len(raw), float(high > 0)) if math.isclose(low, high) else (raw - low) / (high - low)
    return raw, normalized, support


def rank_components(components, weights, order, semantic=None):
    if not order or len(set(order)) != len(order) or set(components) != set(SIGNALS):
        raise ValueError("Invalid candidate identities or components")
    if (set(weights) - set(SIGNALS) or any(not math.isfinite(v) or v < 0 for v in weights.values())
            or not math.isclose(sum(weights.values()), 1., abs_tol=1e-6)):
        raise ValueError("Invalid weights")
    values = dict(components)
    if semantic is not None:
        values["semantic"] = semantic
    for signal in SIGNALS:
        array = np.asarray(values[signal])
        if array.shape != (len(order),) or not np.isfinite(array).all():
            raise ValueError("Invalid signal shape or values")
    scores = [sum(weights.get(s, 0.) * float(values[s][i]) for s in SIGNALS) for i in range(len(order))]
    return scores, sorted(range(len(order)), key=lambda i: (-scores[i], order[i]))
