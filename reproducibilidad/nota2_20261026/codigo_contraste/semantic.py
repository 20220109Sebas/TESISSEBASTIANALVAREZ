from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from .text import tokenize


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def safe_token_chunks(token_ids: list[int], max_sequence_length: int, decode, measure) -> tuple[list[str], list[int], int]:
    """Asigna todos los tokens y comprueba el tamaño tras decode/encode.

    Una partición de WordPieces puede volver a tokenizarse con más piezas al
    quedar una subpalabra al inicio. Reducir el tramo evita truncar el input.
    """
    limit = max_sequence_length - 2
    if limit <= 0:
        raise ValueError("El modelo necesita espacio para tokens especiales")
    chunks, covered, adjustments = [], [], 0
    position = 0
    while position < len(token_ids):
        count = min(limit, len(token_ids) - position)
        while True:
            piece = decode(token_ids[position:position + count])
            actual_length = measure(piece)
            if actual_length <= max_sequence_length:
                break
            adjustments += 1
            if count == 1:
                raise ValueError("Un token aislado supera el límite al recodificarse")
            count = max(1, count - max(1, actual_length - max_sequence_length))
        chunks.append(piece)
        covered.append(count)
        position += count
    if sum(covered) != len(token_ids):
        raise ValueError("Hay tokens sin asignar a un fragmento")
    return chunks or [""], covered, adjustments


class LaBSEEmbedder:
    """Adaptador real. Requiere el extra `semantic` y acceso al modelo."""

    model_name = "sentence-transformers/LaBSE"

    def __init__(self, model_name: str | None = None, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Instale el extra semántico: pip install -e '.[semantic]'") from exc
        self.model_name = model_name or self.model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return vectors.tolist()

    @property
    def model_max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    @property
    def embedding_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model_revision(self) -> str | None:
        first_module = self._model[0]
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        return getattr(config, "_commit_hash", None)

    def token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self._model.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        return [int(value) for value in encoded["length"]]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Codifica textos largos por fragmentos y agrega con media normalizada."""
        chunk_token_limit = self.model_max_seq_length - 2
        chunks: list[str] = []
        document_rows: list[list[int]] = []
        original_lengths: list[int] = []
        document_stats = []
        total_adjustments = 0
        for text in texts:
            token_ids = self._model.tokenizer.encode(
                text, add_special_tokens=False, truncation=False
            )
            original_lengths.append(len(token_ids) + 2)
            def decode(piece_ids):
                piece = self._model.tokenizer.decode(piece_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                return piece or self._model.tokenizer.decode(piece_ids, skip_special_tokens=False)
            pieces, covered, adjustments = safe_token_chunks(
                token_ids, self.model_max_seq_length, decode,
                lambda piece: len(self._model.tokenizer.encode(piece, add_special_tokens=True, truncation=False)))
            rows: list[int] = []
            for piece in pieces:
                rows.append(len(chunks))
                chunks.append(piece)
            document_rows.append(rows)
            total_adjustments += adjustments
            document_stats.append({"original_tokens": len(token_ids) + 2, "chunks": len(pieces),
                                   "original_token_ids_covered": sum(covered), "chunk_boundary_adjustments": adjustments})

        chunk_vectors = self.encode(chunks)
        results: list[list[float]] = []
        for rows in document_rows:
            vectors = [chunk_vectors[row] for row in rows]
            mean = [sum(values) / len(vectors) for values in zip(*vectors)]
            norm = math.sqrt(sum(value * value for value in mean)) or 1.0
            results.append([value / norm for value in mean])
        self.last_encoding_audit = {
            "documents": len(texts),
            "chunks": len(chunks),
            "chunk_token_limit_without_special_tokens": chunk_token_limit,
            "documents_split_into_multiple_chunks": sum(
                len(rows) > 1 for rows in document_rows
            ),
            "maximum_original_tokens_with_special_tokens": max(original_lengths, default=0),
            "chunk_partition_version": "retokenization_checked_v2",
            "chunk_boundary_adjustments": total_adjustments,
        }
        self.last_document_encoding_stats = document_stats
        return results


class PrecomputedEmbedder:
    """Lee un caché verificable de embeddings sin volver a ejecutar el modelo."""

    def __init__(self, cache_dir: str | Path) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("El caché de embeddings requiere numpy") from exc

        root = Path(cache_dir)
        metadata_path = root / "metadata.json"
        index_path = root / "index.json"
        vectors_path = root / "embeddings.npy"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = json.loads(index_path.read_text(encoding="utf-8"))
        expected_sha = self.metadata["embeddings_sha256"]
        actual_sha = hashlib.sha256(vectors_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("El archivo embeddings.npy no coincide con su SHA-256")
        self._vectors = np.load(vectors_path, allow_pickle=False)
        self._rows = {entry["text_sha256"]: int(entry["row"]) for entry in entries}
        if len(self._vectors) != int(self.metadata["vector_count"]):
            raise ValueError("El número de vectores no coincide con metadata.json")
        self.model_name = self.metadata["model_name"]

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if digest not in self._rows:
                raise KeyError(
                    "El texto solicitado no existe en el caché. Reconstruya el caché "
                    "con los mismos archivos de entrada."
                )
            vectors.append(self._vectors[self._rows[digest]].tolist())
        return vectors


class DeterministicTestEmbedder:
    """Feature hashing para pruebas de software; NO sustituye LaBSE en la tesis."""

    model_name = "deterministic-test-embedder-not-for-research"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Los embeddings deben tener igual dimensión")
    return sum(a * b for a, b in zip(left, right))
