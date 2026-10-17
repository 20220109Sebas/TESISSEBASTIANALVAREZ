from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[list[str]] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequency: Counter[str] = Counter()
        self.avgdl = 0.0

    def fit(self, documents: list[str]) -> "BM25":
        self.documents = [tokenize(document) for document in documents]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequency = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))
        self.avgdl = sum(map(len, self.documents)) / len(self.documents) if self.documents else 0.0
        return self

    def score(self, query: str) -> list[float]:
        if not self.documents:
            return []
        query_terms = tokenize(query)
        n_docs = len(self.documents)
        output: list[float] = []
        for document, frequencies in zip(self.documents, self.term_frequencies):
            score = 0.0
            doc_len = len(document)
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * frequency * (self.k1 + 1) / norm
            output.append(score)
        return output
