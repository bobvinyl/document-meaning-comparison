"""NLP processing for semantic document comparison."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from sentence_transformers import SentenceTransformer, util


class DocumentSemanticProcessor:
    """Performs semantic comparison on pre-loaded document text."""

    def __init__(
        self,
        model_name: str,
        embedder: Any | None = None,
        cos_sim_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = embedder if embedder is not None else SentenceTransformer(model_name)
        self.cos_sim_fn = cos_sim_fn if cos_sim_fn is not None else util.cos_sim

    def compare(
        self,
        text1: str,
        text2: str,
        normalize: bool,
        include_alignments: bool,
        top_k: int,
    ) -> dict[str, Any]:
        if normalize:
            text1 = self.normalize_text(text1)
            text2 = self.normalize_text(text2)

        if not text1 or not text2:
            raise ValueError("Both documents must be non-empty after preprocessing.")

        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        score = self._to_float(self.cos_sim_fn(embeddings[0], embeddings[1]))

        result: dict[str, Any] = {
            "cosine_similarity": score,
            "interpretation": self.interpret_score(score),
        }

        if include_alignments:
            result["sentence_alignments"] = self.sentence_alignment(
                text1=text1,
                text2=text2,
                top_k=top_k,
            )

        return result

    @staticmethod
    def normalize_text(text: str) -> str:
        """Apply minimal normalization while preserving meaning-bearing content."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def interpret_score(score: float) -> str:
        if score < 0.4:
            return "low"
        if score < 0.7:
            return "medium"
        return "high"

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into sentence-like units for alignment."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def sentence_alignment(self, text1: str, text2: str, top_k: int) -> list[dict[str, Any]]:
        """Return best-matching sentence pairs from doc1 to doc2."""
        sentences1 = self.split_sentences(text1)
        sentences2 = self.split_sentences(text2)

        if not sentences1 or not sentences2:
            return []

        emb1 = self.model.encode(sentences1, convert_to_tensor=True)
        emb2 = self.model.encode(sentences2, convert_to_tensor=True)
        matrix = self.cos_sim_fn(emb1, emb2)

        matches: list[dict[str, Any]] = []
        for i, sent1 in enumerate(sentences1):
            row = matrix[i]
            best_j = self._argmax_index(row)
            score = self._to_float(row[best_j])
            matches.append(
                {
                    "score": score,
                    "doc1_sentence": sent1,
                    "doc2_sentence": sentences2[best_j],
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[: max(top_k, 1)]

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    @classmethod
    def _argmax_index(cls, row: Any) -> int:
        if hasattr(row, "argmax"):
            index = row.argmax()
            return int(index.item() if hasattr(index, "item") else index)

        best_index = 0
        best_score = cls._to_float(row[0])
        for idx in range(1, len(row)):
            score = cls._to_float(row[idx])
            if score > best_score:
                best_index = idx
                best_score = score
        return best_index
