from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        directory: Path,
        model_name: str,
        hf_token: str = "",
        hf_endpoint: str = "https://huggingface.co",
    ) -> None:
        self.directory = directory
        self.index_path = directory / "index.faiss"
        self.metadata_path = directory / "chunks.json"
        self.model_name = model_name
        self.hf_token = hf_token or None
        self.hf_endpoint = hf_endpoint
        self._lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []
        self._load()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    os.environ["HF_ENDPOINT"] = self.hf_endpoint
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name, token=self.hf_token
                    )
        return self._model

    def _load(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists() != self.metadata_path.exists():
            raise RuntimeError("FAISS index and metadata must both exist or both be absent.")
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self._index.ntotal != len(self._chunks):
                raise RuntimeError("FAISS index and chunk metadata are out of sync.")
        else:
            self._index = None
            self._chunks = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("No text chunks were produced for this document.")
        vectors = self._normalize(
            self.model.encode([item["text"] for item in chunks], convert_to_numpy=True)
        )
        with self._lock:
            dimension = vectors.shape[1]
            if self._index is not None and self._index.d != dimension:
                raise RuntimeError("Embedding dimension does not match the saved FAISS index.")
            if self._index is None:
                self._index = faiss.IndexFlatIP(dimension)
            self._index.add(vectors)
            self._chunks.extend(chunks)
            self._persist()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if not any(chunk["document_id"] == document_id for chunk in self._chunks):
                return False
            keep_indexes = [
                i for i, chunk in enumerate(self._chunks)
                if chunk["document_id"] != document_id
            ]
            if keep_indexes:
                vectors = np.vstack([self._index.reconstruct(i) for i in keep_indexes])
                rebuilt = faiss.IndexFlatIP(self._index.d)
                rebuilt.add(vectors)
                self._index = rebuilt
                self._chunks = [self._chunks[i] for i in keep_indexes]
            else:
                self._index = None
                self._chunks = []
            self._persist()
            return True

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            documents: dict[str, dict[str, Any]] = {}
            for chunk in self._chunks:
                document_id = chunk["document_id"]
                document = documents.setdefault(
                    document_id,
                    {
                        "document_id": document_id,
                        "filename": chunk["filename"],
                        "chunk_count": 0,
                        "uploaded_at": chunk["uploaded_at"],
                    },
                )
                document["chunk_count"] += 1
            return sorted(
                documents.values(),
                key=lambda item: item["uploaded_at"],
                reverse=True,
            )

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            query_vector = self._normalize(
                self.model.encode([query], convert_to_numpy=True)
            )
            scores, indexes = self._index.search(
                query_vector, min(limit, self._index.ntotal)
            )
            return [
                {**self._chunks[int(index)], "vector_score": float(score)}
                for score, index in zip(scores[0], indexes[0])
                if index >= 0
            ]

    def all_chunks(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._chunks)

    def _persist(self) -> None:
        index_tmp = self.index_path.with_suffix(".faiss.tmp")
        metadata_tmp = self.metadata_path.with_suffix(".json.tmp")
        if self._index is None:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            return
        faiss.write_index(self._index, str(index_tmp))
        index_tmp.replace(self.index_path)
        metadata_tmp.write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata_tmp.replace(self.metadata_path)
