from rank_bm25 import BM25Okapi

from app.repositories.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def search(self, query: str, limit: int) -> list[dict]:
        chunks = self.vector_store.all_chunks()
        if not chunks:
            return []
        vector_results = self.vector_store.search(query, limit * 3)
        tokenized_corpus = [item["text"].lower().split() for item in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        keyword_scores = bm25.get_scores(query.lower().split())
        keyword_results = [
            chunks[index]
            for index in sorted(
                range(len(chunks)), key=lambda index: keyword_scores[index], reverse=True
            )[: limit * 3]
        ]

        scores: dict[str, float] = {}
        records: dict[str, dict] = {}
        for weight, results in ((0.65, vector_results), (0.35, keyword_results)):
            for rank, record in enumerate(results, start=1):
                key = f'{record["document_id"]}:{record["chunk"]}'
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
                records[key] = record
        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [records[key] for key in ordered]
