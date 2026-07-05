from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.vectorstore import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, k: int = 5):
        self._vector_store = vector_store
        self._k = k
        self._cache: Dict[Tuple[str, int], List[Tuple[Document, float]]] = {}

    def retrieve(self, query: str) -> List[Document]:
        return self._vector_store.as_retriever(k=self._k).invoke(query)

    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> List[Tuple[Document, float]]:
        k = k or self._k
        key = (query, k)
        if key not in self._cache:
            self._cache[key] = self._vector_store.similarity_search_with_score(
                query, k=k
            )
        return self._cache[key]
