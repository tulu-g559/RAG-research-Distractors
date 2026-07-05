from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings import EmbeddingProvider, OpenRouterEmbeddingProvider

DEFAULT_PERSIST_DIR = Path("data/faiss")


class VectorStore:
    def __init__(
        self,
        index: FAISS | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._index = index
        self._embedding_provider = embedding_provider or OpenRouterEmbeddingProvider()

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "VectorStore":
        provider = embedding_provider or OpenRouterEmbeddingProvider()
        index = FAISS.from_documents(documents, provider)
        return cls(index, provider)

    def save_local(self, path: str | Path = DEFAULT_PERSIST_DIR) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._index.save_local(str(path))

    @classmethod
    def load_local(
        cls,
        path: str | Path = DEFAULT_PERSIST_DIR,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "VectorStore":
        provider = embedding_provider or OpenRouterEmbeddingProvider()
        index = FAISS.load_local(
            str(path), provider, allow_dangerous_deserialization=True
        )
        return cls(index, provider)

    def as_retriever(self, k: int = 5):
        return self._index.as_retriever(search_kwargs={"k": k})

    def similarity_search_with_score(self, query: str, k: int = 5):
        return self._index.similarity_search_with_score(query, k=k)
