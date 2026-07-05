from abc import ABC, abstractmethod
from typing import Dict, List

from langchain_core.embeddings import Embeddings


class EmbeddingProvider(Embeddings, ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings

        self._impl = OpenAIEmbeddings(
            model=model,
            base_url="https://openrouter.ai/api/v1",
        )
        self._query_cache: Dict[str, List[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if text not in self._query_cache:
            self._query_cache[text] = self._impl.embed_query(text)
        return self._query_cache[text]


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-004"):
        self._model = model
        self._query_cache: Dict[str, List[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from google import genai

        client = genai.Client()
        result = client.models.embed_content(model=self._model, contents=texts)
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        if text in self._query_cache:
            return self._query_cache[text]
        from google import genai

        client = genai.Client()
        result = client.models.embed_content(model=self._model, contents=[text])
        emb = result.embeddings[0].values
        self._query_cache[text] = emb
        return emb


def create_embedding_provider(model_name: str) -> EmbeddingProvider:
    if model_name == "text-embedding-3-small":
        return OpenRouterEmbeddingProvider(model=model_name)
    if model_name == "text-embedding-004":
        return GoogleEmbeddingProvider(model=model_name)
    raise ValueError(f"Unknown embedding model: {model_name}")
