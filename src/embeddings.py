from functools import lru_cache
from typing import List

from google import genai
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingProvider(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "openai/text-embedding-3-small"):
        self._impl = OpenAIEmbeddings(
            model=model,
            base_url="https://openrouter.ai/api/v1",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._impl.embed_documents(batch))
        return all_embeddings

    @lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> List[float]:
        return self._impl.embed_query(text)


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-004"):
        self._model = model
        self._client = genai.Client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._client.models.embed_content(
            model=self._model, contents=texts
        )
        if not result.embeddings:
            return []
        return [e.values or [] for e in result.embeddings]

    @lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self._model, contents=text
        )
        if not result.embeddings or not result.embeddings[0].values:
            raise ValueError("No embeddings returned by Google API")
        return result.embeddings[0].values


def create_embedding_provider(model_name: str) -> EmbeddingProvider:
    if model_name in ("text-embedding-3-small", "openai/text-embedding-3-small"):
        return OpenRouterEmbeddingProvider(model="openai/text-embedding-3-small")
    if model_name == "text-embedding-004":
        return GoogleEmbeddingProvider(model=model_name)
    raise ValueError(f"Unknown embedding model: {model_name}")