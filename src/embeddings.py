import time
from functools import lru_cache
from typing import List

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class EmbeddingProvider(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "gemini-embedding-2"):
        self._model = model
        self._client = genai.Client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            contents = [
                types.Content(parts=[types.Part(text=t)]) for t in batch
            ]
            for attempt in range(10):
                try:
                    result = self._client.models.embed_content(
                        model=self._model, contents=contents
                    )
                except Exception as exc:
                    err_msg = str(exc)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        wait = 15 * (attempt + 1)
                    else:
                        wait = 5 * (attempt + 1)
                    if attempt < 9:
                        time.sleep(wait)
                        continue
                    raise
                break
            if not result.embeddings:
                continue
            all_embeddings.extend(e.values or [] for e in result.embeddings)
            time.sleep(1.5)
        return all_embeddings

    @lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> List[float]:
        content = types.Content(parts=[types.Part(text=text)])
        result = self._client.models.embed_content(
            model=self._model, contents=[content]
        )
        if not result.embeddings or not result.embeddings[0].values:
            raise ValueError("No embeddings returned by Google API")
        return result.embeddings[0].values


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small"):
        self._model = model
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/anomalyco/opencode",
            },
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "intfloat/e5-small-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = model
        self._client = SentenceTransformer(model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self._client.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    @lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}"
        embedding = self._client.encode(prefixed)
        return embedding.tolist()


def create_embedding_provider(model_name: str = "intfloat/e5-small-v2") -> EmbeddingProvider:
    if model_name == "gemini-embedding-2":
        return GoogleEmbeddingProvider(model=model_name)
    if model_name == "text-embedding-3-small":
        return OpenAIEmbeddingProvider(model=model_name)
    if model_name == "intfloat/e5-small-v2":
        return LocalEmbeddingProvider(model=model_name)
    raise ValueError(f"Unknown embedding model: {model_name}")
