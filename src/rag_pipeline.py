import time
from typing import Dict, List

from langchain_core.documents import Document

from src.generators import (
    GeminiGenerator,
    GroqGenerator,
    OpenRouterGenerator,
)
from src.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self._retriever = retriever
        self._generators: Dict[str, object] = {
            "gemini": GeminiGenerator(),
            "gpt41mini": OpenRouterGenerator(),
            "llama31": GroqGenerator(),
        }

    def run(self, question: str) -> Dict:
        docs: List[Document] = self._retriever.retrieve(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        responses = {}
        latencies = {}
        for name, gen in self._generators.items():
            t0 = time.time()
            try:
                responses[name] = gen.generate(question=question, context=context)
            except Exception as e:
                responses[name] = f"[{name} error: {e}]"
            latencies[name] = round((time.time() - t0) * 1000)
        return {
            "question": question,
            "context": context,
            "responses": responses,
            "latencies": latencies,
        }
