import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.distractors import DistractorInjector
from src.generators import (
    GeminiGenerator,
    GroqGenerator,
    OpenRouterGenerator,
)
from src.retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        injector: Optional[DistractorInjector] = None,
    ):
        self._retriever = retriever
        self._injector = injector
        self._generators: Dict[str, Any] = {
            "gemini": GeminiGenerator(),
            "gpt4o-mini": OpenRouterGenerator(),
            "llama31": GroqGenerator(),
        }

    def run(
        self,
        question: str,
        distractor_count: int = 0,
        distractor_type: str = "topical",
    ) -> Dict:
        docs: List[Document] = self._retriever.retrieve(question)
        gold_passage = docs[0] if docs else Document(page_content="")

        if self._injector and distractor_count > 0:
            context, metadata = self._injector.inject(
                question=question,
                gold_passage=gold_passage,
                distractor_count=distractor_count,
                distractor_type=distractor_type,
            )
        else:
            context = gold_passage.page_content
            metadata = {
                "distractors": [],
                "distractor_type": distractor_type,
                "distractor_count": 0,
                "gold_passage": gold_passage.page_content,
            }

        responses = {}
        latencies = {}

        for name, gen in self._generators.items():
            t0 = time.perf_counter()
            try:
                responses[name] = gen.generate(question=question, context=context)
            except Exception as e:
                responses[name] = f"[{name} error: {e}]"
            latencies[name] = round((time.perf_counter() - t0) * 1000)

        return {
            "question": question,
            "context": context,
            "responses": responses,
            "latencies": latencies,
            "metadata": metadata,
        }