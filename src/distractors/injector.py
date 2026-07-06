from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.generators.base import BaseGenerator
from src.vectorstore import VectorStore

from .generators import (
    generate_hard_negative,
    generate_paraphrased_contradiction,
    generate_topical,
)


class DistractorInjector:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_generator: Optional[BaseGenerator] = None,
        random_seed: int = 42,
    ):
        self._vector_store = vector_store
        self._llm_generator = llm_generator
        self._random_seed = random_seed

    def inject(
        self,
        question: str,
        gold_passage: Document,
        distractor_count: int = 3,
        distractor_type: str = "topical",
    ) -> Tuple[str, Dict]:
        if distractor_count <= 0:
            return (
                gold_passage.page_content,
                {
                    "distractors": [],
                    "distractor_type": distractor_type,
                    "gold_passage": gold_passage.page_content,
                },
            )

        if distractor_type == "topical":
            if self._vector_store is None:
                raise ValueError("VectorStore required for topical distractors")
            distractors = generate_topical(
                question=question,
                gold_passage=gold_passage,
                vector_store=self._vector_store,
                count=distractor_count,
            )
        elif distractor_type == "paraphrased_contradiction":
            if self._llm_generator is None:
                raise ValueError(
                    "LLM generator required for paraphrased contradiction"
                )
            distractors = generate_paraphrased_contradiction(
                gold_passage=gold_passage,
                llm_generator=self._llm_generator,
                count=distractor_count,
                question=question,
            )
        elif distractor_type == "hard_negative":
            if self._vector_store is None:
                raise ValueError(
                    "VectorStore required for hard negative distractors"
                )
            distractors = generate_hard_negative(
                question=question,
                gold_passage=gold_passage,
                vector_store=self._vector_store,
                count=distractor_count,
            )
        else:
            raise ValueError(f"Unknown distractor_type: {distractor_type}")

        parts = [gold_passage.page_content] + [d.page_content for d in distractors]
        new_context = "\n\n".join(parts)

        metadata = {
            "distractors": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in distractors
            ],
            "distractor_type": distractor_type,
            "distractor_count": len(distractors),
            "gold_passage": gold_passage.page_content,
        }
        return new_context, metadata
