import hashlib
import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from src.cache import ResponseCache
from src.generators.base import BaseGenerator
from src.vectorstore import VectorStore

_CONTRADICTION_CACHE_PATH = Path("cache/distractor_contradictions.json")


def _load_cache() -> dict:
    if _CONTRADICTION_CACHE_PATH.exists():
        with open(_CONTRADICTION_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    _CONTRADICTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONTRADICTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _cache_key(gold_passage: Document, question: str) -> str:
    raw = f"{question}|||{gold_passage.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_topical(
    question: str,
    gold_passage: Document,
    vector_store: VectorStore,
    count: int,
    k: int = 20,
) -> List[Document]:
    topic = gold_passage.metadata.get("title", "")
    if not topic:
        return []

    candidates = vector_store.similarity_search_with_score(
        gold_passage.page_content, k=k
    )
    distractors = []
    for doc, _ in candidates:
        if doc.page_content == gold_passage.page_content:
            continue
        if doc.metadata.get("title") == topic:
            distractors.append(doc)
        if len(distractors) >= count:
            break
    return distractors[:count]


def generate_paraphrased_contradiction(
    gold_passage: Document,
    llm_generator: BaseGenerator,
    count: int,
    question: str = "",
) -> List[Document]:
    cache = _load_cache()
    key = _cache_key(gold_passage, question)

    if key in cache:
        return [
            Document(
                page_content=cache[key],
                metadata={"type": "paraphrased_contradiction", "source": "cache"},
            )
        ]

    prompt = (
        "Generate a version of the passage below that contradicts its "
        "key facts. Keep the same topic and writing style.\n\n"
        f"Passage: {gold_passage.page_content}"
    )
    contradiction = llm_generator.generate(
        question="Generate contradiction", context=prompt
    )
    cache[key] = contradiction
    _save_cache(cache)

    return [
        Document(
            page_content=contradiction,
            metadata={"type": "paraphrased_contradiction", "source": "llm"},
        )
    ]


def generate_hard_negative(
    question: str,
    gold_passage: Document,
    vector_store: VectorStore,
    count: int,
    k: int = 20,
) -> List[Document]:
    gold_answer = gold_passage.metadata.get("answer", "").strip().lower()
    candidates = vector_store.similarity_search_with_score(question, k=k)
    distractors = []
    for doc, _ in candidates:
        if doc.page_content == gold_passage.page_content:
            continue
        doc_answer = doc.metadata.get("answer", "").strip().lower()
        if gold_answer and doc_answer == gold_answer:
            continue
        distractors.append(doc)
        if len(distractors) >= count:
            break
    return distractors[:count]
