import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

from src.distractors import DistractorInjector
from src.loaders import SquadLoader
from src.vectorstore import VectorStore 

SEP = "-" * 72


def print_section(label: str, content: str, max_len: int = 600) -> None:
    print(f"\n{SEP}")
    print(f"{label:^72}")
    print(SEP)
    text = content[:max_len]
    if len(content) > max_len:
        text += "..."
    print(text)
    print()


def main() -> None:
    print("Loading SQuAD documents...")
    docs = SquadLoader().load(limit=50)
    gold: Document = docs[0]

    question = gold.metadata["question"]

    print("Loading vector store...")
    vector_store = VectorStore.load_local(Path("data/faiss_gemini-embedding-2"))

    injector = DistractorInjector(
        vector_store=vector_store,
        random_seed=42,
    )

    test_cases = [
        ("topical", 1),
        ("hard_negative", 1),
    ]

    for dtype, count in test_cases:
        print("\n" + "=" * 72)
        print(f"  {dtype.upper()}  (count={count})".center(72))
        print("=" * 72)

        context, metadata = injector.inject(
            question=question,
            gold_passage=gold,
            distractor_count=count,
            distractor_type=dtype,
        )

        print_section("QUESTION", question)
        print_section("GOLD PASSAGE", gold.page_content)
        for i, d in enumerate(metadata["distractors"]):
            print_section(f"DISTRACTOR {i + 1}", d["content"])
        print_section("FINAL CONTEXT", context)

    # ---- paraphrased_contradiction needs an LLM ----
    print("\n" + "=" * 72)
    print("  PARAPHRASED CONTRADICTION  (count=1)".center(72))
    print("=" * 72)

    from src.generators.gemini import GeminiGenerator

    llm = GeminiGenerator()
    injector_llm = DistractorInjector(
        vector_store=vector_store,
        llm_generator=llm,
        random_seed=42,
    )

    context, metadata = injector_llm.inject(
        question=question,
        gold_passage=gold,
        distractor_count=1,
        distractor_type="paraphrased_contradiction",
    )

    print_section("QUESTION", question)
    print_section("GOLD PASSAGE", gold.page_content)
    for i, d in enumerate(metadata["distractors"]):
        print_section(f"DISTRACTOR {i + 1}", d["content"])
    print_section("FINAL CONTEXT", context)

if __name__ == "__main__":
    main()
