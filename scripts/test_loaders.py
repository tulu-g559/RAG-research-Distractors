import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.loaders import SquadLoader, HotpotLoader

METADATA_KEYS = {"dataset", "question", "answer", "title", "doc_id", "questions"}


def main() -> None:
    print("=" * 70)
    print("SQuAD Loader — limit=5")
    print("=" * 70)

    squad = SquadLoader()
    squad_docs = squad.load(limit=5)
    print(f"Returned {len(squad_docs)} Document(s)\n")

    d: Document = squad_docs[0]
    print(f"page_content (first 200 chars): {d.page_content[:200]}...")
    print(f"metadata: {json.dumps(d.metadata, indent=2, ensure_ascii=False)}")
    print(f"Number of QAs for this paragraph: {len(d.metadata['questions'])}")

    print("\n" + "=" * 70)
    print("HotpotQA Loader — limit=5")
    print("=" * 70)

    hotpot = HotpotLoader()
    hotpot_docs = hotpot.load(limit=5)
    print(f"Returned {len(hotpot_docs)} Document(s)\n")

    d = hotpot_docs[0]
    print(f"page_content (first 200 chars): {d.page_content[:200]}...")
    print(f"metadata: {json.dumps(d.metadata, indent=2, ensure_ascii=False)}")
    print(f"Number of QAs for this paragraph: {len(d.metadata['questions'])}")

    print("\n" + "=" * 70)
    print("Schema Verification")
    print("=" * 70)

    for label, docs in [("squad", squad_docs), ("hotpot", hotpot_docs)]:
        for doc in docs:
            assert isinstance(doc, Document), f"{label}: not a Document"
            assert set(doc.metadata.keys()) == METADATA_KEYS, (
                f"{label}: got keys {set(doc.metadata.keys())}, "
                f"expected {METADATA_KEYS}"
            )
            assert isinstance(doc.metadata["doc_id"], str), f"{label}: doc_id not str"
            assert doc.metadata["dataset"] in ("squad", "hotpot"), f"{label}: bad dataset"
            assert isinstance(doc.metadata["title"], str), f"{label}: title not str"
            assert isinstance(doc.metadata["answer"], str), f"{label}: answer not str"
            assert isinstance(doc.metadata["question"], str), f"{label}: question not str"
            assert isinstance(doc.metadata["questions"], list), f"{label}: questions not list"
            assert len(doc.metadata["questions"]) > 0, f"{label}: empty questions"
            assert isinstance(doc.page_content, str), f"{label}: page_content not str"
    print("All Document-schema checks passed \u2713")


if __name__ == "__main__":
    main()
