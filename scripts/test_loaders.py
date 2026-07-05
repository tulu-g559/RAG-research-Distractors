import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loaders import SquadLoader, HotpotLoader
from src.embeddings import EmbeddingModel


FLAT_SCHEMA_KEYS = {"doc_id", "dataset", "title", "text", "answer", "question"}


def main() -> None:
    print("=" * 70)
    print("SQuAD Loader — limit=5")
    print("=" * 70)

    squad = SquadLoader()
    squad_docs = squad.load(limit=5)
    print(f"Returned {len(squad_docs)} document(s)\n")
    print(json.dumps(squad_docs[0], indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("HotpotQA Loader — limit=5")
    print("=" * 70)

    hotpot = HotpotLoader()
    hotpot_docs = hotpot.load(limit=5)
    print(f"Returned {len(hotpot_docs)} document(s)\n")
    print(json.dumps(hotpot_docs[0], indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Schema Verification")
    print("=" * 70)

    for label, docs in [("squad", squad_docs), ("hotpot", hotpot_docs)]:
        for d in docs:
            assert set(d.keys()) == FLAT_SCHEMA_KEYS, (
                f"{label}: got keys {set(d.keys())}, "
                f"expected {FLAT_SCHEMA_KEYS}"
            )
            assert isinstance(d["doc_id"], str), f"{label}: doc_id not str"
            assert d["dataset"] in ("squad", "hotpot"), f"{label}: bad dataset"
            assert isinstance(d["title"], str), f"{label}: title not str"
            assert isinstance(d["text"], str), f"{label}: text not str"
            assert isinstance(d["answer"], str), f"{label}: answer not str"
            assert isinstance(d["question"], str), f"{label}: question not str"
    print("All flat-schema checks passed ✓")

    print("\n" + "=" * 70)
    print("EmbeddingModel interface check")
    print("=" * 70)
    assert EmbeddingModel.embed.__isabstractmethod__
    print("EmbeddingModel.embed is abstract ✓")


if __name__ == "__main__":
    main()
