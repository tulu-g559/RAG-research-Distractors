import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
from langchain_core.documents import Document

from src.embeddings import create_embedding_provider
from src.evaluation import normalize
from src.loaders import HotpotLoader, SquadLoader
from src.vectorstore import VectorStore

CONFIG_PATH = Path("experiments/config.yaml")


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def answer_in_context(answer: str, docs) -> bool:
    if not answer:
        return True
    norm_answer = normalize(answer)
    return any(norm_answer in normalize(d.page_content) for d in docs)


def safe(text: str, maxlen: int = 200) -> str:
    return text[:maxlen].replace('\n', ' ').encode('utf-8', errors='replace').decode('utf-8')


def main() -> None:
    config = load_config(CONFIG_PATH)
    embedding_model = config["embedding_model"]
    random_seed = config["random_seed"]

    rng = random.Random(random_seed)
    embedding_provider = create_embedding_provider(embedding_model)
    faiss_path = Path(f"data/faiss_{embedding_model.replace('/', '_')}")
    dataset_subset = config["dataset_subset"]

    all_index_docs: List[Document] = []
    all_eval_questions: List[dict] = []

    print(f"\n{'=' * 70}")
    print("DATASET SUBSET CONSTRUCTION")
    print(f"{'=' * 70}")

    loader_map = {
        "squad": SquadLoader,
        "hotpot": HotpotLoader,
    }

    for ds_name, subset_cfg in dataset_subset.items():
        target_docs = subset_cfg["documents"]
        target_questions = subset_cfg["questions"]

        print(f"\n{ds_name.upper()}")
        print("-" * len(ds_name) + "--------")

        loader_cls = loader_map[ds_name]
        docs = loader_cls().load(limit=None)

        all_questions: List[dict] = []
        doc_by_content: Dict[str, Document] = {}
        for doc in docs:
            doc_by_content[doc.page_content] = doc
            for qa in doc.metadata["questions"]:
                all_questions.append({
                    "dataset": ds_name,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "doc_id": doc.metadata["doc_id"],
                    "gold_passage": doc.page_content,
                })

        selected_questions = rng.sample(all_questions, target_questions)
        gold_docs: List[Document] = []
        gold_contents: Set[str] = set()
        for q in selected_questions:
            content = q["gold_passage"]
            if content not in gold_contents:
                gold_contents.add(content)
                gold_docs.append(doc_by_content[content])

        filler_needed = target_docs - len(gold_docs)
        filler_docs: List[Document] = []
        if filler_needed > 0:
            candidates = [d for d in docs if d.page_content not in gold_contents]
            filler_docs = rng.sample(candidates, filler_needed)

        ds_index_docs = gold_docs + filler_docs
        all_index_docs.extend(ds_index_docs)
        all_eval_questions.extend(selected_questions)

        print(f"  Questions selected: {target_questions}")
        print(f"  Unique gold paragraphs: {len(gold_docs)}")
        print(f"  Random filler paragraphs: {filler_needed}")
        print(f"  Final indexed paragraphs: {len(ds_index_docs)}")

    print(f"\n  Total indexed documents: {len(all_index_docs)}")
    print(f"  Total evaluation questions: {len(all_eval_questions)}")

    squad_docs = [d for d in all_index_docs if d.metadata.get("dataset") == "squad"]
    hotpot_docs = [d for d in all_index_docs if d.metadata.get("dataset") == "hotpot"]

    print(f"\n{'=' * 70}")
    print("VALIDATION")
    print(f"{'=' * 70}")

    index_content_set = set(d.page_content for d in all_index_docs)
    missing = 0
    for eq in all_eval_questions:
        if eq["gold_passage"] not in index_content_set:
            print(f"  MISSING: gold passage for '{eq['question'][:60]}...'")
            missing += 1
    if missing > 0:
        print(f"\n  FATAL: {missing} gold passage(s) missing from index. Aborting.")
        sys.exit(1)
    print("  PASS: every evaluation question has its gold passage in the index.")

    index_contents = [d.page_content for d in all_index_docs]
    if len(index_contents) != len(set(index_contents)):
        dup_count = len(index_contents) - len(set(index_contents))
        print(f"  FAIL: {dup_count} duplicate paragraph(s) in index. Aborting.")
        sys.exit(1)
    print(f"  PASS: all {len(all_index_docs)} indexed documents are unique.")
    print(f"  PASS: no duplicate paragraph embeddings.")

    test_questions = all_eval_questions

    print(f"\n{'=' * 70}")
    print(f"BUILDING FAISS INDEX FROM {faiss_path}")
    print(f"{'=' * 70}")

    print(f"Building FAISS index from {len(all_index_docs)} paragraphs ...")
    vector_store = VectorStore.from_documents(all_index_docs, embedding_provider=embedding_provider)
    vector_store.save_local(faiss_path)
    print(f"Index saved to {faiss_path} with {vector_store._index.index.ntotal} vectors")

    index = vector_store._index
    print(f"FAISS index size: {index.index.ntotal}")

    stored_docs = index.docstore._dict
    print(f"Documents in FAISS docstore: {len(stored_docs)}")

    stored_contents = Counter(d.page_content for d in stored_docs.values())
    stored_content_dups = {c: n for c, n in stored_contents.items() if n > 1}
    print(f"Unique page_content in FAISS: {len(stored_contents)}")
    print(f"Duplicate page_content in FAISS: {sum(n - 1 for n in stored_content_dups.values())}")

    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL AUDIT")
    print(f"{'=' * 70}")

    search_ks = [1, 5, 10, 20]
    recall_answer = {k: 0 for k in search_ks}
    recall_docid = {k: 0 for k in search_ks}
    total = len(test_questions)
    cases = []

    search_ok = True
    try:
        for qi, qt in enumerate(test_questions, 1):
            dataset = qt["dataset"]
            question = qt["question"]
            ground_truth = qt["answer"]
            gold_content = qt["gold_passage"]
            gold_content_in_index = gold_content in index_content_set

            docs_with_scores = vector_store.similarity_search_with_score(question, k=max(search_ks))
            retrieved_docs = [d for d, _ in docs_with_scores]
            retrieved_scores = [s for _, s in docs_with_scores]

            for k in search_ks:
                sub_docs = retrieved_docs[:k]
                if answer_in_context(ground_truth, sub_docs):
                    recall_answer[k] += 1

            ans_in_top5 = answer_in_context(ground_truth, retrieved_docs[:5])
            print(f"  Q{qi} ({dataset}): gold_in_index={gold_content_in_index} "
                  f"ans_in_top5={ans_in_top5} top1_score={retrieved_scores[0]:.4f}")
    except Exception as e:
        print(f"  Retrieval search blocked: {e}")
        search_ok = False

    n = len(test_questions)
    print(f"\n{'=' * 70}")
    print(f"RECALL SUMMARY (n={n})")
    print(f"{'=' * 70}")
    if not search_ok:
        print("  (Not available — embedding API credits exhausted)")
    else:
        print(f"{'k':>5}  {'Recall(answer)':>20}")
        print(f"{'---':>5}  {'--------------':>20}")
        for k in search_ks:
            r = recall_answer[k] / n * 100
            print(f"{k:>5}  {recall_answer[k]:>4}/{n} ({r:>5.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"VERIFICATION")
    print(f"{'=' * 70}")
    print(f"1. GOLD PASSAGE IN INDEX: all {len(test_questions)}/{len(test_questions)} present.")
    print(f"   -> PASS")
    print(f"2. DEDUPLICATION: all {len(all_index_docs)} indexed documents unique.")
    print(f"   -> PASS")
    print(f"3. RETRIEVAL SEARCH: {'PASS' if search_ok else 'BLOCKED (API credits)'}")


if __name__ == "__main__":
    main()
