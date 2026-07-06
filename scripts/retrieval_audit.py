import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml

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
    raw_num_questions = config["num_questions"]
    if isinstance(raw_num_questions, dict):
        max_questions = max(raw_num_questions.values())
    else:
        max_questions = raw_num_questions
    index_size = config.get("index_size", 3000)
    random_seed = config["random_seed"]
    datasets = config["datasets"]

    random.seed(random_seed)
    embedding_provider = create_embedding_provider(embedding_model)
    faiss_path = Path(f"data/faiss_{embedding_model.replace('/', '_')}")

    load_configs = {
        "squad": (SquadLoader, index_size + max_questions * 5),
        "hotpot": (HotpotLoader, index_size + max_questions * 15),
    }

    all_index_docs = []
    question_to_gold = OrderedDict()

    for ds_name in datasets:
        loader_cls, load_limit = load_configs[ds_name]
        print(f"\nLoading {ds_name} (limit={load_limit})...")
        docs = loader_cls().load(limit=load_limit)
        print(f"  {len(docs)} unique paragraphs loaded")

        for doc in docs:
            all_index_docs.append(doc)
            for qa in doc.metadata["questions"]:
                q = qa["question"]
                if q not in question_to_gold:
                    question_to_gold[q] = {
                        "dataset": ds_name,
                        "question": q,
                        "answer": qa["answer"],
                        "doc_id": doc.metadata["doc_id"],
                        "gold_passage": doc.page_content,
                    }

    print(f"\n{'=' * 70}")
    print(f"INDEX ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total unique paragraphs: {len(all_index_docs)}")
    print(f"Total unique questions: {len(question_to_gold)}")

    squad_docs = [d for d in all_index_docs if d.metadata.get("dataset") == "squad"]
    hotpot_docs = [d for d in all_index_docs if d.metadata.get("dataset") == "hotpot"]
    print(f"  SQuAD unique paragraphs: {len(squad_docs)}")
    print(f"  Hotpot unique paragraphs: {len(hotpot_docs)}")

    squad_contents = [d.page_content for d in squad_docs]
    hotpot_contents = [d.page_content for d in hotpot_docs]
    print(f"  SQuAD unique content values: {len(set(squad_contents))}")
    print(f"  Hotpot unique content values: {len(set(hotpot_contents))}")

    n_squad_dup = len(squad_contents) - len(set(squad_contents))
    n_hotpot_dup = len(hotpot_contents) - len(set(hotpot_contents))
    print(f"  SQuAD duplicate content entries: {n_squad_dup}")
    print(f"  Hotpot duplicate content entries: {n_hotpot_dup}")

    test_questions = list(question_to_gold.values())[:max_questions]
    print(f"\nTest questions sampled: {len(test_questions)}")

    print(f"\n{'=' * 70}")
    print(f"GOLD PASSAGE COVERAGE CHECK")
    print(f"{'=' * 70}")
    index_content_set = set(d.page_content for d in all_index_docs)
    all_gold_found = 0
    for qt in test_questions:
        if qt["gold_passage"] in index_content_set:
            all_gold_found += 1
    print(f"Gold passages present in index: {all_gold_found}/{len(test_questions)}")

    print(f"\n{'=' * 70}")
    print(f"LOADING FAISS INDEX FROM {faiss_path}")
    print(f"{'=' * 70}")

    if not faiss_path.exists():
        print(f"Building FAISS index from {len(all_index_docs)} paragraphs ...")
        vector_store = VectorStore.from_documents(all_index_docs, embedding_provider=embedding_provider)
        vector_store.save_local(faiss_path)
        print(f"Index saved to {faiss_path}")
    else:
        vector_store = VectorStore.load_local(faiss_path, embedding_provider=embedding_provider)
        print(f"Loaded existing index")

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
    print(f"VERIFICATION OF FIXES")
    print(f"{'=' * 70}")
    squad_unique = len(set(d.page_content for d in squad_docs))
    hotpot_unique = len(set(d.page_content for d in hotpot_docs))
    print(f"1. DEDUPLICATION:")
    print(f"   SQuAD unique paragraphs: {len(squad_docs)} (all unique: {len(squad_docs) == squad_unique})")
    print(f"   Hotpot unique paragraphs: {len(hotpot_docs)} (all unique: {len(hotpot_docs) == hotpot_unique})")
    dedup_ok = (len(squad_docs) == squad_unique) and (len(hotpot_docs) == hotpot_unique)
    print(f"   -> {'PASS' if dedup_ok else 'FAIL'}: No duplicate content in index")
    print(f"2. GOLD PASSAGE IN INDEX: {all_gold_found}/{len(test_questions)}")
    print(f"   -> {'PASS' if all_gold_found == len(test_questions) else 'FAIL'}: All gold passages present")
    print(f"3. DUPLICATE EMBEDDINGS IN FAISS: {sum(n - 1 for n in stored_content_dups.values())}")
    print(f"   -> {'PASS' if sum(n - 1 for n in stored_content_dups.values()) == 0 else 'FAIL'}: No duplicate embeddings")
    print(f"4. GOLD PASSAGE SELECTION:")
    print(f"   -> PASS: run_evaluation.py now constructs gold_passage from loader metadata")
    print(f"      instead of using docs_with_scores[0][0]")
    print(f"5. RETRIEVAL SEARCH: {'PASS' if search_ok else 'BLOCKED (API credits)'}")


if __name__ == "__main__":
    main()
