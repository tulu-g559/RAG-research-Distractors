import csv
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import yaml
from langchain_core.documents import Document

from src.cache import ResponseCache
from src.distractors import DistractorInjector
from src.embeddings import create_embedding_provider
from src.evaluation import exact_match, f1_score, normalize
from src.generators import GeminiGenerator
from src.loaders import HotpotLoader, SquadLoader
from src.retriever import Retriever
from src.vectorstore import VectorStore

CONFIG_PATH = Path("experiments/config.yaml")
CACHE_PATH = Path("cache/api_responses.json")
RESULTS_DIR = Path("results")


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def answer_in_context(answer: str, docs) -> bool:
    if not answer:
        return True
    norm_answer = normalize(answer)
    return any(norm_answer in normalize(d.page_content) for d in docs)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    config = load_config(CONFIG_PATH)
    embedding_model = config["embedding_model"]
    generator_models = config["generator_models"]
    top_k = config["top_k"]
    raw_num_questions = config["num_questions"]
    if isinstance(raw_num_questions, dict):
        max_questions = max(raw_num_questions.values())
        per_model_nq = raw_num_questions
    else:
        max_questions = raw_num_questions
        per_model_nq = {}
    index_size = config.get("index_size", 3000)
    random_seed = config["random_seed"]
    datasets = config["datasets"]
    distractor_counts = config.get("distractor_count", [0])
    distractor_types = config.get("distractor_type", ["topical"])

    experiment_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()

    faiss_path = Path(f"data/faiss_{embedding_model.replace('/', '_')}")
    cache = ResponseCache(CACHE_PATH)

    print("=" * 70)
    print("EXPERIMENT")
    print(f"  experiment_id : {experiment_id}")
    print(f"  timestamp     : {timestamp}")
    print(f"  git_commit    : {git_commit}")
    print(f"  config        : {CONFIG_PATH}")
    print(f"  distractor_counts : {distractor_counts}")
    print(f"  distractor_types  : {distractor_types}")
    print()
    print("CONFIG")
    print(f"  embedding_model : {embedding_model}")
    print(f"  generator_models: {list(generator_models.keys())}")
    print(f"  top_k           : {top_k}")
    print(f"  max_questions   : {max_questions}")
    if per_model_nq:
        print(f"  per_model_limit : {per_model_nq}")
    print(f"  index_size      : {index_size}")
    print(f"  random_seed     : {random_seed}")
    print(f"  datasets        : {datasets}")
    print(f"  (index built from ALL unique paragraphs — no train/test doc split)")

    random.seed(random_seed)

    embedding_provider = create_embedding_provider(embedding_model)

    load_configs: Dict[str, Tuple] = {
        "squad": (SquadLoader, index_size + max_questions * 5),
        "hotpot": (HotpotLoader, index_size + max_questions * 15),
    }

    all_index_docs: List[Document] = []
    question_to_gold: OrderedDict = OrderedDict()

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
                        "gold_title": doc.metadata.get("title", ""),
                    }

    print(f"\nTotal unique paragraphs in index: {len(all_index_docs)}")
    print(f"Total unique questions available: {len(question_to_gold)}")

    test_questions = list(question_to_gold.values())[:max_questions]
    print(f"Test questions sampled: {len(test_questions)}")

    print(f"\nBuilding FAISS index from {len(all_index_docs)} paragraphs ...")
    vector_store = VectorStore.from_documents(
        all_index_docs, embedding_provider=embedding_provider
    )
    vector_store.save_local(faiss_path)
    print(f"Index saved to {faiss_path}")

    retriever = Retriever(vector_store, k=top_k)

    injector = DistractorInjector(
        vector_store=vector_store,
        llm_generator=GeminiGenerator(),
        random_seed=random_seed,
    )

    recall_k_values = [5, 10]
    retrieve_k = max(top_k, max(recall_k_values))

    from src.generators import GroqGenerator, OpenRouterGenerator

    provider_map = {
        "gemini": GeminiGenerator,
        "openrouter": OpenRouterGenerator,
        "groq": GroqGenerator,
    }

    generators = {}
    for label, cfg in generator_models.items():
        provider_cls = provider_map[cfg["provider"]]
        generators[label] = provider_cls(model=cfg["model"])

    model_keys = list(generators.keys())
    all_results = []
    recall_results = []

    for question_index, qt in enumerate(test_questions, 1):
        dataset = qt["dataset"]
        question = qt["question"]
        ground_truth = qt["answer"]
        gold_doc_id = qt["doc_id"]
        gold_content = qt["gold_passage"]

        gold_passage = Document(
            page_content=gold_content,
            metadata={
                "dataset": dataset,
                "question": question,
                "answer": ground_truth,
                "doc_id": gold_doc_id,
                "title": qt.get("gold_title", ""),
            },
        )

        print(f"\n{'=' * 70}")
        print(f"[{question_index}/{len(test_questions)}] {dataset.upper()}")
        print(f"Q: {question}")
        print(f"GT: {ground_truth}")

        t_ret = time.time()
        docs_with_scores = retriever.retrieve_with_scores(
            question, k=retrieve_k
        )
        retrieve_latency = round((time.time() - t_ret) * 1000)

        retrieved_docs = [doc for doc, _ in docs_with_scores]
        retrieved_scores = [score for _, score in docs_with_scores]
        retrieved_doc_ids = [
            doc.metadata.get("doc_id", "?") for doc in retrieved_docs
        ]

        per_q_recall = {
            "dataset": dataset,
            "question": question,
            "ground_truth": ground_truth,
        }
        for k_val in recall_k_values:
            top_k_for_recall = docs_with_scores[:k_val]
            recalled = answer_in_context(
                ground_truth, [d for d, _ in top_k_for_recall]
            )
            per_q_recall[f"recall@{k_val}"] = 1 if recalled else 0

        recall_results.append(per_q_recall)
        recall_at_5 = per_q_recall["recall@5"]
        recall_at_10 = per_q_recall["recall@10"]

        gold_in_top_k = answer_in_context(ground_truth, retrieved_docs)

        print(f"  Retrieval:   {retrieve_latency}ms")
        print(f"  Recall@5:    {recall_at_5}  Recall@10: {recall_at_10}")
        print(f"  Gold in top-{retrieve_k}: {gold_in_top_k}")
        print(f"  Doc IDs:     {retrieved_doc_ids[:top_k]}")
        print(f"  Scores:      {[f'{s:.4f}' for s in retrieved_scores[:top_k]]}")

        top_k_scores = retrieved_scores[:top_k]
        top_k_ids = retrieved_doc_ids[:top_k]

        for dcount in distractor_counts:
            types_to_run = distractor_types if dcount > 0 else ["none"]
            for dtype in types_to_run:
                if dcount > 0:
                    context, inject_meta = injector.inject(
                        question=question,
                        gold_passage=gold_passage,
                        distractor_count=dcount,
                        distractor_type=dtype,
                    )
                    actual_type = dtype
                    actual_count = dcount
                else:
                    context = gold_passage.page_content
                    actual_type = "none"
                    actual_count = 0

                print(f"\n  --- distractors: {actual_type} x{actual_count} ---")

                for key in model_keys:
                    model_limit = per_model_nq.get(key, max_questions)
                    if question_index > model_limit:
                        continue
                    t0 = time.time()
                    cached = cache.get(question, context, key)
                    if cached is not None:
                        pred = cached
                        cached_hit = True
                    else:
                        try:
                            pred = generators[key].generate(
                                question=question, context=context
                            )
                        except Exception as e:
                            pred = f"[{key} error: {e}]"
                        cache.set(question, context, key, pred)
                        cached_hit = False
                    latency_ms = round((time.time() - t0) * 1000)

                    em = exact_match(pred, ground_truth)
                    f1 = f1_score(pred, ground_truth)

                    tag = " [CACHED]" if cached_hit else ""
                    print(f"  [{key}]{tag} EM={1 if em else 0}  F1={f1:.4f}  {latency_ms}ms")

                    all_results.append(
                        {
                            "experiment_id": experiment_id,
                            "timestamp": timestamp,
                            "git_commit": git_commit,
                            "dataset": dataset,
                            "question": question,
                            "ground_truth": ground_truth,
                            "question_index": str(question_index),
                            "retrieved_doc_ids": ";".join(top_k_ids),
                            "retrieved_scores": ";".join(
                                f"{s:.4f}" for s in top_k_scores
                            ),
                            "retrieved_context": context,
                            "prediction": pred,
                            "model": key,
                            "distractor_type": actual_type,
                            "distractor_count": str(actual_count),
                            "em": "1" if em else "0",
                            "f1": f"{f1:.4f}",
                            "faithfulness": "",
                            "latency_ms": f"{latency_ms}",
                            "cached": "1" if cached_hit else "0",
                            "recall@5": str(recall_at_5),
                            "recall@10": str(recall_at_10),
                        }
                    )

    print(f"\n{'=' * 70}")
    print("RETRIEVAL SUMMARY")
    print(f"{'=' * 70}")
    if recall_results:
        for k_val in recall_k_values:
            hits = sum(r[f"recall@{k_val}"] for r in recall_results)
            total = len(recall_results)
            print(f"  Recall@{k_val:<3}: {hits}/{total} = {hits / total:.2%}")

    print(f"\n{'=' * 70}")
    print("GENERATION SUMMARY PER MODEL x DISTRACTOR")
    print(f"{'=' * 70}")
    header = f"  {'model':15s}  {'dcount':6s}  {'dtype':25s}  {'EM':6s}  {'F1':6s}  {'Lat':6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in model_keys:
        for dc in distractor_counts:
            types_to_print = distractor_types if dc > 0 else ["none"]
            for dt in types_to_print:
                subset = [
                    r
                    for r in all_results
                    if r["model"] == key
                    and int(r["distractor_count"]) == dc
                    and r["distractor_type"] == dt
                ]
                if not subset:
                    continue
                ems = [int(r["em"]) for r in subset]
                f1s = [float(r["f1"]) for r in subset]
                lats = [float(r["latency_ms"]) for r in subset]
                print(
                    f"  {key:15s}  {dc:<6d}  {dt:25s}  "
                    f"{sum(ems) / len(ems):<6.2f}  {sum(f1s) / len(f1s):<6.3f}  "
                    f"{sum(lats) / len(lats):<6.0f}ms"
                )

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "baseline.csv"
    fieldnames = [
        "experiment_id",
        "timestamp",
        "git_commit",
        "dataset",
        "question",
        "question_index",
        "ground_truth",
        "retrieved_doc_ids",
        "retrieved_scores",
        "retrieved_context",
        "prediction",
        "model",
        "distractor_type",
        "distractor_count",
        "em",
        "f1",
        "faithfulness",
        "latency_ms",
        "cached",
        "recall@5",
        "recall@10",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    print(f"\n{'=' * 70}")
    print("EM ≈ 0 ANALYSIS")
    print(f"{'=' * 70}")
    total = len(all_results)
    em_zero = [r for r in all_results if r["em"] == "0"]
    em_zero_f1_pos = [r for r in em_zero if float(r["f1"]) > 0]
    recall_zero_at_5 = [r for r in em_zero if r.get("recall@5") == "0"]

    pct_verbose = len(em_zero_f1_pos) / len(em_zero) * 100 if em_zero else 0
    pct_ret_fail = (
        len(recall_zero_at_5) / len(em_zero) * 100 if em_zero else 0
    )

    print(f"  EM=0 samples            : {len(em_zero)}/{total}")
    print(
        f"  EM=0 but F1>0 (verbose) : {len(em_zero_f1_pos)}/{len(em_zero)}"
        f" ({pct_verbose:.0f}%)"
    )
    print(
        f"  EM=0 & Recall@5=0       : {len(recall_zero_at_5)}/{len(em_zero)}"
        f" ({pct_ret_fail:.0f}%)"
    )
    print()


if __name__ == "__main__":
    main()
