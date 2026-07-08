import random
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
from langchain_core.documents import Document

from src.embeddings import create_embedding_provider
from src.loaders import HotpotLoader, SquadLoader
from src.vectorstore import VectorStore

CONFIG_PATH = Path("experiments/config.yaml")

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
embedding_model = config["embedding_model"]
random_seed = config["random_seed"]
dataset_subset = config["dataset_subset"]

rng = random.Random(random_seed)
embedding_provider = create_embedding_provider(embedding_model)
faiss_path = Path(f"data/faiss_{embedding_model.replace('/', '_')}")

loader_map = {
    "squad": SquadLoader,
    "hotpot": HotpotLoader,
}

all_index_docs: List[Document] = []

print("=" * 70)
print("DATASET SUBSET CONSTRUCTION")
print("=" * 70)

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

    print(f"  Questions selected: {target_questions}")
    print(f"  Unique gold paragraphs: {len(gold_docs)}")
    print(f"  Random filler paragraphs: {filler_needed}")
    print(f"  Final indexed paragraphs: {len(ds_index_docs)}")

print(f"\n  Total indexed documents: {len(all_index_docs)}")

print(f"\nBuilding FAISS index from {len(all_index_docs)} paragraphs ...")
vector_store = VectorStore.from_documents(all_index_docs, embedding_provider=embedding_provider)
vector_store.save_local(faiss_path)
print(f"Index saved to {faiss_path} with {vector_store._index.index.ntotal} vectors")
