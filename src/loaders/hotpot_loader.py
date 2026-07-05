from pathlib import Path
from typing import List, Optional

from datasets import load_from_disk
from langchain_core.documents import Document

HOTPOT_PATH = Path("data/raw/hotpot")


class HotpotLoader:
    """Loads HotpotQA (distractor setting) and returns one Document per
    context paragraph."""

    def load(self, limit: Optional[int] = None) -> List[Document]:
        dataset = load_from_disk(str(HOTPOT_PATH))
        train = dataset["train"]

        docs: List[Document] = []
        for sample in train:
            qid = sample["id"]
            question = sample["question"]
            answer = sample["answer"]

            for idx, title in enumerate(sample["context"]["title"]):
                sentences = sample["context"]["sentences"][idx]
                text = " ".join(sentences)
                metadata = {
                    "dataset": "hotpot",
                    "question": question,
                    "answer": answer,
                    "title": title,
                    "doc_id": f"{qid}_p{idx}",
                }
                docs.append(Document(page_content=text, metadata=metadata))

                if limit is not None and len(docs) >= limit:
                    return docs
        return docs
