from pathlib import Path
from typing import List, Optional

from datasets import load_from_disk

HOTPOT_PATH = Path("data/raw/hotpot")


class HotpotLoader:
    """Loads HotpotQA (distractor setting) and returns one flat document
    per context paragraph."""

    def load(self, limit: Optional[int] = None) -> List[dict]:
        """Load up to `limit` documents from the HotpotQA train set.

        Each context paragraph becomes a separate document sharing the
        same question and answer.  Every returned dict follows the flat
        document schema: doc_id, dataset, title, text, answer, question
        """
        dataset = load_from_disk(str(HOTPOT_PATH))
        train = dataset["train"]

        docs: List[dict] = []
        for sample in train:
            qid = sample["id"]
            question = sample["question"]
            answer = sample["answer"]

            for idx, title in enumerate(sample["context"]["title"]):
                sentences = sample["context"]["sentences"][idx]
                text = " ".join(sentences)
                docs.append(
                    {
                        "doc_id": f"{qid}_p{idx}",
                        "dataset": "hotpot",
                        "title": title,
                        "text": text,
                        "answer": answer,
                        "question": question,
                    }
                )

                if limit is not None and len(docs) >= limit:
                    return docs
        return docs
