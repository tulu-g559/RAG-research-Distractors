import json
from pathlib import Path
from typing import List, Optional

SQUAD_PATH = Path("data/raw/squad_v2/train-v2.0.json")


class SquadLoader:
    """Loads SQuAD v2.0 and returns one flat document per QA pair."""

    def load(self, limit: Optional[int] = None) -> List[dict]:
        """Load up to `limit` documents from the SQuAD v2 train set.

        Each returned dict follows the flat document schema:
            doc_id, dataset, title, text, answer, question
        """
        with open(SQUAD_PATH, "r", encoding="utf-8") as f:
            squad = json.load(f)

        docs: List[dict] = []
        for article in squad["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    docs.append(
                        {
                            "doc_id": qa["id"],
                            "dataset": "squad",
                            "title": title,
                            "text": context,
                            "answer": (
                                qa["answers"][0]["text"]
                                if qa["answers"]
                                else ""
                            ),
                            "question": qa["question"],
                        }
                    )
                    if limit is not None and len(docs) >= limit:
                        return docs
        return docs
