import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

SQUAD_PATH = Path("data/raw/squad_v2/train-v2.0.json")


class SquadLoader:
    """Loads SQuAD v2.0 and returns one Document per QA pair."""

    def load(self, limit: Optional[int] = None) -> List[Document]:
        with open(SQUAD_PATH, "r", encoding="utf-8") as f:
            squad = json.load(f)

        docs: List[Document] = []
        for article in squad["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    answer = qa["answers"][0]["text"] if qa["answers"] else ""
                    metadata = {
                        "dataset": "squad",
                        "question": qa["question"],
                        "answer": answer,
                        "title": title,
                        "doc_id": qa["id"],
                    }
                    docs.append(Document(page_content=context, metadata=metadata))
                    if limit is not None and len(docs) >= limit:
                        return docs
        return docs
