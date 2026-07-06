import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

SQUAD_PATH = Path("data/raw/squad_v2/train-v2.0.json")


class SquadLoader:
    """Loads SQuAD v2.0 and returns one Document per unique context paragraph.

    Each Document's metadata includes a ``questions`` list containing all
    Q/A pairs that share that paragraph.
    """

    def load(self, limit: Optional[int] = None) -> List[Document]:
        with open(SQUAD_PATH, "r", encoding="utf-8") as f:
            squad = json.load(f)

        docs: List[Document] = []
        seen = set()

        for article in squad["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if context in seen:
                    continue
                seen.add(context)

                questions = []
                for qa in paragraph["qas"]:
                    answer = qa["answers"][0]["text"] if qa["answers"] else ""
                    questions.append(
                        {
                            "question": qa["question"],
                            "answer": answer,
                            "qa_id": qa["id"],
                        }
                    )

                first = questions[0]
                metadata = {
                    "dataset": "squad",
                    "title": title,
                    "doc_id": first["qa_id"],
                    "question": first["question"],
                    "answer": first["answer"],
                    "questions": questions,
                }
                docs.append(Document(page_content=context, metadata=metadata))

                if limit is not None and len(docs) >= limit:
                    return docs
        return docs
