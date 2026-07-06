from pathlib import Path
from typing import List, Optional

from datasets import load_from_disk
from langchain_core.documents import Document

HOTPOT_PATH = Path("data/raw/hotpot")


class HotpotLoader:
    """Loads HotpotQA (distractor setting) and returns one Document per
    unique paragraph content.

    If the same paragraph text appears in multiple samples, questions
    from all those samples are aggregated in the ``questions`` metadata
    list.
    """

    def load(self, limit: Optional[int] = None) -> List[Document]:
        dataset = load_from_disk(str(HOTPOT_PATH))
        train = dataset["train"]

        docs: List[Document] = []
        content_to_doc: dict[str, Document] = {}

        for sample in train:
            qid = sample["id"]
            question = sample["question"]
            answer = sample["answer"]

            for idx, title in enumerate(sample["context"]["title"]):
                sentences = sample["context"]["sentences"][idx]
                text = " ".join(sentences)

                existing = content_to_doc.get(text)
                if existing is not None:
                    existing.metadata["questions"].append(
                        {
                            "question": question,
                            "answer": answer,
                            "qid": qid,
                        }
                    )
                else:
                    metadata = {
                        "dataset": "hotpot",
                        "title": title,
                        "doc_id": f"{qid}_p{idx}",
                        "question": question,
                        "answer": answer,
                        "questions": [
                            {
                                "question": question,
                                "answer": answer,
                                "qid": qid,
                            }
                        ],
                    }
                    doc = Document(page_content=text, metadata=metadata)
                    content_to_doc[text] = doc
                    docs.append(doc)

                    if limit is not None and len(docs) >= limit:
                        return docs
        return docs
