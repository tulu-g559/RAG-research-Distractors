import hashlib
import json
from pathlib import Path
from typing import Optional


class ResponseCache:
    def __init__(self, cache_path: str | Path = "cache/api_responses.json"):
        self._path = Path(cache_path)
        self._data: dict[str, str] = {}
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, question: str, context: str, model_name: str) -> Optional[str]:
        return self._data.get(self._key(question, context, model_name))

    def set(self, question: str, context: str, model_name: str, response: str) -> None:
        self._data[self._key(question, context, model_name)] = response
        self._save()

    def _key(self, question: str, context: str, model_name: str) -> str:
        raw = f"{question}|||{context}|||{model_name}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
