import time

from google import genai
from google.genai import errors as genai_errors

from .base import BaseGenerator


class GeminiGenerator(BaseGenerator):
    def __init__(self, model: str = "gemini-2.0-flash", max_retries: int = 3):
        self._client = genai.Client()
        self._model = model
        self._max_retries = max_retries

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "Answer the question based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=prompt
                )
                return response.text.strip() if response.text else ""
            except genai_errors.ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < self._max_retries - 1:
                    wait = 2 ** (attempt + 2)
                    print(f"  [gemini] rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return f"[Gemini error: {e}]"
            except Exception as e:
                return f"[Gemini error: {e}]"
        return "[Gemini error: max retries exceeded]"
