from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .base import BaseGenerator

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Answer the question based only on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:",
        )
    ]
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterGenerator(BaseGenerator):
    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        self._llm = ChatOpenAI(
            model=model,
            base_url=OPENROUTER_BASE,
            max_tokens=512,
        )

    def generate(self, question: str, context: str) -> str:
        messages = PROMPT.format_messages(question=question, context=context)
        response = self._llm.invoke(messages)
        return response.content.strip()
