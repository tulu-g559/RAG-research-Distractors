from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

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

class GroqGenerator(BaseGenerator):
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self._llm = ChatGroq(model=model, max_tokens=512)
        # Chain prompt, llm, and parser
        self._chain = PROMPT | self._llm | StrOutputParser()

    def generate(self, question: str, context: str) -> str:
        # Invoke chain directly with dict
        response = self._chain.invoke({"question": question, "context": context})
        return response.strip()