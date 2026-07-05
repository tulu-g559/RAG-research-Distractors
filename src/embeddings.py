from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Interface for embedding models.

    Subclasses implement ``embed`` to convert a list of text strings
    into their vector representations.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts and return a list of vectors."""
        ...
