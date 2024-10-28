import faiss
import numpy as np
from typing import List


class FAISSStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.ids = []

    def add_embeddings(self, embeddings: np.ndarray, ids: List[str]):
        self.index.add(embeddings)
        self.ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        distances, indices = self.index.search(query_embedding, top_k)
        results = [(self.ids[idx], distances[i]) for i, idx in enumerate(indices[0])]
        return results
