from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from backend.utils.logger import logger

class BM25RetrieverManager:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        # Tokenize content for BM25 indexing
        self.corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank documents by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:  # Only return relevant non-zero matches
                results.append((self.documents[idx], float(scores[idx])))
        return results