from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from backend.utils.logger import logger

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Cross-Encoder model.
        Uses a lightweight MS-MARCO model optimized for fast passage re-ranking.
        """
        logger.info(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Reranks a list of retrieved documents based on relevance score to the query.
        """
        if not documents:
            return []

        # Create (query, chunk_text) pairs for the cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Pair documents with their predicted scores
        doc_score_pairs = list(zip(documents, scores))

        # Sort descending by score
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Cross-Encoder reranked {len(documents)} docs -> keeping top {top_k}.")
        return doc_score_pairs[:top_k]

# Global singleton instance for local runtime reuse
reranker_instance = CrossEncoderReranker()