import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from backend.config import settings
from backend.rag.vector_store import VectorStoreManager
from backend.rag.bm25_retriever import BM25RetrieverManager
from backend.rag.reranker import reranker_instance
from backend.kg.graph_store import GraphStoreManager
from backend.utils.logger import logger

class StrategyEngine:
    def __init__(self):
        self.vector_mgr = VectorStoreManager()
        self.graph_mgr = GraphStoreManager()

    def reciprocal_rank_fusion(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document], 
        k: int = 60
    ) -> List[Document]:
        """Combines FAISS and BM25 search results using Reciprocal Rank Fusion (RRF)."""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Score Vector DB results
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:30])
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:30])
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Sort documents by accumulated RRF score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_map[did] for did in sorted_ids]

    def retrieve_context(self, session_id: str, query: str, strategy: str = None) -> Dict[str, Any]:
        """
        Executes retrieval according to selected RAG strategy:
        - baseline       : FAISS vector search only + Knowledge Graph
        - hybrid_rerank : FAISS + BM25 via RRF + Cross-Encoder Re-rank + Knowledge Graph
        - fusion        : Multi-query generation + Hybrid RRF + Cross-Encoder + Knowledge Graph
        - crag          : Corrective RAG + Cross-Encoder + Knowledge Graph
        """
        selected_strategy = strategy or settings.RAG_STRATEGY
        logger.info(f"Executing RAG Strategy: '{selected_strategy}' for query: '{query}'")

        # 1. Base Vector Retrieval
        faiss_results = self.vector_mgr.search_similarity(session_id, query, k=6)
        vector_docs = [doc for doc, score in faiss_results]

        # 2. Knowledge Graph Retrieval
        kg_relations = self.graph_mgr.search_relationships(session_id, query)

        final_chunks: List[Document] = []

        if selected_strategy == "baseline":
            final_chunks = vector_docs[:4]

        elif selected_strategy in ["hybrid_rerank", "fusion", "crag"]:
            # Load stored documents for BM25 search
            faiss_store = self.vector_mgr.load_index(session_id)
            all_docs = list(faiss_store.docstore._dict.values())
            
            bm25_mgr = BM25RetrieverManager(all_docs)
            bm25_results = bm25_mgr.search(query, top_k=6)
            bm25_docs = [doc for doc, score in bm25_results]

            # Fuse Vector + BM25 using RRF
            fused_docs = self.reciprocal_rank_fusion(vector_docs, bm25_docs)

            # Step 9 Addition: Cross-Encoder Re-ranking
            reranked_pairs = reranker_instance.rerank(query, fused_docs, top_k=4)
            final_chunks = [doc for doc, score in reranked_pairs]

        return {
            "strategy": selected_strategy,
            "chunks": final_chunks,
            "kg_context": kg_relations
        }