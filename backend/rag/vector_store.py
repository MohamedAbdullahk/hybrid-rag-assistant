import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from backend.config import settings
from backend.utils.logger import logger

class VectorStoreManager:
    def __init__(self):
        # Default to OpenAI embeddings
        if settings.OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY
            )
        else:
            raise ValueError("OPENAI_API_KEY is required for embedding generation.")

    def _get_session_dir(self, session_id: str) -> str:
        path = os.path.join(settings.DATA_DIR, session_id, "faiss_index")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def create_and_save_index(self, session_id: str, documents: List[Document]) -> str:
        """Embeds document chunks and saves FAISS index to disk per session."""
        if not documents:
            raise ValueError("Cannot create FAISS index with empty document list.")

        logger.info(f"Creating FAISS index for session {session_id} with {len(documents)} chunks...")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        save_path = self._get_session_dir(session_id)
        vectorstore.save_local(save_path)
        logger.info(f"FAISS index saved successfully at: {save_path}")
        return save_path

    def load_index(self, session_id: str) -> FAISS:
        """Loads FAISS index for a specific session."""
        save_path = self._get_session_dir(session_id)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"FAISS index not found for session {session_id}")

        return FAISS.load_local(
            save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def search_similarity(self, session_id: str, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Retrieves top-k most similar document chunks along with similarity scores."""
        vectorstore = self.load_index(session_id)
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        return docs_and_scores