from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from backend.config import settings
from backend.services.intent_router import IntentRouter
from backend.services.guardrails import guardrails
from backend.memory.session_memory import memory_manager
from backend.rag.strategies import StrategyEngine
from backend.utils.timer import StageTimer
from backend.utils.logger import logger

REWRITE_PROMPT = """Given the following conversation history and a follow-up user question, rewrite the question to be a self-contained, standalone question.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

QA_PROMPT = """You are an accurate, helpful AI assistant answering questions based on an uploaded document.

Context Chunks from Document:
{context_chunks}

Knowledge Graph Relationships:
{kg_context}

Previous Conversation History:
{chat_history}

User Question: {question}

Instructions:
1. Answer the question using ONLY the provided Document Chunks and Knowledge Graph Relationships.
2. If the answer cannot be found in the context, explicitly reply: "I couldn't find this in the document."
3. Keep the answer clear, concise, and factual. Do not hallucinate.

Answer:"""

class RAGService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.strategy_engine = StrategyEngine()

    @traceable(name="rewrite_query", run_type="chain")
    def rewrite_query(self, session_id: str, query: str) -> str:
        history_str = memory_manager.get_formatted_history(session_id)
        if not history_str or history_str == "No previous history.":
            return query

        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        # format_messages -> clean chat message (no "Human:" text prefix in the prompt)
        messages = prompt.format_messages(chat_history=history_str, question=query)
        response = self.llm.invoke(messages)
        rewritten = response.content.strip()
        logger.info(f"Original Query: '{query}' -> Rewritten: '{rewritten}'")
        return rewritten

    @traceable(name="answer_question", run_type="chain", tags=["chat"])
    def answer_question(
        self,
        session_id: str,
        query: str,
        strategy: str = None,        # 🆕 எந்த strategy-ல ஓடணும்
        save_memory: bool = True     # 🆕 Eval-ல False, memory-ல சேர்க்க வேண்டாம்
    ) -> Dict[str, Any]:
        timer = StageTimer()
        timer.start("total")

        # Step 13 Guardrails: Input Safety Check
        is_safe, block_msg = guardrails.validate_input(query)
        if not is_safe:
            timer.stop("total")
            return {
                "answer": block_msg,
                "sources": [],
                "timings": timer.get_summary(),
                "is_small_talk": False
            }

        # Stage 1: Intent Routing (Small Talk Check)
        is_small_talk, fast_reply = IntentRouter.is_small_talk(query)
        if is_small_talk:
            timer.stop("total")
            return {
                "answer": fast_reply,
                "sources": [],
                "timings": timer.get_summary(),
                "is_small_talk": True
            }

        # Stage 2: Query Rewrite
        timer.start("rewrite")
        standalone_query = self.rewrite_query(session_id, query)
        timer.stop("rewrite")

        # Stage 3: Context Retrieval
        timer.start("retrieval")
        retrieval_data = self.strategy_engine.retrieve_context(
            session_id, standalone_query, strategy=strategy
        )
        timer.stop("retrieval")

        chunks = retrieval_data["chunks"]
        kg_context = retrieval_data["kg_context"]

        formatted_chunks = ""
        sources = []
        for doc in chunks:
            formatted_chunks += f"- {doc.page_content}\n"
            page = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            sources.append(f"Page {page} ({chunk_id})")

        formatted_kg = "\n".join(kg_context) if kg_context else "No direct relationships found."
        history_str = memory_manager.get_formatted_history(session_id)

        # Stage 4: LLM Generation
        timer.start("llm")
        prompt = ChatPromptTemplate.from_template(QA_PROMPT)
        messages = prompt.format_messages(
            context_chunks=formatted_chunks if formatted_chunks else "No relevant chunks found.",
            kg_context=formatted_kg,
            chat_history=history_str,
            question=standalone_query
        )
        
        response = self.llm.invoke(messages)
        raw_answer = response.content.strip()
        
        # Step 13 Guardrails: Sanitize Output
        answer = guardrails.sanitize_output(raw_answer)
        timer.stop("llm")

        timer.stop("total")

        # Store turn in memory (skip during evaluation)
        if save_memory:
            memory_manager.add_turn(session_id, query, answer)

        return {
            "answer": answer,
            "sources": list(set(sources)),
            "timings": timer.get_summary(),
            "is_small_talk": False,
            "contexts": [doc.page_content for doc in chunks],
            "strategy": retrieval_data["strategy"]  
        }

rag_service_instance = RAGService()