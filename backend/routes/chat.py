from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.rag_service import rag_service_instance
from backend.utils.logger import logger

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]
    timings: Dict[str, float]
    is_small_talk: bool

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        logger.info(f"Received chat request for session: {request.session_id}")
        
        # Execute RAG Service Pipeline
        result = rag_service_instance.answer_question(
            session_id=request.session_id,
            query=request.message
        )

        return ChatResponse(
            session_id=request.session_id,
            answer=result["answer"],
            sources=result["sources"],
            timings=result["timings"],
            is_small_talk=result["is_small_talk"]
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")