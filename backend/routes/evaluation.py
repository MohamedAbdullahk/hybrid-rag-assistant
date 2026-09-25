from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.eval.evaluator import evaluator_instance
from backend.utils.logger import logger

router = APIRouter()


class EvalRequest(BaseModel):
    session_id: str
    question: str
    expected_answer: str = ""     # Optional reference answer


@router.post("/compare")
def compare_strategies(request: EvalRequest):
    """Runs all RAG strategies on one question and returns scores + best strategy."""
    if not request.session_id.strip() or not request.question.strip():
        raise HTTPException(status_code=400, detail="session_id and question are required.")

    try:
        logger.info(f"Evaluation requested for session: {request.session_id}")
        return evaluator_instance.compare_all_strategies(
            session_id=request.session_id,
            query=request.question,
            expected_output=request.expected_answer
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")
    except Exception as e:
        logger.error(f"Evaluation failed for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")