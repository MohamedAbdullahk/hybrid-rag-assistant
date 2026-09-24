import os
import json
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from backend.config import settings
from backend.utils.logger import logger

router = APIRouter()

@router.get("/latency", response_model=Dict[str, Any])
async def get_latency_diagnostics():
    """
    Parses application logs to calculate average, max, and min latency
    across all RAG stages (rewrite, retrieval, llm, total).
    """
    log_file_path = settings.LOG_FILE

    if not os.path.exists(log_file_path):
        return {"status": "no_data", "message": "Log file does not exist yet."}

    stage_totals: Dict[str, List[float]] = {
        "rewrite": [],
        "retrieval": [],
        "llm": [],
        "total": []
    }

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Stage Timings:" in line:
                    # Extract JSON timing object from log line
                    try:
                        json_str = line.split("Stage Timings:")[1].strip()
                        timings = json.loads(json_str.replace("'", '"'))
                        for stage, duration in timings.items():
                            if stage in stage_totals:
                                stage_totals[stage].append(duration)
                    except Exception:
                        continue

        # Calculate statistics
        summary = {}
        for stage, durations in stage_totals.items():
            if durations:
                summary[stage] = {
                    "count": len(durations),
                    "avg_seconds": round(sum(durations) / len(durations), 3),
                    "min_seconds": round(min(durations), 3),
                    "max_seconds": round(max(durations), 3)
                }
            else:
                summary[stage] = {"count": 0, "avg_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0}

        return {
            "status": "success",
            "diagnostics": summary
        }

    except Exception as e:
        logger.error(f"Error reading diagnostics logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing latency logs: {str(e)}")