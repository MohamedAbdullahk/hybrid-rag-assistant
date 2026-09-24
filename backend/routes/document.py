import os
import shutil
import uuid
import time
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from backend.config import settings
from backend.rag.pdf_processor import PDFProcessor
from backend.rag.vector_store import VectorStoreManager
from backend.utils.logger import logger
from backend.kg.extractor import KGExtractor
from backend.kg.graph_store import GraphStoreManager

router = APIRouter()

# In-memory store for processing status and timers per session
processing_jobs: Dict[str, Dict[str, Any]] = {}

def process_pdf_task(session_id: str, file_path: str):
    start_time = time.perf_counter()
    processing_jobs[session_id] = {"status": "processing", "elapsed_time": 0.0, "total_chunks": 0}
    
    try:
        logger.info(f"Starting PDF processing for session {session_id}...")
        processor = PDFProcessor()
        chunks = processor.extract_and_chunk(file_path)
        
        # 1. Index chunks into FAISS Vector DB
        vector_mgr = VectorStoreManager()
        vector_mgr.create_and_save_index(session_id, chunks)
        
        # 2. Extract KG Triples & Build NetworkX Graph
        kg_extractor = KGExtractor()
        graph_mgr = GraphStoreManager()
        
        all_triples = []
        for chunk in chunks[:15]:  # Limit initial chunk batch for fast processing
            triples = kg_extractor.extract_triples(chunk.page_content)
            all_triples.extend(triples)
            
        graph_mgr.build_and_save_graph(session_id, all_triples)
        
        elapsed = round(time.perf_counter() - start_time, 2)
        processing_jobs[session_id] = {
            "status": "completed",
            "elapsed_time": elapsed,
            "total_chunks": len(chunks),
            "message": f"Document processed in {elapsed}s ({len(chunks)} chunks, {len(all_triples)} KG relationships)."
        }
        logger.info(f"Session {session_id} PDF, FAISS & KG processed in {elapsed}s")
    except Exception as e:
        logger.error(f"Error processing PDF for session {session_id}: {str(e)}")
        processing_jobs[session_id] = {
            "status": "failed",
            "error": str(e)
        }

@router.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(settings.DATA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    file_path = os.path.join(session_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger background indexing
    background_tasks.add_task(process_pdf_task, session_id, file_path)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "status": "processing",
        "message": "File uploaded successfully. Processing started."
    }

@router.get("/status/{session_id}")
async def get_processing_status(session_id: str):
    if session_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    return processing_jobs[session_id]