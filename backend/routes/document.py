import os
import shutil
import uuid
import time
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from langsmith import traceable
from backend.config import settings
from backend.rag.pdf_processor import PDFProcessor
from backend.rag.vector_store import VectorStoreManager
from backend.utils.logger import logger
from backend.kg.extractor import KGExtractor
from backend.kg.graph_store import GraphStoreManager

router = APIRouter()

# Knowledge Graph extraction settings
KG_MAX_CHUNKS = 60      # safety cap so a very large PDF doesn't make hundreds of LLM calls
KG_PARALLEL_CALLS = 8   # how many OpenAI calls run at the same time

# In-memory store for processing status and timers per session
processing_jobs: Dict[str, Dict[str, Any]] = {}

@traceable(name="process_pdf", run_type="chain", tags=["upload"])
def process_pdf_task(session_id: str, file_path: str):
    start_time = time.perf_counter()
    processing_jobs[session_id] = {"status": "processing", "elapsed_time": 0.0, "total_chunks": 0}
    
    try:
        logger.info(f"Starting PDF processing for session {session_id}...")
        processor = PDFProcessor()
        chunks = traceable(name="pdf_chunking", run_type="chain")(processor.extract_and_chunk)(file_path)
        
        # 1. Index chunks into FAISS Vector DB
        vector_mgr = VectorStoreManager()
        vector_mgr.create_and_save_index(session_id, chunks)
        
        # 2. Extract KG Triples & Build NetworkX Graph
        kg_extractor = KGExtractor()
        graph_mgr = GraphStoreManager()
        
        # Run KG extraction for many chunks IN PARALLEL (was: 15 chunks, one by one)
        kg_chunks = chunks[:KG_MAX_CHUNKS]
        kg_start = time.perf_counter()
        # Each worker thread gets a copy of the current context, so every KG call
        # appears as a child of "process_pdf" in LangSmith (threads don't inherit it by default)
        contexts = [contextvars.copy_context() for _ in kg_chunks]
        with ThreadPoolExecutor(max_workers=KG_PARALLEL_CALLS) as pool:
            results = list(pool.map(
                lambda pair: pair[0].run(kg_extractor.extract_triples, pair[1].page_content),
                zip(contexts, kg_chunks)
            ))
        all_triples = [t for triples in results for t in triples]
        logger.info(
            f"KG extraction: {len(kg_chunks)} chunks, {len(all_triples)} triples "
            f"in {time.perf_counter() - kg_start:.1f}s ({KG_PARALLEL_CALLS} parallel calls)"
        )
            
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