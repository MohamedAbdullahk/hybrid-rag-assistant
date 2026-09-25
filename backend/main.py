from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import document, chat, diagnostics
from backend.utils.logger import logger
from backend.routes import document, chat, diagnostics, evaluation

app = FastAPI(
    title="Conversational Hybrid RAG API",
    description="Vector + Knowledge Graph + Memory RAG Engine",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(document.router, prefix="/api/v1/document", tags=["Document"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(diagnostics.router, prefix="/api/v1/diagnostics", tags=["Diagnostics"])
app.include_router(evaluation.router, prefix="/api/v1/eval", tags=["Evaluation"])

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed.")
    return {"status": "online", "message": "Hybrid RAG API is operational."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)