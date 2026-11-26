import os
from fastapi import APIRouter, HTTPException, status

from app.schemas.responses import QueryRequest, QueryResponse, HealthResponse
from app.rag.retriever import RAGMetadataRetriever
from app.rag.generator import RAGGeneratorService
from app.api.routes_identification import get_services

router = APIRouter(prefix="/api/v1", tags=["RAG & Health"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DIR = os.path.join(BASE_DIR, "indexes", "rag_index")
INDEX_DIR = os.path.join(BASE_DIR, "indexes", "image_index")


@router.post("/query", response_model=QueryResponse, summary="Design-Scoped Follow-Up Q&A")
async def query_design_rag(request: QueryRequest):
    """
    Answers follow-up questions for a specific Design ID.
    Uses RAG retrieval scoped strictly by design_id with prompt injection defenses.
    """
    design_id = request.design_id.strip()
    question = request.question.strip()

    if not design_id or not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="design_id and question cannot be empty.")

    cls_svc, emb_svc, search_svc, rej_svc, rag_ret, rag_gen = get_services()

    retrieved_context = rag_ret.get_by_design_id(design_id)
    grounded_found = retrieved_context is not None and "metadata" in retrieved_context

    answer = rag_gen.generate_answer(design_id, question, retrieved_context)

    return QueryResponse(
        design_id=design_id,
        question=question,
        answer=answer,
        grounded_context_found=grounded_found
    )


@router.get("/health", response_model=HealthResponse, summary="System Health Check")
async def health_check():
    """Returns system status, loaded classifier status, FAISS vectors indexed, and ChromaDB document counts."""
    cls_svc, emb_svc, search_svc, rej_svc, rag_ret, rag_gen = get_services()

    faiss_count = search_svc.faiss_index.index.ntotal if search_svc and search_svc.faiss_index else 0
    chroma_count = 0

    if rag_ret and rag_ret.collection:
        try:
            chroma_count = rag_ret.collection.count()
        except Exception:
            chroma_count = 0

    return HealthResponse(
        status="healthy",
        classifier_loaded=cls_svc is not None,
        faiss_vectors_indexed=faiss_count,
        chromadb_documents_count=chroma_count,
        version="1.0.0"
    )
