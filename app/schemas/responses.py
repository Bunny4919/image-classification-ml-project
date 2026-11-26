from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ViewMetadata(BaseModel):
    view_id: str
    view_type: str
    filename: Optional[str] = None


class DesignMatch(BaseModel):
    design_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    similarity_score: float
    material: Optional[str] = None
    karat: Optional[int] = None
    stone: Optional[str] = None
    collection: Optional[str] = None
    description: Optional[str] = None
    matched_view_type: Optional[str] = None


class IdentifyResponse(BaseModel):
    status: str = Field(..., description="Operational status: KNOWN_DESIGN, KNOWN_CATEGORY_UNKNOWN_DESIGN, or UNSUPPORTED_OBJECT")
    category: str = Field(..., description="Predicted category: Ring, Earring, Necklace, Bracelet")
    category_confidence: float = Field(..., description="Classification probability score between 0.0 and 1.0")
    category_probabilities: Dict[str, float] = Field(default_factory=dict, description="Probabilities across all 4 categories")
    best_match_design_id: Optional[str] = Field(default=None, description="Resolved catalog Design ID if known design match")
    similarity_score: float = Field(default=0.0, description="Highest visual similarity score across catalog views")
    design_details: Optional[DesignMatch] = Field(default=None, description="Detailed metadata of matched design")
    explanation: str = Field(..., description="Grounded natural-language summary / explanation")


class QueryRequest(BaseModel):
    design_id: str = Field(..., min_length=1, max_length=100, description="Target Design ID")
    question: str = Field(..., min_length=1, max_length=1000, description="Free-form follow-up question")


class QueryResponse(BaseModel):
    design_id: str
    question: str
    answer: str
    grounded_context_found: bool


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    faiss_vectors_indexed: int
    chromadb_documents_count: int
    version: str = "1.0.0"
