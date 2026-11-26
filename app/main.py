import os
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from app.api.routes_identification import router as identification_router
from app.api.routes_rag import router as rag_router

# Environment & Security Configuration
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]

app = FastAPI(
    title="Gold Jewellery AI",
    description="Multimodal AI system for gold jewellery classification, visual vector search, rejection handling, and RAG Q&A.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Explicit CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(identification_router)
app.include_router(rag_router)

# Mount static frontend
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Global Exception Handler (returns generic error messages to clients without leaking sensitive internal stack traces)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled Server Error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Gold Jewellery AI FastAPI server on http://{host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
