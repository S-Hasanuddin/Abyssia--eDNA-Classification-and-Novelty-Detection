"""
main.py — FastAPI application entry point
============================================
Lifecycle:
  1. On startup: load model, start job worker
  2. On request: route through API endpoints
  3. On shutdown: stop job worker

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, REPORT_DIR
from app.routers.analysis import router as analysis_router
from app.services.job_manager import job_manager
from app.services.inference import EDNAInferencePipeline
from app.services.historical_clusters import HistoricalClusterStore
from app.services.report_generator import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifecycle ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("Starting eDNA Biodiversity API...")

    # Load model
    pipeline = EDNAInferencePipeline.get_instance()
    pipeline.load()

    # Historical cluster store
    history_store = HistoricalClusterStore()

    # Wire up job manager with actual callables
    def run_inference(sequences, seq_ids):
        return pipeline.predict_sample(sequences, seq_ids)

    def run_history(job_id, result):
        # Store new clusters
        history_store.add_clusters(
            job_id=job_id,
            novel_report=result.get('novel_taxa'),
            embeddings=result.get('_novel_embeddings'),
            novel_ids=result.get('_novel_ids'),
        )
        # Compare against history
        return history_store.compare_with_history(
            novel_report=result.get('novel_taxa'),
            embeddings=result.get('_novel_embeddings'),
            novel_ids=result.get('_novel_ids'),
        )

    def run_report(job_id, result, historical_comparison, filename):
        # Strip internal fields before passing to report
        clean_result = {k: v for k, v in result.items()
                        if not k.startswith('_')}
        return generate_pdf_report(
            job_id=job_id,
            result=clean_result,
            historical_comparison=historical_comparison,
            filename=filename,
        )

    job_manager.configure(
        inference_fn=run_inference,
        report_fn=run_report,
        history_fn=run_history,
    )
    await job_manager.start()

    if pipeline.is_loaded:
        logger.info("API ready — model loaded, job worker running")
    else:
        logger.warning(
            "API started but model NOT loaded. "
            "Place checkpoint files in the checkpoints/ directory.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    await job_manager.stop()
    logger.info("eDNA Biodiversity API stopped")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deep-Sea eDNA Biodiversity API",
    description=(
        "Taxonomic classification and novelty detection for deep-sea "
        "environmental DNA using a 96M-parameter BERT-style transformer "
        "with bottom-up cascaded classification and HDBSCAN clustering."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # permissive for dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(analysis_router)

# Serve reports directory for direct PDF access
REPORT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")


@app.get("/")
async def root():
    return {
        "service": "Deep-Sea eDNA Biodiversity API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "analyze_file": "POST /api/analyze",
            "analyze_sequence": "POST /api/analyze/sequence",
            "job_status": "GET /api/job/{job_id}",
            "job_result": "GET /api/job/{job_id}/result",
            "download_pdf": "GET /api/report/{job_id}/pdf",
            "history": "GET /api/history",
        },
    }
