"""
analysis.py — API routes for eDNA analysis
=============================================
Endpoints:
  POST /api/analyze         — Upload file, start analysis job
  POST /api/analyze/sequence — Single sequence classification
  GET  /api/job/{job_id}    — Poll job status + results
  GET  /api/job/{job_id}/result — Full results JSON
  GET  /api/report/{job_id}/pdf — Download PDF report
  GET  /api/health          — Health check
  GET  /api/history         — Historical cluster summary

Single-sequence novel storage:
  When /api/analyze/sequence returns specific_enough=False, the sequence
  embedding is stored in HistoricalClusterStore under job_id="single_{seq_id}".
  This allows future batch jobs to find cosine-similar novel candidates across
  both single and batch submissions.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import UPLOAD_DIR, MAX_UPLOAD_SIZE_MB, RANKS
from app.services.job_manager import job_manager, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])


# ── Request / Response schemas ───────────────────────────────────────────────

class SequenceRequest(BaseModel):
    sequence: str = Field(..., min_length=50, max_length=10000,
                          description="DNA sequence string (ACGT)")
    id: str = Field(default="query", description="Sequence identifier")


class JobResponse(BaseModel):
    job_id: str
    status: str
    step_index: int
    step_label: str
    progress: float
    created_at: str
    completed_at: Optional[str] = None
    has_result: bool
    error: Optional[str] = None
    has_pdf: bool
    filename: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    checkpoint: str
    queue_size: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    """Check API health and model status."""
    from app.services.inference import EDNAInferencePipeline
    pipeline = EDNAInferencePipeline.get_instance()
    from app.config import DEVICE, CHECKPOINT_DIR, CHECKPOINT_PRIORITY

    ckpt_name = "none"
    for fname in CHECKPOINT_PRIORITY:
        if (CHECKPOINT_DIR / fname).exists():
            ckpt_name = fname
            break

    return HealthResponse(
        status="ok" if pipeline.is_loaded else "model_not_loaded",
        model_loaded=pipeline.is_loaded,
        device=DEVICE,
        checkpoint=ckpt_name,
        queue_size=job_manager._queue.qsize() if job_manager._queue else 0,
    )


@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Upload an eDNA file and start analysis.
    Supported formats: .fasta, .fa, .fastq, .fq, .csv, .json, .xls, .xlsx, .txt
    Returns a job_id to poll for results.
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    allowed = {'.fasta', '.fa', '.fna', '.fastq', '.fq', '.csv', '.json', '.xls', '.xlsx', '.txt'}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {sorted(allowed)}")

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_SIZE_MB} MB")

    # Save to disk
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, 'wb') as f:
        f.write(content)

    # Create and submit job
    job = job_manager.create_job(
        filename=file.filename,
        file_path=str(file_path),
    )
    await job_manager.submit(job)

    return JSONResponse(content={
        "job_id": job.id,
        "filename": file.filename,
        "file_size_mb": round(size_mb, 2),
        "status": job.status.value,
        "message": "Analysis job created. Poll /api/job/{job_id} for progress.",
    })


@router.post("/analyze/sequence")
async def analyze_sequence(req: SequenceRequest):
    """
    Classify a single DNA sequence (synchronous — no job queue).
    Returns taxonomy prediction immediately.

    If the sequence is flagged as a novel candidate (specific_enough=False),
    its 256-dim embedding is stored in HistoricalClusterStore so future batch
    jobs can find cosine-similar sequences via historical comparison.
    """
    from app.services.inference import EDNAInferencePipeline
    from app.services.historical_clusters import HistoricalClusterStore

    pipeline = EDNAInferencePipeline.get_instance()
    if not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    from app.utils.parsers import validate_sequence
    try:
        clean_seq = validate_sequence(req.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = pipeline.predict_sequence(clean_seq, seq_id=req.id)

    # ── Store novel candidates for historical comparison ──────────────────
    # Sequences that don't reach the minimum classification rank are novel
    # candidates. We store their embeddings so future batch jobs can match
    # against them via cosine similarity in HistoricalClusterStore.
    if not result.get("specific_enough", True):
        try:
            embedding = pipeline.get_embeddings([clean_seq])  # shape: (1, 256)

            store = HistoricalClusterStore()

            # Wrap in the novel_report schema that add_clusters() expects
            novel_report = {
                "clusters": {
                    "cluster_single": {
                        "size": 1,
                        "member_ids": [req.id],
                        "reported_rank": result["reported_rank"],
                        "reported_taxon": result["reported_taxon"],
                        "reported_confidence": result["reported_confidence"],
                        "novelty_score": round(
                            1.0 - float(result["reported_confidence"]), 4),
                        "assessment": (
                            f"Single-sequence novel candidate — best rank: "
                            f"{result['reported_rank']}: {result['reported_taxon']} "
                            f"(conf={result['reported_confidence']:.2f})"
                        ),
                        "taxonomy_profile": result.get("taxonomy", {}),
                    }
                }
            }

            store.add_clusters(
                job_id=f"single_{req.id}",
                novel_report=novel_report,
                embeddings=embedding.tolist(),   # list of lists (1 × 256)
                novel_ids=[req.id],
            )

            logger.info(
                f"Stored single-seq novel candidate '{req.id}' "
                f"(rank={result['reported_rank']}, "
                f"conf={result['reported_confidence']:.2f}) "
                f"in historical store"
            )

            # Surface the storage event back to the caller
            result["stored_as_novel"] = True
            result["history_job_id"]  = f"single_{req.id}"

        except Exception as e:
            # Non-fatal — classification result is still returned
            logger.warning(
                f"Failed to store novel candidate '{req.id}' "
                f"in historical store: {e}"
            )
            result["stored_as_novel"] = False

    return result


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll job status and progress. When complete, includes summary results.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response = job.to_dict()

    # Include summary when complete
    if job.status == JobStatus.COMPLETE and job.result:
        response['summary'] = job.result.get('summary', {})

    return response


@router.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get complete analysis results for a finished job.
    Includes classified sequences, novel taxa, ecosystem analysis,
    and historical comparison.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=409,
            detail=f"Job not complete yet. Status: {job.status.value}")

    if job.result is None:
        raise HTTPException(status_code=500, detail="Job completed but no results")

    # Build response (strip internal fields)
    result = {k: v for k, v in job.result.items() if not k.startswith('_')}
    result['job_id'] = job_id
    result['filename'] = job.filename

    if job.historical_comparison:
        result['historical_comparison'] = job.historical_comparison

    result['has_pdf'] = job.pdf_path is not None

    return result


@router.get("/report/{job_id}/pdf")
async def download_pdf(job_id: str):
    """Download the PDF biodiversity report for a completed job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF report not available")

    if not Path(job.pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    return FileResponse(
        path=job.pdf_path,
        media_type="application/pdf",
        filename=f"edna_report_{job_id}.pdf",
    )


@router.get("/history")
async def get_history():
    """Get summary of all historical novel cluster analyses."""
    from app.services.historical_clusters import HistoricalClusterStore
    store = HistoricalClusterStore()
    return store.get_history_summary()