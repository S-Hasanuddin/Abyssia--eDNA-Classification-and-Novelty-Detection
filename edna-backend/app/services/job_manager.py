"""
job_manager.py — Async job queue for inference
================================================
With a 96M param model, we can't handle concurrent inference requests
without risking GPU OOM. Jobs are processed sequentially via asyncio.
"""

import asyncio
import logging
import uuid
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    PREPROCESSING = "preprocessing"
    EMBEDDING = "embedding"
    CLASSIFYING = "classifying"
    CLUSTERING = "clustering"
    ECOSYSTEM = "ecosystem"
    REPORTING = "reporting"
    COMPLETE = "complete"
    FAILED = "failed"


# Step labels shown in frontend progress
PIPELINE_STEPS = [
    (JobStatus.PREPROCESSING, "Validating and preprocessing reads..."),
    (JobStatus.EMBEDDING, "Generating embeddings via Transformer Encoder (96M params)..."),
    (JobStatus.CLASSIFYING, "Running multi-rank taxonomic classification..."),
    (JobStatus.CLUSTERING, "HDBSCAN novelty detection & clustering..."),
    (JobStatus.ECOSYSTEM, "Computing ecological diversity metrics..."),
    (JobStatus.REPORTING, "Generating biodiversity report..."),
]


class Job:
    def __init__(self, job_id: str, filename: str, file_path: str):
        self.id = job_id
        self.filename = filename
        self.file_path = file_path
        self.status = JobStatus.QUEUED
        self.step_index = -1
        self.step_label = "Queued..."
        self.progress = 0.0
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.pdf_path: Optional[str] = None
        self.historical_comparison: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status.value,
            "step_index": self.step_index,
            "step_label": self.step_label,
            "progress": round(self.progress, 2),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "has_result": self.result is not None,
            "error": self.error,
            "has_pdf": self.pdf_path is not None,
        }

    def advance(self, step_index: int):
        if step_index < len(PIPELINE_STEPS):
            self.status, self.step_label = PIPELINE_STEPS[step_index]
            self.step_index = step_index
            self.progress = (step_index / len(PIPELINE_STEPS)) * 100
        else:
            self.status = JobStatus.COMPLETE
            self.step_label = "Complete"
            self.progress = 100.0

    def fail(self, error: str):
        self.status = JobStatus.FAILED
        self.error = error
        self.step_label = f"Failed: {error[:100]}"


class JobManager:
    """
    Manages a queue of analysis jobs. Processes them one at a time
    to avoid GPU memory contention.
    """

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue = None
        self._worker_task: Optional[asyncio.Task] = None
        self._inference_fn = None
        self._report_fn = None
        self._history_fn = None

    def configure(self, inference_fn, report_fn=None, history_fn=None):
        """Set the actual inference + report generation callables."""
        self._inference_fn = inference_fn
        self._report_fn = report_fn
        self._history_fn = history_fn

    async def start(self):
        """Start the background worker."""
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Job manager worker started")

    async def stop(self):
        """Stop the background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Job manager worker stopped")

    def create_job(self, filename: str, file_path: str) -> Job:
        job_id = str(uuid.uuid4())[:12]
        job = Job(job_id=job_id, filename=filename, file_path=file_path)
        self.jobs[job_id] = job
        return job

    async def submit(self, job: Job):
        if self._queue is None:
            # Auto-start if not yet started
            await self.start()
        await self._queue.put(job.id)
        logger.info(f"Job {job.id} submitted (queue size: {self._queue.qsize()})")

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def _worker(self):
        """Process jobs from the queue one at a time."""
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue

            try:
                await self._process_job(job)
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
                job.fail(str(e))
            finally:
                self._queue.task_done()

    async def _process_job(self, job: Job):
        """Run the full pipeline for a single job."""
        loop = asyncio.get_event_loop()

        # Step 0: Preprocessing (parsing)
        job.advance(0)
        from app.utils.parsers import parse_file, validate_sequence
        records = await loop.run_in_executor(None, parse_file, job.file_path)

        # Validate sequences
        valid_records = []
        for rec in records:
            try:
                rec['sequence'] = validate_sequence(rec['sequence'])
                valid_records.append(rec)
            except ValueError as e:
                logger.warning(f"Skipped {rec.get('id', '?')}: {e}")

        if not valid_records:
            job.fail("No valid sequences found after preprocessing")
            return

        sequences = [r['sequence'] for r in valid_records]
        seq_ids = [r['id'] for r in valid_records]

        # Steps 1-4: inference (runs in thread to not block event loop)
        job.advance(1)  # embedding

        def run_inference():
            # The inference pipeline handles steps 1-4 internally
            return self._inference_fn(sequences, seq_ids)

        result = await loop.run_in_executor(None, run_inference)

        job.advance(2)  # classifying
        job.advance(3)  # clustering
        job.advance(4)  # ecosystem

        job.result = result

        # Step 5: Historical comparison + Report generation
        job.advance(5)

        # Historical cluster comparison
        if self._history_fn and result.get('novel_taxa'):
            try:
                comparison = await loop.run_in_executor(
                    None,
                    self._history_fn,
                    job.id,
                    result,
                )
                job.historical_comparison = comparison
            except Exception as e:
                logger.warning(f"Historical comparison failed: {e}")

        # PDF report
        if self._report_fn:
            try:
                pdf_path = await loop.run_in_executor(
                    None,
                    self._report_fn,
                    job.id,
                    result,
                    job.historical_comparison,
                    job.filename,
                )
                job.pdf_path = pdf_path
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")

        # Done
        job.status = JobStatus.COMPLETE
        job.step_label = "Analysis complete"
        job.progress = 100.0
        job.completed_at = datetime.utcnow().isoformat()
        logger.info(
            f"Job {job.id} complete: {len(sequences)} seqs, "
            f"{result['summary']['classified']} classified, "
            f"{result['summary']['novel_candidates']} novel"
        )


# Singleton
job_manager = JobManager()
