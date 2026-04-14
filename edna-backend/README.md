# Deep-Sea eDNA Biodiversity API — FastAPI Backend

Backend service for the deep-sea eDNA taxonomy classification and novelty
detection pipeline. Wraps a 96M-parameter BERT-style transformer model with
HDBSCAN clustering, deployed via FastAPI with async job processing.

## Architecture

```
React SPA (port 8080) ←→ FastAPI Backend (port 8000)
                           ├── POST /api/analyze           Upload file → job
                           ├── POST /api/analyze/sequence   Single sequence
                           ├── GET  /api/job/{job_id}       Poll progress
                           ├── GET  /api/job/{job_id}/result Full results
                           ├── GET  /api/report/{job_id}/pdf Download PDF
                           ├── GET  /api/health              Model status
                           └── GET  /api/history             Historical clusters
```

## Quick Start

### 1. Install Dependencies

```bash
cd edna-backend
pip install -r requirements.txt
```

### 2. Place Model Checkpoint

Copy these files into `checkpoints/`:
- `ft_final.pt` (or `phase3_final.pt` or `phase2_latest.pt`)
- `label_to_id.json`

```bash
cp /path/to/ft_final.pt checkpoints/
cp /path/to/label_to_id.json checkpoints/
```

### 3. Run the Server

```bash
# CPU (default)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# GPU
DEVICE=cuda uvicorn app.main:app --host 0.0.0.0 --port 8000

# Custom checkpoint directory
CHECKPOINT_DIR=/path/to/checkpoints uvicorn app.main:app --port 8000
```

### 4. Configure Frontend

In the React frontend, point API calls to `http://localhost:8000/api/`.
Add to `vite.config.ts`:

```ts
server: {
  proxy: {
    '/api': 'http://localhost:8000',
  },
},
```

## API Reference

### POST /api/analyze
Upload an eDNA file for analysis. Returns a `job_id`.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@sample.fasta"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "filename": "sample.fasta",
  "status": "queued",
  "message": "Analysis job created. Poll /api/job/{job_id} for progress."
}
```

### GET /api/job/{job_id}
Poll job progress.

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "classifying",
  "step_index": 2,
  "step_label": "Running multi-rank taxonomic classification...",
  "progress": 50.0,
  "has_result": false,
  "has_pdf": false
}
```

### GET /api/job/{job_id}/result
Full results (only when `status === "complete"`).

Returns classified sequences, novel taxa clusters with novelty scores,
ecosystem diversity metrics with dominant taxa, and historical comparison.

### POST /api/analyze/sequence
Synchronous single-sequence classification.

```bash
curl -X POST http://localhost:8000/api/analyze/sequence \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATCGATCG...", "id": "my_seq"}'
```

### GET /api/report/{job_id}/pdf
Download the PDF biodiversity report.

### GET /api/health
Check model and API status.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHECKPOINT_DIR` | `./checkpoints` | Path to model weights |
| `CONFIDENCE_THRESHOLD` | `0.5` | Classification vs novelty routing |
| `BATCH_SIZE` | `32` | Inference batch size |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated CORS origins |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max upload file size |

## Project Structure

```
edna-backend/
├── app/
│   ├── main.py                  # FastAPI app + lifecycle
│   ├── config.py                # Configuration
│   ├── models/
│   │   └── dna_model.py         # Model architecture (FIXED proj head)
│   ├── routers/
│   │   └── analysis.py          # API endpoints
│   ├── services/
│   │   ├── inference.py          # Core inference pipeline
│   │   ├── job_manager.py        # Async job queue
│   │   ├── report_generator.py   # PDF report generation
│   │   └── historical_clusters.py # Cross-run cluster comparison
│   └── utils/
│       └── parsers.py            # File parsers (FASTA/FASTQ/CSV/JSON)
├── tokenizer.py                  # KmerTokenizer (6-mer, vocab 4101)
├── checkpoints/                  # Place ft_final.pt + label_to_id.json here
├── requirements.txt
└── README.md
```

## Critical: Projection Head Architecture

The checkpoint `ft_final.pt` was trained with a Dropout layer in the
projection head:

```python
nn.Sequential(
    nn.Linear(768, 512),   # index 0
    nn.ReLU(),             # index 1
    nn.Dropout(0.5),       # index 2  ← MUST EXIST for weight loading
    nn.Linear(512, 256),   # index 3
)
```

The original `inference_api.py` OMITS this Dropout (uses indices 0,1,2 only),
which causes the second Linear layer's weights to load into the Dropout
"layer" position — silently corrupting the model. This backend uses the
correct 4-layer architecture.
