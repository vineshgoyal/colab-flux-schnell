"""
FastAPI Backend Server for AI Image Studio.
Exposes asynchronous 202 endpoints, authenticated status polling, cancellation, and idempotency
to completely eliminate proxy timeouts (Cloudflare 524).
"""
import os
import sys
import logging
from typing import Optional, Dict, Any, List
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import FastAPI, HTTPException, Security, Depends, status, Response, Request, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    GenerationRequest,
    ModelLoadRequest,
    HealthResponse,
    JobResponse,
    CancelResponse,
    DEFAULT_MODELS
)
from backend.inference import InferenceManager

logger = logging.getLogger("studio.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

app = FastAPI(
    title="AI Image Studio - Asynchronous GPU Backend",
    version="2.1.0",
    description="Production-grade asynchronous Diffusers GPU inference server for Google Colab"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)
API_TOKEN = os.getenv("STUDIO_API_TOKEN", "").strip()

# Initialize Inference Manager
CACHE_DIR = os.getenv("HF_HOME", "/content/huggingface_cache")
STORAGE_DIR = os.getenv("STUDIO_STORAGE_DIR", "/tmp/studio_jobs")
inference_manager = InferenceManager(cache_dir=CACHE_DIR, storage_dir=STORAGE_DIR)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Validates Bearer token authentication."""
    if not API_TOKEN:
        return True
    if not credentials or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token in Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

@app.get("/")
def read_root():
    return {
        "service": "AI Image Studio GPU Backend",
        "version": "2.1.0",
        "queue_architecture": "asynchronous_sequential",
        "status": "online"
    }

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check(auth: bool = Depends(verify_token)):
    hw = inference_manager.get_hardware_info()
    return HealthResponse(
        status="healthy",
        version="2.1.0",
        gpu_available=hw["gpu_available"],
        device_name=hw["device_name"],
        cuda_version=hw["cuda_version"],
        cuda_device_count=hw.get("cuda_device_count", 0),
        cuda_visible_devices=hw.get("cuda_visible_devices"),
        vram=hw["vram"],
        current_model=hw["current_model"],
        model_loaded=hw.get("model_loaded", False),
        generation_ready=hw.get("generation_ready", False),
        active_jobs=hw.get("active_jobs", 0),
        queue_length=hw.get("queue_length", 0),
        running_job_id=hw.get("running_job_id")
    )

@app.get("/models")
@app.get("/api/models")
def list_models(auth: bool = Depends(verify_token)):
    models_list = list(DEFAULT_MODELS.values())
    return {
        "current_model": inference_manager.current_model_id,
        "models": models_list
    }

# -------------------------------------------------------------
# Asynchronous Job Submission Endpoints (Returns 202 immediately)
# -------------------------------------------------------------
@app.post("/models/load", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
@app.post("/api/models/load", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
def submit_model_load(
    req: ModelLoadRequest,
    idempotency_hdr: Optional[str] = Header(None, alias="Idempotency-Key"),
    x_idempotency_hdr: Optional[str] = Header(None, alias="X-Idempotency-Key"),
    auth: bool = Depends(verify_token)
):
    idempotency_key = req.idempotency_key or idempotency_hdr or x_idempotency_hdr
    try:
        job_id = inference_manager.submit_model_load_job(
            req.model_id,
            hf_token=req.hf_token,
            idempotency_key=idempotency_key
        )
        job = inference_manager.get_job(job_id)
        return job
    except Exception as exc:
        logger.error(f"Failed to submit model load: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

@app.post("/generate", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
@app.post("/api/generate", status_code=status.HTTP_202_ACCEPTED, response_model=JobResponse)
def submit_generate(
    req: GenerationRequest,
    idempotency_hdr: Optional[str] = Header(None, alias="Idempotency-Key"),
    x_idempotency_hdr: Optional[str] = Header(None, alias="X-Idempotency-Key"),
    auth: bool = Depends(verify_token)
):
    idempotency_key = req.idempotency_key or idempotency_hdr or x_idempotency_hdr
    if idempotency_key and not req.idempotency_key:
        req.idempotency_key = idempotency_key

    try:
        job_id = inference_manager.submit_generation_job(req)
        job = inference_manager.get_job(job_id)
        return job
    except ValueError as ve:
        err_msg = str(ve)
        status_code = status.HTTP_409_CONFLICT if "Model mismatch" in err_msg or "No model is currently loaded" in err_msg else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=err_msg)
    except RuntimeError as re:
        err_msg = str(re)
        status_code = status.HTTP_429_TOO_MANY_REQUESTS if "queue is full" in err_msg.lower() else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=err_msg)
    except Exception as e:
        logger.error(f"Generate submission error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# -------------------------------------------------------------
# Authenticated Status & Job Management Endpoints
# -------------------------------------------------------------
@app.get("/jobs", response_model=List[JobResponse])
@app.get("/api/jobs", response_model=List[JobResponse])
def list_jobs(limit: int = Query(20, ge=1, le=100), auth: bool = Depends(verify_token)):
    all_jobs = list(inference_manager.jobs.values())
    all_jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return [inference_manager.get_job(j["id"]) for j in all_jobs[:limit]]

@app.get("/jobs/{job_id}", response_model=JobResponse)
@app.get("/api/jobs/{job_id}", response_model=JobResponse)
@app.get("/generation/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str, auth: bool = Depends(verify_token)):
    job = inference_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. Jobs are stored in-process and reset if the Colab runtime restarts."
        )
    return job

@app.post("/jobs/{job_id}/cancel", response_model=CancelResponse)
@app.post("/api/jobs/{job_id}/cancel", response_model=CancelResponse)
def cancel_job_endpoint(job_id: str, auth: bool = Depends(verify_token)):
    success, message = inference_manager.cancel_job(job_id)
    if not success:
        job = inference_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)

    return CancelResponse(
        job_id=job_id,
        status="cancelled" if "cancelled" in message else "cancelling",
        message=message
    )

@app.get("/jobs/{job_id}/result")
@app.get("/api/jobs/{job_id}/result")
@app.get("/jobs/{job_id}/image")
@app.get("/api/jobs/{job_id}/image")
@app.get("/generation/{job_id}/image")
def get_job_result_image(
    job_id: str,
    index: int = Query(0, ge=0, le=10),
    auth: bool = Depends(verify_token)
):
    job = inference_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found."
        )
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' is not yet completed (current status: {job.get('status')}, progress: {job.get('progress')})."
        )

    image_bytes = inference_manager.get_job_image(job_id, index=index)
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image index {index} unavailable for job '{job_id}'.")

    # Locate seed for this index
    seed_val = str(job.get("seed", ""))
    results = job.get("results") or []
    for r in results:
        if r.get("index") == index:
            seed_val = str(r.get("seed", seed_val))
            break

    headers = {
        "Content-Disposition": f"inline; filename=\"{job_id}_{index}.png\"",
        "X-Generation-Seed": seed_val,
        "X-Model-Id": str(job.get("model_id", ""))
    }
    return Response(content=image_bytes, media_type="image/png", headers=headers)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.server:app", host="0.0.0.0", port=port, reload=False)
