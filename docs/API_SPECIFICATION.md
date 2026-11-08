# API Specification: AI Image Studio Backend (v2.1)

This document specifies the REST API contract between the client CLI and the FastAPI inference server running on Google Colab.

- **Base URL**: `https://<tunnel-domain>.trycloudflare.com` or `http://localhost:8000`
- **Authentication**: HTTP Authorization Header with Bearer Token:
  ```http
  Authorization: Bearer <STUDIO_API_TOKEN>
  ```
- **Content-Type**: `application/json` (for JSON requests) or `image/png` (for binary image responses).
- **Idempotency**: Clients may pass `Idempotency-Key: <unique-uuid>` in HTTP headers to prevent duplicate jobs on network retries.

---

## Summary of Endpoints

| Method | Path | Description | Auth Required | Status Code |
|---|---|---|---|---|
| `GET` | `/` | Basic service info & version | No | 200 OK |
| `GET` | `/api/health` (or `/health`) | Server health, GPU device name, VRAM status, queue status | Yes | 200 OK |
| `GET` | `/api/models` (or `/models`) | Configured models and active VRAM model | Yes | 200 OK |
| `POST` | `/api/models/load` (or `/models/load`) | Queue asynchronous model download & VRAM allocation | Yes | 202 Accepted |
| `POST` | `/api/generate` (or `/generate`) | Queue asynchronous image generation job | Yes | 202 Accepted |
| `GET` | `/api/jobs` (or `/jobs`) | List recent jobs on the GPU backend | Yes | 200 OK |
| `GET` | `/api/jobs/{job_id}` (or `/jobs/{job_id}`) | Poll job status, progress percentage, and stage | Yes | 200 OK |
| `POST` | `/api/jobs/{job_id}/cancel` (or `/jobs/{job_id}/cancel`) | Request cooperative cancellation of a job | Yes | 200 OK / 409 |
| `GET` | `/api/jobs/{job_id}/result` (or `/jobs/{job_id}/result`) | Download result PNG image (with `?index=0` query) | Yes | 200 OK |

---

## 1. Authentication & Security

All requests to protected endpoints must supply a valid bearer token in the `Authorization` header:

```http
Authorization: Bearer vinesh123
```

### Authentication Error Response (401 Unauthorized)
```json
{
  "detail": "Invalid or missing API token in Authorization header"
}
```

---

## 2. Endpoints

### 2.1 GET `/api/health` (alias: `/health`)

Inspects server health, GPU hardware, active model, and queue metrics.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "gpu_available": true,
  "device_name": "Tesla T4",
  "cuda_version": "12.2",
  "cuda_device_count": 1,
  "cuda_visible_devices": "0",
  "vram": {
    "total_gb": 14.75,
    "allocated_gb": 5.82,
    "reserved_gb": 6.50,
    "free_gb": 8.93
  },
  "current_model": "runwayml/stable-diffusion-v1-5",
  "model_loaded": true,
  "generation_ready": true,
  "active_jobs": 0,
  "queue_length": 0,
  "running_job_id": null
}
```

---

### 2.2 GET `/api/models` (alias: `/models`)

Returns configured model catalog and active model in VRAM.

#### Response (200 OK)
```json
{
  "current_model": "runwayml/stable-diffusion-v1-5",
  "models": [
    {
      "id": "sd-1.5",
      "name": "Stable Diffusion 1.5",
      "repo_id": "runwayml/stable-diffusion-v1-5",
      "pipeline": "StableDiffusionPipeline",
      "default_resolution": [512, 512],
      "min_vram_gb": 4,
      "recommended_steps": 25,
      "fp16": true
    },
    {
      "id": "sdxl-base-1.0",
      "name": "Stable Diffusion XL Base 1.0",
      "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
      "pipeline": "StableDiffusionXLPipeline",
      "default_resolution": [1024, 1024],
      "min_vram_gb": 8,
      "recommended_steps": 30,
      "fp16": true
    }
  ]
}
```

---

### 2.3 POST `/api/models/load` (alias: `/models/load`)

Submits an asynchronous model load job. Returns immediately with HTTP 202 Accepted.

#### Request Headers (Optional)
```http
Idempotency-Key: load-sd15-uuid-001
```

#### Request Body
```json
{
  "model_id": "sd-1.5",
  "hf_token": null
}
```

#### Response (202 Accepted)
```json
{
  "job_id": "load_3f8a91c2b0",
  "type": "model_load",
  "status": "queued",
  "stage": "waiting",
  "progress": 0,
  "message": "Queued for GPU model loading...",
  "model_id": "sd-1.5",
  "created_at": 1730912000.12
}
```

---

### 2.4 POST `/api/generate` (alias: `/generate`)

Submits an asynchronous image generation job. Validates model loaded and parameter boundaries.

#### Request Headers (Optional)
```http
Idempotency-Key: gen-uuid-88849
```

#### Request Body
```json
{
  "prompt": "Beauty portrait of fictional model, natural skin pores, 85mm lens",
  "negative_prompt": "blurry, plastic skin, cartoon",
  "width": 512,
  "height": 768,
  "steps": 25,
  "guidance_scale": 7.0,
  "seed": 987654,
  "num_images": 1,
  "model_id": "sd-1.5",
  "preset": "beauty-portrait"
}
```

#### Response (202 Accepted)
```json
{
  "job_id": "gen_a1b2c3d4e5",
  "idempotency_key": "gen-uuid-88849",
  "type": "generation",
  "status": "queued",
  "stage": "waiting",
  "progress": 0,
  "message": "Queued in GPU worker line...",
  "model_id": "sd-1.5",
  "created_at": 1730912010.50
}
```

#### Model Mismatch Error (409 Conflict)
```json
{
  "detail": "Model mismatch: active GPU model is 'runwayml/stable-diffusion-v1-5', but 'sdxl-base-1.0' was requested. Switching models automatically during generation is disabled. Please load 'sdxl-base-1.0' first via POST /api/models/load."
}
```

---

### 2.5 GET `/api/jobs/{job_id}` (alias: `/jobs/{job_id}`)

Polls job status. Status moves through: `queued` → `running` (stage: `diffusion`) → `completed` (or `failed` / `cancelled`).

#### Response (200 OK - In Progress)
```json
{
  "job_id": "gen_a1b2c3d4e5",
  "type": "generation",
  "status": "running",
  "stage": "diffusion",
  "progress": 48,
  "message": "Denoising step 12/25 (48%)...",
  "model_id": "sd-1.5",
  "created_at": 1730912010.50,
  "started_at": 1730912012.00,
  "queue_wait_seconds": 1.50
}
```

#### Response (200 OK - Completed)
```json
{
  "job_id": "gen_a1b2c3d4e5",
  "type": "generation",
  "status": "completed",
  "stage": "completed",
  "progress": 100,
  "message": "Generation completed successfully (1 image in 10.4s).",
  "model_id": "sd-1.5",
  "created_at": 1730912010.50,
  "started_at": 1730912012.00,
  "completed_at": 1730912022.40,
  "queue_wait_seconds": 1.50,
  "execution_duration_seconds": 10.40,
  "results": [
    {
      "index": 0,
      "seed": 987654,
      "url": "/api/jobs/gen_a1b2c3d4e5/result?index=0",
      "width": 512,
      "height": 768
    }
  ]
}
```

---

### 2.6 POST `/api/jobs/{job_id}/cancel` (alias: `/jobs/{job_id}/cancel`)

Cooperatively cancels a queued or running generation job.

#### Response (200 OK)
```json
{
  "job_id": "gen_a1b2c3d4e5",
  "status": "cancelled",
  "message": "Queued job 'gen_a1b2c3d4e5' successfully cancelled."
}
```

---

### 2.7 GET `/api/jobs/{job_id}/result` (aliases: `/jobs/{job_id}/result`, `/jobs/{job_id}/image`)

Downloads result PNG binary for completed job.

#### Query Parameters
- `index`: integer (default `0`), image index within the generated batch.

#### Response Headers
```http
Content-Type: image/png
Content-Disposition: inline; filename="gen_a1b2c3d4e5_0.png"
X-Generation-Seed: 987654
X-Model-Id: sd-1.5
```
