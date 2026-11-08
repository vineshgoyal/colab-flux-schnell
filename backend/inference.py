"""
Asynchronous GPU Inference and Model Lifecycle Manager for AI Image Studio.
Executes jobs sequentially in a background worker thread to prevent GPU thrashing and Cloudflare 524 timeouts.
"""
import io
import os
import gc
import time
import uuid
import queue
import shutil
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

import torch
from backend.models import DEFAULT_MODELS, DEFAULT_NEGATIVE_PROMPT, GenerationRequest, VramMetrics, JobResultItem, get_model_specs

logger = logging.getLogger("studio.inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

class InferenceManager:
    def __init__(self, cache_dir: Optional[str] = None, storage_dir: Optional[str] = None, max_queue_size: int = 20):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir or os.getenv("HF_HOME", "/content/huggingface_cache")
        self.storage_dir = storage_dir or os.getenv("STUDIO_STORAGE_DIR", "/tmp/studio_jobs")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.current_model_id: Optional[str] = None
        self.current_load_source: Optional[str] = None
        self.pipeline = None

        # Thread-safe in-process job registry & sequential queue
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.abort_events: Dict[str, threading.Event] = {}
        self.idempotency_map: Dict[str, str] = {}
        self.max_queue_size = max_queue_size
        self.job_queue = queue.Queue(maxsize=max_queue_size)
        self.gpu_lock = threading.Lock()
        self.active_running_job_id: Optional[str] = None

        # Start single sequential background worker
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"Initialized InferenceManager on device: {self.device}. Background queue active (max {max_queue_size}). Storage: {self.storage_dir}")

    def get_vram_metrics(self) -> Optional[VramMetrics]:
        if not torch.cuda.is_available():
            return None
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return VramMetrics(
                total_gb=round(total, 2),
                allocated_gb=round(allocated, 2),
                reserved_gb=round(reserved, 2),
                free_gb=round(total - allocated, 2)
            )
        except Exception:
            return None

    def get_hardware_info(self) -> Dict[str, Any]:
        gpu_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        model_ok = self.pipeline is not None and self.current_model_id is not None
        return {
            "gpu_available": gpu_ok,
            "device": "cuda" if gpu_ok else "cpu",
            "device_name": torch.cuda.get_device_name(0) if gpu_ok else "CPU",
            "cuda_version": torch.version.cuda if gpu_ok else None,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "torch_version": torch.__version__,
            "current_model": self.current_model_id,
            "model_loaded": model_ok,
            "generation_ready": gpu_ok and model_ok,
            "active_jobs": self.job_queue.qsize() + (1 if self.active_running_job_id else 0),
            "queue_length": self.job_queue.qsize(),
            "running_job_id": self.active_running_job_id,
            "vram": self.get_vram_metrics()
        }

    # ---------------------------------------------------------
    # Idempotency & Cleanup Helpers
    # ---------------------------------------------------------
    def _prune_old_jobs(self, max_retained: int = 50, ttl_seconds: int = 3600):
        """Removes expired job records and disk files to protect Colab RAM and disk."""
        now = time.time()
        sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.get("created_at", 0))

        # Check TTL and count limits
        expired_ids = []
        for j in sorted_jobs:
            if j.get("status") in ("completed", "failed", "cancelled"):
                if (now - j.get("created_at", now) > ttl_seconds) or (len(self.jobs) - len(expired_ids) > max_retained):
                    expired_ids.append(j["id"])

        for jid in expired_ids:
            self.jobs.pop(jid, None)
            self.abort_events.pop(jid, None)
            # Remove disk folder
            job_dir = os.path.join(self.storage_dir, jid)
            if os.path.exists(job_dir):
                try:
                    shutil.rmtree(job_dir, ignore_errors=True)
                except Exception:
                    pass

        # Prune idempotency map
        active_ids = set(self.jobs.keys())
        self.idempotency_map = {k: v for k, v in self.idempotency_map.items() if v in active_ids}

    # ---------------------------------------------------------
    # Public Asynchronous Job Submission (Returns immediately)
    # ---------------------------------------------------------
    def submit_model_load_job(self, model_key_or_repo: str, hf_token: Optional[str] = None, idempotency_key: Optional[str] = None) -> str:
        self._prune_old_jobs()

        if idempotency_key and idempotency_key in self.idempotency_map:
            existing_id = self.idempotency_map[idempotency_key]
            if existing_id in self.jobs:
                logger.info(f"Idempotency hit for model load: reusing existing job [{existing_id}]")
                return existing_id

        if self.job_queue.qsize() >= self.max_queue_size:
            raise RuntimeError(f"Job queue is full ({self.max_queue_size} pending). Please wait for active jobs to finish.")

        job_id = f"load_{uuid.uuid4().hex[:10]}"
        now = time.time()
        job_record = {
            "id": job_id,
            "job_id": job_id,
            "idempotency_key": idempotency_key,
            "type": "model_load",
            "model_id": model_key_or_repo,
            "hf_token": hf_token,
            "status": "queued",
            "stage": "waiting",
            "progress": 0,
            "message": "Queued for GPU model loading...",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "queue_wait_seconds": None,
            "execution_duration_seconds": None,
            "duration_seconds": None,
            "error": None
        }

        self.jobs[job_id] = job_record
        if idempotency_key:
            self.idempotency_map[idempotency_key] = job_id

        try:
            self.job_queue.put_nowait(job_record)
        except queue.Full:
            self.jobs.pop(job_id, None)
            raise RuntimeError("Job queue capacity exceeded.")

        logger.info(f"Enqueued model load job [{job_id}] for model: {model_key_or_repo}")
        return job_id

    def submit_generation_job(self, req: GenerationRequest) -> str:
        self._prune_old_jobs()

        if req.idempotency_key and req.idempotency_key in self.idempotency_map:
            existing_id = self.idempotency_map[req.idempotency_key]
            if existing_id in self.jobs:
                logger.info(f"Idempotency hit for generation: reusing existing job [{existing_id}]")
                return existing_id

        if not torch.cuda.is_available():
            raise RuntimeError("GPU generation requires an active CUDA GPU.")

        if req.width % 8 != 0 or req.height % 8 != 0:
            raise ValueError(f"Width and height must be multiples of 8. Got {req.width}x{req.height}")

        # Model readiness & mismatch check
        if self.pipeline is None or self.current_model_id is None:
            raise ValueError(
                "No model is currently loaded in GPU VRAM. "
                "Automatic loading during generation is disabled. "
                "Please load a model first using POST /api/models/load (or 'node src/cli.js models load <model-id>')."
            )

        target_model = req.model_id or self.current_model_id
        configured = DEFAULT_MODELS.get(target_model)
        target_repo = configured["repo_id"] if configured else target_model

        if self.current_model_id != target_repo and self.current_model_id != target_model:
            raise ValueError(
                f"Model mismatch: active GPU model is '{self.current_model_id}', but '{target_model}' was requested. "
                f"Automatic switching during generation is disabled to prevent pipeline thrashing. "
                f"Please load '{target_model}' first via POST /api/models/load."
            )

        if self.job_queue.qsize() >= self.max_queue_size:
            raise RuntimeError(f"Job queue is full ({self.max_queue_size} pending). Please wait for active jobs to finish.")

        seed = req.seed if req.seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
        job_id = f"gen_{uuid.uuid4().hex[:10]}"
        now = time.time()

        job_record = {
            "id": job_id,
            "job_id": job_id,
            "idempotency_key": req.idempotency_key,
            "type": "generation",
            "status": "queued",
            "stage": "waiting",
            "progress": 0,
            "message": "Queued in GPU worker line...",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "queue_wait_seconds": None,
            "execution_duration_seconds": None,
            "duration_seconds": None,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": seed,
            "num_images": req.num_images or 1,
            "model_id": self.current_model_id,
            "character_id": req.character_id,
            "preset": req.preset,
            "vram": None,
            "error": None,
            "results": [],
            "image_url": None
        }

        self.jobs[job_id] = job_record
        self.abort_events[job_id] = threading.Event()
        if req.idempotency_key:
            self.idempotency_map[req.idempotency_key] = job_id

        try:
            self.job_queue.put_nowait(job_record)
        except queue.Full:
            self.jobs.pop(job_id, None)
            self.abort_events.pop(job_id, None)
            raise RuntimeError("Job queue capacity exceeded.")

        logger.info(f"Enqueued generation job [{job_id}] (model: {self.current_model_id}, seed: {seed}, count: {job_record['num_images']})")
        return job_id

    def cancel_job(self, job_id: str) -> Tuple[bool, str]:
        job = self.jobs.get(job_id)
        if not job:
            return False, f"Job '{job_id}' not found."

        current_status = job.get("status")
        if current_status in ("completed", "failed", "cancelled"):
            return False, f"Job '{job_id}' is already in terminal state '{current_status}'."

        if current_status == "queued":
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["progress"] = None
            job["message"] = "Job cancelled before execution started."
            job["completed_at"] = time.time()
            return True, f"Queued job '{job_id}' successfully cancelled."

        # Job is actively running
        if job.get("type") == "model_load":
            return False, f"Cannot safely cancel model load [{job_id}] while downloading/allocating weights."

        if job_id in self.abort_events:
            self.abort_events[job_id].set()
            job["message"] = "Interruption signal delivered to GPU worker..."
            return True, f"Cancellation requested for active job '{job_id}'."

        return False, f"Unable to interrupt running job '{job_id}'."

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {k: v for k, v in job.items() if k not in ("hf_token",)}

    def get_job_image(self, job_id: str, index: int = 0) -> Optional[bytes]:
        job = self.jobs.get(job_id)
        if not job or job.get("status") != "completed":
            return None

        # Check disk storage first
        job_dir = os.path.join(self.storage_dir, job_id)
        img_path = os.path.join(job_dir, f"result_{index}.png")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                return f.read()

        return None

    # ---------------------------------------------------------
    # Sequential Background Worker Loop (Executes 1 at a time)
    # ---------------------------------------------------------
    def _worker_loop(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break

            job_id = job["id"]
            # If job was cancelled while waiting in queue, skip it
            if job.get("status") == "cancelled":
                self.job_queue.task_done()
                continue

            self.active_running_job_id = job_id
            job_type = job["type"]
            job["status"] = "running"
            job["stage"] = "loading" if job_type == "model_load" else "diffusion"
            job["started_at"] = time.time()
            job["queue_wait_seconds"] = round(job["started_at"] - job["created_at"], 2)

            try:
                with self.gpu_lock:
                    if job_type == "model_load":
                        self._execute_model_load(job)
                    elif job_type == "generation":
                        self._execute_generation(job)
            except InterruptedError:
                logger.info(f"Job [{job_id}] was cooperatively interrupted and cancelled.")
                job["status"] = "cancelled"
                job["stage"] = "cancelled"
                job["progress"] = None
                job["message"] = "Generation was successfully interrupted and cancelled by user."
                job["completed_at"] = time.time()
                job["execution_duration_seconds"] = round(job["completed_at"] - job["started_at"], 2)
                job["duration_seconds"] = job["execution_duration_seconds"]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.error(f"Worker exception in job [{job_id}]: {exc}", exc_info=True)
                job["status"] = "failed"
                job["stage"] = "failed"
                job["error"] = str(exc)
                job["message"] = f"Execution failed: {exc}"
                job["completed_at"] = time.time()
                job["execution_duration_seconds"] = round(job["completed_at"] - job["started_at"], 2)
                job["duration_seconds"] = job["execution_duration_seconds"]
            finally:
                self.active_running_job_id = None
                self.job_queue.task_done()

    def _execute_model_load(self, job: Dict[str, Any]):
        model_key_or_repo = job["model_id"]
        hf_token = job.get("hf_token")
        model_info = DEFAULT_MODELS.get(model_key_or_repo)
        repo_id = model_info["repo_id"] if model_info else model_key_or_repo
        pipeline_type = model_info.get("pipeline") if model_info else None

        if self.current_model_id == repo_id and self.pipeline is not None:
            job["status"] = "completed"
            job["stage"] = "completed"
            job["progress"] = 100
            job["message"] = f"Model {repo_id} was already active in VRAM."
            job["completed_at"] = time.time()
            job["execution_duration_seconds"] = 0.0
            job["duration_seconds"] = 0.0
            job["vram"] = self.get_vram_metrics().dict() if self.get_vram_metrics() else None
            return

        job["status"] = "loading"
        job["stage"] = "weights_download"
        job["progress"] = 10
        job["message"] = f"Freeing previous pipeline and clearing CUDA cache..."

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.current_model_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        job["status"] = "downloading"
        job["stage"] = "weights_download"
        job["progress"] = 30

        # Check local SSD cache first, then Google Drive persistent cache
        hf_snapshot_base = os.path.join(self.cache_dir, f"models--{repo_id.replace('/', '--')}", "snapshots")
        candidates = []
        if os.path.exists(hf_snapshot_base):
            try:
                for s in sorted(os.listdir(hf_snapshot_base), reverse=True):
                    cand = os.path.join(hf_snapshot_base, s)
                    if os.path.exists(os.path.join(cand, "model_index.json")):
                        candidates.append(cand)
            except Exception:
                pass

        drive_cache = os.getenv("STUDIO_DRIVE_CACHE", "/content/drive/MyDrive/ai_image_studio_cache")
        candidates.extend([
            os.path.join(drive_cache, model_key_or_repo),
            os.path.join(drive_cache, repo_id.split("/")[-1]),
            os.path.join(drive_cache, "flux-schnell") if ("flux" in str(model_key_or_repo).lower() or "flux" in str(repo_id).lower()) else "",
            os.path.join(drive_cache, "FLUX.1-schnell"),
        ])
        drive_path = None
        for cand in candidates:
            if cand and os.path.exists(cand) and os.path.exists(os.path.join(cand, "model_index.json")):
                drive_path = cand
                break

        load_source = drive_path if drive_path else repo_id
        is_drive = drive_path is not None and not drive_path.startswith(self.cache_dir)
        self.current_load_source = load_source

        if is_drive:
            job["status"] = "loading"
            job["stage"] = "drive_cache_load"
            job["message"] = f"Loading model from Google Drive persistent cache ({drive_path})..."
            logger.info(f"Loading weights from Google Drive cache: {drive_path}")
        elif drive_path:
            job["status"] = "loading"
            job["stage"] = "local_cache_load"
            job["message"] = f"Loading model from local SSD snapshot cache ({drive_path})..."
            logger.info(f"Loading weights from local SSD snapshot: {drive_path}")
        else:
            job["message"] = f"Fetching weights for {repo_id} from Hugging Face Hub..."

        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True
        }
        if not is_drive:
            load_kwargs["cache_dir"] = self.cache_dir
            if hf_token:
                load_kwargs["token"] = hf_token

        is_flux = pipeline_type == "FluxPipeline" or "flux" in repo_id.lower()
        if is_flux:
            from diffusers import FluxPipeline
            job["message"] = f"Loading FLUX pipeline for {repo_id}..."
            flux_kwargs = {
                "torch_dtype": torch.bfloat16,
            }
            if not is_drive:
                flux_kwargs["cache_dir"] = self.cache_dir
                if hf_token:
                    flux_kwargs["token"] = hf_token

            # If repo is already a pre-quantized package (like nf4-pkg), load directly
            if "nf4-pkg" in repo_id.lower():
                pipe = FluxPipeline.from_pretrained(load_source, **flux_kwargs)
            else:
                try:
                    import bitsandbytes
                except ImportError:
                    logger.info("bitsandbytes not found in environment. Dynamically installing bitsandbytes, sentencepiece, protobuf...")
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43.0", "sentencepiece", "protobuf"], check=True)
                    import importlib
                    importlib.invalidate_caches()

                try:
                    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                    from diffusers.quantizers import PipelineQuantizationConfig
                    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

                    quant_config = PipelineQuantizationConfig(
                        quant_mapping={
                            "transformer": DiffusersBitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16
                            ),
                            "text_encoder_2": TransformersBitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16
                            ),
                        }
                    )
                    flux_kwargs["quantization_config"] = quant_config
                    pipe = FluxPipeline.from_pretrained(load_source, **flux_kwargs)
                except Exception as quant_err:
                    logger.error(f"Quantized FluxPipeline loading failed: {quant_err}")
                    raise RuntimeError(
                        f"Failed to load FLUX model with 4-bit NF4 quantization: {quant_err}. "
                        f"Unquantized 32GB fallback blocked to prevent GPU OOM on 16GB VRAM."
                    ) from quant_err

            if self.device == "cuda":
                try:
                    pipe.vae.to("cuda")
                    pipe.text_encoder.to("cuda")
                    # Explicitly enable VAE tiling and slicing directly on pipe.vae
                    if hasattr(pipe, "vae") and pipe.vae is not None:
                        if hasattr(pipe.vae, "enable_tiling"):
                            pipe.vae.enable_tiling()
                        if hasattr(pipe.vae, "enable_slicing"):
                            pipe.vae.enable_slicing()
                    if hasattr(pipe, "enable_vae_slicing"):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, "enable_vae_tiling"):
                        pipe.enable_vae_tiling()
                except Exception as e:
                    logger.warning(f"Could not configure VAE tiling/slicing: {e}")

        elif pipeline_type == "StableDiffusionXLPipeline" or "xl" in repo_id.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(load_source, **load_kwargs)
            if self.device == "cuda":
                pipe.to("cuda")
                try:
                    pipe.enable_attention_slicing(slice_size="auto")
                except Exception:
                    pass
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
        elif pipeline_type == "StableDiffusionPipeline" or "1-5" in repo_id or "1.5" in repo_id or "v5.1" in repo_id.lower():
            pipe = StableDiffusionPipeline.from_pretrained(load_source, **load_kwargs)
            if self.device == "cuda":
                pipe.to("cuda")
                try:
                    pipe.enable_attention_slicing(slice_size="auto")
                except Exception:
                    pass
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(load_source, **load_kwargs)
            if self.device == "cuda":
                pipe.to("cuda")
                try:
                    pipe.enable_attention_slicing(slice_size="auto")
                except Exception:
                    pass

        self.pipeline = pipe
        self.current_model_id = repo_id

        duration = round(time.time() - job["started_at"], 2)
        job["status"] = "completed"
        job["stage"] = "completed"
        job["progress"] = 100
        job["message"] = f"Model {repo_id} successfully loaded in VRAM."
        job["completed_at"] = time.time()
        job["execution_duration_seconds"] = duration
        job["duration_seconds"] = duration
        job["vram"] = self.get_vram_metrics().dict() if self.get_vram_metrics() else None
        logger.info(f"Model load job [{job['id']}] finished in {duration}s")

    def _execute_generation(self, job: Dict[str, Any]):
        job_id = job["id"]
        abort_event = self.abort_events.get(job_id)
        if abort_event and abort_event.is_set():
            raise InterruptedError("Generation cancelled prior to diffusion.")

        total_steps = job["steps"]
        num_images = job.get("num_images", 1)
        base_seed = job["seed"]

        job_dir = os.path.join(self.storage_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        specs = get_model_specs(str(self.current_model_id))
        is_flux = specs["family"] == "flux"

        # 1. Resolve steps with transparent reporting (schnell is distilled for 1-4 steps, max recommended 8)
        requested_steps = job.get("steps")
        step_adjustment = None
        if is_flux:
            max_schnell = specs.get("max_recommended_steps", 8)
            if requested_steps and requested_steps > max_schnell:
                actual_steps = max_schnell
                step_adjustment = f"Requested {requested_steps} steps capped to {actual_steps} (FLUX.1-schnell is guidance-distilled for 1-4 steps)."
                logger.info(f"[Inference] {step_adjustment}")
            elif requested_steps and requested_steps > 0:
                actual_steps = requested_steps
            else:
                actual_steps = specs["default_steps"]
        else:
            actual_steps = requested_steps if (requested_steps and requested_steps > 0) else specs["default_steps"]

        # 2. Resolve guidance scale (FLUX.1-schnell requires 0.0)
        requested_guidance = job.get("guidance_scale")
        guidance_adjustment = None
        if is_flux:
            actual_guidance = 0.0
            if requested_guidance is not None and requested_guidance != 0.0:
                guidance_adjustment = f"Requested CFG scale {requested_guidance} adjusted to 0.0 (FLUX.1-schnell requires guidance_scale=0.0)."
                logger.info(f"[Inference] {guidance_adjustment}")
        else:
            actual_guidance = requested_guidance if (requested_guidance is not None and requested_guidance >= 0.0) else specs["default_guidance_scale"]

        # 3. Resolve and normalize resolution (multiples of 16 for FLUX, 8 for SDXL)
        step_mult = specs.get("step_multiple", 16 if is_flux else 8)
        raw_w = job.get("width") or specs["default_width"]
        raw_h = job.get("height") or specs["default_height"]
        req_w = max(256, (int(raw_w) // step_mult) * step_mult)
        req_h = max(256, (int(raw_h) // step_mult) * step_mult)
        res_adjustment = f"Resolution adjusted from {raw_w}x{raw_h} to {req_w}x{req_h} (multiple of {step_mult})." if (raw_w != req_w or raw_h != req_h) else None

        # 4. Handle negative prompt transparency
        neg_prompt = (job.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
        neg_adjustment = None
        if not specs["supports_negative_prompt"] and neg_prompt:
            neg_adjustment = f"Negative prompt bypassed (unsupported by {specs['variant']} when guidance_scale={actual_guidance})."
            logger.info(f"[Inference] {neg_adjustment}")

        # Attach effective parameters for client visibility
        adjustments = [a for a in [step_adjustment, guidance_adjustment, res_adjustment, neg_adjustment] if a]
        effective_params = {
            "model_id": str(self.current_model_id),
            "model_variant": specs["variant"],
            "width": req_w,
            "height": req_h,
            "steps": actual_steps,
            "guidance_scale": actual_guidance,
            "max_sequence_length": specs.get("max_sequence_length", 256),
            "negative_prompt_applied": bool(specs["supports_negative_prompt"] and neg_prompt),
            "adjustments": adjustments
        }
        job["effective_parameters"] = effective_params
        job["effective_steps"] = actual_steps
        job["effective_guidance_scale"] = actual_guidance

        # Clean VRAM before generation loop
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        results = []

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if abort_event and abort_event.is_set():
                raise InterruptedError("Generation cancelled by user request.")
            pct = int((step_index + 1) / actual_steps * 100)
            job["progress"] = min(pct, 99)
            job["stage"] = "diffusion"
            job["message"] = f"Denoising step {step_index + 1}/{actual_steps} ({job['progress']}%)..."
            return callback_kwargs

        for img_idx in range(num_images):
            if abort_event and abort_event.is_set():
                raise InterruptedError("Generation cancelled during batch processing.")

            current_seed = base_seed + img_idx
            gen_device = "cpu" if is_flux else self.device
            generator = torch.Generator(device=gen_device).manual_seed(current_seed)

            if is_flux:
                kwargs = {
                    "prompt": job["prompt"],
                    "width": req_w,
                    "height": req_h,
                    "num_inference_steps": actual_steps,
                    "guidance_scale": actual_guidance,
                    "max_sequence_length": specs.get("max_sequence_length", 256),
                    "generator": generator
                }
            else:
                kwargs = {
                    "prompt": job["prompt"],
                    "negative_prompt": neg_prompt if specs["supports_negative_prompt"] else None,
                    "width": req_w,
                    "height": req_h,
                    "num_inference_steps": actual_steps,
                    "guidance_scale": actual_guidance,
                    "generator": generator
                }

            try:
                kwargs["callback_on_step_end"] = step_callback
            except Exception:
                pass

            # Structured diagnostics logging (No secrets/tokens logged)
            logger.info(
                f"[Inference] Job [{job_id}] Image {img_idx + 1}/{num_images} -> "
                f"Model: {self.current_model_id} ({specs['variant']}) | Source: {self.current_load_source or 'Drive/Hub'} | "
                f"Res: {req_w}x{req_h} | Steps: {actual_steps} | Guidance: {actual_guidance} | Seed: {current_seed}"
            )
            logger.info(f"[Inference] Executed Prompt: {kwargs['prompt']!r}")
            if kwargs.get("negative_prompt"):
                logger.info(f"[Inference] Executed Negative Prompt: {kwargs['negative_prompt']!r}")

            job["status"] = "running"
            job["stage"] = "diffusion"
            job["message"] = f"Starting diffusion loop (image {img_idx + 1}/{num_images})..."

            # Ensure VAE tiling and slicing are active before diffusion and decoding
            if hasattr(self.pipeline, "vae") and self.pipeline.vae is not None:
                try:
                    if hasattr(self.pipeline.vae, "enable_tiling"):
                        self.pipeline.vae.enable_tiling()
                    if hasattr(self.pipeline.vae, "enable_slicing"):
                        self.pipeline.vae.enable_slicing()
                except Exception:
                    pass

            try:
                output = self.pipeline(**kwargs)
                image: Image.Image = output.images[0]

                job["stage"] = "encoding"
                job["message"] = f"Saving result image {img_idx + 1}/{num_images} to disk..."

                file_path = os.path.join(job_dir, f"result_{img_idx}.png")
                image.save(file_path, format="PNG")

                result_item = {
                    "index": img_idx,
                    "seed": current_seed,
                    "url": f"/api/jobs/{job_id}/result?index={img_idx}",
                    "width": req_w,
                    "height": req_h
                }
                results.append(result_item)

                del output
                del image
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as oom:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                job["status"] = "failed"
                job["stage"] = "failed"
                job["error"] = "CUDA Out Of Memory during generation. Lower resolution or reduce step count."
                job["completed_at"] = time.time()
                job["execution_duration_seconds"] = round(job["completed_at"] - job["started_at"], 2)
                job["duration_seconds"] = job["execution_duration_seconds"]
                raise RuntimeError(job["error"]) from oom

        duration = round(time.time() - job["started_at"], 2)
        vram_metrics = self.get_vram_metrics()
        job["status"] = "completed"
        job["stage"] = "completed"
        job["progress"] = 100
        job["message"] = f"Generation completed successfully ({num_images} image{'s' if num_images > 1 else ''} in {duration}s)."
        job["completed_at"] = time.time()
        job["execution_duration_seconds"] = duration
        job["duration_seconds"] = duration
        job["vram"] = vram_metrics.dict() if vram_metrics else None
        job["results"] = results
        job["image_url"] = f"/api/jobs/{job_id}/result?index=0"

        # Structured diagnostics output
        job["diagnostics"] = {
            "model_id": str(self.current_model_id),
            "model_variant": specs["variant"],
            "checkpoint_source": self.current_load_source or "Drive/Hub",
            "resolution": f"{req_w}x{req_h}",
            "inference_steps": actual_steps,
            "guidance_scale": actual_guidance,
            "seed": base_seed,
            "duration_seconds": duration,
            "prompt_length_chars": len(job["prompt"]),
            "vram": job["vram"]
        }
        logger.info(f"Generation job [{job['id']}] finished in {duration}s")
