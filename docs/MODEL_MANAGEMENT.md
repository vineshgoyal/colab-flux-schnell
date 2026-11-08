# Model Management Specification: AI Image Studio

## 1. Overview & GPU Constraints

Google Colab free and standard runtime typically provisions an **NVIDIA Tesla T4 GPU (15.0 GB to 16.0 GB VRAM)**. 
Running large diffusion models such as Stable Diffusion XL Base 1.0 (6.6 billion parameters total, ~6.6 GB FP16 weights) at 1024x1024 resolution creates significant VRAM pressure when intermediate attention maps and VAE decoding buffers are allocated.

This document details the memory budget, pipeline loading strategies, optimization mechanisms, and caching rules.

---

## 2. Memory Budget on Tesla T4 (15 GB VRAM)

| State / Model | Base Weights (FP16) | Peak VRAM (Standard) | Peak VRAM with Optimizations | T4 Feasibility |
|---|---|---|---|---|
| **SD 1.5** (512x512) | ~3.4 GB | ~4.5 GB | ~3.0 GB | Excellent |
| **SSD-1B** (1024x1024) | ~4.8 GB | ~8.0 GB | ~5.5 GB | Excellent |
| **SDXL Base 1.0** (1024x1024) | ~6.6 GB | ~13.5 GB | ~7.2 GB | Fully Supported |
| **SDXL Base 1.0** (1280x1280) | ~6.6 GB | > 16.0 GB (OOM) | ~11.5 GB | Supported with VAE Tiling |

---

## 3. Supported Model Catalog

The system ships with default support for the following verified public models:

### 3.1 Stable Diffusion XL Base 1.0
- **Hugging Face ID**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Architecture**: `StableDiffusionXLPipeline`
- **Default Resolution**: 1024 x 1024
- **Precision**: `torch.float16`
- **Variant**: `fp16`
- **Safetensors**: Required (`use_safetensors=True`)
- **License**: CreativeML OpenRAIL++-M License

### 3.2 Stable Diffusion 1.5
- **Hugging Face ID**: `runwayml/stable-diffusion-v1-5`
- **Architecture**: `StableDiffusionPipeline`
- **Default Resolution**: 512 x 512 (can upscale to 768x768)
- **Precision**: `torch.float16`
- **Safetensors**: Required
- **License**: CreativeML OpenRAIL-M

### 3.3 Segmind SSD-1B (Distilled SDXL)
- **Hugging Face ID**: `segmind/SSD-1B`
- **Architecture**: `StableDiffusionXLPipeline`
- **Default Resolution**: 1024 x 1024
- **Benefit**: 50% faster inference and 40% lower VRAM usage than SDXL Base.
- **License**: Apache 2.0 compatible weights.

---

## 4. Diffusers Pipeline Optimization Suite

When loading SDXL or SD 1.5 on Colab GPUs, the following optimizations are applied systematically in `backend/inference.py`:

```python
# 1. FP16 Weight Precision
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_repo_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=cache_directory
)

# 2. Model CPU Offloading
# Offloads submodules to CPU RAM when inactive, streaming to GPU only when computed
pipe.enable_model_cpu_offload()

# 3. Attention Slicing
# Splits cross-attention computation into sequential slices to reduce peak allocation
pipe.enable_attention_slicing(slice_size="auto")

# 4. VAE Slicing & Tiling
# Prevents OOM during high-resolution decoding step
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

---

## 5. Weights Caching Strategy

- Downloaded models are stored in `/content/huggingface_cache` (or custom `$HF_HOME`).
- Once downloaded, Diffusers verifies existing cached files by commit hash without redownloading.
- If a user switches between models during a session, existing weights remain on disk and load in seconds without network retransfer.
- Only one model pipeline resides in active GPU VRAM at any time. When switching models, the previous pipeline is unloaded, garbage collection is triggered, and `torch.cuda.empty_cache()` is called before allocating the new model.

---

## 6. OOM Recovery Protocol

If an out-of-memory error occurs (e.g. attempting 1536x1536 on T4):

1. Catch `torch.cuda.OutOfMemoryError`.
2. Clear CUDA cache: `torch.cuda.empty_cache()` and `gc.collect()`.
3. Mark job as failed with specific status code `CUDA_OOM`.
4. Suggest remedies in the error payload:
   - Lower generation resolution (e.g., from 1024x1024 to 768x1024 or 512x512).
   - Ensure VAE tiling is active.
   - Switch to distilled model `segmind/SSD-1B`.
5. The server process remains operational and ready for subsequent requests without restarting.

