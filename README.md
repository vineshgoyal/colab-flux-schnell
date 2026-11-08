# 🎨 AI Image Studio (colab-flux-schnell)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vineshgoyal/colab-flux-schnell/blob/main/notebooks/colab_flux_schnell.ipynb)

> 💡 **Persistent Colab Notebook**: Click the badge above to open the saved notebook directly in Google Colab.
> Click **File ➔ Save a copy in Drive** in Colab to permanently keep this notebook in your Google Account. Weights and outputs are persisted to Google Drive so you never need to recreate files or redownload models on demand.

Production-quality AI image-generation studio combining a high-performance **Google Colab GPU inference backend** (PyTorch, Hugging Face Diffusers, FastAPI, Cloudflare Tunnel) with a zero-dependency **JavaScript (Node.js 20+) CLI client**, character consistency engine, and browser studio UI.

Executes public Hugging Face model weights directly on the assigned GPU (NVIDIA Tesla T4 / A100 / V100) using FP16 precision, model CPU offloading, attention slicing, and VAE tiling—with **zero dependence** on paid third-party APIs such as fal.ai or Hugging Face hosted inference endpoints.

---

## 📑 Detailed Specifications

Comprehensive architectural and engineering specifications are located in the [`docs/`](file:///Users/vinesh/Documents/project/movie-gen-colab/docs) directory:
- [**Architecture Specification**](file:///Users/vinesh/Documents/project/movie-gen-colab/docs/ARCHITECTURE.md): System components, data flows, sequence diagrams, and security architecture.
- [**API Specification**](file:///Users/vinesh/Documents/project/movie-gen-colab/docs/API_SPECIFICATION.md): Full REST contract (`/health`, `/models`, `/models/load`, `/generate`, `/generation/{id}`, `/generation/{id}/image`), Bearer token authentication, status codes, and JSON error schemas.
- [**Model Management**](file:///Users/vinesh/Documents/project/movie-gen-colab/docs/MODEL_MANAGEMENT.md): Tesla T4 VRAM budget (~15 GB), FP16 precision, model CPU offload, attention slicing, VAE tiling, and OOM recovery protocol.
- [**Character Consistency System**](file:///Users/vinesh/Documents/project/movie-gen-colab/docs/CHARACTER_SYSTEM.md): Reusable adult fictional persona schema (strictly 18+), deterministic seed anchoring, and multi-scene prompt synthesis.
- [**CLI Reference**](file:///Users/vinesh/Documents/project/movie-gen-colab/docs/CLI_REFERENCE.md): Command-line reference for `doctor`, `models`, `generate`, `character`, `photoshoot`, `history`, and `serve`.
- [**Google Colab Setup Guide**](file:///Users/vinesh/Documents/project/movie-gen-colab/docs/COLAB_SETUP_GUIDE.md): Step-by-step Colab GPU provisioning and authenticated Cloudflare Tunnel deployment.

---

## 🚀 Live GPU Verification & Performance Benchmarks

The stack is verified on an **NVIDIA Tesla T4 GPU (14.56 GB VRAM)** on Google Colab:

| Metric / Parameter | Live Verified Measurement |
|---|---|
| **GPU Hardware** | NVIDIA Tesla T4 (CUDA 13.0, 14.56 GB VRAM) |
| **Model Evaluated** | `runwayml/stable-diffusion-v1-5` (FP16 with CPU offload) |
| **Model Weight Download & Allocation** | ~108s (one-time download from Hugging Face Hub, cached in `/content/huggingface_cache`) |
| **Active VRAM Consumption** | **0.58 GB** allocated / 14.00 GB free |
| **Inference Resolution & Steps** | 512 x 512 pixels, 25 denoising steps |
| **GPU Computation Duration** | **10.8s** per image |
| **Total Turnaround (including tunnel transfer)** | **15.3s** (End-to-end CLI request to local PNG write) |
| **Output Image & Metadata** | 457 KB PNG + synchronized JSON metadata sidecar |

---

## 🛠️ Project Structure

```text
ai-image-studio/
├── README.md
├── package.json                 # ES module config, node >= 20.0.0, zero runtime dependencies
├── .env.example                 # Environment configuration template
├── config/
│   ├── models.json              # Model catalog (SDXL 1.0, SD 1.5, SSD-1B, SDXL Turbo)
│   └── generation.json          # Aspect ratio presets, parameter limits, negative prompts
├── src/
│   ├── cli.js                   # Executable CLI client with subcommands
│   ├── api-client.js            # Native Fetch API client with Bearer auth & timeout handling
│   ├── config.js                # Configuration, zero-dep .env loader, parameter resolver
│   ├── characters.js            # Character engine with strict 18+ adult validation
│   └── output.js                # Atomic image PNG and JSON sidecar writer, history scanner
├── backend/
│   ├── server.py                # FastAPI HTTP REST gateway with Bearer auth
│   ├── inference.py             # Diffusers pipeline manager, FP16, offload, OOM safety
│   ├── models.py                # Pydantic schemas and model registry
│   └── requirements.txt         # Pinned Python GPU dependencies
├── notebooks/
│   └── colab_setup.ipynb        # Ready-to-run Google Colab GPU notebook
├── characters/
│   ├── elena-vance.json         # Example fictional adult female profile
│   └── marcus-reid.json         # Example fictional adult male profile
├── docs/                        # Detailed architectural and API specifications
├── public/
│   └── index.html               # Lightweight single-page browser studio UI
├── outputs/                     # Downloaded PNG images and JSON metadata sidecars
└── tests/                       # Automated test suite using Node.js native test runner
```

---

## ⚡ Quick Start Guide

### Step 1: Open Google Colab & Select T4 GPU

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the local notebook: [`notebooks/colab_setup.ipynb`](file:///Users/vinesh/Documents/project/movie-gen-colab/notebooks/colab_setup.ipynb).
3. In Colab top menu: click **Runtime** > **Change runtime type**.
4. Select **T4 GPU** and click **Save**.
5. Click **Runtime** > **Run all**.

The final cell starts the FastAPI server, creates an encrypted Cloudflare Tunnel, and outputs:
```text
======================================================================
🚀 AI Image Studio Backend is Live on Tesla T4 GPU!
======================================================================
Public Tunnel URL: https://example-tunnel.trycloudflare.com
Studio API Token : vinesh123
======================================================================
```

> [!IMPORTANT]
> Use the **`Public Tunnel URL`** (`https://...trycloudflare.com`) printed in the cell output. Do **not** use the Colab browser URL from your browser address bar.

---

### Step 2: Configure Local Workstation

```bash
# 1. Copy environment template (if not already done)
cp .env.example .env
```

Edit `.env` with your tunnel URL and token:
```env
STUDIO_API_URL=https://example-tunnel.trycloudflare.com
STUDIO_API_TOKEN=vinesh123
STUDIO_OUTPUT_DIR=./outputs
STUDIO_CHARACTERS_DIR=./characters
```

---

### Step 3: Verify Health with the Doctor Command

```bash
node src/cli.js doctor
```

Output:
```text
======================================================
           AI Image Studio: System Doctor             
======================================================
[✓] Node.js runtime: v22.x.x (compatible >= 20.0.0)
[✓] Output directory:     /path/to/outputs
[✓] Characters directory: /path/to/characters
[✓] Configured models:    4 loaded from config/models.json
[✓] Generation defaults:  verified

Connecting to backend:    https://example-tunnel.trycloudflare.com
Configured Auth Token:    vines...h123
[✓] Backend HTTP Status:   Healthy & Connected
[✓] GPU Acceleration:     YES (Tesla T4, Devices: 1)
[✓] CUDA Version:          13.0
[✓] VRAM Status:           0.58 GB allocated / 14.56 GB total (14.00 GB free)
[✓] Loaded Model:          runwayml/stable-diffusion-v1-5 (Active in memory)

------------------------------------------------------
              Generation Readiness Status             
------------------------------------------------------
STATUS: [READY] Fully operational and ready for GPU generation!
Active Model: runwayml/stable-diffusion-v1-5
```

---

## 🎨 CLI Command Reference

### 1. Model Inspection & Switching

```bash
# List configured models
node src/cli.js models

# Load or switch active model in GPU VRAM
node src/cli.js models load sd-1.5
node src/cli.js models load sdxl-base-1.0
node src/cli.js models load ssd-1b
```

---

### 2. Image Generation & Photography Presets

```bash
# Text prompt with glamour/fashion photography preset
node src/cli.js generate --prompt "Fictional fashion model in Paris" --preset beauty-portrait

# Custom dimensions, steps, seed, and batch size
node src/cli.js generate \
  --prompt "Athletic fitness portrait in modern gym" \
  --preset fitness-glamour \
  --model sd-1.5 \
  --num-images 2 \
  --steps 25 \
  --seed 12345

# Reusable Photography Presets:
#  • beauty-portrait     (extreme close-up beauty, 85mm f/1.4 lens, natural skin microtextures)
#  • luxury-fashion      (medium-length haute couture editorial, 50mm f/2.0 lens)
#  • fitness-glamour     (athletic definition, rim lighting, 70-200mm f/2.8 telephoto)
#  • evening-editorial   (evening gown gala, chandelier bokeh, 35mm f/1.4 lens)
#  • beach-fashion       (sun-drenched resortwear, ocean breeze, golden hour rim)
#  • cinematic-portrait  (35mm anamorphic film still, chiaroscuro lighting, Vision3 grain)
#  • natural-lifestyle   (authentic documentary candid, soft window daylight)
#  • studio-full-body    (full-length standing stance, 70mm lens, cyclorama backdrop)

# Generate using a JSON job file
node src/cli.js generate --config my-job.json
```

Outputs are automatically saved locally:
- `outputs/YYYYMMDD_HHMMSS_gen_xxxxxxxxxx.png` (Raw image)
- `outputs/YYYYMMDD_HHMMSS_gen_xxxxxxxxxx.json` (Companion metadata sidecar with prompt, seed, model, execution duration, and GPU VRAM usage)

---

### 3. Background Job Management

Track, monitor, cancel, and retrieve asynchronous GPU background jobs:

```bash
# List recent jobs running or completed on the backend
node src/cli.js jobs

# Inspect job progress percentage and stages
node src/cli.js jobs status gen_a1b2c3d4e5

# Cancel an active or queued job
node src/cli.js jobs cancel gen_a1b2c3d4e5

# Re-download a result image from a completed job
node src/cli.js jobs download gen_a1b2c3d4e5 [--index 0]
```

---

### 4. Character Consistency Engine

Generate consistent imagery of reusable adult fictional personas across varying scenes, poses, and outfits without training LoRA weights:

```bash
# List saved character profiles
node src/cli.js character list

# View character profile details
node src/cli.js character show elena-vance

# Generate persona in dynamic scene (locks facial structure & features)
node src/cli.js generate \
  --character elena-vance \
  --scene "sipping espresso at an outdoor Parisian cafe during golden hour" \
  --clothing "minimalist black cashmere turtleneck" \
  --preset luxury-fashion

# Run an automated 5-shot consistent photoshoot series:
# (1. Beauty Close-Up, 2. Medium Fashion, 3. Full-Body Standing, 4. Walking Lifestyle, 5. Cinematic Environmental)
node src/cli.js photoshoot elena-vance --theme "autumn luxury street style" --shots 5

# Create a new character profile interactively
node src/cli.js character create
```

---

### 4. Generation History

```bash
# View recent generations
node src/cli.js history

# View last 25 generations
node src/cli.js history --limit 25
```

---

### 5. Interactive Local Web Studio UI

```bash
node src/cli.js serve --port 3000
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser to access the graphical studio, adjust parameters with sliders, select character profiles, and preview generated images.

---

## 🧪 Running Automated Tests

AI Image Studio utilizes Node.js's native test runner (`node:test`) with **zero external dependencies**:

```bash
npm test
```

Test coverage:
```text
✔ API Client: connects to server, handles Bearer auth and asynchronous polling flow
✔ Characters: rejects profile missing adult age or underage (<18)
✔ Characters: validates valid adult character profile
✔ Characters: synthesizes character prompt combining identity, traits, and scene
✔ Characters: can read pre-existing character profiles from disk
✔ CLI: --help prints command reference with jobs and presets
✔ CLI: character list displays saved profiles
✔ CLI: character show outputs full json
✔ CLI: character create via flags creates and validates profile
✔ CLI: doctor reports local health metrics cleanly
✔ Config: modelsConfig loads default model and list
✔ Config: resolveGenerationParams applies portrait preset by default
✔ Config: resolveGenerationParams applies square preset
✔ Config: resolveGenerationParams rounds dimensions to multiples of 8 and clamps bounds
✔ Config: resolveGenerationParams applies photography presets with model-specific resolution
✔ Output: saveGenerationOutput writes image and companion json metadata
✔ Acceptance Suite: Asynchronous Job Lifecycle, Idempotency, Model Mismatch, & Cloudflare 524 Protection
ℹ tests 17, pass 17, fail 0
```

---

## 🛡️ Safety & Policy Compliance

- **Adult Fictional Subjects Only**: Character profile schemas strictly enforce an explicit adult age (`18+`). Profiles lacking or below age 18 are rejected.
- **Child Protection**: Generation of imagery depicting minors in sexual or non-consensual contexts is strictly forbidden.
- **No Non-Consensual Real Person Imagery**: Built exclusively for fictional characters and authorized figure studies; do not generate intimate depictions of identifiable real people.
- **Open Weights Licensing**: All models comply with their respective open licenses (CreativeML OpenRAIL++-M, CreativeML OpenRAIL-M, Apache 2.0).
- **Secure Authentication**: Backend endpoints enforce `STUDIO_API_TOKEN` Bearer authentication. Unauthenticated public endpoints are disallowed.

---

## 💻 Cross-Platform Compatibility (macOS / Linux / Windows)

- **Node.js**: Requires Node.js 20.0.0 or higher.
- **Zero NPM Dependencies**: Built using ES Modules and standard Node built-in modules (`node:fs`, `node:path`, `node:crypto`, `node:http`, `node:readline`, native Fetch API). `npm install` requires no external packages.

# colab-flux-schnell
