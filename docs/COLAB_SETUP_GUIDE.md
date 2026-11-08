# Google Colab Setup & Deployment Guide

This guide explains how to spin up the GPU inference backend on Google Colab and link it securely with your local JavaScript CLI.

---

## Prerequisites

1. A Google account to access [Google Colab](https://colab.research.google.com).
2. Free or Colab Pro subscription with access to a GPU runtime (Tesla T4, V100, or A100).
3. (Optional) A Hugging Face account only if accessing gated models (public models like SDXL 1.0 and SD 1.5 do not require any token).

---

## Step 1: Open the Notebook in Google Colab

1. Navigate to Google Colab: https://colab.research.google.com/
2. Click **Upload** and upload the file from this repository:
   `notebooks/colab_setup.ipynb`
3. Alternatively, open a fresh notebook and run the setup script cells.

---

## Step 2: Select a GPU Runtime

1. In the Colab menu, go to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or A100 if you have Colab Pro).
3. Click **Save**.
4. Run the first cell to verify GPU availability:
   ```bash
   !nvidia-smi
   ```
   You should see output indicating an NVIDIA Tesla T4 with ~15 GB VRAM.

---

## Step 3: Run the Colab Setup Cell

The notebook will:
1. Clone or download the backend Python scripts:
   - `backend/server.py`
   - `backend/inference.py`
   - `backend/models.py`
   - `backend/requirements.txt`
2. Install the optimized PyTorch & Diffusers stack:
   ```bash
   pip install -q diffusers transformers accelerate safetensors fastapi uvicorn pydantic python-multipart
   ```
3. Install Cloudflare Tunnel (`cloudflared` binary) for a zero-configuration, secure HTTPS reverse tunnel.
4. Generate a random API token (`STUDIO_API_TOKEN`) or let you specify one.
5. Launch Uvicorn in the background on port `8000`.
6. Launch Cloudflare Tunnel and print your public endpoint.

---

## Step 4: Configure Your Local Environment

When the notebook completes the tunnel startup cell, it will output:

```text
======================================================================
 AI Image Studio Backend is Live!
======================================================================
 Public Tunnel URL: https://bright-random-words.trycloudflare.com
 Studio API Token : sk_studio_a7b8c9d0e1f2
======================================================================
 Copy the above into your local .env file:
 STUDIO_API_URL=https://bright-random-words.trycloudflare.com
 STUDIO_API_TOKEN=sk_studio_a7b8c9d0e1f2
======================================================================
```

On your local machine in the `ai-image-studio` directory, copy `.env.example` to `.env` and paste the values:

```bash
cp .env.example .env
```

Edit `.env`:
```env
STUDIO_API_URL=https://bright-random-words.trycloudflare.com
STUDIO_API_TOKEN=sk_studio_a7b8c9d0e1f2
```

---

## Step 5: Test the Connection Locally

Run the CLI doctor command:

```bash
node src/cli.js doctor
```

Output:
```text
[✓] Node.js version v22.x.x
[✓] Configuration files verified
[✓] Backend connected: https://bright-random-words.trycloudflare.com
[✓] GPU detected: Tesla T4 (VRAM: 14.75 GB Total, 7.25 GB Free)
[✓] Active model: stabilityai/stable-diffusion-xl-base-1.0
System ready for generation!
```

---

## Step 6: Generate Your First Image

```bash
node src/cli.js generate --prompt "Photorealistic cinematic portrait of a fictional 28-year-old woman, emerald green eyes, golden hour lighting" --preset portrait
```

The image and its companion JSON metadata will be downloaded to `outputs/`.

---

## Troubleshooting & FAQ

### 1. Colab Disconnects / Idle Timeout
Colab free tier can disconnect after periods of inactivity. If disconnected, simply re-run the notebook cells. The Hugging Face cache persists during the session on `/content/huggingface_cache` so model weights will not re-download if the disk is still mounted.

### 2. CUDA Out of Memory (OOM)
If you attempt resolutions beyond 1024x1024 or higher batch sizes, the server will catch the OOM error and return:
```json
{
  "error": "CUDA_OOM",
  "message": "CUDA out of memory. Try lowering width/height to 768 or 1024."
}
```
The server clears PyTorch cache automatically and remains online.

### 3. Alternative Tunnels
If Cloudflare Tunnel is blocked by your network, the notebook also provides an alternative cell using `ngrok` (requires a free ngrok authtoken).

