# CLI Reference: AI Image Studio (v2.1)

The AI Image Studio CLI (`src/cli.js`) provides complete command-line control over asynchronous GPU image generation, model switching, background job management, photography presets, and character consistency.

## Environment Variables

| Variable | Description | Default | Required |
|---|---|---|---|
| `STUDIO_API_URL` | Base URL of the backend (e.g. `https://xxx.trycloudflare.com` or `http://localhost:8000`) | `http://localhost:8000` | Yes (for remote Colab) |
| `STUDIO_API_TOKEN` | Bearer authentication token configured on the server | `""` | Yes (when auth is active) |
| `STUDIO_OUTPUT_DIR`| Local output path for downloaded PNG images and JSON metadata | `./outputs` | No |
| `STUDIO_CHARACTERS_DIR`| Local directory storing character JSON profiles | `./characters` | No |

These variables can be placed in a `.env` file at the root of the project.

---

## Commands

### 1. `doctor`
Performs health diagnostics against local environment, config files, and the remote GPU backend.

```bash
node src/cli.js doctor
```

**Checks executed**:
- Node.js runtime version (>= 20.0.0).
- Presence of `config/models.json` and `config/generation.json`.
- Accessibility of `outputs/` and `characters/` directories.
- Ping backend `/api/health` endpoint.
- Report GPU availability, device model (e.g. Tesla T4), and allocated/free VRAM.
- Distinct 4-condition generation readiness breakdown:
  1. Backend Online
  2. GPU Available
  3. Model Loaded
  4. Generation Ready

---

### 2. `models`
Inspects models configured in the backend and switches the active model on the GPU.

```bash
# List available and currently loaded models
node src/cli.js models

# Load a specific model on the GPU
node src/cli.js models load sd-1.5
node src/cli.js models load sdxl-base-1.0
node src/cli.js models load ssd-1b
```

---

### 3. `jobs`
Inspects, monitors, cancels, and downloads asynchronous GPU background jobs.

```bash
# List recent jobs running or completed on the backend
node src/cli.js jobs

# Inspect detailed status, progress %, and stages of a specific job
node src/cli.js jobs status gen_a1b2c3d4e5

# Cancel an active or queued job
node src/cli.js jobs cancel gen_a1b2c3d4e5

# Re-download result image from a completed job
node src/cli.js jobs download gen_a1b2c3d4e5 [--index 0]
```

---

### 4. `generate`
Submits a text-to-image generation job, monitors progress without holding long HTTP requests open, downloads the finished PNG, and saves the companion JSON metadata sidecar.

```bash
# Basic prompt with photography preset
node src/cli.js generate --prompt "Fictional fashion model in Paris" --preset beauty-portrait

# Custom parameters and model selection
node src/cli.js generate \
  --prompt "Athletic fitness portrait in modern gym" \
  --preset fitness-glamour \
  --model sd-1.5 \
  --num-images 2 \
  --timeout 600

# Available photography presets:
# - beauty-portrait     (extreme close-up beauty, 85mm lens, natural skin microtextures)
# - luxury-fashion      (medium-length haute couture editorial, 50mm lens)
# - fitness-glamour     (athletic definition, rim lighting, 70-200mm lens)
# - evening-editorial   (evening gown gala, chandelier bokeh, 35mm lens)
# - beach-fashion       (sun-drenched resortwear, ocean breeze, golden hour rim)
# - cinematic-portrait  (35mm anamorphic film still, chiaroscuro lighting)
# - natural-lifestyle   (authentic documentary candid, soft window daylight)
# - studio-full-body    (full-length standing stance, 70mm lens, cyclorama backdrop)

# Generate using a reusable character profile
node src/cli.js generate \
  --character elena-vance \
  --scene "sitting at an outdoor café in Rome during golden hour" \
  --preset luxury-fashion

# Generate using a JSON job file
node src/cli.js generate --config my-job.json
```

---

### 5. `photoshoot`
Runs a multi-shot professional photography series for a character across distinct camera framings and lighting setups.

```bash
node src/cli.js photoshoot elena-vance --theme "autumn luxury street style" --shots 5
```

**Standard Shots executed**:
1. Beauty Close-Up Portrait (`beauty-portrait`)
2. Medium Fashion Portrait (`luxury-fashion`)
3. Full-Body Standing Pose (`studio-full-body`)
4. Walking Lifestyle Photograph (`natural-lifestyle`)
5. Cinematic Environmental Photograph (`cinematic-portrait`)

---

### 6. `character`
Manages reusable adult character profiles (strictly 18+).

```bash
# List all saved character profiles
node src/cli.js character list

# Display full profile details
node src/cli.js character show elena-vance

# Interactively create a new character profile
node src/cli.js character create

# Create non-interactively via flags
node src/cli.js character create \
  --id "marcus-reid" \
  --name "Marcus Reid" \
  --age 32 \
  --facial "strong jawline, subtle stubble" \
  --eyes "blue-grey"
```

---

### 7. `history`
Lists downloaded outputs and metadata stored locally in `outputs/`.

```bash
node src/cli.js history [--limit 20]
```

---

### 8. `serve`
Launches the local browser-based studio UI on `http://localhost:3000`.

```bash
node src/cli.js serve [--port 3000]
```
