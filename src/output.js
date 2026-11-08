import fs from 'node:fs';
import path from 'node:path';
import { paths } from './config.js';

/**
 * Generates an ISO-like timestamp safe for filenames: YYYYMMDD_HHMMSS
 */
function getTimestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const min = pad(d.getMinutes());
  const ss = pad(d.getSeconds());
  return `${yyyy}${mm}${dd}_${hh}${min}${ss}`;
}

/**
 * Saves generation image binary and metadata sidecar file atomically.
 */
export async function saveGenerationOutput(imageBuffer, metadata = {}) {
  const timestamp = getTimestamp();
  const id = metadata.job_id || metadata.id || `gen_${Date.now().toString(36)}`;
  const indexSuffix = metadata.index !== undefined && metadata.index !== null ? `_${metadata.index}` : '';
  const baseName = `${timestamp}_${id}${indexSuffix}`;

  const imagePath = path.join(paths.outputs, `${baseName}.png`);
  const metaPath = path.join(paths.outputs, `${baseName}.json`);

  // Enrich metadata
  const fullMetadata = {
    id,
    job_id: id,
    index: metadata.index ?? 0,
    timestamp: new Date().toISOString(),
    image_file: path.basename(imagePath),
    metadata_file: path.basename(metaPath),
    prompt: metadata.prompt,
    negative_prompt: metadata.negative_prompt,
    width: metadata.width,
    height: metadata.height,
    steps: metadata.steps,
    guidance_scale: metadata.guidance_scale,
    seed: metadata.seed,
    model_id: metadata.model_id,
    preset: metadata.preset || null,
    character_id: metadata.character_id || null,
    queue_wait_seconds: metadata.queue_wait_seconds ?? null,
    execution_duration_seconds: metadata.execution_duration_seconds ?? metadata.duration_seconds ?? null,
    duration_seconds: metadata.duration_seconds ?? metadata.execution_duration_seconds ?? null,
    vram: metadata.vram || null
  };

  // Write image buffer
  await fs.promises.writeFile(imagePath, Buffer.from(imageBuffer));

  // Write JSON sidecar
  await fs.promises.writeFile(metaPath, JSON.stringify(fullMetadata, null, 2), 'utf8');

  return {
    imagePath,
    metaPath,
    metadata: fullMetadata
  };
}

/**
 * Reads local generation history by scanning the outputs directory.
 */
export async function getGenerationHistory(limit = 20) {
  if (!fs.existsSync(paths.outputs)) {
    return [];
  }

  const files = await fs.promises.readdir(paths.outputs);
  const jsonFiles = files
    .filter((f) => f.endsWith('.json'))
    .sort()
    .reverse();

  const history = [];
  for (const f of jsonFiles.slice(0, limit)) {
    try {
      const content = await fs.promises.readFile(path.join(paths.outputs, f), 'utf8');
      const data = JSON.parse(content);
      history.push(data);
    } catch (e) {
      // Skip corrupt sidecar files
    }
  }

  return history;
}
