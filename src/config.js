import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

/**
 * Lightweight, zero-dependency .env parser.
 * Reads .env file if it exists and loads keys into process.env if not already set.
 */
export function loadEnv(envPath = path.join(ROOT_DIR, '.env')) {
  if (fs.existsSync(envPath)) {
    try {
      const content = fs.readFileSync(envPath, 'utf8');
      const lines = content.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const eqIdx = trimmed.indexOf('=');
        if (eqIdx !== -1) {
          const key = trimmed.slice(0, eqIdx).trim();
          let value = trimmed.slice(eqIdx + 1).trim();
          // Remove wrapping quotes if present
          if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }
          if (process.env[key] === undefined) {
            process.env[key] = value;
          }
        }
      }
    } catch (err) {
      console.warn(`[Config] Notice: Could not read .env at ${envPath}: ${err.message}`);
    }
  }
}

// Load env upon import
loadEnv();

/**
 * Loads JSON config file safely with fallback.
 */
function loadJson(filePath, fallback = {}) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }
  } catch (err) {
    console.error(`[Config] Error loading ${filePath}: ${err.message}`);
  }
  return fallback;
}

export const modelsConfig = loadJson(path.join(ROOT_DIR, 'config', 'models.json'), {
  default_model: 'sdxl-base-1.0',
  models: []
});

export const DEFAULT_NEGATIVE_PROMPT = 'worst quality, low quality, blurry, out of focus, pixelated, low resolution, cartoon, anime, illustration, CGI, 3D render, plastic skin, waxy skin, overly smooth skin, artificial appearance, unrealistic anatomy, distorted body proportions, malformed body, deformed face, asymmetrical eyes, crossed eyes, poorly drawn hands, extra fingers, missing fingers, fused fingers, extra arms, extra legs, duplicate body parts, unnatural pose, twisted limbs, broken anatomy, floating limbs, cropped head, cropped feet, incomplete body, cut-off legs, unnatural lighting, harsh shadows, overexposed, underexposed, oversaturated, excessive skin smoothing, distorted clothing, unnatural fabric, text, watermark, logo, signature';

export const generationConfig = loadJson(path.join(ROOT_DIR, 'config', 'generation.json'), {
  defaults: { steps: 30, guidance_scale: 7.0, format: 'png', preset: 'portrait', batch_size: 1 },
  parameter_limits: {
    width: { min: 256, max: 1536, step: 8 },
    height: { min: 256, max: 1536, step: 8 },
    steps: { min: 10, max: 60 },
    guidance_scale: { min: 1.0, max: 15.0 },
    batch_size: { min: 1, max: 4 }
  },
  aspect_presets: {
    portrait: { width: 768, height: 1024 }
  },
  photography_presets: {},
  default_negative_prompt: DEFAULT_NEGATIVE_PROMPT
});

export const paths = {
  root: ROOT_DIR,
  outputs: path.resolve(ROOT_DIR, process.env.STUDIO_OUTPUT_DIR || './outputs'),
  characters: path.resolve(ROOT_DIR, process.env.STUDIO_CHARACTERS_DIR || './characters'),
  config: path.join(ROOT_DIR, 'config'),
  docs: path.join(ROOT_DIR, 'docs')
};

// Ensure directories exist
for (const dir of [paths.outputs, paths.characters]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

export const serverConfig = {
  baseUrl: (process.env.STUDIO_API_URL || 'http://localhost:8000').replace(/\/+$/, ''),
  token: (process.env.STUDIO_API_TOKEN || '').trim(),
  timeoutMs: parseInt(process.env.STUDIO_REQUEST_TIMEOUT_MS || '1800000', 10)
};

/**
 * Validates and normalizes generation parameters.
 * Supports aspect ratio presets and the 8 glamour/photography presets.
 */
export function resolveGenerationParams(userOptions = {}) {
  const defaults = generationConfig.defaults || {};
  const aspectPresets = generationConfig.aspect_presets || generationConfig.presets || {};
  const photoPresets = generationConfig.photography_presets || {};
  const limits = generationConfig.parameter_limits || {};

  const modelId = userOptions.model || userOptions.model_id || modelsConfig.default_model;
  const isFlux = Boolean(modelId && modelId.toLowerCase().includes('flux'));
  const isFluxDev = Boolean(isFlux && modelId.toLowerCase().includes('dev'));
  const isFluxSchnell = Boolean(isFlux && !isFluxDev);
  const isSd15 = Boolean(modelId && (modelId.includes('1-5') || modelId.includes('1.5') || modelId.includes('v5.1')));
  const isSdxlTurbo = Boolean(modelId && modelId.toLowerCase().includes('turbo'));

  const presetName = userOptions.preset;
  const photoPreset = presetName && photoPresets[presetName] ? photoPresets[presetName] : null;
  const aspectPreset = presetName && aspectPresets[presetName] ? aspectPresets[presetName] : null;

  let width = userOptions.width;
  let height = userOptions.height;
  let steps = userOptions.steps;
  let guidance = userOptions.guidance || userOptions.guidance_scale;
  let negativePrompt = userOptions.negative || userOptions.negative_prompt;
  let promptText = (userOptions.prompt || '').trim();

  // If photography preset selected, apply model-aware defaults
  if (photoPreset) {
    const configTarget = isSd15 ? photoPreset.sd15 : photoPreset.sdxl;
    if (configTarget) {
      width = width || configTarget.width;
      height = height || configTarget.height;
      steps = steps || configTarget.steps;
      guidance = guidance !== undefined ? guidance : configTarget.guidance_scale;
    }

    // Blend photography negative prompts
    const photoNeg = photoPreset.negative_prompt || '';
    if (negativePrompt === undefined) {
      negativePrompt = [generationConfig.default_negative_prompt || DEFAULT_NEGATIVE_PROMPT, photoNeg].filter(Boolean).join(', ');
    } else if (photoNeg && !negativePrompt.includes(photoNeg.slice(0, 20))) {
      negativePrompt = [negativePrompt, photoNeg].filter(Boolean).join(', ');
    }
  } else if (aspectPreset) {
    width = width || aspectPreset.width;
    height = height || aspectPreset.height;
  }

  // Fallbacks based on model: SDXL and FLUX default to 768x1024, SD 1.5 to 512x768
  if (!width || !height) {
    if (isSd15) {
      width = width || 512;
      height = height || 768;
    } else {
      width = width || 768;
      height = height || 1024;
    }
  }

  // Round to nearest multiple (16 for FLUX patch size, 8 for SDXL/SD15)
  const multiple = isFlux ? 16 : 8;
  width = Math.round(width / multiple) * multiple;
  height = Math.round(height / multiple) * multiple;

  // Clamp width and height
  const minW = limits.width?.min || 256;
  const maxW = limits.width?.max || 1536;
  width = Math.max(minW, Math.min(maxW, width));

  const minH = limits.height?.min || 256;
  const maxH = limits.height?.max || 1536;
  height = Math.max(minH, Math.min(maxH, height));

  // Steps by model architecture
  let defaultSteps = isFluxSchnell ? 4 : isSd15 ? 25 : (defaults.steps || 30);
  const minSteps = isFluxSchnell ? 1 : (limits.steps?.min || 10);
  const maxSteps = limits.steps?.max || 60;
  steps = parseInt(steps || defaultSteps, 10);
  steps = Math.max(minSteps, Math.min(maxSteps, steps));

  // Guidance Scale by model architecture
  let defaultGuidance = isFluxSchnell ? 0.0 : (defaults.guidance_scale || 7.0);
  const minG = isFluxSchnell ? 0.0 : (limits.guidance_scale?.min || 1.0);
  const maxG = limits.guidance_scale?.max || 15.0;
  guidance = parseFloat(guidance !== undefined ? guidance : defaultGuidance);
  guidance = Math.max(minG, Math.min(maxG, guidance));

  // Seed
  let seed = userOptions.seed !== undefined && userOptions.seed !== null ? parseInt(userOptions.seed, 10) : null;
  if (seed !== null && (isNaN(seed) || seed < 0)) {
    seed = null;
  }

  // Negative prompt default
  if (negativePrompt === undefined) {
    negativePrompt = generationConfig.default_negative_prompt || DEFAULT_NEGATIVE_PROMPT || '';
  }

  // Num images (batch)
  let numImages = parseInt(userOptions.num_images || userOptions['num-images'] || defaults.batch_size || 1, 10);
  numImages = Math.max(1, Math.min(4, isNaN(numImages) ? 1 : numImages));

  return {
    prompt: promptText,
    negative_prompt: negativePrompt,
    width,
    height,
    steps,
    guidance_scale: guidance,
    seed,
    num_images: numImages,
    format: userOptions.format || 'png',
    model_id: modelId,
    preset: presetName || undefined,
    character_id: userOptions.character || userOptions.character_id || undefined,
    idempotency_key: userOptions.idempotency_key || userOptions['idempotency-key'] || undefined,
    max_sequence_length: isFluxSchnell ? 256 : isFluxDev ? 512 : undefined
  };
}
