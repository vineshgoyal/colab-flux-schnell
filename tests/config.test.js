import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveGenerationParams, modelsConfig, generationConfig } from '../src/config.js';

test('Config: modelsConfig loads default model and list', () => {
  assert.ok(modelsConfig.default_model);
  assert.ok(Array.isArray(modelsConfig.models));
  assert.ok(modelsConfig.models.length >= 2);
  const sdxl = modelsConfig.models.find(m => m.id === 'sdxl-base-1.0');
  assert.ok(sdxl, 'SDXL base model should be in models.json');
  assert.equal(sdxl.pipeline, 'StableDiffusionXLPipeline');
});

test('Config: resolveGenerationParams applies portrait preset by default', () => {
  const params = resolveGenerationParams({ prompt: 'A stylish portrait' });
  assert.equal(params.prompt, 'A stylish portrait');
  assert.equal(params.width, 768);
  assert.equal(params.height, 1024);
  assert.equal(params.steps, 30);
  assert.equal(params.format, 'png');
});

test('Config: resolveGenerationParams applies square preset', () => {
  const params = resolveGenerationParams({ prompt: 'Square test', preset: 'square' });
  assert.equal(params.width, 1024);
  assert.equal(params.height, 1024);
});

test('Config: resolveGenerationParams rounds dimensions to multiples of 8 and clamps bounds', () => {
  const params = resolveGenerationParams({
    prompt: 'Bounds test',
    width: 771, // should round to 768
    height: 1021, // should round to 1024
    steps: 120, // should clamp to 60
    guidance: 30 // should clamp to 15.0
  });
  assert.equal(params.width % 8, 0);
  assert.equal(params.height % 8, 0);
  assert.equal(params.steps, 60);
  assert.equal(params.guidance_scale, 15.0);
});

test('Config: resolveGenerationParams applies photography presets with model-specific resolution', () => {
  // Test beauty-portrait on SDXL
  const sdxlParams = resolveGenerationParams({
    prompt: 'Beauty test',
    preset: 'beauty-portrait',
    model: 'sdxl-base-1.0'
  });
  assert.equal(sdxlParams.width, 768);
  assert.equal(sdxlParams.height, 1024);
  assert.equal(sdxlParams.steps, 30);
  assert.equal(sdxlParams.guidance_scale, 5.5);
  assert.ok(sdxlParams.negative_prompt.includes('plastic skin'));

  // Test beauty-portrait on SD 1.5
  const sd15Params = resolveGenerationParams({
    prompt: 'Beauty test SD15',
    preset: 'beauty-portrait',
    model: 'sd-1.5'
  });
  assert.equal(sd15Params.width, 512);
  assert.equal(sd15Params.height, 768);
  assert.equal(sd15Params.steps, 25);
  assert.equal(sd15Params.guidance_scale, 7.0);

  // Test all 8 photography presets exist
  const requiredPresets = [
    'beauty-portrait',
    'luxury-fashion',
    'fitness-glamour',
    'evening-editorial',
    'beach-fashion',
    'cinematic-portrait',
    'natural-lifestyle',
    'studio-full-body'
  ];
  for (const p of requiredPresets) {
    assert.ok(generationConfig.photography_presets[p], `Preset '${p}' must exist in generationConfig`);
  }
});

test('Config: resolveGenerationParams applies FLUX.1-schnell specific defaults', () => {
  const fluxParams = resolveGenerationParams({
    prompt: 'FLUX portrait test',
    model: 'flux-schnell'
  });
  assert.equal(fluxParams.width, 768);
  assert.equal(fluxParams.height, 1024);
  assert.equal(fluxParams.steps, 4);
  assert.equal(fluxParams.guidance_scale, 0.0);
  assert.equal(fluxParams.max_sequence_length, 256);
  assert.equal(fluxParams.model_id, 'flux-schnell');
});

test('Config: resolveGenerationParams applies enhanced default negative prompt', () => {
  const params = resolveGenerationParams({
    prompt: 'Testing negative prompt defaults'
  });
  assert.ok(params.negative_prompt.includes('worst quality, low quality'));
  assert.ok(params.negative_prompt.includes('unrealistic anatomy'));
  assert.ok(params.negative_prompt.includes('distorted body proportions'));
  assert.ok(params.negative_prompt.includes('watermark, logo, signature'));
});
