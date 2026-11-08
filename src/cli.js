#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {
  serverConfig,
  modelsConfig,
  generationConfig,
  paths,
  resolveGenerationParams
} from './config.js';
import { StudioApiClient } from './api-client.js';
import { saveGenerationOutput, getGenerationHistory } from './output.js';
import {
  listCharacterProfiles,
  getCharacterProfile,
  saveCharacterProfile,
  synthesizeCharacterPrompt,
  validateCharacterProfile
} from './characters.js';
import { createStudioServer } from './server.js';

// Minimal CLI argument parser
function parseArgs(args) {
  const parsed = { _: [], flags: {} };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
        parsed.flags[key] = args[++i];
      } else {
        parsed.flags[key] = true;
      }
    } else if (arg.startsWith('-')) {
      const key = arg.slice(1);
      if (i + 1 < args.length && !args[i + 1].startsWith('-')) {
        parsed.flags[key] = args[++i];
      } else {
        parsed.flags[key] = true;
      }
    } else {
      parsed._.push(arg);
    }
  }
  return parsed;
}

function maskSecret(str) {
  if (!str) return '(none)';
  if (str.length <= 8) return '****';
  return str.slice(0, 5) + '...' + str.slice(-4);
}

function parseWaitMs(timeoutFlag) {
  if (!timeoutFlag) return undefined;
  const num = parseInt(timeoutFlag, 10);
  if (isNaN(num)) return undefined;
  // If user passes small number like 1800, assume seconds -> ms
  return num < 10000 ? num * 1000 : num;
}

// -------------------------------------------------------------
// Command: doctor
// -------------------------------------------------------------
async function cmdDoctor() {
  console.log('\n======================================================');
  console.log('           AI Image Studio: System Doctor             ');
  console.log('======================================================');

  // 1. Check Node.js
  const nodeVersion = process.version;
  const major = parseInt(nodeVersion.slice(1).split('.')[0], 10);
  if (major >= 20) {
    console.log(`[✓] Node.js runtime:      ${nodeVersion} (compatible >= 20.0.0)`);
  } else {
    console.warn(`[!] Node.js version ${nodeVersion} is below the recommended v20+`);
  }

  // 2. Check local directories
  console.log(`[✓] Output directory:     ${paths.outputs}`);
  console.log(`[✓] Characters directory: ${paths.characters}`);

  // 3. Check Configurations
  console.log(`[✓] Configured models:    ${modelsConfig.models?.length || 0} loaded from config/models.json`);
  const photoPresetCount = Object.keys(generationConfig.photography_presets || {}).length;
  console.log(`[✓] Presets loaded:       ${photoPresetCount} photography presets + aspect presets`);

  // 4. Remote Backend Connectivity Check
  console.log(`\nConnecting to backend:    ${serverConfig.baseUrl}`);
  console.log(`Configured Auth Token:    ${maskSecret(serverConfig.token)}`);

  const client = new StudioApiClient();
  try {
    const health = await client.getHealth();
    console.log('[✓] Backend HTTP Status:   Healthy & Connected');

    // 4a. GPU & CUDA Device Diagnostics
    const gpuOk = Boolean(health.gpu_available);
    if (gpuOk) {
      console.log(`[✓] GPU Acceleration:     YES (${health.device_name || 'CUDA Device'}, Devices: ${health.cuda_device_count ?? 1})`);
      if (health.cuda_version) {
        console.log(`[✓] CUDA Version:          ${health.cuda_version}`);
      }
      if (health.vram) {
        console.log(`[✓] VRAM Status:           ${health.vram.allocated_gb} GB allocated / ${health.vram.total_gb} GB total (${health.vram.free_gb} GB free)`);
      }
    } else {
      console.log('[!] GPU Acceleration:     NO (Backend process has no CUDA GPU)');
      console.log(`    Compute Device:        ${health.device_name || 'CPU'}`);
      console.log(`    CUDA Device Count:     ${health.cuda_device_count ?? 0}`);
      console.log(`    CUDA_VISIBLE_DEVICES:  ${health.cuda_visible_devices || 'not set'}`);
    }

    // 4b. Distinguish Configured vs Loaded Model
    const loadedModel = health.current_model;
    if (loadedModel) {
      console.log(`[✓] Loaded Model:          ${loadedModel} (Active in VRAM)`);
    } else {
      console.log('[!] Loaded Model:          NONE (Weights not loaded into VRAM)');
    }

    // 4c. Queue and active jobs
    console.log(`[✓] Job Queue Status:      ${health.queue_length ?? 0} queued, ${health.active_jobs ?? 0} active`);
    if (health.running_job_id) {
      console.log(`    Currently Running:     Job ID ${health.running_job_id}`);
    }

    // 4d. Four-Condition Generation Readiness Breakdown
    console.log('\n------------------------------------------------------');
    console.log('              Generation Readiness Status             ');
    console.log('------------------------------------------------------');
    console.log(` 1. Backend Online:        YES`);
    console.log(` 2. GPU Available:         ${gpuOk ? 'YES' : 'NO'}`);
    console.log(` 3. Model Loaded:          ${loadedModel ? `YES (${loadedModel})` : 'NO'}`);
    console.log(` 4. Generation Ready:      ${health.generation_ready ? 'YES' : 'NO'}`);

    if (!gpuOk) {
      console.log('\nSTATUS: [WARNING] NOT READY FOR GPU GENERATION');
      console.log('Reason: The Colab backend runtime is operating on CPU only.');
      console.log('\nTroubleshooting:');
      console.log(' 1. Open Google Colab -> Runtime -> Change runtime type.');
      console.log(' 2. Select "T4 GPU" and Save.');
      console.log(' 3. Run the notebook setup cell to re-attach the server to GPU.\n');
    } else if (!loadedModel) {
      console.log('\nSTATUS: [WARNING] GPU DETECTED, BUT NO MODEL LOADED');
      console.log('Reason: Model weights have not been downloaded or loaded into VRAM.');
      console.log('\nAction Required:');
      console.log(' Run: node src/cli.js models load sd-1.5');
      console.log(' (or node src/cli.js models load sdxl-base-1.0)\n');
    } else {
      console.log('\nSTATUS: [READY] Fully operational and ready for GPU generation!');
      console.log(`Active Model: ${loadedModel}`);
      console.log('You can now run: node src/cli.js generate --prompt "..."\n');
    }
  } catch (err) {
    console.error('\n[✗] Backend connection failed:');
    if (err.isCloudflareTimeout || err.status === 524) {
      console.error('    Reason: Cloudflare gateway timeout (HTTP 524).');
      console.error('    The backend took too long to respond to the health check.');
    } else {
      console.error(`    ${err.message}`);
    }
    console.error('\nTroubleshooting suggestions:');
    console.error(' 1. Check if your Google Colab notebook cell is running.');
    console.error(' 2. Verify STUDIO_API_URL and STUDIO_API_TOKEN in your .env file.');
    console.error(' 3. If using Cloudflare Tunnel, ensure the tunnel URL has not changed.');
    console.error(' 4. Run "node src/cli.js doctor" again after updating .env.\n');

    console.log('------------------------------------------------------');
    console.log('              Generation Readiness Status             ');
    console.log('------------------------------------------------------');
    console.log('STATUS: [OFFLINE] Backend is unreachable. Please verify Colab.\n');
  }
}

// -------------------------------------------------------------
// Command: models
// -------------------------------------------------------------
async function cmdModels(args, flags) {
  const subCommand = args[1];
  const timeoutMs = parseWaitMs(flags.timeout);
  const client = new StudioApiClient({ timeoutMs });

  if (subCommand === 'load') {
    const requested = args[2] || flags.id || modelsConfig.default_model;
    const resolved = modelsConfig.models.find(m => m.id === requested);
    const modelToLoad = resolved ? (resolved.repo_id || resolved.id) : requested;
    const hfToken = flags.token || flags['hf-token'] || process.env.HF_TOKEN || null;

    console.log(`\nSubmitting model load job for ${requested} (${modelToLoad}) to GPU queue...`);
    try {
      const res = await client.loadModel(modelToLoad, hfToken, (msg) => {
        console.log(`  ${msg}`);
      });
      console.log(`\n[✓] Model successfully loaded: ${res.model_id || modelToLoad}`);
      if (res.execution_duration_seconds !== null && res.execution_duration_seconds !== undefined) {
        console.log(`    Total duration: ${res.execution_duration_seconds}s`);
      } else if (res.duration_seconds !== null && res.duration_seconds !== undefined) {
        console.log(`    Total duration: ${res.duration_seconds}s`);
      }
      if (res.vram) {
        console.log(`    VRAM Allocated: ${res.vram.allocated_gb} GB / ${res.vram.total_gb} GB`);
      }
      console.log('');
    } catch (err) {
      if (err.isCloudflareTimeout || err.status === 524) {
        console.error('\n[✗] Model load request timed out via Cloudflare (HTTP 524).');
        console.error('    The model may still be downloading in the background on Colab.');
        console.error('    Check status: node src/cli.js doctor\n');
      } else if (err.message && (err.message.includes('401') || err.message.includes('gated') || err.message.includes('restricted'))) {
        console.error(`\n[✗] Failed to load model: Gated Repository Access Restricted (HTTP 401)\n`);
        console.error(`    ${modelToLoad} is a gated model requiring Hugging Face authentication.`);
        console.error('\nQuick 2-step setup:');
        console.error(` 1. Visit: https://huggingface.co/${modelToLoad}`);
        console.error('    Click "Agree and access repository" (free instant access).');
        console.error(' 2. Get your free Hugging Face Read token at:');
        console.error('    https://huggingface.co/settings/tokens');
        console.error('\nThen run:');
        console.error(`    node src/cli.js models load ${requested} --token hf_xxxxxxxxxxxxxxxxxxxxxxxx`);
        console.error('    (or set HF_TOKEN=hf_... in your .env file)\n');
      } else {
        console.error(`\n[✗] Failed to load model: ${err.message}\n`);
      }
      process.exit(1);
    }
    return;
  }

  console.log('\n======================================================');
  console.log('              Configured Model Catalog                ');
  console.log('======================================================');

  try {
    const backendData = await client.getModels().catch(() => null);
    const activeModel = backendData?.current_model;

    const list = modelsConfig.models;
    for (const m of list) {
      const isActive = activeModel && (activeModel === m.repo_id || activeModel === m.id);
      const indicator = isActive ? ' [CURRENTLY ACTIVE IN VRAM]' : '';
      console.log(`\n• ID:          ${m.id}${indicator}`);
      console.log(`  Name:        ${m.name}`);
      console.log(`  Repository:  ${m.repo_id}`);
      console.log(`  Pipeline:    ${m.pipeline || 'Auto'}`);
      console.log(`  Resolution:  ${m.default_resolution ? (Array.isArray(m.default_resolution) ? m.default_resolution.join('x') : `${m.default_resolution.width}x${m.default_resolution.height}`) : '1024x1024'}`);
      console.log(`  Min VRAM:    ${m.min_vram_gb || 8} GB`);
      console.log(`  License:     ${m.license || 'Open weights'}`);
    }
    console.log('\nTo load a model on the GPU, run:');
    console.log('  node src/cli.js models load <id>\n');
  } catch (err) {
    console.error(`Error listing models: ${err.message}`);
  }
}

// -------------------------------------------------------------
// Command: jobs
// -------------------------------------------------------------
async function cmdJobs(args, flags) {
  const subCommand = args[1];
  const client = new StudioApiClient();

  if (subCommand === 'status') {
    const jobId = args[2] || flags.id;
    if (!jobId) {
      console.error('[✗] Usage: node src/cli.js jobs status <job-id>');
      process.exit(1);
    }
    try {
      const job = await client.getJobStatus(jobId);
      console.log('\n======================================================');
      console.log(`                 Job Details: ${job.job_id || job.id} `);
      console.log('======================================================');
      console.log(`Type:       ${job.type}`);
      console.log(`Status:     ${job.status.toUpperCase()}`);
      console.log(`Stage:      ${job.stage || 'n/a'}`);
      console.log(`Progress:   ${job.progress !== null && job.progress !== undefined ? `${job.progress}%` : 'n/a'}`);
      console.log(`Message:    ${job.message || 'n/a'}`);
      console.log(`Model:      ${job.model_id || 'n/a'}`);
      if (job.seed !== undefined) console.log(`Seed:       ${job.seed}`);
      if (job.num_images) console.log(`Images:     ${job.num_images}`);
      if (job.created_at) console.log(`Created:    ${new Date(job.created_at * 1000).toLocaleString()}`);
      if (job.queue_wait_seconds) console.log(`Queue Wait: ${job.queue_wait_seconds}s`);
      if (job.execution_duration_seconds) console.log(`Execution:  ${job.execution_duration_seconds}s`);
      if (job.error) console.log(`Error:      ${job.error}`);
      if (job.results && job.results.length > 0) {
        console.log(`\nOutputs (${job.results.length} image(s)):`);
        for (const r of job.results) {
          console.log(`  • [Index ${r.index}] Seed: ${r.seed} (${r.width}x${r.height})`);
        }
      }
      console.log('');
    } catch (err) {
      console.error(`[✗] Failed to fetch job status: ${err.message}`);
      process.exit(1);
    }
    return;
  }

  if (subCommand === 'cancel') {
    const jobId = args[2] || flags.id;
    if (!jobId) {
      console.error('[✗] Usage: node src/cli.js jobs cancel <job-id>');
      process.exit(1);
    }
    try {
      const res = await client.cancelJob(jobId);
      console.log(`\n[✓] ${res.message} (Status: ${res.status})\n`);
    } catch (err) {
      console.error(`\n[✗] Cancellation failed: ${err.message}\n`);
      process.exit(1);
    }
    return;
  }

  if (subCommand === 'download') {
    const jobId = args[2] || flags.id;
    if (!jobId) {
      console.error('[✗] Usage: node src/cli.js jobs download <job-id> [--index 0]');
      process.exit(1);
    }
    const index = parseInt(flags.index || '0', 10);
    try {
      console.log(`Fetching job status for ${jobId}...`);
      const job = await client.getJobStatus(jobId);
      if (job.status !== 'completed') {
        console.error(`[✗] Job ${jobId} is not completed (current status: ${job.status}).`);
        process.exit(1);
      }
      console.log(`Downloading result image (index ${index})...`);
      const buffer = await client.getJobImage(jobId, index);
      const saved = await saveGenerationOutput(buffer, {
        ...job,
        index
      });
      console.log(`[✓] Downloaded and saved successfully:`);
      console.log(`    Image:    ${saved.imagePath}`);
      console.log(`    Metadata: ${saved.metaPath}\n`);
    } catch (err) {
      console.error(`[✗] Download failed: ${err.message}\n`);
      process.exit(1);
    }
    return;
  }

  // Default: list recent jobs
  try {
    const limit = parseInt(flags.limit || '15', 10);
    const jobs = await client.listJobs(limit);
    console.log('\n======================================================');
    console.log(`              Recent Backend GPU Jobs (${jobs.length}) `);
    console.log('======================================================');
    if (jobs.length === 0) {
      console.log('No recent jobs recorded on the backend.\n');
      return;
    }
    for (const j of jobs) {
      const statusIcon = j.status === 'completed' ? '✓' : j.status === 'running' ? '⏳' : j.status === 'failed' ? '✗' : j.status === 'cancelled' ? '🚫' : '•';
      const duration = j.execution_duration_seconds ? `${j.execution_duration_seconds}s` : j.status === 'running' ? 'running' : 'queued';
      console.log(`\n${statusIcon} Job ID:    ${j.job_id || j.id}`);
      console.log(`  Type:      ${j.type} | Model: ${j.model_id || 'n/a'}`);
      console.log(`  Status:    ${j.status.toUpperCase()} (${j.stage || ''} ${j.progress !== null && j.progress !== undefined ? `${j.progress}%` : ''})`);
      console.log(`  Duration:  ${duration}`);
      if (j.error) console.log(`  Error:     ${j.error}`);
    }
    console.log('\nTo inspect a job:    node src/cli.js jobs status <job-id>');
    console.log('To cancel a job:     node src/cli.js jobs cancel <job-id>');
    console.log('To download result:  node src/cli.js jobs download <job-id>\n');
  } catch (err) {
    console.error(`[✗] Could not list jobs: ${err.message}\n`);
    process.exit(1);
  }
}

// -------------------------------------------------------------
// Command: generate
// -------------------------------------------------------------
async function cmdGenerate(flags) {
  const timeoutMs = parseWaitMs(flags.timeout);
  const pollIntervalMs = parseWaitMs(flags['poll-interval']);
  const client = new StudioApiClient({ timeoutMs, pollIntervalMs });

  // 1. Discover active model on GPU backend
  let activeModel = null;
  let modelLoaded = false;
  try {
    const health = await client.getHealth();
    activeModel = health.current_model;
    modelLoaded = Boolean(health.model_loaded && activeModel);
  } catch (err) {
    // If backend is unreachable or failing, will handle later
  }

  // 2. Resolve target model per Section 5 requirements
  let selectedModel = flags.model || flags.model_id;
  if (!selectedModel) {
    if (modelLoaded && activeModel) {
      selectedModel = activeModel;
    } else {
      console.log('\n======================================================');
      console.log('              Model Readiness Check                   ');
      console.log('======================================================');
      console.log(`Active GPU Model: none`);
      console.log(`Selected Model:   none`);
      console.log(`Status:           Not ready. No model is loaded in GPU VRAM.\n`);
      console.log('[!] Action Required: Please load a model into VRAM first.');
      console.log('    Example: node src/cli.js models load sd-1.5');
      console.log('             node src/cli.js models load sdxl-base-1.0\n');
      process.exit(1);
    }
  }

  let params = {};

  // Check if config file supplied
  if (flags.config) {
    const configPath = path.resolve(process.cwd(), flags.config);
    if (!fs.existsSync(configPath)) {
      console.error(`[✗] Config file not found: ${configPath}`);
      process.exit(1);
    }
    const fileData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    params = resolveGenerationParams({ ...fileData, ...flags, model: selectedModel });
  } else if (flags.character) {
    // Generate using a character profile
    try {
      const profile = await getCharacterProfile(flags.character);
      const synthesized = synthesizeCharacterPrompt(profile, {
        scene: flags.scene || flags.prompt,
        clothing: flags.clothing,
        style: flags.style,
        negative_prompt: flags.negative,
        preset: flags.preset,
        seed: flags.seed ? parseInt(flags.seed, 10) : undefined
      });

      params = resolveGenerationParams({
        ...flags,
        model: selectedModel,
        preset: flags.preset,
        prompt: synthesized.prompt,
        negative_prompt: synthesized.negative_prompt,
        seed: synthesized.seed
      });
      params.character_id = profile.id;
      console.log(`\n[Character Mode] Active Profile: ${profile.name} (${profile.age}yo)`);
    } catch (err) {
      console.error(`[✗] Character error: ${err.message}`);
      process.exit(1);
    }
  } else {
    if (!flags.prompt) {
      console.error('[✗] Error: Please provide a prompt via --prompt "..." or a job file via --config <file.json>');
      console.error('Example: node src/cli.js generate --prompt "Beauty portrait of fictional model" --preset beauty-portrait');
      process.exit(1);
    }
    params = resolveGenerationParams({ ...flags, model: selectedModel });
  }

  // Display Section 5 model and generation banner
  console.log('\n======================================================');
  console.log('               Image Generation Job                   ');
  console.log('======================================================');
  console.log(`Active GPU Model: ${activeModel || 'none'}`);
  console.log(`Selected Model:   ${params.model_id}`);
  console.log(`Resolution:       ${params.width} x ${params.height}`);
  console.log(`Steps:            ${params.steps}`);
  console.log(`CFG Scale:        ${params.guidance_scale}`);
  console.log(`Seed:             ${params.seed ?? 'Random'}`);
  console.log(`Images:           ${params.num_images || 1}`);
  if (params.preset) {
    console.log(`Preset:           ${params.preset}`);
  }
  console.log(`Status:           Ready for generation`);
  console.log('------------------------------------------------------');
  console.log(`Prompt:           "${params.prompt}"`);
  if (params.negative_prompt) {
    console.log(`Negative:         "${params.negative_prompt}"`);
  }
  console.log('------------------------------------------------------');

  try {
    const startTime = Date.now();
    const result = await client.generateAndDownload(params, (msg) => {
      console.log(`[Status] ${msg}`);
    });

    const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);
    const downloadedImages = result.images || [{ index: 0, seed: result.jobRecord.seed, imageBuffer: result.imageBuffer }];

    console.log('\n[✓] Generation Succeeded!');
    for (const img of downloadedImages) {
      const saveResult = await saveGenerationOutput(img.imageBuffer, {
        ...params,
        ...result.jobRecord,
        index: img.index,
        seed: img.seed,
        duration_seconds: result.jobRecord.execution_duration_seconds || parseFloat(elapsedSec)
      });
      console.log(`[✓] Output Image [${img.index}]:    ${saveResult.imagePath}`);
      console.log(`[✓] Metadata Record:    ${saveResult.metaPath}`);
      console.log(`[✓] Seed:               ${img.seed}`);
    }

    if (result.jobRecord.vram) {
      console.log(`[✓] VRAM Allocated:     ${result.jobRecord.vram.allocated_gb || '?'} GB`);
    }
    console.log(`[✓] Total Turnaround:   ${elapsedSec}s\n`);
  } catch (err) {
    if (err.isCloudflareTimeout || err.status === 524) {
      console.error('\n[✗] Generation failed');
      console.error('\nReason: Cloudflare gateway timeout');
      console.error('HTTP Status: 524\n');
      console.error('The backend may still be processing the job.');
      console.error('Check the job status before retrying: node src/cli.js jobs\n');
    } else {
      console.error(`\n[✗] Generation failed: ${err.message}`);
      if (err.details && typeof err.details === 'object' && err.details.detail) {
        console.error(`Details: ${err.details.detail}`);
      }
      console.log('');
    }
    process.exit(1);
  }
}

// -------------------------------------------------------------
// Command: character
// -------------------------------------------------------------
async function cmdCharacter(args, flags) {
  const subCommand = args[1] || 'list';

  if (subCommand === 'list') {
    const profiles = await listCharacterProfiles();
    console.log('\n======================================================');
    console.log('              Saved Character Profiles                ');
    console.log('======================================================');
    if (profiles.length === 0) {
      console.log('No character profiles found in characters/.');
      console.log('Create one using: node src/cli.js character create');
      return;
    }
    for (const p of profiles) {
      console.log(`\n• ID:         ${p.id}`);
      console.log(`  Name:       ${p.name} (Age: ${p.age}, Adult Fictional)`);
      console.log(`  Eyes:       ${p.eye_color} | Hair: ${p.hair?.length || ''} ${p.hair?.color || ''}`);
      console.log(`  Facial:     ${p.facial_features}`);
      console.log(`  Pref. Seed: ${p.preferred_seed ?? 'none'}`);
    }
    console.log('\nTo generate with a character, run:');
    console.log('  node src/cli.js generate --character <id> --scene "..."\n');
    return;
  }

  if (subCommand === 'show') {
    const id = args[2] || flags.id;
    if (!id) {
      console.error('Please specify a character id: node src/cli.js character show <id>');
      return;
    }
    try {
      const profile = await getCharacterProfile(id);
      console.log('\n' + JSON.stringify(profile, null, 2));
    } catch (err) {
      console.error(`[✗] ${err.message}`);
    }
    return;
  }

  if (subCommand === 'create') {
    // Check if passed via flags
    if (flags.id && flags.name && flags.age) {
      const profile = {
        id: flags.id,
        name: flags.name,
        age: parseInt(flags.age, 10),
        gender: flags.gender || 'female',
        ethnicity: flags.ethnicity || '',
        facial_features: flags.facial || flags.facial_features || 'symmetrical defined features',
        eye_color: flags.eyes || flags.eye_color || 'brown',
        hair: {
          color: flags['hair-color'] || 'dark brown',
          style: flags['hair-style'] || 'natural waves',
          length: flags['hair-length'] || 'medium'
        },
        body: flags.body || 'athletic proportion',
        clothing_preferences: flags.clothing || 'minimalist high-fashion',
        photography_style: flags.style || '85mm portrait, cinematic lighting, 8k',
        preferred_seed: flags.seed ? parseInt(flags.seed, 10) : null,
        negative_prompt_additions: 'plastic skin, cartoon, anime',
        reference_images: []
      };

      try {
        const saved = await saveCharacterProfile(profile);
        console.log(`[✓] Character profile saved successfully to: ${saved.path}`);
      } catch (err) {
        console.error(`[✗] Character creation failed: ${err.message}`);
      }
      return;
    }

    // Interactive creation prompt
    console.log('\n======================================================');
    console.log('       Interactive Character Profile Creator          ');
    console.log('======================================================');
    console.log('Note: Character must represent an adult (18+) fictional persona.\n');

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const ask = (q, def = '') =>
      new Promise((resolve) => {
        rl.question(`${q}${def ? ` [${def}]` : ''}: `, (ans) => resolve(ans.trim() || def));
      });

    try {
      const name = await ask('Character Name', 'Aria Stirling');
      const id = await ask('Identifier Slug (e.g. aria-stirling)', name.toLowerCase().replace(/\s+/g, '-'));
      const ageStr = await ask('Adult Age (18+)', '26');
      const age = parseInt(ageStr, 10);
      if (isNaN(age) || age < 18) {
        console.error('[✗] Safety Requirement: Age must be an adult (18 or older).');
        rl.close();
        return;
      }
      const gender = await ask('Gender', 'female');
      const ethnicity = await ask('Ethnicity / Heritage', 'Scandinavian-Mediterranean');
      const facial = await ask('Facial Features', 'high cheekbones, subtle freckles, softly defined jaw');
      const eyeColor = await ask('Eye Color', 'deep hazel with amber flecks');
      const hairColor = await ask('Hair Color', 'chestnut brown');
      const hairStyle = await ask('Hair Style', 'wavy textured blowout');
      const hairLength = await ask('Hair Length', 'long');
      const body = await ask('Body Build & Proportions', 'slender athletic build, 5ft 8in');
      const clothing = await ask('Default Clothing Preference', 'tailored blazers, elegant evening silk');
      const style = await ask('Photography Style', 'Hasselblad 85mm f/1.4, golden hour rim lighting, 8k');
      const seedStr = await ask('Preferred Seed (Optional, press Enter for random)', '');

      rl.close();

      const profile = {
        id,
        name,
        age,
        gender,
        ethnicity,
        facial_features: facial,
        eye_color: eyeColor,
        hair: { color: hairColor, style: hairStyle, length: hairLength },
        body,
        clothing_preferences: clothing,
        photography_style: style,
        preferred_seed: seedStr ? parseInt(seedStr, 10) : null,
        negative_prompt_additions: 'plastic skin, anime, distorted eyes',
        reference_images: []
      };

      const saved = await saveCharacterProfile(profile);
      console.log(`\n[✓] Character '${profile.name}' saved to ${saved.path}!`);
      console.log(`Try generating with: node src/cli.js generate --character ${profile.id} --scene "sitting in a rooftop cafe"`);
    } catch (e) {
      rl.close();
      console.error(`[✗] ${e.message}`);
    }
    return;
  }

  console.log(`Unknown character subcommand: ${subCommand}`);
  console.log('Available subcommands: list, show <id>, create');
}

// -------------------------------------------------------------
// Command: photoshoot (multi-shot character series)
// -------------------------------------------------------------
async function cmdPhotoshoot(args, flags) {
  const characterId = args[1] || flags.character;
  if (!characterId) {
    console.error('[✗] Please specify character: node src/cli.js photoshoot <character-id> [--theme "winter street style"]');
    process.exit(1);
  }

  const profile = await getCharacterProfile(characterId);
  const theme = flags.theme || 'high fashion editorial';

  const timeoutMs = parseWaitMs(flags.timeout);
  const client = new StudioApiClient({ timeoutMs });

  // Query backend for active model per Section 5
  let activeModel = null;
  let modelLoaded = false;
  try {
    const health = await client.getHealth();
    activeModel = health.current_model;
    modelLoaded = Boolean(health.model_loaded && activeModel);
  } catch (err) {
    // backend offline or unreachable handled on generate
  }

  let selectedModel = flags.model || flags.model_id;
  if (!selectedModel) {
    if (modelLoaded && activeModel) {
      selectedModel = activeModel;
    } else {
      console.log('\n======================================================');
      console.log('              Model Readiness Check                   ');
      console.log('======================================================');
      console.log(`Active GPU Model: none`);
      console.log(`Selected Model:   none`);
      console.log(`Status:           Not ready. No model is loaded in GPU VRAM.\n`);
      console.log('[!] Action Required: Please load a model into VRAM first.');
      console.log('    Example: node src/cli.js models load sd-1.5');
      console.log('             node src/cli.js models load sdxl-base-1.0\n');
      process.exit(1);
    }
  }

  console.log('\n======================================================');
  console.log(`        Character Photoshoot Series: ${profile.name}         `);
  console.log('======================================================');
  console.log(`Active GPU Model: ${activeModel || 'none'}`);
  console.log(`Selected Model:   ${selectedModel}`);
  console.log(`Theme:            ${theme}`);
  console.log(`Persona:          ${profile.name} (Age: ${profile.age}, ${profile.gender})`);
  console.log(`Anchored Seed:    ${profile.preferred_seed ?? 'Random'}`);
  console.log('------------------------------------------------------');

  // The 5 standard shots per Section 8
  const standardShots = [
    {
      title: '1. Beauty Close-Up Portrait',
      scene: `extreme close-up beauty portrait in ${theme}, capturing intricate facial microtexture and eye reflections`,
      preset: 'beauty-portrait'
    },
    {
      title: '2. Medium Fashion Portrait',
      scene: `medium half-body fashion portrait in ${theme}, showcasing tailored styling, posture, and fabric drape`,
      preset: 'luxury-fashion'
    },
    {
      title: '3. Full-Body Standing Pose',
      scene: `full-length standing fashion pose in ${theme}, head to toe visible on clean backdrop`,
      preset: 'studio-full-body'
    },
    {
      title: '4. Walking Lifestyle Photograph',
      scene: `candid walking lifestyle photograph in ${theme}, relaxed authentic stride in natural daylight`,
      preset: 'natural-lifestyle'
    },
    {
      title: '5. Cinematic Environmental Photograph',
      scene: `cinematic atmospheric shot in ${theme}, dynamic background depth of field and filmic lighting`,
      preset: 'cinematic-portrait'
    }
  ];

  const requestedCount = parseInt(flags.shots || flags.count || '5', 10);
  const shotsToRun = standardShots.slice(0, Math.min(requestedCount, standardShots.length));

  let successCount = 0;
  let failCount = 0;

  for (let i = 0; i < shotsToRun.length; i++) {
    const shot = shotsToRun[i];
    console.log(`\n[Shot ${i + 1}/${shotsToRun.length}] ${shot.title}...`);

    const synthesized = synthesizeCharacterPrompt(profile, {
      scene: flags.scene || shot.scene,
      clothing: flags.clothing,
      style: flags.style,
      negative_prompt: flags.negative,
      preset: shot.preset,
      seed: profile.preferred_seed || (flags.seed ? parseInt(flags.seed, 10) : undefined)
    });

    const params = resolveGenerationParams({
      ...flags,
      model: selectedModel,
      preset: shot.preset,
      prompt: synthesized.prompt,
      negative_prompt: synthesized.negative_prompt,
      seed: synthesized.seed
    });
    params.character_id = profile.id;

    try {
      const startTime = Date.now();
      const result = await client.generateAndDownload(params, (msg) => {
        console.log(`  [Status] ${msg}`);
      });

      const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);
      const saveResult = await saveGenerationOutput(result.imageBuffer, {
        ...params,
        ...result.jobRecord,
        duration_seconds: parseFloat(elapsedSec)
      });

      console.log(`  [✓] Shot ${i + 1} succeeded: ${saveResult.imagePath} (${elapsedSec}s)`);
      successCount++;
    } catch (err) {
      console.error(`  [✗] Shot ${i + 1} failed: ${err.message}`);
      failCount++;
    }
  }

  console.log('\n======================================================');
  console.log(`Photoshoot Summary: ${successCount} succeeded, ${failCount} failed.`);
  console.log('View generated shots in outputs/\n');
}

// -------------------------------------------------------------
// Command: serve (Local interactive Studio Web UI)
// -------------------------------------------------------------
async function cmdServe(flags) {
  const port = parseInt(flags.port || process.env.PORT || '3000', 10);
  const studio = createStudioServer({ port });
  await studio.start();
}

// -------------------------------------------------------------
// Command: history
// -------------------------------------------------------------
async function cmdHistory(flags) {
  const limit = parseInt(flags.limit || '15', 10);
  const items = await getGenerationHistory(limit);

  console.log('\n======================================================');
  console.log(`         Recent Generations History (Last ${items.length})      `);
  console.log('======================================================');

  if (items.length === 0) {
    console.log('No generation history found. Run a generation first!\n');
    return;
  }

  for (const item of items) {
    console.log(`\n• File:       ${item.image_file}`);
    console.log(`  Date:       ${item.timestamp}`);
    console.log(`  Dimensions: ${item.width}x${item.height} | Steps: ${item.steps} | Seed: ${item.seed}`);
    console.log(`  Model:      ${item.model_id}`);
    if (item.preset) console.log(`  Preset:     ${item.preset}`);
    if (item.character_id) console.log(`  Character:  ${item.character_id}`);
    console.log(`  Prompt:     "${item.prompt?.slice(0, 90)}${item.prompt?.length > 90 ? '...' : ''}"`);
  }
  console.log('');
}

// -------------------------------------------------------------
// Main Dispatcher
// -------------------------------------------------------------
async function main() {
  const rawArgs = process.argv.slice(2);
  const { _, flags } = parseArgs(rawArgs);
  const command = _[0];

  switch (command) {
    case 'doctor':
      await cmdDoctor();
      break;

    case 'models':
      await cmdModels(_, flags);
      break;

    case 'jobs':
      await cmdJobs(_, flags);
      break;

    case 'generate':
      await cmdGenerate(flags);
      break;

    case 'character':
    case 'characters':
      await cmdCharacter(_, flags);
      break;

    case 'history':
      await cmdHistory(flags);
      break;

    case 'photoshoot':
      await cmdPhotoshoot(_, flags);
      break;

    case 'serve':
      await cmdServe(flags);
      break;

    case 'help':
    case '--help':
    case '-h':
    default:
      console.log(`
AI Image Studio CLI - Production GPU Image Generation Client

Usage:
  node src/cli.js doctor                   Check system, GPU backend, and config health
  node src/cli.js models                   List configured models
  node src/cli.js models load <id>         Switch active model on GPU
  node src/cli.js jobs                     List recent backend GPU jobs
  node src/cli.js jobs status <job-id>     Inspect job progress and status
  node src/cli.js jobs cancel <job-id>     Cancel active or queued job
  node src/cli.js jobs download <job-id>   Download result PNG for completed job
  node src/cli.js generate --prompt "..."  Generate image from prompt
  node src/cli.js generate --config <json> Generate image from job config file
  node src/cli.js generate --character <id> --scene "..."  Generate character in scene
  node src/cli.js photoshoot <character-id> [--shots 5]    Run multi-shot character photo series
  node src/cli.js character list           List reusable fictional character profiles
  node src/cli.js character create         Create a new character profile
  node src/cli.js character show <id>      View full character profile
  node src/cli.js history                  List local generation history
  node src/cli.js serve [--port 3000]      Launch local interactive web studio UI

Options:
  --preset <preset-name>                   Photography preset or aspect ratio preset:
                                           (beauty-portrait, luxury-fashion, fitness-glamour,
                                            evening-editorial, beach-fashion, cinematic-portrait,
                                            natural-lifestyle, studio-full-body, portrait, square, landscape)
  --width <pixels>, --height <pixels>      Custom dimensions (divisible by 8)
  --steps <int>                            Diffusion inference steps (10-60)
  --guidance <float>                       CFG scale (1.0-15.0)
  --seed <int>                             Deterministic seed
  --num-images <1-4>                       Number of images to generate (batch size)
  --negative "..."                         Negative prompt
  --model <id>                             Model ID override
  --timeout <seconds>                      Max wait time for long-running GPU operations
  --poll-interval <seconds>                Status polling frequency (default: 3s)
`);
      break;
  }
}

main().catch((err) => {
  console.error('\nFatal error:', err.message);
  process.exit(1);
});
