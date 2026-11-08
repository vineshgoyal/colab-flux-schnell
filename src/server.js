import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { URL } from 'node:url';
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
  synthesizeCharacterPrompt
} from './characters.js';

// Cache for completed jobs that have been saved locally
const savedJobCache = new Map();
// Cache for submitted job parameters to preserve prompt metadata
const activeJobParams = new Map();

function sendJson(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
  });
  res.end(JSON.stringify(data));
}

function parseJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 5 * 1024 * 1024) {
        reject(new Error('Request body payload too large (> 5MB)'));
      }
    });
    req.on('end', () => {
      if (!body.trim()) return resolve({});
      try {
        resolve(JSON.parse(body));
      } catch (err) {
        reject(new Error(`Malformed JSON body: ${err.message}`));
      }
    });
    req.on('error', reject);
  });
}

/**
 * Creates and starts the Node.js Studio Web UI server.
 */
export function createStudioServer(options = {}) {
  const defaultPort = process.env.PORT ? parseInt(process.env.PORT, 10) : 3001;
  const port = parseInt(options.port || defaultPort, 10);
  const publicDir = path.resolve(paths.root, 'public');
  const client = new StudioApiClient();

  const server = http.createServer(async (req, res) => {
    const parsedUrl = new URL(req.url, `http://localhost:${port}`);
    const pathname = parsedUrl.pathname;

    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      });
      res.end();
      return;
    }

    try {
      // 1. API: System Status & Hardware
      if (pathname === '/api/status' && req.method === 'GET') {
        let health = null;
        let online = false;
        try {
          health = await client.getHealth();
          online = true;
        } catch (err) {
          health = { error: err.message, status: 'offline' };
        }

        const characters = await listCharacterProfiles().catch(() => []);
        return sendJson(res, 200, {
          online,
          backendUrl: serverConfig.baseUrl,
          health,
          models: modelsConfig.models || [],
          defaultModel: modelsConfig.default_model,
          activeModel: health?.current_model || health?.loaded_model || null,
          gpu: health?.device_name ? { name: health.device_name, vram: health.vram } : (health?.gpu || null),
          queue: { active: health?.active_jobs || 0, length: health?.queue_length || 0 },
          charactersCount: characters.length
        });
      }

      // 2. API: List Models
      if (pathname === '/api/models' && req.method === 'GET') {
        let health = null;
        try {
          health = await client.getHealth();
        } catch (_) {}
        return sendJson(res, 200, {
          active_model: health?.current_model || health?.loaded_model || null,
          default_model: modelsConfig.default_model,
          models: modelsConfig.models || []
        });
      }

      // 3. API: Load Model into VRAM
      if (pathname === '/api/models/load' && req.method === 'POST') {
        const body = await parseJsonBody(req);
        const modelId = body.model_id || body.model;
        if (!modelId) {
          return sendJson(res, 400, { error: 'Missing model_id parameter' });
        }
        const loadResult = await client.submitModelLoad(modelId, body.hf_token);
        return sendJson(res, 202, loadResult);
      }

      // 4. API: Presets & Aspect Ratios
      if (pathname === '/api/presets' && req.method === 'GET') {
        return sendJson(res, 200, {
          aspect_presets: generationConfig.aspect_presets || generationConfig.presets || {},
          photography_presets: generationConfig.photography_presets || {}
        });
      }

      // 5. API: Characters
      if (pathname === '/api/characters' && req.method === 'GET') {
        const profiles = await listCharacterProfiles();
        return sendJson(res, 200, profiles);
      }

      // 6. API: Generate Image (Submit Job)
      if (pathname === '/api/generate' && req.method === 'POST') {
        const body = await parseJsonBody(req);

        let finalPrompt = (body.prompt || '').trim();
        let character = null;

        if (body.character_id) {
          character = await getCharacterProfile(body.character_id);
          if (character) {
            const synthesized = synthesizeCharacterPrompt(character, {
              scene: finalPrompt,
              negative_prompt: body.negative_prompt,
              preset: body.preset,
              seed: body.seed
            });
            finalPrompt = synthesized.prompt;
            if (synthesized.negative_prompt && !body.negative_prompt) {
              body.negative_prompt = synthesized.negative_prompt;
            }
          }
        }

        if (!finalPrompt) {
          return sendJson(res, 400, { error: 'Prompt cannot be empty' });
        }

        const resolved = resolveGenerationParams({
          ...body,
          prompt: finalPrompt
        });

        const jobParams = {
          prompt: resolved.prompt,
          negative_prompt: resolved.negative_prompt,
          width: resolved.width,
          height: resolved.height,
          steps: resolved.steps,
          guidance_scale: resolved.guidance_scale,
          seed: resolved.seed,
          num_images: resolved.num_images || 1,
          preset: body.preset || null,
          character_id: body.character_id || null
        };

        const jobSubmission = await client.submitGeneration(jobParams);
        const jobId = jobSubmission.job_id || jobSubmission.id;
        activeJobParams.set(jobId, jobParams);

        return sendJson(res, 202, {
          ok: true,
          job_id: jobId,
          params: jobParams,
          submission: jobSubmission
        });
      }

      // 7. API: Job Status & Result Polling
      if (pathname.startsWith('/api/jobs/') && req.method === 'GET') {
        const jobId = pathname.replace('/api/jobs/', '').trim();
        if (!jobId) {
          return sendJson(res, 400, { error: 'Missing job ID' });
        }

        // Check if already finalized and cached
        if (savedJobCache.has(jobId)) {
          return sendJson(res, 200, savedJobCache.get(jobId));
        }

        const jobStatus = await client.getJobStatus(jobId);

        if (jobStatus.status === 'completed') {
          // Download and persist images locally if this is an image generation job
          const isGeneration = jobStatus.type === 'generation' || (!jobId.startsWith('load_') && (jobStatus.results?.length > 0 || jobStatus.image_url));
          const resultUrls = [];
          const savedMetadataList = [];
          const numImages = isGeneration ? (jobStatus.num_images || (jobStatus.results?.length || 1)) : 0;

          const originalParams = activeJobParams.get(jobId) || {};
          for (let i = 0; i < numImages; i++) {
            try {
              const imageBuffer = await client.getJobImage(jobId, i);
              const saveRes = await saveGenerationOutput(imageBuffer, {
                ...originalParams,
                ...jobStatus,
                index: i,
                job_id: jobId
              });
              resultUrls.push(`/outputs/${saveRes.metadata.image_file}`);
              savedMetadataList.push(saveRes.metadata);
            } catch (dlErr) {
              console.error(`[Server] Error saving image result ${i} for job ${jobId}: ${dlErr.message}`);
            }
          }
          activeJobParams.delete(jobId);

          const finalized = {
            ...jobStatus,
            result_urls: resultUrls,
            local_metadata: savedMetadataList
          };
          savedJobCache.set(jobId, finalized);
          return sendJson(res, 200, finalized);
        }

        // Return current running/queued state
        return sendJson(res, 200, jobStatus);
      }

      // 8. API: Cancel Job
      if (pathname.startsWith('/api/jobs/') && pathname.endsWith('/cancel') && req.method === 'POST') {
        const parts = pathname.split('/');
        const jobId = parts[3];
        const cancelRes = await client.cancelJob(jobId);
        return sendJson(res, 200, cancelRes);
      }

      // 9. API: Generation History
      if (pathname === '/api/history' && req.method === 'GET') {
        const limit = parseInt(parsedUrl.searchParams.get('limit') || '30', 10);
        const history = await getGenerationHistory(limit);
        return sendJson(res, 200, history);
      }

      // 10. Static Output Files (/outputs/:file)
      if (pathname.startsWith('/outputs/') && req.method === 'GET') {
        const fileName = path.basename(pathname);
        const filePath = path.join(paths.outputs, fileName);
        if (fs.existsSync(filePath)) {
          const ext = path.extname(filePath).toLowerCase();
          const mimeTypes = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.json': 'application/json'
          };
          res.writeHead(200, {
            'Content-Type': mimeTypes[ext] || 'application/octet-stream',
            'Cache-Control': 'public, max-age=86400'
          });
          fs.createReadStream(filePath).pipe(res);
          return;
        }
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Output file not found');
        return;
      }

      // 11. Static Frontend Assets (public/index.html, etc.)
      let reqPath = pathname === '/' ? '/index.html' : pathname;
      const safePath = path.normalize(reqPath).replace(/^(\.\.[\/\\])+/, '');
      const assetPath = path.join(publicDir, safePath);

      if (fs.existsSync(assetPath) && fs.statSync(assetPath).isFile()) {
        const ext = path.extname(assetPath).toLowerCase();
        const mimeTypes = {
          '.html': 'text/html; charset=utf-8',
          '.css': 'text/css; charset=utf-8',
          '.js': 'application/javascript; charset=utf-8',
          '.json': 'application/json',
          '.svg': 'image/svg+xml',
          '.png': 'image/png',
          '.ico': 'image/x-icon'
        };
        res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
        fs.createReadStream(assetPath).pipe(res);
        return;
      }

      // Fallback: serve index.html for SPA routing
      const indexPath = path.join(publicDir, 'index.html');
      if (fs.existsSync(indexPath)) {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        fs.createReadStream(indexPath).pipe(res);
        return;
      }

      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
    } catch (err) {
      console.error(`[Server Error] ${req.method} ${pathname}: ${err.message}`);
      sendJson(res, 500, { error: err.message });
    }
  });

  return {
    server,
    start: () =>
      new Promise((resolve, reject) => {
        server.listen(port, () => {
          console.log('\n======================================================');
          console.log('       🎨 AI Image Studio Local Web Server            ');
          console.log('======================================================');
          console.log(`Local Web Studio: http://localhost:${port}`);
          console.log(`Outputs Directory: ${paths.outputs}`);
          console.log(`Backend Endpoint:  ${serverConfig.baseUrl}`);
          console.log('\nPress Ctrl+C to stop the server.\n');
          resolve({ port, server });
        });
        server.on('error', reject);
      })
  };
}

// Allow direct CLI execution: node src/server.js
if (import.meta.url === `file://${process.argv[1]}`) {
  const port = parseInt(process.argv[2] || process.env.PORT || '3000', 10);
  const studioServer = createStudioServer({ port });
  studioServer.start().catch(err => {
    console.error('Failed to start studio server:', err.message);
    process.exit(1);
  });
}
