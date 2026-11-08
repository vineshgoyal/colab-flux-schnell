import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { StudioApiClient } from '../src/api-client.js';
import { resolveGenerationParams } from '../src/config.js';
import { synthesizeCharacterPrompt, getCharacterProfile } from '../src/characters.js';
import { saveGenerationOutput } from '../src/output.js';
import fs from 'node:fs';

test('Acceptance Suite: Asynchronous Job Lifecycle, Idempotency, Model Mismatch, & Cloudflare 524 Protection', async () => {
  const MOCK_TOKEN = 'studio_secret_token_123';
  const dummyPng = Buffer.from(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6360000000020001e221bc330000000049454e44ae426082',
    'hex'
  );

  let activeModel = 'sd-1.5';
  const submittedJobs = new Map();
  const idempotencyMap = new Map();
  let pollCounter = 0;

  const server = http.createServer((req, res) => {
    const authHeader = req.headers['authorization'];
    const idempotencyKey = req.headers['idempotency-key'];

    // 10. Reject unauthenticated requests
    if (authHeader !== `Bearer ${MOCK_TOKEN}`) {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ detail: 'Invalid or missing API token in Authorization header' }));
      return;
    }

    const url = new URL(req.url, 'http://127.0.0.1');
    const pathname = url.pathname;

    // Health
    if ((pathname === '/api/health' || pathname === '/health') && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          status: 'healthy',
          version: '2.1.0',
          gpu_available: true,
          device_name: 'Tesla T4',
          current_model: activeModel,
          model_loaded: Boolean(activeModel),
          generation_ready: Boolean(activeModel),
          active_jobs: submittedJobs.size,
          queue_length: 0,
          vram: { total_gb: 15.0, allocated_gb: 5.2, reserved_gb: 6.0, free_gb: 9.8 }
        })
      );
      return;
    }

    // Models
    if ((pathname === '/api/models' || pathname === '/models') && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          current_model: activeModel,
          models: [{ id: 'sd-1.5', name: 'SD 1.5' }, { id: 'sdxl-base-1.0', name: 'SDXL 1.0' }]
        })
      );
      return;
    }

    // 4 & 5. Reject generation when requested model is not loaded (Model mismatch)
    if ((pathname === '/api/generate' || pathname === '/generate') && req.method === 'POST') {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        const payload = JSON.parse(body);

        // 7. Handle duplicate submissions using idempotency keys
        if (idempotencyKey && idempotencyMap.has(idempotencyKey)) {
          const existingJobId = idempotencyMap.get(idempotencyKey);
          res.writeHead(202, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(submittedJobs.get(existingJobId)));
          return;
        }

        if (!activeModel) {
          res.writeHead(409, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ detail: 'No model is currently loaded in GPU VRAM.' }));
          return;
        }

        if (payload.model_id && payload.model_id !== activeModel) {
          res.writeHead(409, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              detail: `Model mismatch: active GPU model is '${activeModel}', but '${payload.model_id}' was requested.`
            })
          );
          return;
        }

        // 1. Accept generation job and return HTTP 202
        const jobId = `gen_${Date.now().toString(36)}`;
        const jobRecord = {
          job_id: jobId,
          id: jobId,
          type: 'generation',
          status: 'queued',
          stage: 'waiting',
          progress: 0,
          message: 'Queued in GPU worker line...',
          prompt: payload.prompt,
          seed: payload.seed || 12345,
          num_images: payload.num_images || 1,
          model_id: activeModel,
          created_at: Date.now() / 1000
        };

        submittedJobs.set(jobId, jobRecord);
        if (idempotencyKey) {
          idempotencyMap.set(idempotencyKey, jobId);
        }

        res.writeHead(202, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(jobRecord));
      });
      return;
    }

    // List Jobs
    if ((pathname === '/api/jobs' || pathname === '/jobs') && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(Array.from(submittedJobs.values())));
      return;
    }

    // Cancel Job
    if (pathname.includes('/cancel') && req.method === 'POST') {
      const parts = pathname.split('/');
      const jobId = parts[parts.indexOf('cancel') - 1];
      if (submittedJobs.has(jobId)) {
        const job = submittedJobs.get(jobId);
        job.status = 'cancelled';
        job.message = 'Cancelled by user request';
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ job_id: jobId, status: 'cancelled', message: 'Job successfully cancelled' }));
      } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Job not found' }));
      }
      return;
    }

    // 2. Poll job status
    if ((pathname.startsWith('/api/jobs/') || pathname.startsWith('/jobs/')) && req.method === 'GET' && !pathname.includes('/result') && !pathname.includes('/image')) {
      const jobId = pathname.split('/').pop();
      const job = submittedJobs.get(jobId);
      if (!job) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Job not found' }));
        return;
      }

      pollCounter++;
      if (job.status !== 'cancelled') {
        if (pollCounter === 1) {
          job.status = 'running';
          job.stage = 'diffusion';
          job.progress = 50;
          job.message = 'Denoising step 12/25 (50%)...';
        } else {
          job.status = 'completed';
          job.stage = 'completed';
          job.progress = 100;
          job.execution_duration_seconds = 8.5;
          job.message = 'Generation completed successfully.';
          job.results = [{ index: 0, seed: job.seed, url: `/api/jobs/${jobId}/result?index=0`, width: 512, height: 768 }];
        }
      }

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(job));
      return;
    }

    // 9. Download result
    if ((pathname.includes('/result') || pathname.includes('/image')) && req.method === 'GET') {
      const segments = pathname.split('/');
      const jidIdx = segments.findIndex(s => s === 'result' || s === 'image') - 1;
      const jobId = segments[jidIdx];

      if (jobId === 'fail_download') {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Image data unavailable' }));
        return;
      }

      res.writeHead(200, {
        'Content-Type': 'image/png',
        'X-Generation-Seed': '12345'
      });
      res.end(dummyPng);
      return;
    }

    // Simulated Cloudflare 524 HTML error
    if (pathname === '/test-cloudflare-524') {
      res.writeHead(524, { 'Content-Type': 'text/html' });
      res.end('<!DOCTYPE html><html><head><title>A timeout occurred | Cloudflare</title></head><body><h1>Error 524 Gateway Timeout</h1></body></html>');
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ detail: 'Not found' }));
  });

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  const client = new StudioApiClient({
    baseUrl,
    token: MOCK_TOKEN,
    pollIntervalMs: 50 // Fast polling for test speed
  });

  try {
    // Test 10: Reject unauthenticated requests
    const badClient = new StudioApiClient({ baseUrl, token: 'wrong_secret' });
    await assert.rejects(async () => {
      await badClient.getHealth();
    }, /Invalid or missing API token/);

    // Test 1 & 2: Accept generation job, return 202, and poll to completion
    pollCounter = 0;
    const acceptedJob = await client.submitGeneration({
      prompt: 'High fashion portrait',
      model_id: 'sd-1.5'
    });
    assert.ok(acceptedJob.job_id);
    assert.equal(acceptedJob.status, 'queued');

    const completed = await client.waitForJob(acceptedJob.job_id);
    assert.equal(completed.status, 'completed');
    assert.equal(completed.progress, 100);

    // Test 4: Reject model mismatch
    await assert.rejects(async () => {
      await client.submitGeneration({
        prompt: 'Fashion shot',
        model_id: 'sdxl-base-1.0' // Active is sd-1.5
      });
    }, /Model mismatch/);

    // Test 7: Handle duplicate submissions using idempotency keys
    const testIdempKey = 'idemp_key_unique_test_1';
    const firstSubmission = await client.submitGeneration({
      prompt: 'Idempotency test',
      model_id: 'sd-1.5',
      idempotency_key: testIdempKey
    });
    const secondSubmission = await client.submitGeneration({
      prompt: 'Idempotency test',
      model_id: 'sd-1.5',
      idempotency_key: testIdempKey
    });
    assert.equal(firstSubmission.job_id, secondSubmission.job_id, 'Duplicate idempotency key must return same job ID');

    // Test Cancellation
    const jobToCancel = await client.submitGeneration({ prompt: 'Cancel test', model_id: 'sd-1.5' });
    const cancelRes = await client.cancelJob(jobToCancel.job_id);
    assert.equal(cancelRes.status, 'cancelled');

    // Test 9: Handle failed download
    await assert.rejects(async () => {
      await client.getJobImage('fail_download');
    }, /Image data unavailable/);

    // Test 13: Intercept Cloudflare 524 HTML error cleanly without dumping raw HTML
    await assert.rejects(async () => {
      await client.request('/test-cloudflare-524');
    }, (err) => {
      assert.equal(err.status, 524);
      assert.ok(err.isCloudflareTimeout);
      assert.ok(!err.message.includes('<!DOCTYPE html>'), 'Must not dump full raw HTML in error message');
      assert.ok(err.message.includes('Cloudflare gateway timeout'), 'Must provide concise human-readable error');
      return true;
    });

    // Test 11 & 12: Character Profile synthesis and metadata verification
    const profile = await getCharacterProfile('elena-vance');
    const synthesized = synthesizeCharacterPrompt(profile, {
      scene: 'sitting at a luxury penthouse terrace at dusk',
      preset: 'beauty-portrait'
    });
    assert.ok(synthesized.prompt.includes('Elena Vance'));
    assert.ok(synthesized.prompt.includes('28-year-old'));
    assert.ok(synthesized.negative_prompt.includes('plastic skin'));

    const dummyBuffer = Buffer.from('test_png_bytes');
    const saveMeta = await saveGenerationOutput(dummyBuffer, {
      id: 'gen_test_output_1',
      prompt: synthesized.prompt,
      seed: 8888,
      model_id: 'sd-1.5',
      character_id: profile.id
    });
    assert.ok(fs.existsSync(saveMeta.imagePath));
    assert.ok(fs.existsSync(saveMeta.metaPath));
    const diskMeta = JSON.parse(fs.readFileSync(saveMeta.metaPath, 'utf8'));
    assert.equal(diskMeta.character_id, 'elena-vance');
    assert.equal(diskMeta.seed, 8888);

    // Cleanup saved test file
    fs.unlinkSync(saveMeta.imagePath);
    fs.unlinkSync(saveMeta.metaPath);

  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});

