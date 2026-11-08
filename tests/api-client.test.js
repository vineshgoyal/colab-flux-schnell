import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { StudioApiClient } from '../src/api-client.js';

test('API Client: connects to server, handles Bearer auth and asynchronous polling flow', async () => {
  const MOCK_TOKEN = 'sk_test_studio_token';
  const dummyPng = Buffer.from('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6360000000020001e221bc330000000049454e44ae426082', 'hex');

  let pollCount = 0;

  // Spin up an in-process mock HTTP server
  const server = http.createServer((req, res) => {
    const authHeader = req.headers['authorization'];

    // Verify token
    if (authHeader !== `Bearer ${MOCK_TOKEN}`) {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ detail: 'Invalid or missing API token' }));
      return;
    }

    const url = req.url.startsWith('/api/') ? req.url.slice(4) : req.url;

    if (url === '/health' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'healthy',
        version: '2.0.0',
        gpu_available: true,
        device_name: 'Tesla T4',
        current_model: 'sdxl-base-1.0',
        vram: { total_gb: 15.0, allocated_gb: 6.5, reserved_gb: 7.0, free_gb: 8.5 }
      }));
      return;
    }

    if (url === '/models' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        current_model: 'sdxl-base-1.0',
        models: [{ id: 'sdxl-base-1.0', name: 'SDXL Base 1.0', repo_id: 'stabilityai/stable-diffusion-xl-base-1.0' }]
      }));
      return;
    }

    if (url === '/generate' && req.method === 'POST') {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        const payload = JSON.parse(body);
        // Return 202 with initial queued job
        res.writeHead(202, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          id: 'gen_mock123',
          status: 'processing',
          progress: 'Denoising step 10/25 (40%)...',
          prompt: payload.prompt,
          seed: payload.seed || 999
        }));
      });
      return;
    }

    if ((url === '/jobs/gen_mock123' || url === '/generation/gen_mock123') && req.method === 'GET') {
      pollCount++;
      const isDone = pollCount >= 2;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        id: 'gen_mock123',
        status: isDone ? 'completed' : 'processing',
        progress: isDone ? 'Finished' : 'Denoising step 20/25 (80%)...',
        seed: 999,
        duration_seconds: 2.1,
        vram: { allocated_gb: 6.8 }
      }));
      return;
    }

    if ((url === '/jobs/gen_mock123/image' || url === '/jobs/gen_mock123/result' || url === '/jobs/gen_mock123/result?index=0' || url === '/generation/gen_mock123/image') && req.method === 'GET') {
      res.writeHead(200, {
        'Content-Type': 'image/png',
        'X-Generation-Seed': '999'
      });
      res.end(dummyPng);
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ detail: 'Not found' }));
  });

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    // 1. Test unauthorized call
    const unauthClient = new StudioApiClient({ baseUrl, token: 'wrong_token' });
    await assert.rejects(async () => {
      await unauthClient.getHealth();
    }, /Invalid or missing API token/);

    // 2. Test authorized health check
    const client = new StudioApiClient({ baseUrl, token: MOCK_TOKEN, timeoutMs: 10000 });
    const health = await client.getHealth();
    assert.equal(health.status, 'healthy');
    assert.equal(health.device_name, 'Tesla T4');

    // 3. Test list models
    const modelsResp = await client.getModels();
    assert.equal(modelsResp.current_model, 'sdxl-base-1.0');

    // 4. Test async polling generateAndDownload flow
    const progressLogs = [];
    const genResult = await client.generateAndDownload(
      { prompt: 'Testing portrait generation', seed: 999 },
      (msg) => progressLogs.push(msg)
    );

    assert.equal(genResult.jobRecord.id, 'gen_mock123');
    assert.equal(genResult.jobRecord.status, 'completed');
    assert.ok(Buffer.isBuffer(genResult.imageBuffer));
    assert.equal(genResult.imageBuffer.length, dummyPng.length);
    assert.ok(progressLogs.length >= 2, 'Should have logged polling progress');

  } finally {
    server.close();
  }
});
