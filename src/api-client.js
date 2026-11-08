import { serverConfig } from './config.js';

export class StudioApiClient {
  constructor(options = {}) {
    this.baseUrl = (options.baseUrl || serverConfig.baseUrl).replace(/\/+$/, '');
    this.token = (options.token || serverConfig.token || '').trim();
    // Default 30 minutes (1,800,000 ms) for long-running jobs
    this.timeoutMs = options.timeoutMs || serverConfig.timeoutMs || 1800000;
    this.pollIntervalMs = options.pollIntervalMs || 3000;
  }

  /**
   * Internal request helper with Bearer token injection and Cloudflare HTML error protection.
   * Short HTTP timeout per request (30s) because endpoints return immediately (202).
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
    const headers = {
      ...(options.headers || {})
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const controller = new AbortController();
    const reqTimeoutMs = options.reqTimeoutMs || 30000;
    const timeoutId = setTimeout(() => controller.abort(), reqTimeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        let errBody = null;
        let rawText = '';
        try {
          rawText = await response.text();
          try {
            errBody = JSON.parse(rawText);
          } catch {
            errBody = null;
          }
        } catch {
          errBody = null;
        }

        let message = '';
        if (errBody && (errBody.detail || errBody.message)) {
          message = errBody.detail || errBody.message;
        } else if (response.status === 524) {
          message = 'Cloudflare gateway timeout (HTTP 524). The GPU backend may still be processing the job.';
        } else if (response.status === 502 || response.status === 504) {
          message = `Cloudflare upstream gateway error (HTTP ${response.status}). Backend unreachable or tunnel dropped.`;
        } else if (rawText && rawText.includes('<html')) {
          const titleMatch = rawText.match(/<title>([^<]+)<\/title>/i);
          const cleanTitle = titleMatch ? titleMatch[1].trim() : `HTTP ${response.status} error`;
          message = `HTTP ${response.status}: ${cleanTitle}`;
        } else {
          message = rawText || `HTTP ${response.status}: ${response.statusText}`;
        }

        const error = new Error(this._sanitize(message));
        error.status = response.status;
        error.details = errBody || { detail: message };
        error.isCloudflareTimeout = response.status === 524;
        throw error;
      }

      return response;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error(`HTTP request to ${this._sanitize(url)} timed out after ${reqTimeoutMs}ms.`);
      }
      if (err.code === 'ECONNREFUSED' || (err.message && err.message.includes('fetch failed'))) {
        throw new Error(`Could not connect to AI Studio backend at ${this._sanitize(this.baseUrl)}. Is Colab running?`);
      }
      err.message = this._sanitize(err.message);
      throw err;
    }
  }

  _sanitize(text) {
    if (!text || typeof text !== 'string') return text;
    if (this.token && this.token.length > 4) {
      text = text.replaceAll(this.token, '****');
    }
    return text;
  }

  /**
   * Health and hardware check.
   */
  async getHealth() {
    try {
      const res = await this.request('/api/health');
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        const res = await this.request('/health');
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * List configured and available models.
   */
  async getModels() {
    try {
      const res = await this.request('/api/models');
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        const res = await this.request('/models');
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * Submit an asynchronous model loading job. Returns immediately (202 Accepted).
   */
  async submitModelLoad(modelId, hfToken = null, idempotencyKey = null) {
    const headers = { 'Content-Type': 'application/json' };
    if (idempotencyKey) {
      headers['Idempotency-Key'] = idempotencyKey;
    }
    const body = JSON.stringify({
      model_id: modelId,
      hf_token: hfToken,
      idempotency_key: idempotencyKey
    });

    try {
      const res = await this.request('/api/models/load', { method: 'POST', headers, body });
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        const res = await this.request('/models/load', { method: 'POST', headers, body });
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * Submit an asynchronous image generation job. Returns immediately (202 Accepted).
   */
  async submitGeneration(generationParams) {
    const headers = { 'Content-Type': 'application/json' };
    if (generationParams.idempotency_key) {
      headers['Idempotency-Key'] = generationParams.idempotency_key;
    }
    const body = JSON.stringify(generationParams);

    try {
      const res = await this.request('/api/generate', { method: 'POST', headers, body });
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        const res = await this.request('/generate', { method: 'POST', headers, body });
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * List recent jobs on the backend.
   */
  async listJobs(limit = 20) {
    try {
      const res = await this.request(`/api/jobs?limit=${limit}`);
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        const res = await this.request(`/jobs?limit=${limit}`);
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * Poll status of any asynchronous job.
   */
  async getJobStatus(jobId) {
    try {
      const res = await this.request(`/api/jobs/${jobId}`);
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        // Fallback to legacy routes
        const res = await this.request(`/jobs/${jobId}`).catch(() => this.request(`/generation/${jobId}`));
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * Cancel an active or queued job.
   */
  async cancelJob(jobId) {
    try {
      const res = await this.request(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      if (err.status === 404) {
        const res = await this.request(`/jobs/${jobId}/cancel`, { method: 'POST' });
        return await res.json();
      }
      throw err;
    }
  }

  /**
   * Download the raw PNG bytes for a completed job result.
   */
  async getJobImage(jobId, index = 0) {
    let res;
    try {
      res = await this.request(`/api/jobs/${jobId}/result?index=${index}`);
    } catch (err) {
      if (err.status === 404) {
        res = await this.request(`/jobs/${jobId}/image?index=${index}`).catch(() =>
          this.request(`/generation/${jobId}/image`)
        );
      } else {
        throw err;
      }
    }
    const arrayBuffer = await res.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }

  /**
   * Backwards-compatible aliases
   */
  async generate(generationParams) {
    return await this.submitGeneration(generationParams);
  }

  async getGenerationStatus(jobId) {
    return await this.getJobStatus(jobId);
  }

  async getGenerationImage(jobId) {
    return await this.getJobImage(jobId, 0);
  }

  /**
   * Polls job status every 3 seconds for up to maxWaitMs without holding long HTTP requests open.
   */
  async waitForJob(jobId, onProgress = null, maxWaitMs = null, pollIntervalMs = null) {
    const maxWait = maxWaitMs || this.timeoutMs;
    const interval = pollIntervalMs || this.pollIntervalMs;
    const startTime = Date.now();
    let lastProgressMsg = '';
    let consecutiveErrors = 0;

    while (true) {
      if (Date.now() - startTime > maxWait) {
        throw new Error(
          `Job ${jobId} exceeded maximum wait time of ${(maxWait / 60000).toFixed(1)} minutes. ` +
          `The backend may still be running this task. You can inspect it with: node src/cli.js jobs status ${jobId}`
        );
      }

      let jobRecord;
      try {
        jobRecord = await this.getJobStatus(jobId);
        consecutiveErrors = 0;
      } catch (err) {
        if (err.status === 404) {
          throw new Error(
            `Job ${jobId} not found. The Colab runtime may have restarted. Jobs are kept in-process and reset upon restart.`
          );
        }
        consecutiveErrors++;
        if (consecutiveErrors >= 5) {
          throw err;
        }
        if (onProgress) {
          onProgress(`[POLL] Transient network warning (${err.message}). Retrying in ${interval / 1000}s...`);
        }
        await new Promise((resolve) => setTimeout(resolve, interval));
        continue;
      }

      const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(0);
      const stage = jobRecord.stage || jobRecord.status || 'processing';
      const progressPct = jobRecord.progress !== null && jobRecord.progress !== undefined ? ` (${jobRecord.progress}%)` : '';
      const messageText = jobRecord.message || `Stage: ${stage}${progressPct}`;

      if (onProgress && messageText !== lastProgressMsg) {
        onProgress(`[${(jobRecord.status || 'RUNNING').toUpperCase()}] ${messageText} (${elapsedSec}s)`);
        lastProgressMsg = messageText;
      }

      if (jobRecord.status === 'completed') {
        return jobRecord;
      }

      if (jobRecord.status === 'cancelled') {
        throw new Error(`Job ${jobId} was cancelled (${jobRecord.message || 'user cancellation'}).`);
      }

      if (jobRecord.status === 'failed') {
        throw new Error(jobRecord.error || jobRecord.message || 'Job failed during GPU execution.');
      }

      await new Promise((resolve) => setTimeout(resolve, interval));
    }
  }

  /**
   * Loads a model with asynchronous queuing and status polling.
   */
  async loadModel(modelId, hfToken = null, onProgress = null, options = {}) {
    if (onProgress) onProgress(`Submitting model load task for ${modelId}...`);
    const idempotencyKey = options.idempotencyKey || `load_${modelId}_${Date.now()}`;
    const initial = await this.submitModelLoad(modelId, hfToken, idempotencyKey);

    const initialId = initial.job_id || initial.id;
    if (initial.status === 'completed') {
      return initial;
    }

    const completed = await this.waitForJob(initialId, onProgress, options.timeoutMs || this.timeoutMs, options.pollIntervalMs);
    return completed;
  }

  /**
   * Generates an image and retrieves result with asynchronous queuing,
   * status polling, and binary download. Supports batch downloads.
   */
  async generateAndDownload(params, onProgress = null, options = {}) {
    if (onProgress) onProgress('Submitting generation task to GPU queue...');
    const initial = await this.submitGeneration(params);
    const jobId = initial.job_id || initial.id;

    if (onProgress) onProgress(`Job accepted (ID: ${jobId}). Beginning status polling...`);
    const completedRecord = await this.waitForJob(jobId, onProgress, options.timeoutMs || this.timeoutMs, options.pollIntervalMs);

    const numImages = completedRecord.num_images || (completedRecord.results ? completedRecord.results.length : 1);
    const downloadedImages = [];

    for (let i = 0; i < numImages; i++) {
      if (onProgress && numImages > 1) {
        onProgress(`Downloading image ${i + 1}/${numImages} for job ${jobId}...`);
      }
      const imageBuffer = await this.getJobImage(jobId, i);
      let seed = completedRecord.seed;
      if (completedRecord.results && completedRecord.results[i]) {
        seed = completedRecord.results[i].seed;
      }
      downloadedImages.push({
        index: i,
        seed,
        imageBuffer
      });
    }

    return {
      jobRecord: completedRecord,
      imageBuffer: downloadedImages[0]?.imageBuffer, // Backwards compatibility for single image
      images: downloadedImages
    };
  }
}
