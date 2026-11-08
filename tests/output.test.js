import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { saveGenerationOutput, getGenerationHistory } from '../src/output.js';

test('Output: saveGenerationOutput writes image and companion json metadata', async () => {
  // 1x1 dummy PNG buffer
  const dummyBuffer = Buffer.from('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6360000000020001e221bc330000000049454e44ae426082', 'hex');

  const metadata = {
    id: `test_${Date.now()}`,
    prompt: 'Unit test image prompt',
    negative_prompt: 'bad quality',
    width: 768,
    height: 1024,
    steps: 25,
    guidance_scale: 7.0,
    seed: 42,
    model_id: 'sdxl-base-1.0',
    duration_seconds: 5.2
  };

  const res = await saveGenerationOutput(dummyBuffer, metadata);

  assert.ok(fs.existsSync(res.imagePath), 'Image file must exist on disk');
  assert.ok(fs.existsSync(res.metaPath), 'Metadata file must exist on disk');

  const loadedMeta = JSON.parse(await fs.promises.readFile(res.metaPath, 'utf8'));
  assert.equal(loadedMeta.seed, 42);
  assert.equal(loadedMeta.prompt, 'Unit test image prompt');
  assert.equal(loadedMeta.model_id, 'sdxl-base-1.0');

  // Verify history contains this record
  const history = await getGenerationHistory(10);
  assert.ok(history.length > 0);
  assert.ok(history.some(item => item.id === metadata.id));

  // Clean up test files
  await fs.promises.unlink(res.imagePath).catch(() => {});
  await fs.promises.unlink(res.metaPath).catch(() => {});
});

