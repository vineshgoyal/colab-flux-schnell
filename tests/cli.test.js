import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLI_PATH = path.resolve(__dirname, '../src/cli.js');

test('CLI: --help prints command reference with jobs and presets', async () => {
  const { stdout } = await execFileAsync('node', [CLI_PATH, '--help']);
  assert.ok(stdout.includes('AI Image Studio CLI'));
  assert.ok(stdout.includes('doctor'));
  assert.ok(stdout.includes('generate'));
  assert.ok(stdout.includes('character'));
  assert.ok(stdout.includes('jobs'));
  assert.ok(stdout.includes('beauty-portrait'));
  assert.ok(stdout.includes('photoshoot'));
});

test('CLI: character list displays saved profiles', async () => {
  const { stdout } = await execFileAsync('node', [CLI_PATH, 'character', 'list']);
  assert.ok(stdout.includes('Elena Vance'));
  assert.ok(stdout.includes('Marcus Reid'));
});

test('CLI: character show outputs full json', async () => {
  const { stdout } = await execFileAsync('node', [CLI_PATH, 'character', 'show', 'elena-vance']);
  const parsed = JSON.parse(stdout);
  assert.equal(parsed.id, 'elena-vance');
  assert.equal(parsed.age, 28);
});

test('CLI: character create via flags creates and validates profile', async () => {
  const testCharId = 'cli-test-char';
  const charPath = path.resolve(__dirname, `../characters/${testCharId}.json`);

  try {
    const { stdout } = await execFileAsync('node', [
      CLI_PATH, 'character', 'create',
      '--id', testCharId,
      '--name', 'CLI Test Persona',
      '--age', '30',
      '--facial', 'sharp features',
      '--eyes', 'blue'
    ]);

    assert.ok(stdout.includes('saved successfully'));
    assert.ok(fs.existsSync(charPath));
    const data = JSON.parse(fs.readFileSync(charPath, 'utf8'));
    assert.equal(data.age, 30);
  } finally {
    if (fs.existsSync(charPath)) {
      fs.unlinkSync(charPath);
    }
  }
});

test('CLI: doctor reports local health metrics cleanly', async () => {
  const { stdout, stderr } = await execFileAsync('node', [CLI_PATH, 'doctor']).catch(e => e);
  const output = (stdout || '') + (stderr || '');
  assert.ok(output.includes('AI Image Studio: System Doctor'));
  assert.ok(output.includes('Node.js runtime'));
  assert.ok(output.includes('Output directory'));
  assert.ok(output.includes('Generation Readiness Status'));
});
