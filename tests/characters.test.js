import test from 'node:test';
import assert from 'node:assert/strict';
import {
  validateCharacterProfile,
  synthesizeCharacterPrompt,
  getCharacterProfile,
  listCharacterProfiles
} from '../src/characters.js';

test('Characters: rejects profile missing adult age or underage (<18)', () => {
  assert.throws(() => {
    validateCharacterProfile({
      id: 'test-child',
      name: 'Test Child',
      age: 15,
      facial_features: 'freckles'
    });
  }, /Compliance violation: Character age must be an explicit adult age \(18\+\)/);

  assert.throws(() => {
    validateCharacterProfile({
      id: 'test-no-age',
      name: 'No Age',
      facial_features: 'freckles'
    });
  }, /Compliance violation/);
});

test('Characters: validates valid adult character profile', () => {
  const profile = validateCharacterProfile({
    id: 'victoria-cross',
    name: 'Victoria Cross',
    age: 29,
    facial_features: 'sharp jawline, cheekbones',
    eye_color: 'amber',
    hair: 'dark brown wavy'
  });
  assert.equal(profile.id, 'victoria-cross');
  assert.equal(profile.age, 29);
  assert.equal(typeof profile.hair, 'object');
  assert.equal(profile.hair.color, 'dark brown wavy');
});

test('Characters: synthesizes character prompt combining identity, traits, and scene', () => {
  const profile = {
    id: 'elena-vance',
    name: 'Elena Vance',
    age: 28,
    ethnicity: 'Mediterranean',
    gender: 'female',
    facial_features: 'high cheekbones, light freckles',
    eye_color: 'emerald green',
    hair: { color: 'chestnut', style: 'waves', length: 'long' },
    body: 'slender athletic build',
    clothing_preferences: 'tailored trench coat',
    photography_style: 'Vogue 85mm f/1.4',
    preferred_seed: 718293,
    negative_prompt_additions: 'plastic skin'
  };

  const synthesized = synthesizeCharacterPrompt(profile, {
    scene: 'drinking espresso at a cafe in Paris',
    clothing: 'black cashmere turtleneck',
    negative_prompt: 'blurry, bad anatomy'
  });

  assert.ok(synthesized.prompt.includes('Elena Vance'));
  assert.ok(synthesized.prompt.includes('28-year-old'));
  assert.ok(synthesized.prompt.includes('emerald green eyes'));
  assert.ok(synthesized.prompt.includes('drinking espresso at a cafe in Paris'));
  assert.ok(synthesized.prompt.includes('black cashmere turtleneck'));
  assert.ok(synthesized.negative_prompt.includes('plastic skin'));
  assert.equal(synthesized.seed, 718293);
});

test('Characters: avoids adding default outfit when user explicitly specifies clothing in prompt', () => {
  const profile = {
    id: 'elena-vance',
    name: 'Elena Vance',
    age: 28,
    facial_features: 'high cheekbones',
    clothing_preferences: 'tailored trench coat'
  };

  const synthesized = synthesizeCharacterPrompt(profile, {
    scene: 'wearing a fitted red mini dress on the red carpet'
  });

  assert.ok(synthesized.prompt.includes('wearing a fitted red mini dress'));
  assert.ok(!synthesized.prompt.includes('tailored trench coat'), 'Should not add default trench coat when user explicitly specified dress');
});

test('Characters: can read pre-existing character profiles from disk', async () => {
  const elena = await getCharacterProfile('elena-vance');
  assert.equal(elena.name, 'Elena Vance');
  assert.equal(elena.age, 28);
  assert.ok(elena.facial_features);

  const all = await listCharacterProfiles();
  assert.ok(all.length >= 2);
});

