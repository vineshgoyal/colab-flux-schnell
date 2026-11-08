import fs from 'node:fs';
import path from 'node:path';
import { paths, generationConfig } from './config.js';

/**
 * Validates and normalizes character profile schema.
 * Enforces explicit adult age (18+) to ensure safety compliance.
 */
export function validateCharacterProfile(profile) {
  if (!profile.id || typeof profile.id !== 'string') {
    throw new Error("Character 'id' is required and must be a string slug.");
  }
  // Sanitize id for file safety
  profile.id = profile.id.toLowerCase().replace(/[^a-z0-9_-]/g, '-');

  if (!profile.name || typeof profile.name !== 'string') {
    throw new Error("Character 'name' is required.");
  }

  const age = parseInt(profile.age, 10);
  if (isNaN(age) || age < 18) {
    throw new Error(`Compliance violation: Character age must be an explicit adult age (18+). Received: ${profile.age}`);
  }
  profile.age = age;

  if (!profile.facial_features) {
    throw new Error("Character 'facial_features' is required for facial consistency.");
  }

  // Ensure nested hair object
  if (typeof profile.hair === 'string') {
    profile.hair = { color: profile.hair, style: 'natural', length: 'medium' };
  } else if (!profile.hair) {
    profile.hair = { color: 'dark', style: 'natural', length: 'medium' };
  }

  if (!Array.isArray(profile.reference_images)) {
    profile.reference_images = [];
  }

  return profile;
}

/**
 * Saves a character profile JSON to characters directory.
 */
export async function saveCharacterProfile(profile) {
  const validated = validateCharacterProfile(profile);
  const filePath = path.join(paths.characters, `${validated.id}.json`);
  await fs.promises.writeFile(filePath, JSON.stringify(validated, null, 2), 'utf8');
  return { path: filePath, profile: validated };
}

/**
 * Loads a character profile by id.
 */
export async function getCharacterProfile(id) {
  const cleanId = id.toLowerCase().replace(/[^a-z0-9_-]/g, '-');
  const filePath = path.join(paths.characters, `${cleanId}.json`);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Character profile '${cleanId}' not found at ${filePath}`);
  }
  const content = await fs.promises.readFile(filePath, 'utf8');
  return JSON.parse(content);
}

/**
 * Lists all available character profiles.
 */
export async function listCharacterProfiles() {
  if (!fs.existsSync(paths.characters)) {
    return [];
  }
  const files = await fs.promises.readdir(paths.characters);
  const profiles = [];
  for (const f of files.filter((file) => file.endsWith('.json'))) {
    try {
      const content = await fs.promises.readFile(path.join(paths.characters, f), 'utf8');
      profiles.push(JSON.parse(content));
    } catch {
      // Skip invalid JSON
    }
  }
  return profiles;
}

/**
 * Synthesizes a unified diffusion prompt blending the character's physical likeness
 * with dynamic scene, pose, wardrobe, and lighting descriptors.
 */
export function synthesizeCharacterPrompt(profile, sceneOptions = {}) {
  const opts = typeof sceneOptions === 'string' ? { scene: sceneOptions } : (sceneOptions || {});
  const parts = [];

  // 1. Identity & Age Anchor
  const ethnicity = profile.ethnicity ? `${profile.ethnicity} ` : '';
  const gender = profile.gender || 'person';
  parts.push(`Photorealistic portrait of a fictional ${profile.age}-year-old ${ethnicity}${gender} named ${profile.name}`);

  // 2. Physical Identity Core (anchors facial structure across images)
  const hairDesc = `${profile.hair?.length || ''} ${profile.hair?.style || ''} ${profile.hair?.color || ''} hair`.trim();
  const eyeDesc = profile.eye_color ? `${profile.eye_color} eyes` : '';
  const bodyDesc = profile.body || '';

  const physicalTraits = [profile.facial_features, eyeDesc, hairDesc, bodyDesc]
    .filter(Boolean)
    .join(', ');
  if (physicalTraits) {
    parts.push(physicalTraits);
  }

  // 3. Dynamic Scene, Action, and Setting (User prompt)
  const sceneText = (opts.scene || '').trim();
  if (sceneText) {
    parts.push(sceneText);
  }

  // 4. Wardrobe / Outfit (Only add if user didn't already specify clothing)
  const hasUserClothing = Boolean(
    opts.clothing ||
    /\b(wearing|dressed in|outfit|dress|gown|suit|bikini|swimsuit|jacket|jeans|skirt|costume|uniform|lingerie|attire|apparel|wardrobe|top|blouse|pants|shorts|monokini|bodysuit|corset|leotard|robe|coat)\b/i.test(sceneText)
  );

  let clothing = null;
  if (opts.clothing) {
    clothing = opts.clothing;
  } else if (!hasUserClothing && profile.clothing_preferences) {
    clothing = profile.clothing_preferences;
  }

  if (clothing) {
    const formattedClothing = clothing.toLowerCase().startsWith('wearing') ? clothing : `wearing ${clothing}`;
    parts.push(formattedClothing);
  }

  // 5. Lighting, Camera, and Photographic Aesthetics
  const photoPreset = opts.preset && generationConfig.photography_presets ? generationConfig.photography_presets[opts.preset] : null;
  let photoStyle = opts.style;
  const isFullBody = /\b(full-body|full body|head-to-toe|full length|wide shot)\b/i.test(sceneText);

  if (!photoStyle && photoPreset) {
    photoStyle = `${photoPreset.framing}, ${photoPreset.lens}, ${photoPreset.lighting}, ${photoPreset.style}`;
  } else if (!photoStyle) {
    if (isFullBody) {
      photoStyle = 'Hasselblad medium format editorial photography, 50mm lens, natural studio rim lighting, authentic skin texture, full head-to-toe framing, 8k resolution';
    } else {
      photoStyle = profile.photography_style || 
        'Hasselblad medium format editorial photography, 85mm portrait lens, natural studio rim light, authentic skin texture, 8k resolution';
    }
  }
  if (photoStyle) {
    parts.push(photoStyle);
  }

  // Clean, normalize, and deduplicate parts to prevent contradictory tokens
  const seen = new Set();
  const cleanParts = [];
  for (const part of parts) {
    const trimmed = (part || '').trim().replace(/^[,;\s]+|[,;\s]+$/g, '');
    if (!trimmed) continue;
    const lower = trimmed.toLowerCase();
    if (!seen.has(lower)) {
      seen.add(lower);
      cleanParts.push(trimmed);
    }
  }
  const fullPrompt = cleanParts.join(', ');

  // Synthesize negative prompt
  const baseNegative = opts.negative_prompt || '';
  const charNegative = profile.negative_prompt_additions || '';
  const presetNegative = photoPreset?.negative_prompt || '';
  const fullNegative = [baseNegative, charNegative, presetNegative].filter(Boolean).join(', ');

  // Deterministic seed
  const seed = opts.seed !== undefined && opts.seed !== null 
    ? opts.seed 
    : (profile.preferred_seed || null);

  return {
    prompt: fullPrompt,
    negative_prompt: fullNegative,
    seed,
    character_id: profile.id
  };
}
