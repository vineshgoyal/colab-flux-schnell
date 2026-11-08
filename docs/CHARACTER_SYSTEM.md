# Character Consistency System Specification

## 1. Objective

Provide consistent, photorealistic fictional adult characters across varying scenes, outfits, camera angles, backgrounds, and lighting conditions using pure prompt composition, deterministic seed anchoring, and photographic styling tokens—without requiring heavy LoRA training or third-party proprietary APIs.

---

## 2. Character Profile Schema

Character profiles are stored as standalone JSON files under the `characters/` directory (e.g. `characters/elena_vance.json`).

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CharacterProfile",
  "type": "object",
  "required": ["id", "name", "age", "facial_features", "eye_color", "hair", "body"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier slug (e.g., 'elena-vance')"
    },
    "name": {
      "type": "string",
      "description": "Full fictional character name"
    },
    "age": {
      "type": "integer",
      "minimum": 18,
      "description": "Explicit adult age (Must be 18+ for compliance)"
    },
    "gender": {
      "type": "string",
      "description": "Character gender"
    },
    "ethnicity": {
      "type": "string",
      "description": "Ethnicity / cultural heritage description"
    },
    "facial_features": {
      "type": "string",
      "description": "Distinctive facial attributes (e.g., 'high cheekbones, subtle freckles across nose bridge, soft jawline')"
    },
    "eye_color": {
      "type": "string",
      "description": "Precise eye color and pupil characteristics"
    },
    "hair": {
      "type": "object",
      "properties": {
        "color": { "type": "string" },
        "style": { "type": "string" },
        "length": { "type": "string" }
      },
      "required": ["color", "style", "length"]
    },
    "body": {
      "type": "string",
      "description": "Height, build, and proportions"
    },
    "clothing_preferences": {
      "type": "string",
      "description": "Default wardrobe style if unspecified in generation"
    },
    "photography_style": {
      "type": "string",
      "description": "Default lens, lighting, camera sensor, and film stock aesthetics"
    },
    "preferred_seed": {
      "type": ["integer", "null"],
      "description": "Deterministic seed known to yield consistent facial likeness"
    },
    "negative_prompt_additions": {
      "type": "string",
      "description": "Character-specific negative tokens"
    },
    "reference_images": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Paths or URLs to reference images (for future IP-Adapter / ControlNet extensions)"
    }
  }
}
```

---

## 3. Prompt Synthesis Algorithm

To maintain consistency without weight fine-tuning, the `src/characters.js` engine constructs prompts following an ordered priority hierarchy:

```
[Identity Anchor] + [Physical Core] + [Action / Pose] + [Wardrobe] + [Environment & Background] + [Lighting & Composition] + [Camera & Lens Render Directives]
```

### Synthesis Formula:
1. **Identity Anchor**: `"Photorealistic photographic portrait of a fictional {age}-year-old {ethnicity} {gender} named {name}"`
2. **Physical Core**: `"{facial_features}, {eye_color} eyes, {hair.length} {hair.style} {hair.color} hair, {body}"`
3. **Scene / Subject Action**: `"{scene_prompt}"` (e.g. `"sipping espresso at a sunlit Parisian sidewalk cafe"`)
4. **Wardrobe**: `"{clothing}"` (defaults to character's `clothing_preferences` if not provided)
5. **Lighting & Photography**: `"{photography_style}"` (e.g. `"Hasselblad medium format photography, 85mm f/1.4 lens, natural golden hour rim light, 8k resolution, authentic skin texture"`)

---

## 4. Multi-Scene Consistency Strategy

| Strategy Element | Implementation |
|---|---|
| **Deterministic Seed Anchoring** | When generating variations of a character, the user can lock the character's `preferred_seed` across slight prompt mutations to keep facial geometry stable. |
| **Fixed Trigger Keywords** | Keep identical descriptor tokens (e.g., `"subtle freckles across nose bridge, almond-shaped emerald green eyes"`) constant across every generation. |
| **Negative Token Hardening** | Standard negative prompt (`"bad anatomy, mutated hands, plastic skin, cartoon, low quality"`) augmented by character exclusions. |
| **Future Extension Hooks** | Profiles store `reference_images` array ready for IP-Adapter / ControlNet once extended. |

