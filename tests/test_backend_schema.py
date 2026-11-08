"""
Tests for FastAPI backend schemas and models catalog.
"""
import unittest

try:
    from backend.models import GenerationRequest, ModelLoadRequest, DEFAULT_MODELS, JobResponse, CancelResponse
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    DEFAULT_MODELS = None

class TestBackendModels(unittest.TestCase):
    def test_default_models_catalog(self):
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed on local host machine (intended for Colab environment)")
        self.assertIn("sdxl-base-1.0", DEFAULT_MODELS)
        self.assertIn("sd-1.5", DEFAULT_MODELS)
        self.assertIn("ssd-1b", DEFAULT_MODELS)
        sdxl = DEFAULT_MODELS["sdxl-base-1.0"]
        self.assertEqual(sdxl["pipeline"], "StableDiffusionXLPipeline")
        self.assertTrue(sdxl["fp16"])

    def test_generation_request_defaults(self):
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed on local host machine (intended for Colab environment)")
        req = GenerationRequest(prompt="Photorealistic portrait")
        self.assertEqual(req.width, 768)
        self.assertEqual(req.height, 1024)
        self.assertEqual(req.steps, 4)
        self.assertEqual(req.guidance_scale, 0.0)
        self.assertEqual(req.num_images, 1)
        self.assertIsNone(req.seed)
        self.assertIsNone(req.idempotency_key)

    def test_job_response_schema(self):
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed on local host machine (intended for Colab environment)")
        job = JobResponse(
            job_id="gen_test",
            type="generation",
            status="queued",
            created_at=1730000000.0
        )
        self.assertEqual(job.job_id, "gen_test")
        self.assertEqual(job.status, "queued")

if __name__ == "__main__":
    unittest.main()
