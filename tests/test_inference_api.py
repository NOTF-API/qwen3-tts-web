import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np

from qwen3_tts_web import server
from qwen3_tts_web.config import Settings
from qwen3_tts_web.inference import InferenceScheduler
from qwen3_tts_web.library import AudioLibrary
from test_inference import FakeMemory


class ApiModel:
    device = "cuda:0"

    def __init__(self):
        self.entered = threading.Event()
        self.gate = None
        self.calls = []

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        self.calls.append(list(text))
        self.entered.set()
        if self.gate:
            self.gate.wait(5)
        return [np.full(2400, 0.01 * (i + 1)) for i in range(len(text))], 24000

    def generate_voice_design(self, text, language, instruct):
        return self.generate_voice_clone(text, language, instruct)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = ApiModel()
        self.scheduler = InferenceScheduler(Settings(self.root, batch_wait_ms=20), lambda *_: self.model,
                                            threading.Lock(), FakeMemory)
        self.patches = [patch.object(server, "OUT_DIR", self.root),
                        patch.object(server, "model", self.model),
                        patch.object(server, "prepare_tts", return_value=[SimpleNamespace()]),
                        patch.object(server.app.state, "scheduler", self.scheduler)]
        for item in self.patches:
            item.start()
        self.scheduler.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")

    async def asyncTearDown(self):
        if self.model.gate:
            self.model.gate.set()
        await self.client.aclose()
        await asyncio.to_thread(self.scheduler.close)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def request(self, **kwargs):
        return dict(text="测试音频", language="Chinese", role="测试", **kwargs)

    async def test_batch_mixed_modes_preserves_order_and_saved_metadata(self):
        response = await self.client.post("/api/generate-batch", json={"items": [
            self.request(synthesis_mode="clone"),
            dict(synthesis_mode="design", text="不同的文本", instruct="温暖", language="Chinese"),
        ]})
        self.assertEqual(response.status_code, 200)
        results = response.json()["items"]
        self.assertEqual([r["status"] for r in results], ["ok", "ok"])
        self.assertEqual([r["result"]["clip"]["text"] for r in results], ["测试音频", "不同的文本"])
        self.assertEqual([r["result"]["clip"]["synthesis_mode"] for r in results], ["clone", "design"])
        self.assertEqual(len(AudioLibrary(self.root).list()), 2)
        for result in results:
            stats = result["result"]["clip"]["generation_stats"]
            self.assertEqual(stats["batch_size"], 1)
            self.assertGreaterEqual(stats["batch_seconds"], 0)
        records = (await self.client.get('/api/clips')).json()['clips']
        self.assertTrue(all(record.get('generation_stats') for record in records))

    async def test_single_file_api_remains_compatible(self):
        response = await self.client.post("/api/tts", json=self.request(mode="file"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))

    async def test_same_clip_conflict_is_isolated_within_batch(self):
        record = AudioLibrary(self.root).create({"text": "旧文本"})
        item = self.request(synthesis_mode="clone", clip_id=record["id"])
        response = await self.client.post("/api/generate-batch", json={"items": [item, item]})
        results = response.json()["items"]
        self.assertEqual([r["status"] for r in results], ["ok", "error"])
        self.assertEqual(results[1]["status_code"], 409)
        self.assertEqual(len(AudioLibrary(self.root).list()), 1)
        self.assertFalse(server.generation_claims)

    async def test_status_and_library_respond_while_generation_is_blocked(self):
        self.model.gate = threading.Event()
        task = asyncio.create_task(self.client.post("/api/tts", json=self.request()))
        self.assertTrue(await asyncio.to_thread(self.model.entered.wait, 2))
        status = await asyncio.wait_for(self.client.get("/api/inference/status"), 1)
        library = await asyncio.wait_for(self.client.get("/api/clips"), 1)
        self.assertEqual(status.json()["active"], 1)
        self.assertEqual(library.status_code, 200)
        self.model.gate.set()
        self.assertEqual((await task).status_code, 200)

    async def test_invalid_batch_size_and_text_are_rejected(self):
        response = await self.client.post("/api/generate-batch", json={"items": [self.request(synthesis_mode="clone")] * 33})
        self.assertEqual(response.status_code, 422)
        response = await self.client.post("/api/tts", json={"text": "长" * 10001})
        self.assertEqual(response.status_code, 422)

    async def test_mac_capabilities_keep_browser_serial(self):
        with patch.object(server, "model", SimpleNamespace(device="mps")):
            response = await self.client.get("/api/capabilities")
        capabilities = response.json()["inference"]
        self.assertEqual(capabilities["mode"], "serial")
        self.assertEqual(capabilities["max_batch_size"], 1)
        self.assertEqual(capabilities["max_request_items"], 1)



if __name__ == "__main__":
    unittest.main()
