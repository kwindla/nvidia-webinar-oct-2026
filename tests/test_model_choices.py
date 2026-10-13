import os
import unittest
from unittest import mock

import aiohttp

from bot import default_model, model_names


class ModelChoicesTest(unittest.IsolatedAsyncioTestCase):
    async def names(self, **env):
        base = {"LOCAL_LLM_URL": "http://127.0.0.1:9/v1"}  # nothing listens on the discard port
        with mock.patch.dict(os.environ, {**base, **env}, clear=False):
            for key in ("TUNED_LLM_URL", "TUNED_LLM_NAME"):
                if key not in env:
                    os.environ.pop(key, None)
            async with aiohttp.ClientSession() as session:
                return await model_names(session)

    async def test_unreachable_local_is_hidden(self):
        self.assertEqual(await self.names(), {"cloud": "GPT-5.6 Terra"})

    async def test_tuned_name_reflects_where_it_runs(self):
        local = await self.names(TUNED_LLM_URL="http://127.0.0.1:8083/v1")
        remote = await self.names(TUNED_LLM_URL="https://example.modal.run/v1")
        self.assertEqual(local["tuned"], "ATC tutor · local")
        self.assertEqual(remote["tuned"], "ATC tutor · remote")

    async def test_tuned_name_override(self):
        names = await self.names(TUNED_LLM_URL="http://127.0.0.1:8083/v1", TUNED_LLM_NAME="Round 5")
        self.assertEqual(names["tuned"], "Round 5")


class DefaultModelTest(unittest.TestCase):
    names = {"cloud": "GPT-5.6 Terra", "tuned": "ATC tutor · local"}

    def test_client_choice_wins_when_offered(self):
        with mock.patch.dict(os.environ, {"DEFAULT_LLM": "cloud"}):
            self.assertEqual(default_model(self.names, "tuned"), "tuned")

    def test_unoffered_or_invalid_choice_falls_back(self):
        with mock.patch.dict(os.environ, {"DEFAULT_LLM": "tuned"}):
            self.assertEqual(default_model(self.names, "local"), "tuned")
            self.assertEqual(default_model(self.names, ["tuned"]), "tuned")
        with mock.patch.dict(os.environ, {"DEFAULT_LLM": "local"}):
            self.assertEqual(default_model(self.names), "cloud")


if __name__ == "__main__":
    unittest.main()
