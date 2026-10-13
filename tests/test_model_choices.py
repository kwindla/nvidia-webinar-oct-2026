import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from pipecat.services.openai.llm import OpenAILLMService

from model_choices import default_model, model_catalog, tuned_api_key


class ModelChoicesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def session(self, *, status=200, data=None, error=None):
        session = MagicMock()
        response = MagicMock(status=status)
        response.json = AsyncMock(return_value=data if data is not None else {"data": [{"id": "gb-phone"}]})
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=response, side_effect=error)
        manager.__aexit__ = AsyncMock(return_value=False)
        session.get.return_value = manager
        return session

    async def test_ready_model_id_and_normalized_url(self):
        os.environ['LOCAL_LLM_URL'] = 'http://localhost:8080/v1/'
        session = self.session()
        catalog = await model_catalog(session)
        self.assertIn('local', catalog.names)
        self.assertEqual(session.get.call_args.args[0], 'http://localhost:8080/v1/models')
        self.assertEqual(session.get.call_args.kwargs['timeout'].total, 2)

    async def test_unreachable_slow_invalid_missing_and_http_errors(self):
        for session in (
            self.session(error=aiohttp.ClientConnectionError()),
            self.session(error=TimeoutError()),
            self.session(data={"data": [{"id": "other-model"}]}),
            self.session(data={"not": "a catalog"}),
            self.session(status=503),
        ):
            with self.subTest(session=session):
                catalog = await model_catalog(session)
                self.assertEqual(catalog.names, {'cloud': 'GPT-5.6 Terra'})
                selected, fallback = catalog.initial_selection('local')
                self.assertEqual(selected, 'cloud')
                self.assertEqual(fallback['requested'], 'gb-phone')
                self.assertEqual(fallback['actual'], 'GPT-5.6 Terra')
                self.assertTrue(fallback['reason'])

    async def test_tuned_loopback_missing_or_blank_key_uses_dummy(self):
        os.environ['TUNED_LLM_URL'] = 'http://127.0.0.1:8083/v1'
        for key in (None, '', '   '):
            os.environ.pop('TUNED_LLM_API_KEY', None)
            if key is not None:
                os.environ['TUNED_LLM_API_KEY'] = key
            catalog = await model_catalog(self.session())
            self.assertEqual(catalog.names['tuned'], 'ATC tutor · local')
            self.assertEqual(tuned_api_key(os.environ['TUNED_LLM_URL']), 'not-required')
            # SDK construction must succeed even without a real OpenAI key; no network call.
            OpenAILLMService(base_url=os.environ['TUNED_LLM_URL'], api_key=tuned_api_key(os.environ['TUNED_LLM_URL']))

    async def test_remote_missing_key_is_not_offered_and_explains_fallback(self):
        os.environ['TUNED_LLM_URL'] = 'https://example.invalid/v1'
        session = self.session()
        catalog = await model_catalog(session)
        self.assertNotIn('tuned', catalog.names)
        session.get.assert_called_once()  # only Local: no unauthenticated remote probe
        _, fallback = catalog.initial_selection('tuned')
        self.assertIn('TUNED_LLM_API_KEY', fallback['reason'])

    async def test_remote_auth_and_display_override(self):
        os.environ.update(TUNED_LLM_URL='https://example.invalid/v1', TUNED_LLM_API_KEY='test-key', TUNED_LLM_NAME='Round 5')
        session = self.session()
        catalog = await model_catalog(session)
        self.assertEqual(catalog.names['tuned'], 'Round 5')
        self.assertEqual(session.get.call_args_list[-1].kwargs['headers'], {'Authorization': 'Bearer test-key'})
        self.assertNotIn('test-key', str(catalog.payload()))

    async def test_requested_and_default_chain(self):
        catalog = await model_catalog(self.session())
        os.environ['DEFAULT_LLM'] = 'local'
        self.assertEqual(catalog.initial_selection(), ('local', None))
        self.assertEqual(catalog.initial_selection('cloud'), ('cloud', None))
        for invalid in ([], {}, 42, 'unknown'):
            selected, fallback = catalog.initial_selection(invalid)
            self.assertEqual(selected, 'local')
            self.assertEqual(fallback['requested'], 'Unknown model')
        os.environ['DEFAULT_LLM'] = 'tuned'
        self.assertEqual(default_model(catalog.names), 'cloud')
