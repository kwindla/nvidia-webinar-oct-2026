"""Real Pipecat runner routing with mocked WebRTC negotiation and bot execution."""
import asyncio
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from pipecat.runner import run as runner
from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler

from model_choices import ModelCatalog


class ModelRoutesTest(unittest.TestCase):
    def test_catalog_route_coexists_with_runner_and_start_body_reaches_bot(self):
        app = FastAPI()
        invoked = AsyncMock()
        catalog = ModelCatalog({'cloud': 'Terra', 'local': 'Local'}, {})

        async def negotiate(self, request, webrtc_connection_callback):
            await webrtc_connection_callback(SimpleNamespace())
            return {'sdp': 'mock answer', 'type': 'answer'}

        async def requests():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                listing = await client.get('/models')
                self.assertEqual(listing.status_code, 200)
                self.assertEqual(listing.json(), catalog.payload())
                for body in ({'model': 'local'}, {'model': ['invalid']}, {}):
                    start = await client.post('/start', json={'transport': 'webrtc', 'body': body})
                    self.assertEqual(start.status_code, 200)
                    result = await client.post(f"/sessions/{start.json()['sessionId']}/api/offer", json={'sdp': 'mock offer', 'type': 'offer'})
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(invoked.call_args.args[0].body, body)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch('dotenv.load_dotenv'),
            patch.object(runner, 'app', app),
            patch.object(sys, 'argv', ['bot.py', '-t', 'webrtc', '--host', '127.0.0.1', '--port', '19999']),
            patch('uvicorn.run'),
            patch('cloudflare_turn.configure_runner_ice', new_callable=AsyncMock),
            patch('model_choices.model_catalog', new=AsyncMock(return_value=catalog)),
            patch.object(runner, '_get_bot_module', return_value=SimpleNamespace(bot=invoked)),
            patch.object(SmallWebRTCRequestHandler, 'handle_web_request', new=negotiate),
        ):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bot.py'), run_name='__main__')
            asyncio.run(requests())
