"""Mint temporary Cloudflare TURN credentials for Pipecat's dev runner."""

import json
import os

import aiohttp
from loguru import logger

TTL = 86400  # Restart the demo daily to renew credentials on both peers.


def select_servers(payload):
    """Prefer UDP; include TCP/TLS 443 fallback, excluding browser-blocked ports."""
    servers = []
    allowed = {
        "stun:stun.cloudflare.com:3478",
        "turn:turn.cloudflare.com:3478?transport=udp",
        "turn:turn.cloudflare.com:3478?transport=tcp",
        "turns:turn.cloudflare.com:443?transport=tcp",
    }
    for server in payload.get("iceServers", []):
        urls = server.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        urls = [url for url in urls if url in allowed]
        if not urls:
            continue
        entry = {"urls": urls}
        if any(url.startswith(("turn:", "turns:")) for url in urls):
            if not server.get("username") or not server.get("credential"):
                raise ValueError("Cloudflare returned missing TURN credentials")
            entry.update(username=server["username"], credential=server["credential"])
        servers.append(entry)
    if not any("credential" in server for server in servers):
        raise ValueError("Cloudflare returned no usable TURN servers")
    return servers


async def configure_runner_ice():
    key_id = os.getenv("CLOUDFLARE_TURN_KEY_ID")
    token = os.getenv("CLOUDFLARE_TURN_API_TOKEN")
    if not key_id and not token:
        os.environ.setdefault("PIPECAT_ICE_SERVERS", "stun:stun.l.google.com:19302")
        return
    if not key_id or not token:
        raise RuntimeError("Set both Cloudflare TURN variables in .env")
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session,
            session.post(
                f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
                headers={"Authorization": f"Bearer {token}"},
                json={"ttl": TTL},
            ) as response,
        ):
            if response.status != 201:
                raise RuntimeError(
                    f"Cloudflare credential request failed (HTTP {response.status})"
                )
            servers = select_servers(await response.json())
    except (aiohttp.ClientError, TimeoutError, ValueError):
        raise RuntimeError("Could not obtain Cloudflare TURN credentials") from None
    # Runner gives this configuration to both aiortc and the browser via /start.
    # Never print credentials or pass them as command-line arguments.
    os.environ["PIPECAT_ICE_SERVERS"] = json.dumps(servers)
    logger.info(
        "Cloudflare TURN enabled for SmallWebRTC; credentials expire in 24 hours. Restart daily."
    )
