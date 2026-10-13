"""Bounded model discovery and explicit session-start fallback metadata."""

import asyncio
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import aiohttp

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MODEL_TIMEOUT_SECONDS = 2


def display_names() -> dict[str, str]:
    tuned_url = os.getenv("TUNED_LLM_URL", "")
    where = "local" if urlparse(tuned_url).hostname in LOOPBACK_HOSTS else "remote"
    return {
        "cloud": "GPT-5.6 Terra",
        "local": os.getenv("LOCAL_LLM_NAME") or os.getenv("LOCAL_LLM_MODEL", "gb-phone"),
        "tuned": os.getenv("TUNED_LLM_NAME") or f"ATC tutor · {where}",
    }


def tuned_api_key(url: str) -> str | None:
    key = os.getenv("TUNED_LLM_API_KEY", "").strip()
    if key:
        return key
    return "not-required" if urlparse(url).hostname in LOOPBACK_HOSTS else None


@dataclass
class ModelCatalog:
    names: dict[str, str]
    unavailable: dict[str, str]

    def payload(self):
        return {
            "models": [{"id": key, "name": name} for key, name in self.names.items()],
            "default": default_model(self.names),
        }

    def initial_selection(self, requested: object = None):
        selected = default_model(self.names, requested)
        fallback = None
        if requested is not None and requested != selected:
            valid = isinstance(requested, str) and requested in display_names()
            fallback = {
                "requested": display_names()[requested] if valid else "Unknown model",
                "actual": self.names[selected],
                "reason": self.unavailable.get(requested, "This model is not offered.")
                if valid else "The requested model ID is invalid.",
            }
        return selected, fallback


async def endpoint_problem(session, url: str, model: str, key: str | None = None) -> str | None:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with session.get(
            f"{url.rstrip('/')}/models", headers=headers,
            timeout=aiohttp.ClientTimeout(total=MODEL_TIMEOUT_SECONDS),
        ) as response:
            if response.status != 200:
                return f"The model server returned HTTP {response.status}."
            data = await response.json()
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                return "The model server returned an invalid model list."
            if not any(isinstance(row, dict) and row.get("id") == model for row in data["data"]):
                return "The configured model is not loaded on the server."
    except TimeoutError:
        return f"The model server did not respond within {MODEL_TIMEOUT_SECONDS} seconds."
    except (aiohttp.ClientError, ValueError):
        return "The model server is unreachable or returned an invalid response."
    return None


async def model_catalog(session: aiohttp.ClientSession) -> ModelCatalog:
    labels = display_names()
    problems = {}
    checks = {
        "local": endpoint_problem(
            session, os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1"),
            os.getenv("LOCAL_LLM_MODEL", "gb-phone"),
        ),
    }
    if url := os.getenv("TUNED_LLM_URL"):
        if key := tuned_api_key(url):
            checks["tuned"] = endpoint_problem(session, url, "gb-phone", key)
        else:
            problems["tuned"] = "The remote tuned model requires TUNED_LLM_API_KEY."
    else:
        problems["tuned"] = "The tuned model is not configured."
    for model, problem in zip(checks, await asyncio.gather(*checks.values())):
        if problem:
            problems[model] = problem
    return ModelCatalog({key: name for key, name in labels.items() if key not in problems}, problems)


def default_model(names: dict[str, str], requested: object = None) -> str:
    for choice in (requested, os.getenv("DEFAULT_LLM", "cloud")):
        if isinstance(choice, str) and choice in names:
            return choice
    return "cloud"


def register_model_routes(app):
    @app.get("/models")
    async def models():
        async with aiohttp.ClientSession() as session:
            return (await model_catalog(session)).payload()
