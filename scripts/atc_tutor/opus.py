"""Claude Opus 5.5 client: adaptive thinking, schema-constrained JSON, retries, token ledger."""

import asyncio
import json
import time
from pathlib import Path

import anthropic
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/atc-tutor"
LEDGER = OUT / "ledger.jsonl"
MODEL = "claude-opus-5-5"
# List prices, USD per million tokens (standard API).
PRICE_IN, PRICE_OUT = 4.0, 20.0
PRICE_CACHE_WRITE, PRICE_CACHE_READ = 5.0, 0.4


def client():
    key = dotenv_values(ROOT.parent / "pipecat/.env").get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY missing from ../pipecat/.env")
    return anthropic.AsyncAnthropic(api_key=key, max_retries=4)


def cost(usage):
    return (
        usage.input_tokens * PRICE_IN
        + usage.output_tokens * PRICE_OUT
        + (usage.cache_creation_input_tokens or 0) * PRICE_CACHE_WRITE
        + (usage.cache_read_input_tokens or 0) * PRICE_CACHE_READ
    ) / 1e6


def strict(schema):
    """Structured outputs need closed objects: add additionalProperties: false."""
    if isinstance(schema, dict):
        schema = {k: strict(v) for k, v in schema.items()}
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        return schema
    if isinstance(schema, list):
        return [strict(v) for v in schema]
    return schema


async def call_tool(api, *, stage, system, user, tool, max_tokens=32000, effort="high"):
    """One schema-constrained JSON response with adaptive thinking at the given effort.

    `tool` supplies the name and JSON schema (kept for the existing call sites).
    Usage is appended to the ledger, including thinking tokens in `output`.
    """
    for wait in range(30):  # the account auto-tops-up only after the balance reaches zero
        try:
            return await _stream(api, stage, system, user, tool, max_tokens, effort)
        except anthropic.BadRequestError as error:
            if "credit balance" not in str(error):
                raise
            print(f"{stage}: credit balance low, waiting for the auto top-up ({wait + 1}/30)", flush=True)
            await asyncio.sleep(60)
    raise RuntimeError(f"{stage}: credit balance still low after 30 minutes")


async def _stream(api, stage, system, user, tool, max_tokens, effort):
    for attempt in range(3):
        started = time.monotonic()
        async with api.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive", "display": "omitted"},
            output_config={"effort": effort, "format": {
                "type": "json_schema", "schema": strict(tool["input_schema"])}},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            response = await stream.get_final_message()
        usage = response.usage
        OUT.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as ledger:
            ledger.write(json.dumps({
                "stage": stage,
                "model": response.model,
                "effort": effort,
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_write": usage.cache_creation_input_tokens or 0,
                "cache_read": usage.cache_read_input_tokens or 0,
                "usd": round(cost(usage), 5),
                "seconds": round(time.monotonic() - started, 2),
                "stop": response.stop_reason,
            }) + "\n")
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "max_tokens":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{stage}: no complete JSON response after retries")


def items(result, key):
    """Return result[key] as a list of dicts; tolerate JSON-encoded strings."""
    value = result.get(key, [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            out.append(item)
    return out


def ledger_summary():
    totals = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            row = json.loads(line)
            stage = totals.setdefault(row["stage"], {"calls": 0, "input": 0, "output": 0, "usd": 0.0})
            stage["calls"] += 1
            stage["input"] += row["input"] + row["cache_write"] + row["cache_read"]
            stage["output"] += row["output"]
            stage["usd"] += row["usd"]
    return totals
