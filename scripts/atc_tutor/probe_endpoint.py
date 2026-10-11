"""Stream one chat completion from a Nemotron endpoint and report TTFT and content.

Usage: uv run python scripts/atc_tutor/probe_endpoint.py URL KEY_ENV_FILE ["utterance"]
"""

import sys
import time

from dotenv import dotenv_values
from openai import OpenAI

from export import SHORT_PROMPT

url, env_file = sys.argv[1], sys.argv[2]
utterance = sys.argv[3] if len(sys.argv) > 3 else "qantas four ninety eight climb flight level one two zero"
client = OpenAI(base_url=url.rstrip("/") + "/v1", api_key=dotenv_values(env_file)["GB_SERVING_API_KEY"], timeout=1800)
started = time.monotonic()
first = None
text = ""
stream = client.chat.completions.create(
    model="gb-phone", stream=True, max_tokens=150, temperature=0,
    messages=[{"role": "system", "content": SHORT_PROMPT}, {"role": "user", "content": utterance}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        first = first or time.monotonic()
        text += delta.content
print(f"ttft_s={first - started if first else None:.2f} total_s={time.monotonic() - started:.2f}")
print("content:", repr(text))
