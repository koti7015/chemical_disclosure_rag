"""One place that talks to Groq. Every agent calls llm.ask_llm(), so tests can swap it out."""
import json
import os
import time

import groq

from app.core import config


class LLMUnavailable(RuntimeError):
    """The LLM could not be reached (no API key, rate limit, network)."""


def _client() -> groq.Groq:
    if not os.getenv("GROQ_API_KEY"):
        raise LLMUnavailable("GROQ_API_KEY is not set")
    return groq.Groq()


def ask_llm(system: str, user: str) -> dict:
    """Send one prompt and return the JSON reply. Raises ValueError if the reply is not a JSON object."""
    client = _client()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=config.GROQ_MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        except groq.RateLimitError:
            time.sleep(2 ** attempt)                 # back off and retry
            continue
        except groq.APIError as e:
            raise LLMUnavailable(str(e)) from e
        reply = json.loads(r.choices[0].message.content)    # JSONDecodeError is a ValueError
        if not isinstance(reply, dict):
            raise ValueError("LLM reply is not a JSON object")
        return reply
    raise LLMUnavailable("Groq rate limit reached, try again in a minute")
