"""LLM client for any OpenAI-compatible chat endpoint (Gemini, OpenRouter, Groq).

Built for free tiers and for outages (acceptance criterion A11):
  * throttles to LLM_RPM requests per minute;
  * retries 429/5xx/timeouts with exponential backoff, honouring Retry-After (bounded);
  * a circuit breaker: after repeated failures the client stops calling the provider for a cool-down
    period, so a full outage costs milliseconds per ticket rather than a timeout per ticket;
  * an on-disk response cache keyed by (model, prompt), which saves quota and makes runs reproducible.
Every failure surfaces as `LLMUnavailable`; callers degrade to extractive answers or escalation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import requests

from .config import Settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings, cache_dir: Path | None = None, max_retries: int = 2,
                 timeout_s: float = 25.0, breaker_threshold: int = 3, breaker_cooldown_s: float = 60.0,
                 session: requests.Session | None = None):
        self.s = settings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown_s = breaker_cooldown_s
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._consecutive_failures = 0
        self._open_until = 0.0
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "breaker_skips": 0, "throttle_wait_s": 0.0}

    @property
    def available(self) -> bool:
        return self.s.llm_enabled

    # ----------------------------------------------------------------- cache
    def _key(self, messages: list[dict], temperature: float) -> str:
        blob = json.dumps({"m": self.s.llm_model, "msgs": messages, "t": temperature}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        f = self.cache_dir / f"{key}.json"
        try:
            return json.loads(f.read_text(encoding="utf-8"))["text"] if f.exists() else None
        except Exception:
            return None

    def _cache_put(self, key: str, text: str) -> None:
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps({"text": text}), encoding="utf-8")
        except Exception as exc:
            log.debug("cache write failed: %s", exc)

    # ----------------------------------------------------------------- calls
    def _throttle(self) -> None:
        if self.s.llm_rpm <= 0:
            return
        with self._lock:
            wait = (60.0 / self.s.llm_rpm) - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
                self.stats["throttle_wait_s"] += wait
            self._last_call = time.monotonic()

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700) -> str:
        if not self.available:
            raise LLMUnavailable("no LLM configured (LLM_PROVIDER=none or API key missing)")
        key = self._key(messages, temperature)
        cached = self._cache_get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        if time.monotonic() < self._open_until:
            self.stats["breaker_skips"] += 1
            raise LLMUnavailable("circuit breaker open: provider recently failing")

        url = f"{self.s.llm_base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}", "Content-Type": "application/json"}
        payload = {"model": self.s.llm_model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        last_err = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats["calls"] += 1
            try:
                resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"] or ""
                    self._consecutive_failures = 0
                    self._cache_put(key, text)
                    return text
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code in (400, 401, 403, 404):   # our fault - retrying will not help
                    break
                if resp.status_code == 429:
                    delay = min(float(resp.headers.get("Retry-After", 2 ** attempt * 2)), 20.0)
                    time.sleep(delay)
                    continue
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 8))
        self.stats["failures"] += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.breaker_threshold:
            self._open_until = time.monotonic() + self.breaker_cooldown_s
            log.warning("LLM circuit breaker opened for %.0fs after %s", self.breaker_cooldown_s, last_err)
        raise LLMUnavailable(last_err)
