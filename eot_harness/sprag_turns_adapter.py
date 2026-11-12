"""Streaming adapter for Sprag Turns, Sprag's audio end-of-turn model.

Sprag Turns is a hosted service. This adapter replays each turn in real time over the
``/v1/turns`` WebSocket and scores the model's ``turn.update`` readings: one end-of-turn
probability roughly every 100 ms of silence.

The server's own turn decision is switched off (``eot_threshold`` 1.0, the longest
``eot_timeout_ms``), so it never closes a turn on its own and keeps reading through every
silence. The harness's policy sweep then decides when to act, as for every other model.

The model hears the turn's audio and, as context, the agent's last message from
``row["messages"]``. It never reads ``row["words"]``. A unit test asserts this.

Requires SPRAG_API_KEY. Keys are issued at https://sprag.ai.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib.parse import urlencode

from .io import DEFAULT_INFERENCE_INTERVAL
from .languages import supports_language_code
from .streaming_stt import (
    DEFAULT_CHUNK_MS,
    SAMPLE_RATE,
    build_event_prediction_rows,
    chunk_size_bytes,
    import_websockets,
    prepare_pcm16_audio,
    resolve_api_key,
)

DEFAULT_URL = "wss://api.sprag.ai/v1/turns"
DEFAULT_MODEL = "sprag-turns"
DEFAULT_CONCURRENCY = 4
SPRAG_TURNS_SUPPORTED_LANGUAGES = {"en"}
# The server never ends a turn by itself, so the harness's policy sweep is the only policy.
SESSION_CONFIG = {"eot_threshold": 1.0, "min_silence_ms": 0, "eot_timeout_ms": 60000}
CLOSE_TIMEOUT_S = 2.0


class SpragTurnsAdapter:
    """Replays full turns through Sprag Turns and scores its end-of-turn readings."""

    display_name = "Sprag Turns"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str | None = None,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if max_retries < 0 or retry_backoff < 0:
            raise ValueError("retry settings must be non-negative")
        self.model = model
        self.url = url or os.environ.get("SPRAG_TURNS_URL") or DEFAULT_URL
        self.chunk_ms = int(chunk_ms)
        self.concurrency = int(concurrency)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)

    @property
    def adapter_id(self) -> str:
        return f"sprag/{self.model}"

    def supports_language(self, lang_code: str) -> bool:
        return supports_language_code(lang_code, SPRAG_TURNS_SUPPORTED_LANGUAGES)

    async def predict_turn(
        self,
        row: dict[str, Any],
        *,
        inference_interval: float = DEFAULT_INFERENCE_INTERVAL,
    ) -> dict[str, Any]:
        api_key = resolve_api_key("SPRAG_API_KEY")
        attempt = 0
        while True:
            try:
                events, audio_sec = await self._replay_turn(row, api_key)
                break
            except (ConnectionError, OSError, asyncio.TimeoutError):
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                await asyncio.sleep(self.retry_backoff * attempt)

        return {
            "id": row["id"],
            "audio_sec": audio_sec,
            "events": events,
            "prediction_rows": build_event_prediction_rows(
                row,
                events,
                inference_interval=inference_interval,
            ),
        }

    async def _replay_turn(self, row: dict[str, Any], api_key: str) -> tuple[list[dict[str, Any]], float]:
        websockets = import_websockets()
        audio_bytes, audio_sec = prepare_pcm16_audio(row, sample_rate=SAMPLE_RATE)
        chunk_size = chunk_size_bytes(sample_rate=SAMPLE_RATE, chunk_ms=self.chunk_ms)
        events: list[dict[str, Any]] = []

        async with websockets.connect(
            self._session_url(),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            await _configure(ws, row_id=row["id"])
            await ws.send(json.dumps({"type": "context.update", "agent_text": agent_text(row.get("messages"))}))
            started = time.perf_counter()
            recv_task = asyncio.create_task(_recv_events(ws, row_id=row["id"], events=events, started=started))
            for index, start in enumerate(range(0, len(audio_bytes), chunk_size)):
                if recv_task.done():
                    break
                await ws.send(audio_bytes[start : start + chunk_size])
                # Pace against the start time so send overhead does not accumulate as drift.
                delay = started + (index + 1) * self.chunk_ms / 1000.0 - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            if not recv_task.done():
                await ws.send(json.dumps({"type": "session.close"}))
            try:
                await asyncio.wait_for(asyncio.shield(recv_task), timeout=CLOSE_TIMEOUT_S)
            except asyncio.TimeoutError:
                # Every reading for the replayed audio has arrived by now; stop listening.
                recv_task.cancel()
                await asyncio.gather(recv_task, return_exceptions=True)

        return events, audio_sec

    def _session_url(self) -> str:
        separator = "&" if "?" in self.url else "?"
        return f"{self.url}{separator}{urlencode({'model': self.model})}"


def agent_text(messages: list[dict[str, Any]] | None) -> str:
    """Return the agent's last message, the only context Sprag Turns takes."""
    for message in reversed(messages or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def turn_update_event(data: dict[str, Any], *, received_audio_sec: float) -> dict[str, Any] | None:
    """Map a ``turn.update`` reading to a scored event, timed by when it arrived."""
    if data.get("type") != "turn.update" or data.get("p_eot") is None:
        return None
    return {
        "event": "TurnUpdate",
        "timestamp": float(received_audio_sec),
        "p_eot": float(data["p_eot"]),
        "silence_ms": data.get("silence_ms"),
    }


async def _configure(ws, *, row_id: Any) -> None:
    await ws.send(json.dumps({"type": "session.update", "session": SESSION_CONFIG}))
    async for message in ws:
        if not isinstance(message, str):
            continue
        data = json.loads(message)
        if data.get("type") == "session.updated":
            return
        if data.get("type") == "error":
            raise RuntimeError(f"Sprag Turns refused the session for row {row_id!r}: {data}")
    raise ConnectionError(f"Sprag Turns closed before the session was configured for row {row_id!r}.")


async def _recv_events(ws, *, row_id: Any, events: list[dict[str, Any]], started: float) -> None:
    async for message in ws:
        if not isinstance(message, str):
            continue
        data = json.loads(message)
        if data.get("type") == "error":
            raise RuntimeError(f"Sprag Turns error for row {row_id!r}: {data}")
        event = turn_update_event(data, received_audio_sec=time.perf_counter() - started)
        if event is not None:
            events.append(event)
