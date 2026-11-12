from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets

from eot_harness.sprag_turns_adapter import (
    SESSION_CONFIG,
    SpragTurnsAdapter,
    agent_text,
    turn_update_event,
)


class FakeTurnsServer:
    """A /v1/turns double: records what the client sends and replays scripted readings."""

    def __init__(self, readings=None, *, refuse=False):
        self.readings = readings if readings is not None else [{"p_eot": 0.2}, {"p_eot": 0.9}]
        self.refuse = refuse
        self.path = None
        self.headers = None
        self.messages: list[dict] = []
        self.audio_bytes = 0

    async def handle(self, ws):
        self.path = ws.request.path
        self.headers = ws.request.headers
        async for message in ws:
            if isinstance(message, bytes):
                self.audio_bytes += len(message)
                if self.readings and self.audio_bytes >= 3200 * 2:
                    reading = self.readings.pop(0)
                    await ws.send(json.dumps({"type": "turn.update", "silence_ms": 128, **reading}))
                continue
            event = json.loads(message)
            self.messages.append(event)
            if event["type"] == "session.update":
                if self.refuse:
                    await ws.send(json.dumps({"type": "error", "error": {"message": "bad session"}}))
                    return
                await ws.send(json.dumps({"type": "session.updated", "session": event["session"]}))
            elif event["type"] == "session.close":
                return


def _row(**over):
    row = {
        "id": "turn-1",
        "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
        "silence_spans": [{"start": 0.1, "end": 0.9}],
        "messages": [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "how can I help?"},
        ],
        "words": [{"word": "leaked", "end": 0.4}],
    }
    row.update(over)
    return row


def _replay(server: FakeTurnsServer, row=None, **adapter_kw):
    async def run():
        async with websockets.serve(server.handle, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            adapter = SpragTurnsAdapter(url=f"ws://127.0.0.1:{port}/v1/turns", chunk_ms=20, **adapter_kw)
            return await adapter.predict_turn(row or _row())

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SPRAG_API_KEY", "test-key")
    monkeypatch.delenv("SPRAG_TURNS_URL", raising=False)


# ---- identity ---------------------------------------------------------------


def test_defaults():
    adapter = SpragTurnsAdapter()
    assert adapter.adapter_id == "sprag/sprag-turns"
    assert adapter.display_name == "Sprag Turns"
    assert adapter.url == "wss://api.sprag.ai/v1/turns"
    assert adapter.chunk_ms == 100
    assert adapter.concurrency == 4


def test_url_override_from_environment(monkeypatch):
    monkeypatch.setenv("SPRAG_TURNS_URL", "wss://example.test/v1/turns")
    assert SpragTurnsAdapter().url == "wss://example.test/v1/turns"


def test_supports_only_english():
    adapter = SpragTurnsAdapter()
    assert adapter.supports_language("en")
    assert not adapter.supports_language("de")


def test_constructor_validation():
    with pytest.raises(ValueError):
        SpragTurnsAdapter(model="")
    with pytest.raises(ValueError):
        SpragTurnsAdapter(chunk_ms=0)
    with pytest.raises(ValueError):
        SpragTurnsAdapter(concurrency=0)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SPRAG_API_KEY")
    with pytest.raises(RuntimeError, match="SPRAG_API_KEY"):
        asyncio.run(SpragTurnsAdapter().predict_turn(_row()))


# ---- event mapping ------------------------------------------------------------


def test_turn_update_uses_realtime_receive_position():
    event = turn_update_event({"type": "turn.update", "p_eot": 0.73, "silence_ms": 224}, received_audio_sec=1.25)
    assert event == {"event": "TurnUpdate", "timestamp": 1.25, "p_eot": 0.73, "silence_ms": 224}


def test_other_events_are_not_scored():
    assert turn_update_event({"type": "turn.started"}, received_audio_sec=1.0) is None
    assert turn_update_event({"type": "turn.ended", "p_eot": 0.9}, received_audio_sec=1.0) is None
    assert turn_update_event({"type": "turn.update"}, received_audio_sec=1.0) is None


def test_agent_text_is_the_last_assistant_message():
    assert agent_text(_row()["messages"]) == "how can I help?"
    assert agent_text([{"role": "user", "content": "hi"}]) == ""
    assert agent_text([{"role": "assistant", "content": None}]) == ""
    assert agent_text(None) == ""


# ---- replay against a fake server ---------------------------------------------------


def test_session_disables_the_server_turn_decision():
    server = FakeTurnsServer()
    _replay(server)
    assert server.messages[0] == {"type": "session.update", "session": SESSION_CONFIG}
    assert SESSION_CONFIG["eot_threshold"] == 1.0


def test_request_carries_model_and_key():
    server = FakeTurnsServer()
    _replay(server)
    assert server.path == "/v1/turns?model=sprag-turns"
    assert server.headers["Authorization"] == "Bearer test-key"


def test_only_the_agent_message_is_sent_never_the_in_progress_words():
    """Sprag Turns must not see the words of the turn it is asked to judge."""
    server = FakeTurnsServer()
    _replay(server)
    context = [m for m in server.messages if m["type"] == "context.update"]
    assert context == [{"type": "context.update", "agent_text": "how can I help?"}]
    assert "leaked" not in json.dumps(server.messages)


def test_all_audio_is_streamed_then_the_session_is_closed():
    server = FakeTurnsServer()
    _replay(server)
    assert server.audio_bytes == 16000 * 2
    assert server.messages[-1] == {"type": "session.close"}


def test_readings_become_prediction_rows():
    server = FakeTurnsServer([{"p_eot": 0.2}, {"p_eot": 0.9}])
    result = _replay(server)
    assert [e["p_eot"] for e in result["events"]] == [0.2, 0.9]
    assert result["audio_sec"] == pytest.approx(1.0)
    rows = result["prediction_rows"]
    assert rows and {r["label"] for r in rows} == {"eot"}
    assert max(r["p_eot"] for r in rows) == pytest.approx(0.9)


def test_a_refused_session_raises():
    with pytest.raises(RuntimeError, match="refused the session"):
        _replay(FakeTurnsServer(refuse=True))
