from types import SimpleNamespace

import pytest
import httpx

from autorealize.llm.client import LLMClient, _degraded_request_kwargs


def chunk(content=None, *, finish=None, usage=None):
    return SimpleNamespace(model="gpt-5.6-luna", system_fingerprint="provider-a", usage=usage,
        choices=[SimpleNamespace(index=0, finish_reason=finish,
            delta=SimpleNamespace(content=content, reasoning_content=None))] if content is not None or finish else [])


def test_stream_assembles_exact_json_and_preserves_final_usage():
    usage = {"prompt_tokens": 5000, "completion_tokens": 1200,
             "prompt_tokens_details": {"cached_tokens": 4608}}
    response = LLMClient._consume_completion_stream(iter([
        chunk('{"rule":"No assumed '), chunk('dates"}'), chunk(finish="stop"), chunk(usage=usage)]))
    assert response.choices[0].message.content == '{"rule":"No assumed dates"}'
    assert response.choices[0].finish_reason == "stop"
    assert response.usage == usage
    assert response.model == "gpt-5.6-luna"


def test_stream_without_terminal_frame_cannot_be_accepted():
    with pytest.raises(ConnectionError, match="terminal finish_reason"):
        LLMClient._consume_completion_stream(iter([chunk('{"looks_complete":true}')]))


def test_terminal_usage_does_not_wait_for_provider_to_close_connection():
    def frames():
        yield chunk('{"ok":true}', finish="stop")
        yield chunk(usage={"prompt_tokens": 12, "completion_tokens": 5})
        pytest.fail("The response is complete; do not wait for more frames")

    response = LLMClient._consume_completion_stream(frames(), timeout_seconds=1)
    assert response.choices[0].message.content == '{"ok":true}'


def test_wall_deadline_closes_stalled_stream_without_automatic_paid_retry():
    from threading import Event
    from autorealize.llm.client import StreamDeadlineExceeded

    class StalledStream:
        def __init__(self):
            self.closed = Event()

        def close(self):
            self.closed.set()

        def __iter__(self):
            yield chunk('{"incomplete":')
            assert self.closed.wait(2), "Deadline did not close the stream"

    stream = StalledStream()
    with pytest.raises(StreamDeadlineExceeded) as error:
        LLMClient._consume_completion_stream(stream, timeout_seconds=0.02)
    assert stream.closed.is_set()
    assert not LLMClient._is_retryable_llm_error(error.value)


def test_incomplete_chunked_stream_is_retryable_transport_failure():
    error = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body (incomplete chunked read)"
    )
    assert LLMClient._is_retryable_llm_error(error)


def test_incomplete_chunked_response_retries_once_and_recovers(tmp_path, monkeypatch):
    from autorealize.config import AutoRealizeConfig

    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    calls = 0

    def fake_create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("peer closed connection (incomplete chunked read)")
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr("autorealize.llm.client.time.sleep", lambda _seconds: None)

    result = client._chat_completion_with_network_retry(
        prompt_name="stream_disconnect_retry_test",
        mode="text",
        create_kwargs={"model": cfg.llm.model_name, "messages": []},
    )

    assert calls == 2
    assert result.choices[0].finish_reason == "stop"


def test_deadline_interrupts_real_sdk_socket_read():
    import json
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Event, Thread
    from openai import OpenAI
    from autorealize.llm.client import StreamDeadlineExceeded

    release = Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            frame = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
                     "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
            self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
            self.wfile.flush()
            release.wait(2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with OpenAI(api_key="test", base_url=f"http://127.0.0.1:{server.server_port}/v1", max_retries=0, timeout=3) as client:
            stream = client.chat.completions.create(model="test", messages=[{"role": "user", "content": "test"}], stream=True)
            started = time.monotonic()
            with pytest.raises(StreamDeadlineExceeded):
                LLMClient._consume_completion_stream(stream, timeout_seconds=0.05)
            assert time.monotonic() - started < 1, "Socket read survived the total response deadline"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_usage_option_fallback_preserves_stream_and_requested_reasoning():
    params = {"model": "gpt-5.6-luna", "reasoning_effort": "xhigh", "stream": True,
              "stream_options": {"include_usage": True}}
    result, key = _degraded_request_kwargs(params, ValueError("unsupported parameter: stream_options"))
    assert key == "stream_options"
    assert result == {"model": "gpt-5.6-luna", "reasoning_effort": "xhigh", "stream": True}
    assert _degraded_request_kwargs(params, ValueError("unsupported parameter: reasoning_effort")) == (None, "")
