"""OpenAI 兼容判断器的行为测试（不联网：urlopen 用假响应替换）。"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.error

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from nano_jev.nj_engine import Label  # noqa: E402
from nano_jev.nj_llm import (  # noqa: E402
    OpenAICompatJudge,
    extract_json_object,
    label_from_mapping,
    make_hybrid_labeler,
)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fake_urlopen(payload, captured):
    def opener(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    return opener


def test_extract_json_object_tolerates_code_fences_and_chatter():
    text = '好的：\n```json\n{"sentiment": "positive"}\n```\n以上。'
    assert extract_json_object(text) == {"sentiment": "positive"}


def test_extract_json_object_rejects_non_json():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("positive")
    with pytest.raises(ValueError, match="must be an object"):
        extract_json_object("[1, 2]")


def test_label_from_mapping_normalizes_vocabulary():
    label = label_from_mapping(
        "c0",
        {
            "is_relevant": True,
            "sentiment": "Positive",
            "intent": "praise!!",
            "aspects": ["capability", "nope"],
            "emotion": ["curiosity"],
            "confidence": "high",
        },
    )

    assert label.sentiment == "positive"
    assert label.intent == "other"
    assert label.aspects == ["capability"]
    assert label.emotion == ["期待"]
    assert label.confidence == 0.5


def test_judge_posts_prompt_and_parses_usage(monkeypatch):
    captured = {}
    payload = {
        "choices": [{"message": {"content": '{"sentiment": "positive"}'}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", _fake_urlopen(payload, captured)
    )
    judge = OpenAICompatJudge("http://host/v1", "m", api_key="k")

    label = judge("这个模型很好用", Label("c0", True, "neutral", 0.0, 0.5, "other"))

    assert label.sentiment == "positive"
    assert captured["url"] == "http://host/v1/chat/completions"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["temperature"] == 0
    assert judge.last_input_tokens == 123
    assert judge.last_output_tokens == 45
    assert judge.last_duration_ms >= 0


def test_prompt_lists_controlled_vocabulary():
    judge = OpenAICompatJudge("http://host/v1", "m")
    prompt = judge.build_prompt("文本")

    assert "positive" in prompt
    assert "complaint" in prompt
    assert "<content>\n文本\n</content>" in prompt


def test_namespace_encodes_model_and_prompt():
    first = OpenAICompatJudge("http://host/v1", "m")
    second = OpenAICompatJudge("http://host/v1", "m", prompt_template="其他 {content}")

    assert first.namespace.startswith("m/")
    assert first.namespace != second.namespace


def test_labeler_routes_through_hybrid(monkeypatch):
    calls = []

    def fake_judge(content, hint):
        calls.append(content)
        return label_from_mapping(hint.comment_id, {"sentiment": "positive"})

    labeler = make_hybrid_labeler(fake_judge)
    label = labeler({"comment_id": "c0", "content": "这个模型还行"})

    assert label.sentiment == "positive"
    assert calls == ["这个模型还行"]


def _http_error(code: int, retry_after: str | None = None):
    import email.message
    import urllib.error

    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "http://host/v1/chat/completions", code, "err", headers, io.BytesIO(b"")
    )


def test_retries_transient_429_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    calls = {"n": 0}
    payload = {"choices": [{"message": {"content": '{"sentiment": "positive"}'}}]}

    def opener(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, retry_after="2")
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", opener)
    monkeypatch.setattr("time.sleep", sleeps.append)
    judge = OpenAICompatJudge("http://host/v1", "m", retries=2)

    label = judge("文本", Label("c0", True, "neutral", 0.0, 0.5, "other"))

    assert label.sentiment == "positive"
    assert calls["n"] == 2
    assert sleeps == [2.0]  # 用了 Retry-After


def test_does_not_retry_client_errors(monkeypatch):
    calls = {"n": 0}

    def opener(request, timeout=None):
        calls["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr("urllib.request.urlopen", opener)
    judge = OpenAICompatJudge("http://host/v1", "m", retries=3)

    with pytest.raises(urllib.error.HTTPError, match="400"):
        judge("文本", Label("c0", True, "neutral", 0.0, 0.5, "other"))
    assert calls["n"] == 1


def test_retry_exhaustion_reraises(monkeypatch):
    calls = {"n": 0}

    def opener(request, timeout=None):
        calls["n"] += 1
        raise _http_error(503)

    monkeypatch.setattr("urllib.request.urlopen", opener)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    judge = OpenAICompatJudge("http://host/v1", "m", retries=2)

    with pytest.raises(urllib.error.HTTPError, match="503"):
        judge("文本", Label("c0", True, "neutral", 0.0, 0.5, "other"))
    assert calls["n"] == 3


def test_backoff_is_exponential_without_retry_after(monkeypatch):
    sleeps: list[float] = []
    calls = {"n": 0}

    def opener(request, timeout=None):
        calls["n"] += 1
        raise _http_error(500)

    monkeypatch.setattr("urllib.request.urlopen", opener)
    monkeypatch.setattr("time.sleep", sleeps.append)
    judge = OpenAICompatJudge("http://host/v1", "m", retries=2, backoff_s=0.25)

    with pytest.raises(urllib.error.HTTPError, match="500"):
        judge("文本", Label("c0", True, "neutral", 0.0, 0.5, "other"))
    assert sleeps == [0.25, 0.5]


def test_retry_settings_are_validated():
    with pytest.raises(ValueError, match="retries"):
        OpenAICompatJudge("http://h/v1", "m", retries=-1)
    with pytest.raises(ValueError, match="backoff_s"):
        OpenAICompatJudge("http://h/v1", "m", backoff_s=-1)
