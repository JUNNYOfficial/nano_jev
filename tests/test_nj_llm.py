"""OpenAI 兼容判断器的行为测试（不联网：urlopen 用假响应替换）。"""

from __future__ import annotations

import io
import json
import pathlib
import sys

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
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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
