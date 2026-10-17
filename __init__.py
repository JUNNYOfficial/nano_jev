"""``nano_jev``：可整包复制的 **Jev 平替**服务（纳秒级粗筛判断层）。

## 这是什么

一个**零依赖、确定性、可整包复制**的独立子系统：把「用一次极快的粗筛先给判断、
难例再升级」的思路，做成一个能直接接进 `jev-arena <https://github.com/NanmiCoder/jev-arena>`_
对比场的服务。它实现 Jev 侧的原生 ``/alpha/decisions`` 契约，另附 OpenAI 兼容的
``/chat/completions``，两种方式都能让 jev-arena 的左道指向本服务。

## 与仓库根 ``nano/`` 的关系

``nano/`` 是**几何数值内核**（float32 的 C++ 加速层 + 芯片预算模型）——它算的是
体积、包含性这类**数值量**，本身不携带任何任务语义。本目录沿用它的**理念**
（快速、粗粒度、只做初筛、不承诺精选），但落在**语义判断**这一侧：输入是文本与
结构化问题，输出是受控词表上的标签与逐问题答案。二者**不可互相替代**：
``nano/`` 回答「这个数值量大约是多少」，``nano_jev/`` 回答「这条内容大概属于哪一类」。

## 它是什么、不是什么（务必先读）

* **是**：确定性词典 + 启发式的粗筛器，单条判断在微秒—毫秒量级，**可复现**，
  不联网、不调模型。
* **不是**：语义理解模型。它按关键词与规则打分，**准确率上限明显低于真模型**。
  它的定位是「便宜地先筛一遍」，用于省下昂贵调用，而不是取代昂贵调用。

详见 [README.md](README.md)。
"""

from __future__ import annotations

from .nj_arbitrate import FIELDS, field_agreement, merge_labels
from .nj_cache import HybridCache
from .nj_contract import (
    DEFAULT_TOPICS,
    DecisionsRequest,
    build_answers,
    build_decisions_response,
    parse_decisions_request,
)
from .nj_engine import Label, classify, estimate_tokens
from .nj_guard import (
    GuardPolicy,
    PiiMatch,
    PreparedContent,
    PromptInjectionError,
    detect_injection,
    redact_pii,
    wrap_untrusted,
)
from .nj_hybrid import (
    BudgetExceededError,
    HybridBudget,
    HybridPolicy,
    HybridResult,
    LLMJudge,
    judge_hybrid,
    judge_many,
)
from .nj_limits import LimitExceededError, ServiceLimits
from .nj_llm import OpenAICompatJudge, label_from_mapping
from .nj_validate import (
    LabelIssue,
    is_valid_label,
    require_valid_label,
    validate_label,
    validate_label_mapping,
)
from .nj_vocab import ASPECTS, EMOTIONS, INTENTS, MAX_ASPECTS, MAX_EMOTIONS, SENTIMENTS

__all__ = [
    "ASPECTS",
    "DEFAULT_TOPICS",
    "EMOTIONS",
    "FIELDS",
    "INTENTS",
    "MAX_ASPECTS",
    "MAX_EMOTIONS",
    "SENTIMENTS",
    "VERSION",
    "BudgetExceededError",
    "DecisionsRequest",
    "GuardPolicy",
    "HybridBudget",
    "HybridCache",
    "HybridPolicy",
    "HybridResult",
    "LLMJudge",
    "Label",
    "LabelIssue",
    "LimitExceededError",
    "OpenAICompatJudge",
    "PiiMatch",
    "PreparedContent",
    "PromptInjectionError",
    "ServiceLimits",
    "build_answers",
    "build_decisions_response",
    "classify",
    "detect_injection",
    "estimate_tokens",
    "field_agreement",
    "is_valid_label",
    "judge_hybrid",
    "judge_many",
    "label_from_mapping",
    "merge_labels",
    "parse_decisions_request",
    "redact_pii",
    "require_valid_label",
    "validate_label",
    "validate_label_mapping",
    "wrap_untrusted",
]

#: 运行时版本号，随 ``/health`` 与 ``/v1/models`` 返回，便于排查「装的是哪版」。
VERSION = "1.2.0"
