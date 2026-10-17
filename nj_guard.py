"""进入大模型之前的输入防护：PII 脱敏、注入检测与不可信内容包裹。

三件事都属于"把用户文本交给模型之前"的一侧，故集中在一个模块：

* :class:`GuardPolicy` —— 是否脱敏、是否限制长度、遇到注入怎么办；
* :func:`detect_injection` —— 命中常见的"忽略以上指令 / 输出系统提示词"模式；
* :func:`wrap_untrusted` —— 用定界符包裹原文，并转义定界符，降低"文本被当成
  新指令"的机会。

**它不是安全边界**：模式匹配只能挡住已知措辞，绕过方式无穷无尽。真正的防线
是"模型输出必须回到受控词表与原文校验"（见 :mod:`nano_jev.nj_hybrid`），以及
调用方不要把模型输出直接当代码/命令执行。这里只是把明显不该送出去的文本先
挡掉或脱敏，降低泄露面。

**零依赖**：只用标准库正则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: PII 规则：(类型, 预编译模式, 替换标记)。
#: 顺序即替换顺序；长模式（密钥/身份证）放在手机号之前，避免被更短的规则先切碎。
_PII_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "api_key",
        re.compile(r"\b(?:sk|pk|ghp|gho|glpat|xox[baprs])[-_][A-Za-z0-9_-]{16,}\b"),
        "[API_KEY]",
    ),
    (
        "id_number",
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        "[ID_NUMBER]",
    ),
    (
        "phone",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "[PHONE]",
    ),
    (
        "email",
        re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
        "[EMAIL]",
    ),
    (
        "bank_card",
        re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
        "[BANK_CARD]",
    ),
)

#: 注入措辞：(标签, 模式)。只收录高置信、低误伤的写法；命中不等于一定恶意，
#: 故默认只标记不拦截（见 :attr:`GuardPolicy.injection_action`）。
_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?"
            r"(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt",
        re.compile(
            r"(?:reveal|show|print|repeat|output)\s+(?:your\s+)?"
            r"(?:system|developer|hidden)\s+prompt",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+"
            r"(?:a\s+)?(?:dan|developer|admin|root|system)",
            re.IGNORECASE,
        ),
    ),
    (
        "ignore_zh",
        re.compile(r"忽略(?:以上|之前|前面|上述|所有)?.{0,6}(?:指令|提示词?|规则|要求)"),
    ),
    (
        "system_prompt_zh",
        re.compile(
            r"(?:输出|显示|打印|重复|告诉我).{0,6}(?:系统|开发者|隐藏).{0,4}"
            r"(?:提示词|指令|prompt)"
        ),
    ),
    (
        "role_override_zh",
        re.compile(r"你现在(?:是|扮演|作为)"),
    ),
)


class PromptInjectionError(ValueError):
    """输入命中注入模式，且策略要求直接拒绝。"""


@dataclass(frozen=True)
class PiiMatch:
    """一处 PII 命中的位置与类型。"""

    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class PreparedContent:
    """送进模型之前的文本，以及做过的处理记录。"""

    text: str
    pii: tuple[PiiMatch, ...] = ()
    injections: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def redacted(self) -> bool:
        """是否发生过 PII 脱敏。"""
        return bool(self.pii)


@dataclass(frozen=True)
class GuardPolicy:
    """输入防护策略。默认值面向"只标记、不改写语义"的保守用法。"""

    max_chars: int = 0
    redact_pii: bool = True
    detect_injections: bool = True
    injection_action: Literal["flag", "block", "raise"] = "flag"

    def __post_init__(self) -> None:
        if self.max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        if self.injection_action not in ("flag", "block", "raise"):
            raise ValueError("injection_action must be 'flag', 'block' or 'raise'")

    def prepare(self, text: str) -> PreparedContent:
        """脱敏、截断并检测注入；不抛异常（拒绝由调用方按策略决定）。"""
        body = text
        truncated = False
        if self.max_chars and len(body) > self.max_chars:
            body = body[: self.max_chars]
            truncated = True

        matches: list[PiiMatch] = []
        if self.redact_pii:
            for kind, pattern, token in _PII_RULES:
                for match in pattern.finditer(body):
                    matches.append(PiiMatch(kind, match.start(), match.end()))
                body = pattern.sub(token, body)
            matches.sort(key=lambda item: (item.start, item.end))

        injections = detect_injection(body) if self.detect_injections else ()
        return PreparedContent(body, tuple(matches), injections, truncated)


def detect_injection(text: str) -> tuple[str, ...]:
    """返回命中的注入模式标签（去重、保持规则顺序）；空元组表示未命中。"""
    hits: list[str] = []
    for tag, pattern in _INJECTION_RULES:
        if pattern.search(text) and tag not in hits:
            hits.append(tag)
    return tuple(hits)


def redact_pii(text: str) -> tuple[str, tuple[PiiMatch, ...]]:
    """按固定规则把常见 PII 替换成占位符，返回 ``(脱敏文本, 命中列表)``。"""
    prepared = GuardPolicy().prepare(text)
    return prepared.text, prepared.pii


def wrap_untrusted(text: str, tag: str = "content") -> str:
    """用定界符包裹不可信文本，并转义重复的定界符。

    模型看到的是"数据块"，而不是可被续写的指令。``tag`` 由调用方决定；若担心
    用户文本里恰好出现同名标签，可传一个随机 nonce（例如 ``content_7f3a``）。
    """
    closing = f"</{tag}>"
    opening = f"<{tag}>"
    body = text.replace(closing, f"<\\/{tag}>").replace(opening, f"<\\{tag}>")
    return f"{opening}\n{body}\n{closing}"


def build_guarded_prompt(
    text: str,
    *,
    policy: GuardPolicy | None = None,
    tag: str = "content",
) -> tuple[str, PreparedContent]:
    """一站式：按策略处理文本，再包成不可信数据块。

    返回 ``(提示词片段, 处理记录)``；调用方据此决定是否继续调用模型。
    """
    prepared = (policy or GuardPolicy()).prepare(text)
    return wrap_untrusted(prepared.text, tag), prepared
