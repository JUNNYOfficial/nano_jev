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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

#: 中国居民身份证的加权因子与校验码表（GB 11643-1999）。
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECKSUM = "10X98765432"


def luhn_valid(number: str) -> bool:
    """Luhn 校验（银行卡/信用卡）。非数字或长度不在 12–19 之间一律为假。"""
    digits = number.replace(" ", "").replace("-", "")
    if not digits.isdigit() or not 12 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


#: 身份证出生日期的合理年份下界（1980 年前签发的号码已不再使用）。
_ID_EARLIEST_YEAR = 1900


def _valid_birth_date(year: int, month: int, day: int) -> bool:
    """出生日期段是否合法：年份合理、月份 1–12、日与月份/闰年匹配。"""
    if not _ID_EARLIEST_YEAR <= year <= datetime.now().year:
        return False
    if not 1 <= month <= 12:
        return False
    if day < 1:
        return False
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return day <= (29 if leap else 28)
    return day <= (30 if month in (4, 6, 9, 11) else 31)


def chinese_id_valid(number: str) -> bool:
    """中国 18 位身份证：校验位 + 出生日期段。

    只校验校验位是不够的——`11010119901301...` 这类 13 月的号码能凑出正确校验位，
    但它不可能是真实证件。日期段校验把这类"形状合法但语义不可能"的号码挡掉，
    避免把它们当成 PII 替换（那会破坏原文，还可能让下游误判）。
    """
    candidate = number.strip().upper()
    if len(candidate) != 18 or not candidate[:17].isdigit():
        return False
    if candidate[17] not in "0123456789X":
        return False
    year = int(candidate[6:10])
    month = int(candidate[10:12])
    day = int(candidate[12:14])
    if not _valid_birth_date(year, month, day):
        return False
    total = sum(int(char) * weight for char, weight in zip(candidate[:17], _ID_WEIGHTS))
    return _ID_CHECKSUM[total % 11] == candidate[17]


#: 中国大陆手机号：可带 +86 / 86 前缀。
_MOBILE_RE = re.compile(r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d{9}(?!\d)")
#: 国际号码：必须有显式 ``+``，否则 8–15 位数字会把时间戳/订单号全卷进来。
_E164_RE = re.compile(r"(?<![\w+])\+[1-9]\d{7,14}(?!\d)")
#: 中国固话：区号 + 连字符 + 号码；没有连字符的纯数字不猜。
_LANDLINE_RE = re.compile(r"(?<!\d)0\d{2,3}-\d{7,8}(?!\d)")
_ID_RE = re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])")
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")


def _valid_email(value: str) -> bool:
    """邮箱的最低校验：本地部分非空、域名含点且顶级域至少 2 个字母。"""
    local, _, domain = value.partition("@")
    if not local or not domain or domain.startswith(".") or ".." in value:
        return False
    labels = domain.split(".")
    return len(labels) >= 2 and all(labels) and len(labels[-1]) >= 2


#: 邮箱的替换回调：形状不合法就保留原文（`re.sub` 不接受 lambda 的宽泛类型）。
_EMAIL_TOKEN = "[EMAIL]"


def _replace_valid_email(match: re.Match[str]) -> str:
    return _EMAIL_TOKEN if _valid_email(match.group(0)) else match.group(0)


#: 可校验规则：(类型, 模式, 校验函数, 替换标记)。
#: 顺序即优先级；身份证在银行卡之前——18 位号码内部可能凑出合法的 16 位 Luhn 串。
_VERIFIED_RULES: tuple[
    tuple[str, re.Pattern[str], Callable[[str], bool], str], ...
] = (
    ("id_number", _ID_RE, chinese_id_valid, "[ID_NUMBER]"),
    ("bank_card", _CARD_RE, luhn_valid, "[BANK_CARD]"),
    ("phone", _MOBILE_RE, lambda value: True, "[PHONE]"),
    ("phone", _E164_RE, lambda value: True, "[PHONE]"),
    ("phone", _LANDLINE_RE, lambda value: True, "[PHONE]"),
)

#: 邮箱形状（是否合法由 :func:`_valid_email` 再判）。
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

#: 形状规则：没有校验位可用，只能按形状匹配（密钥、邮箱）。
_PATTERN_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "api_key",
        re.compile(r"\b(?:sk|pk|ghp|gho|glpat|xox[baprs])[-_][A-Za-z0-9_-]{16,}\b"),
        "[API_KEY]",
    ),
    (
        "email",
        _EMAIL_RE,
        _EMAIL_TOKEN,
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


#: 内置可脱敏的 PII 类型，供 ``GuardPolicy.pii_kinds`` 选择。
PII_KINDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [kind for kind, _pattern, _check, _token in _VERIFIED_RULES]
        + [kind for kind, _pattern, _token in _PATTERN_RULES]
    )
)


@dataclass(frozen=True)
class CustomRule:
    """调用方自定义的脱敏规则：正则 + 替换标记。

    ``validator`` 可选：给了就先跑它，返回 ``False`` 时不替换（用于"形状像但
    校验不通过"的场景，例如合作方自己的会员号校验位）。
    """

    kind: str
    pattern: re.Pattern[str]
    token: str
    validator: Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind must be non-empty")
        if not self.token:
            raise ValueError("token must be non-empty")

    def replace(self, match: re.Match[str]) -> str:
        """``re.sub`` 的替换回调：校验不过就保留原文。"""
        value = match.group(0)
        if self.validator is not None and not self.validator(value):
            return value
        return self.token


@dataclass(frozen=True)
class GuardPolicy:
    """输入防护策略。默认值面向"只标记、不改写语义"的保守用法。

    ``pii_kinds`` 选择启用哪些内置规则（默认全部）；``extra_rules`` 追加自定义
    规则。两类规则都在形状规则之前执行，避免被更粗的模式抢先切碎。
    """

    max_chars: int = 0
    redact_pii: bool = True
    detect_injections: bool = True
    injection_action: Literal["flag", "block", "raise"] = "flag"
    pii_kinds: tuple[str, ...] = PII_KINDS
    extra_rules: tuple[CustomRule, ...] = ()

    def __post_init__(self) -> None:
        if self.max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        if self.injection_action not in ("flag", "block", "raise"):
            raise ValueError("injection_action must be 'flag', 'block' or 'raise'")
        unknown = [kind for kind in self.pii_kinds if kind not in PII_KINDS]
        if unknown:
            raise ValueError(
                f"unknown PII kinds {unknown}; expected a subset of {list(PII_KINDS)}"
            )

    def _apply_verified(self, body: str, matches: list[PiiMatch]) -> str:
        """按模式 + 校验函数筛选并替换；重叠时保留更长的匹配。

        身份证 18 位、银行卡 16–19 位、手机 11 位会互相包含，故先收集全部候选，
        按长度降序占位，避免同一段文本被替换两次或留下半截残片。
        """
        spans: list[tuple[int, int, str, str]] = []
        for kind, pattern, check, token in _VERIFIED_RULES:
            if kind not in self.pii_kinds:
                continue
            for match in pattern.finditer(body):
                if check(match.group(0)):
                    spans.append((match.start(), match.end(), kind, token))

        spans.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
        occupied: list[tuple[int, int]] = []
        chosen: list[tuple[int, int, str, str]] = []
        for start, end, kind, token in spans:
            if any(not (end <= left or start >= right) for left, right in occupied):
                continue
            occupied.append((start, end))
            chosen.append((start, end, kind, token))

        for start, end, kind, _token in sorted(chosen, key=lambda item: item[0]):
            matches.append(PiiMatch(kind, start, end))
        for start, end, _kind, token in sorted(chosen, key=lambda item: -item[0]):
            body = body[:start] + token + body[end:]
        return body

    def _apply_custom(self, body: str, matches: list[PiiMatch]) -> str:
        """应用自定义规则；``validator`` 返回假时保留原文。"""
        for rule in self.extra_rules:
            for match in list(rule.pattern.finditer(body)):
                if rule.validator is not None and not rule.validator(match.group(0)):
                    continue
                matches.append(PiiMatch(rule.kind, match.start(), match.end()))
            body = rule.pattern.sub(rule.replace, body)
        return body

    def _apply_patterns(self, body: str, matches: list[PiiMatch]) -> str:
        """形状规则（密钥、邮箱）；邮箱额外做最低限度的格式校验。"""
        for kind, pattern, token in _PATTERN_RULES:
            if kind not in self.pii_kinds:
                continue
            for match in pattern.finditer(body):
                if kind == "email" and not _valid_email(match.group(0)):
                    continue
                matches.append(PiiMatch(kind, match.start(), match.end()))
            if kind == "email":
                body = _EMAIL_RE.sub(_replace_valid_email, body, count=0)
            else:
                body = pattern.sub(token, body)
        return body

    def prepare(self, text: str) -> PreparedContent:
        """脱敏、截断并检测注入；不抛异常（拒绝由调用方按策略决定）。

        处理顺序：截断 → 可校验规则 → 自定义规则 → 形状规则 → 注入检测。
        校验失败的数字段**保持原样**：宁可漏脱敏，也不要把订单号、时间戳
        误标成银行卡（那会破坏文本语义，还可能让下游误判）。
        """
        body = text
        truncated = False
        if self.max_chars and len(body) > self.max_chars:
            body = body[: self.max_chars]
            truncated = True

        matches: list[PiiMatch] = []
        if self.redact_pii:
            body = self._apply_verified(body, matches)
            body = self._apply_custom(body, matches)
            body = self._apply_patterns(body, matches)
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
