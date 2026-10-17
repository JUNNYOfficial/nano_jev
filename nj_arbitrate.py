"""字段级升级与冲突仲裁。

整条样本一起升级（"要么全信粗筛，要么全信模型"）在两种情况下都不划算：

* 粗筛对 ``is_relevant`` 很准，但 ``sentiment`` 在反讽上常错——只想复核后者；
* 模型返回的 ``aspects`` 更细，但 ``is_relevant`` 偶尔过度判真。

本模块把标签拆成字段，允许只采纳模型的部分字段（:func:`merge_labels`），
并记录两边在每个字段上是否一致（:func:`field_agreement`）。仲裁规则是**显式**
的：默认模型字段优先，可改为冲突时保留粗筛值；不做"两边置信度加权平均"——
两个分数都不是校准概率，平均出来的数没有可信含义。
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Literal

from .nj_engine import Label

#: 可独立升级/仲裁的字段。``confidence`` 刻意不在其中：粗筛与模型的分数量纲
#: 不同，混用会制造"看起来更确定"的假象。
FIELDS: tuple[str, ...] = (
    "is_relevant",
    "sentiment",
    "sentiment_score",
    "intent",
    "aspects",
    "emotion",
    "evidence_quote",
)

#: 仲裁规则：冲突时保留模型值，或保留粗筛值。
ConflictPolicy = Literal["prefer_model", "prefer_coarse"]


def _copy(label: Label) -> Label:
    return replace(
        label,
        aspects=list(label.aspects),
        emotion=list(label.emotion),
        meta=dict(label.meta),
    )


def field_agreement(coarse: Label, model: Label) -> dict[str, bool]:
    """逐字段比较两个标签是否一致；``aspects`` 按集合比较（与顺序无关）。"""
    agreement: dict[str, bool] = {}
    for name in FIELDS:
        left = getattr(coarse, name)
        right = getattr(model, name)
        if name == "aspects":
            agreement[name] = set(left) == set(right)
        else:
            agreement[name] = left == right
    return agreement


def merge_labels(
    coarse: Label,
    model: Label,
    fields: tuple[str, ...] | None = None,
    *,
    conflict: ConflictPolicy = "prefer_model",
) -> Label:
    """按 ``fields`` 从模型标签取字段，其余保留粗筛值。

    ``fields=None`` 表示**全部**字段都取模型值（整条升级）。``conflict`` 只在
    逐字段模式下生效：``"prefer_coarse"`` 时，两边不一致的字段保留粗筛值。
    返回的标签带 ``meta["arbitration"]``，记录采纳字段与冲突字段。
    """
    selected = tuple(FIELDS) if fields is None else tuple(fields)
    unknown = [name for name in selected if name not in FIELDS]
    if unknown:
        raise ValueError(
            f"unknown field(s) {unknown}; expected a subset of {list(FIELDS)}"
        )
    if conflict not in ("prefer_model", "prefer_coarse"):
        raise ValueError("conflict must be 'prefer_model' or 'prefer_coarse'")

    agreement = field_agreement(coarse, model)
    adopted: list[str] = []
    conflicts: list[str] = []

    merged = _copy(coarse)
    for name in selected:
        if not agreement[name]:
            conflicts.append(name)
            if conflict == "prefer_coarse":
                continue
        value = getattr(model, name)
        if name in ("aspects", "emotion"):
            value = list(value)
        setattr(merged, name, value)
        adopted.append(name)

    # 置信度跟随"是否整条采纳模型结果"：整条采纳时用模型值，否则保留粗筛值，
    # 避免"只采纳了 aspects，却继承模型的高置信度"。
    if fields is None and conflict == "prefer_model":
        merged.confidence = model.confidence

    merged.meta["arbitration"] = {
        "fields": list(selected),
        "adopted": adopted,
        "conflicts": conflicts,
        "conflict_policy": conflict,
    }
    return merged


def arbitration_fields(label: Label) -> tuple[str, ...]:
    """从标签的 ``meta["arbitration"]`` 读回实际采纳的字段（无记录则空）。"""
    record = label.meta.get("arbitration")
    if not isinstance(record, dict):
        return ()
    adopted = record.get("adopted")
    if not isinstance(adopted, list):
        return ()
    return tuple(name for name in adopted if isinstance(name, str))


def label_fields() -> tuple[str, ...]:
    """``Label`` 的数据字段名（不含 ``meta``），供调用方做形状校验。"""
    return tuple(item.name for item in dataclass_fields(Label) if item.name != "meta")
