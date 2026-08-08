"""Explainable heuristic mapping from messy column names to econometric roles."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Mapping

import pandas as pd


ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "outcome": (
        "outcome", "dependent", "response", "result", "y", "被説明変数",
        "目的変数", "成果", "結果", "アウトカム", "因变量", "结果变量",
    ),
    "treatment": (
        "treated", "treatment", "intervention", "policy_group", "exposed",
        "処置", "介入", "政策対象", "処理群", "实验组", "处理组", "政策组",
    ),
    "post": (
        "post", "after", "post_policy", "after_treatment", "政策後", "介入後",
        "事後", "改革後", "政策后", "处理后", "事后",
    ),
    "entity": (
        "entity", "unit", "individual", "firm", "company", "person", "id",
        "自治体", "企業", "個人", "主体", "地区", "个体", "企业", "单位",
    ),
    "time": (
        "time", "year", "date", "period", "quarter", "month", "wave",
        "時点", "年", "年度", "日付", "期間", "年份", "时间", "季度",
    ),
    "subgroup": (
        "subgroup", "eligible", "third_group", "target_group", "group3",
        "第三群", "資格", "対象属性", "サブグループ", "子群", "第三维",
    ),
}


@dataclass(frozen=True)
class Candidate:
    column: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RoleSuggestion:
    role: str
    column: str | None
    confidence: float
    reasons: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def _tokens(value: str) -> list[str]:
    return [part for part in re.split(r"[^\w]+", value.lower()) if part]


def _name_score(column: str, aliases: tuple[str, ...]) -> tuple[float, list[str]]:
    lowered = column.lower()
    tokens = _tokens(lowered)
    best = 0.0
    reason = ""
    for alias in aliases:
        a = alias.lower()
        if lowered == a:
            return 0.75, [f"column name exactly matches ‘{alias}’"]
        if a in tokens or (len(a) >= 2 and a in lowered):
            if 0.62 > best:
                best, reason = 0.62, f"column name contains ‘{alias}’"
        similarity = SequenceMatcher(None, lowered, a).ratio()
        value = 0.42 * similarity
        if value > best:
            best, reason = value, f"column name resembles ‘{alias}’"
    return best, [reason] if reason else []


def _profile_score(series: pd.Series, role: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    non_missing = series.dropna()
    unique = int(non_missing.nunique())
    n = max(len(non_missing), 1)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    if role == "outcome" and is_numeric and unique > 2:
        score += 0.25
        reasons.append("numeric with variation")
    elif role in {"treatment", "post", "subgroup"} and unique == 2:
        score += 0.28
        reasons.append("has exactly two observed values")
    elif role == "time":
        if is_datetime:
            score += 0.32
            reasons.append("datetime type")
        elif is_numeric and 2 <= unique <= max(2, min(100, n // 2)):
            score += 0.15
            reasons.append("numeric repeated periods")
    elif role == "entity" and 2 < unique < n and unique <= int(n * 0.8):
        score += 0.18
        reasons.append("identifiers repeat across rows")
    return score, reasons


def _description_score(column: str, description: str) -> tuple[float, list[str]]:
    if not description.strip():
        return 0.0, []
    c_tokens = set(_tokens(column))
    d_tokens = set(_tokens(description))
    overlap = c_tokens & d_tokens
    if overlap:
        return min(0.24, 0.08 * len(overlap)), ["matches words in the user's description"]
    similarity = SequenceMatcher(None, column.lower(), description.lower()).ratio()
    if similarity >= 0.65:
        return 0.1, ["resembles the user's description"]
    return 0.0, []


def suggest_roles(
    df: pd.DataFrame,
    descriptions: Mapping[str, str] | None = None,
    require_subgroup: bool = False,
) -> dict[str, RoleSuggestion]:
    """Rank columns for each DID/DDD role and return explainable suggestions.

    This is intentionally advisory rather than an opaque classifier. Every role
    includes ranked alternatives and reasons so users can confirm or correct it.
    """

    descriptions = descriptions or {}
    roles = ["outcome", "treatment", "post", "entity", "time"]
    if require_subgroup:
        roles.append("subgroup")
    suggestions: dict[str, RoleSuggestion] = {}
    already_used: set[str] = set()

    for role in roles:
        ranked: list[Candidate] = []
        for column in df.columns:
            name = str(column)
            name_value, name_reasons = _name_score(name, ROLE_ALIASES[role])
            profile_value, profile_reasons = _profile_score(df[column], role)
            desc_value, desc_reasons = _description_score(
                name, descriptions.get(role, "")
            )
            # Econometric roles are semantically distinct. A strong reuse
            # penalty prevents one binary column from being suggested as both
            # treatment and subgroup merely because no true subgroup exists.
            penalty = 0.55 if name in already_used else 0.0
            score = max(0.0, min(1.0, name_value + profile_value + desc_value - penalty))
            ranked.append(
                Candidate(
                    column=name,
                    score=score,
                    reasons=tuple(name_reasons + profile_reasons + desc_reasons),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        best = ranked[0] if ranked and ranked[0].score >= 0.18 else None
        if best is not None:
            already_used.add(best.column)
        suggestions[role] = RoleSuggestion(
            role=role,
            column=best.column if best else None,
            confidence=best.score if best else 0.0,
            reasons=best.reasons if best else ("no credible automatic match",),
            candidates=tuple(ranked[:5]),
        )
    return suggestions
