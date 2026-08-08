"""DID and DDD model presets with diagnostics and robust inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
import warnings as python_warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PresetName = Literal["did_basic", "did_twfe", "did_custom_fe", "ddd"]

PRESETS: dict[str, dict[str, str]] = {
    "did_basic": {
        "label": "Basic 2×2 DID",
        "description": "Treatment, post-period, and their interaction; robust HC1 standard errors.",
    },
    "did_twfe": {
        "label": "Panel DID with entity and time fixed effects",
        "description": "Two-way fixed effects DID; normally cluster standard errors by entity.",
    },
    "did_custom_fe": {
        "label": "DID with custom fixed effects",
        "description": "Basic DID plus any categorical fixed effects selected by the user.",
    },
    "ddd": {
        "label": "Triple differences (DDD)",
        "description": "Full treatment × post × subgroup interaction with optional fixed effects.",
    },
}


@dataclass(frozen=True)
class ModelConfig:
    """User-confirmed semantic roles and estimator options."""

    preset: PresetName
    outcome: str
    treatment: str
    post: str
    subgroup: str | None = None
    entity: str | None = None
    time: str | None = None
    controls: tuple[str, ...] = ()
    fixed_effects: tuple[str, ...] = ()
    cluster: str | None = None
    treatment_value: Any | None = None
    post_value: Any | None = None
    subgroup_value: Any | None = None
    alpha: float = 0.05


@dataclass
class AnalysisResult:
    """Model output plus the exact data and decisions used to create it."""

    config: ModelConfig
    formula: str
    target_term: str
    model: Any
    analysis_data: pd.DataFrame
    diagnostics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def estimate(self) -> float:
        return float(self.model.params[self.target_term])

    @property
    def standard_error(self) -> float:
        return float(self.model.bse[self.target_term])

    @property
    def p_value(self) -> float:
        return float(self.model.pvalues[self.target_term])

    @property
    def confidence_interval(self) -> tuple[float, float]:
        interval = self.model.conf_int(alpha=self.config.alpha).loc[self.target_term]
        return float(interval.iloc[0]), float(interval.iloc[1])

    def coefficient_table(self) -> pd.DataFrame:
        interval = self.model.conf_int(alpha=self.config.alpha)
        table = pd.DataFrame(
            {
                "term": self.model.params.index,
                "estimate": self.model.params.values,
                "std_error": self.model.bse.values,
                "statistic": self.model.tvalues.values,
                "p_value": self.model.pvalues.values,
                "ci_lower": interval.iloc[:, 0].values,
                "ci_upper": interval.iloc[:, 1].values,
            }
        )
        table["is_target"] = table["term"] == self.target_term
        return table

    def headline(self) -> dict[str, Any]:
        lower, upper = self.confidence_interval
        return {
            "preset": self.config.preset,
            "target_term": self.target_term,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "p_value": self.p_value,
            "ci_lower": lower,
            "ci_upper": upper,
            "n_observations": int(self.model.nobs),
            "r_squared": float(self.model.rsquared),
            "formula": self.formula,
            "covariance": str(self.model.cov_type),
            "warnings": self.warnings,
        }

    def summary_text(self) -> str:
        # Some clustered models emit a rank warning while constructing the
        # omnibus F-test shown in the text summary. Coefficient inference and
        # our explicit singularity diagnostic are unaffected.
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("ignore")
            return self.model.summary().as_text()


POSITIVE_LABELS = {
    "1", "true", "yes", "y", "treated", "treatment", "after", "post",
    "exposed", "eligible", "target", "介入", "処置", "政策後", "対象",
    "是", "处理组", "政策后",
}
NEGATIVE_LABELS = {
    "0", "false", "no", "n", "control", "untreated", "before", "pre",
    "unexposed", "ineligible", "対照", "政策前", "非対象", "否", "控制组",
    "政策前",
}


def _quoted(column: str) -> str:
    escaped = column.replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{escaped}")'


def _resolve_positive(values: Sequence[Any], requested: Any) -> Any:
    for value in values:
        try:
            if value == requested or str(value) == str(requested):
                return value
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Selected positive value {requested!r} does not occur in the column.")


def _coerce_indicator(
    series: pd.Series, role: str, positive_value: Any | None
) -> tuple[pd.Series, str]:
    non_missing = series.dropna()
    values = list(pd.unique(non_missing))
    if len(values) != 2:
        raise ValueError(
            f"{role} must contain exactly two non-missing values; found {len(values)}: {values[:8]}"
        )
    if positive_value is not None:
        positive = _resolve_positive(values, positive_value)
        return series.map(lambda value: np.nan if pd.isna(value) else int(value == positive)), (
            f"{role}: {positive!r}=1, all other observed values=0 (user confirmed)"
        )

    if pd.api.types.is_bool_dtype(series):
        return series.astype(float), f"{role}: False=0, True=1"
    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.notna().all() and set(float(v) for v in numeric.unique()) == {0.0, 1.0}:
        return pd.to_numeric(series, errors="coerce").astype(float), f"{role}: 0/1 retained"

    lowered = {str(value).strip().lower(): value for value in values}
    positives = [raw for label, raw in lowered.items() if label in POSITIVE_LABELS]
    negatives = [raw for label, raw in lowered.items() if label in NEGATIVE_LABELS]
    if len(positives) == 1 and len(negatives) == 1:
        positive = positives[0]
        return series.map(lambda value: np.nan if pd.isna(value) else int(value == positive)), (
            f"{role}: recognised {positive!r} as 1 and {negatives[0]!r} as 0"
        )
    raise ValueError(
        f"The direction of {role} is ambiguous ({values}). Select which value should mean 1."
    )


def _required_columns(config: ModelConfig) -> list[str]:
    required = [config.outcome, config.treatment, config.post]
    if config.preset == "ddd":
        if not config.subgroup:
            raise ValueError("DDD requires a subgroup column.")
        required.append(config.subgroup)
    if config.preset == "did_twfe":
        if not config.entity or not config.time:
            raise ValueError("Two-way fixed effects DID requires entity and time columns.")
        required.extend([config.entity, config.time])
    required.extend(config.controls)
    required.extend(config.fixed_effects)
    if config.cluster:
        required.append(config.cluster)
    return list(dict.fromkeys(required))


def _prepare_data(df: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    required = _required_columns(config)
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")
    data = df.loc[:, required].copy()
    warnings: list[str] = []

    data[config.outcome] = pd.to_numeric(data[config.outcome], errors="coerce")
    for control in config.controls:
        if not pd.api.types.is_numeric_dtype(data[control]):
            converted = pd.to_numeric(data[control], errors="coerce")
            if converted.notna().sum() < data[control].notna().sum() * 0.85:
                raise ValueError(f"Control column {control!r} is not reliably numeric.")
            data[control] = converted

    data["_treated"], treatment_note = _coerce_indicator(
        data[config.treatment], "treatment", config.treatment_value
    )
    data["_post"], post_note = _coerce_indicator(data[config.post], "post", config.post_value)
    indicator_notes = [treatment_note, post_note]
    data["_did"] = data["_treated"] * data["_post"]
    if config.preset == "ddd" and config.subgroup:
        data["_subgroup"], subgroup_note = _coerce_indicator(
            data[config.subgroup], "subgroup", config.subgroup_value
        )
        data["_ddd"] = data["_did"] * data["_subgroup"]
        indicator_notes.append(subgroup_note)

    before = len(data)
    data = data.dropna().copy()
    dropped = before - len(data)
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing or unparseable required values.")
    if data.empty:
        raise ValueError("No complete observations remain for estimation.")

    cells = data.groupby(["_treated", "_post"], observed=True).size().to_dict()
    expected_cells = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    missing_cells = expected_cells - set(cells)
    if missing_cells:
        raise ValueError(f"DID needs all four treatment/post cells; missing {sorted(missing_cells)}")

    if config.preset == "ddd":
        ddd_cells = data.groupby(["_treated", "_post", "_subgroup"], observed=True).size()
        if len(ddd_cells) < 8:
            raise ValueError("DDD needs observations in all eight treatment/post/subgroup cells.")

    if config.entity and config.time:
        duplicate_pairs = int(data.duplicated([config.entity, config.time]).sum())
        if duplicate_pairs:
            warnings.append(
                f"Found {duplicate_pairs} duplicate entity-time rows; confirm whether aggregation is needed."
            )
    diagnostics = {
        "input_rows": before,
        "estimation_rows": len(data),
        "dropped_rows": dropped,
        "did_cell_counts": {str(key): int(value) for key, value in cells.items()},
        "indicator_coding": indicator_notes,
    }
    return data, warnings, diagnostics


def _build_formula(config: ModelConfig) -> tuple[str, str]:
    outcome = _quoted(config.outcome)
    controls = [_quoted(column) for column in config.controls]
    fixed = [f"C({_quoted(column)})" for column in config.fixed_effects]

    if config.preset == "did_basic":
        terms = ["_treated", "_post", "_did", *controls]
        target = "_did"
    elif config.preset == "did_twfe":
        assert config.entity is not None and config.time is not None
        terms = ["_did", *controls, f"C({_quoted(config.entity)})", f"C({_quoted(config.time)})"]
        target = "_did"
    elif config.preset == "did_custom_fe":
        terms = ["_treated", "_post", "_did", *controls, *fixed]
        target = "_did"
    elif config.preset == "ddd":
        terms = ["_treated * _post * _subgroup", *controls, *fixed]
        target = "_treated:_post:_subgroup"
    else:
        raise ValueError(f"Unknown preset: {config.preset}")
    return f"{outcome} ~ " + " + ".join(terms), target


def fit_model(df: pd.DataFrame, config: ModelConfig) -> AnalysisResult:
    """Fit a user-confirmed DID/DDD specification with robust inference."""

    if config.preset not in PRESETS:
        raise ValueError(f"Unknown preset {config.preset!r}; choose from {list(PRESETS)}")
    data, warnings, diagnostics = _prepare_data(df, config)
    formula, target = _build_formula(config)
    base = smf.ols(formula=formula, data=data)

    cluster = config.cluster
    if cluster is None and config.preset == "did_twfe":
        cluster = config.entity
    if cluster:
        n_groups = int(data[cluster].nunique())
        diagnostics["cluster_column"] = cluster
        diagnostics["cluster_groups"] = n_groups
        if n_groups >= 2:
            fitted = base.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
        else:
            warnings.append("Only one cluster was available; used HC1 standard errors instead.")
            fitted = base.fit(cov_type="HC1")
    else:
        fitted = base.fit(cov_type="HC1")

    if target not in fitted.params.index:
        raise RuntimeError(f"Target coefficient {target!r} was not estimated.")
    condition_number = float(getattr(fitted, "condition_number", np.nan))
    diagnostics["condition_number"] = condition_number
    if np.isfinite(condition_number) and condition_number > 1e10:
        warnings.append(
            "The design matrix is close to singular. Check collinearity and redundant fixed effects."
        )
    return AnalysisResult(
        config=config,
        formula=formula,
        target_term=target,
        model=fitted,
        analysis_data=data,
        diagnostics=diagnostics,
        warnings=warnings,
    )
