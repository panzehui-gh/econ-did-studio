"""Independent numerical checks for the one-click DID/DDD estimators.

The comparator in this module deliberately does not call statsmodels or reuse
the fitted model's design matrix.  It constructs the relevant regression
matrix directly, solves OLS with NumPy, and implements HC1 or one-way clustered
sandwich covariance.  This makes it useful as an executable cross-check of the
application's normal estimation path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import clean_dataframe
from .mapping import suggest_roles
from .models import AnalysisResult, ModelConfig, fit_model
from .synthetic import generate_ddd_data, generate_did_data, make_messy_example


def _dummy_columns(series: pd.Series) -> list[np.ndarray]:
    """Return drop-first indicator columns in deterministic category order."""

    categories = sorted(pd.unique(series), key=lambda value: str(value))
    return [(series == category).to_numpy(dtype=float) for category in categories[1:]]


def _manual_design(data: pd.DataFrame, config: ModelConfig) -> tuple[np.ndarray, int]:
    """Build a design matrix without formulas and return its target index."""

    columns: list[np.ndarray] = [np.ones(len(data), dtype=float)]
    target_index: int

    if config.preset == "did_basic":
        columns.extend(
            [
                data["_treated"].to_numpy(dtype=float),
                data["_post"].to_numpy(dtype=float),
                data["_did"].to_numpy(dtype=float),
            ]
        )
        target_index = 3
    elif config.preset == "did_twfe":
        columns.append(data["_did"].to_numpy(dtype=float))
        target_index = 1
    elif config.preset == "did_custom_fe":
        columns.extend(
            [
                data["_treated"].to_numpy(dtype=float),
                data["_post"].to_numpy(dtype=float),
                data["_did"].to_numpy(dtype=float),
            ]
        )
        target_index = 3
    elif config.preset == "ddd":
        treated = data["_treated"].to_numpy(dtype=float)
        post = data["_post"].to_numpy(dtype=float)
        subgroup = data["_subgroup"].to_numpy(dtype=float)
        columns.extend(
            [
                treated,
                post,
                treated * post,
                subgroup,
                treated * subgroup,
                post * subgroup,
                treated * post * subgroup,
            ]
        )
        target_index = 7
    else:  # pragma: no cover - guarded by ModelConfig/fit_model
        raise ValueError(f"Unsupported preset: {config.preset}")

    for control in config.controls:
        columns.append(data[control].to_numpy(dtype=float))

    fixed_effects: list[str] = list(config.fixed_effects)
    if config.preset == "did_twfe":
        assert config.entity is not None and config.time is not None
        fixed_effects.extend([config.entity, config.time])
    for fixed_effect in fixed_effects:
        columns.extend(_dummy_columns(data[fixed_effect]))

    return np.column_stack(columns), target_index


def manual_ols(result: AnalysisResult) -> dict[str, float]:
    """Independently reproduce the target OLS coefficient and robust SE.

    The same already-confirmed analysis sample is used so that this function
    tests the estimator rather than making a different missing-data decision.
    """

    data = result.analysis_data
    config = result.config
    x, target_index = _manual_design(data, config)
    y = data[config.outcome].to_numpy(dtype=float)

    pinv_x = np.linalg.pinv(x, rcond=1e-15)
    beta = pinv_x @ y
    residual = y - x @ beta
    n_observations, n_parameters = x.shape

    cluster = config.cluster
    if cluster is None and config.preset == "did_twfe":
        cluster = config.entity

    if cluster and data[cluster].nunique() >= 2:
        groups = data[cluster].to_numpy()
        scores = x * residual[:, None]
        meat = np.zeros((n_parameters, n_parameters), dtype=float)
        for group in pd.unique(groups):
            group_score = scores[groups == group].sum(axis=0)
            meat += np.outer(group_score, group_score)
        bread = pinv_x @ pinv_x.T
        n_groups = int(pd.Series(groups).nunique())
        correction = (n_groups / (n_groups - 1)) * (
            (n_observations - 1) / (n_observations - n_parameters)
        )
        covariance = correction * bread @ meat @ bread
    else:
        rank = int(np.linalg.matrix_rank(x))
        scale = residual**2 * n_observations / (n_observations - rank)
        covariance = (pinv_x * scale[None, :]) @ pinv_x.T

    return {
        "estimate": float(beta[target_index]),
        "std_error": float(np.sqrt(max(covariance[target_index, target_index], 0.0))),
    }


def _config_for(preset: str) -> ModelConfig:
    if preset == "did_basic":
        return ModelConfig(
            preset="did_basic",
            outcome="outcome",
            treatment="treated",
            post="post",
            controls=("covariate",),
        )
    if preset == "did_twfe":
        return ModelConfig(
            preset="did_twfe",
            outcome="outcome",
            treatment="treated",
            post="post",
            entity="unit_id",
            time="year",
            controls=("covariate",),
            cluster="unit_id",
        )
    if preset == "did_custom_fe":
        return ModelConfig(
            preset="did_custom_fe",
            outcome="outcome",
            treatment="treated",
            post="post",
            controls=("covariate",),
            fixed_effects=("region", "year"),
            cluster="unit_id",
        )
    if preset == "ddd":
        return ModelConfig(
            preset="ddd",
            outcome="outcome",
            treatment="treated",
            post="post",
            subgroup="eligible",
            fixed_effects=("unit_id", "year"),
            cluster="unit_id",
        )
    raise ValueError(f"Unsupported preset: {preset}")


def _random_dataset(preset: str, seed: int) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(70_000 + seed)
    true_effect = float(rng.uniform(-2.5, 3.0))
    n_entities = int(rng.integers(44, 73))
    n_periods = int(rng.integers(5, 8))
    if preset == "ddd":
        return (
            generate_ddd_data(
                n_entities=max(48, n_entities),
                n_periods=n_periods,
                effect=true_effect,
                seed=90_000 + seed,
            ),
            true_effect,
        )
    data = generate_did_data(
        n_entities=n_entities,
        n_periods=n_periods,
        effect=true_effect,
        seed=80_000 + seed,
    )
    data["region"] = data["unit_id"].str.extract(r"(\d+)", expand=False).astype(int) % 4
    return data, true_effect


def run_estimator_cross_validation(
    seeds: Iterable[int] = range(50),
    presets: Iterable[str] = ("did_basic", "did_twfe", "did_custom_fe", "ddd"),
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Compare every one-click preset with independent NumPy regression code."""

    records: list[dict[str, float | int | str | bool]] = []
    for preset in presets:
        config = _config_for(preset)
        for seed in seeds:
            data, true_effect = _random_dataset(preset, int(seed))
            result = fit_model(data, config)
            manual = manual_ols(result)
            estimate_difference = abs(result.estimate - manual["estimate"])
            se_difference = abs(result.standard_error - manual["std_error"])
            records.append(
                {
                    "preset": preset,
                    "seed": int(seed),
                    "observations": len(result.analysis_data),
                    "true_effect": true_effect,
                    "one_click_estimate": result.estimate,
                    "manual_estimate": manual["estimate"],
                    "estimate_abs_difference": estimate_difference,
                    "one_click_std_error": result.standard_error,
                    "manual_std_error": manual["std_error"],
                    "std_error_abs_difference": se_difference,
                    "passed": estimate_difference <= tolerance and se_difference <= tolerance,
                }
            )
    return pd.DataFrame(records)


def run_dirty_pipeline_validation(seeds: Iterable[int] = range(30)) -> pd.DataFrame:
    """Check role inference, cleaning, column reordering, and estimates end to end."""

    records: list[dict[str, float | int | str | bool]] = []
    for model in ("DID", "DDD"):
        for seed in seeds:
            seed = int(seed)
            preset = "did_twfe" if model == "DID" else "ddd"
            clean, _ = _random_dataset(preset, 20_000 + seed)
            baseline = fit_model(clean, _config_for(preset))

            messy = make_messy_example(clean)
            messy = messy.sample(frac=1, axis=1, random_state=seed)
            cleaned, audit = clean_dataframe(messy)
            suggestions = suggest_roles(cleaned, require_subgroup=model == "DDD")
            expected = {
                "outcome": "Outcome_value",
                "treatment": "Treatment_group",
                "post": "After_policy",
                "entity": "Unit_ID",
                "time": "Year",
            }
            if model == "DDD":
                expected["subgroup"] = "eligible"
            mapping_passed = all(
                suggestions[role].column == column for role, column in expected.items()
            )

            config = ModelConfig(
                preset=preset,
                outcome=expected["outcome"],
                treatment=expected["treatment"],
                post=expected["post"],
                subgroup=expected.get("subgroup"),
                entity=expected["entity"],
                time=expected["time"],
                controls=("covariate",) if model == "DID" else (),
                fixed_effects=(expected["entity"], expected["time"])
                if model == "DDD"
                else (),
                cluster=expected["entity"],
            )
            dirty_result = fit_model(cleaned, config)
            difference = abs(dirty_result.estimate - baseline.estimate)
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "mapping_passed": mapping_passed,
                    "duplicate_removed": audit.dropped_duplicate_rows == 1,
                    "column_order_changed": list(messy.columns) != list(clean.columns),
                    "baseline_estimate": baseline.estimate,
                    "dirty_pipeline_estimate": dirty_result.estimate,
                    "estimate_abs_difference": difference,
                    "passed": mapping_passed
                    and audit.dropped_duplicate_rows == 1
                    and difference <= 1e-3,
                }
            )
    return pd.DataFrame(records)


def cross_validation_summary(
    estimator_results: pd.DataFrame, dirty_results: pd.DataFrame
) -> pd.DataFrame:
    """Return compact evidence suitable for README and report tables."""

    rows: list[dict[str, float | int | str]] = []
    for preset, group in estimator_results.groupby("preset", sort=False):
        rows.append(
            {
                "check": f"manual_vs_one_click:{preset}",
                "replications": len(group),
                "pass_rate": float(group["passed"].mean()),
                "max_estimate_abs_difference": float(
                    group["estimate_abs_difference"].max()
                ),
                "max_std_error_abs_difference": float(
                    group["std_error_abs_difference"].max()
                ),
            }
        )
    for model, group in dirty_results.groupby("model", sort=False):
        rows.append(
            {
                "check": f"dirty_pipeline:{model}",
                "replications": len(group),
                "pass_rate": float(group["passed"].mean()),
                "max_estimate_abs_difference": float(
                    group["estimate_abs_difference"].max()
                ),
                "max_std_error_abs_difference": np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_cross_validation_figure(results: pd.DataFrame, path: str | Path) -> None:
    """Save a vector-friendly manual-versus-one-click comparison plot."""

    presets = list(results["preset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    for ax, preset in zip(axes.flat, presets):
        group = results[results["preset"] == preset]
        low = float(
            min(group["one_click_estimate"].min(), group["manual_estimate"].min())
        )
        high = float(
            max(group["one_click_estimate"].max(), group["manual_estimate"].max())
        )
        padding = max((high - low) * 0.06, 0.05)
        ax.scatter(
            group["manual_estimate"],
            group["one_click_estimate"],
            s=18,
            alpha=0.7,
            color="#16697A",
        )
        ax.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            linestyle="--",
            color="#D1495B",
            linewidth=1.2,
        )
        ax.set_title(preset)
        ax.set_xlabel("Independent NumPy estimate")
        ax.set_ylabel("One-click estimate")
        ax.grid(alpha=0.2)
    fig.suptitle("Independent regression cross-check across random datasets")
    fig.tight_layout()
    destination = Path(path)
    fig.savefig(
        destination,
        format=destination.suffix.lower().lstrip(".") or "pdf",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
