"""Monte Carlo validation against synthetic datasets with known effects."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import ModelConfig, fit_model
from .synthetic import generate_ddd_data, generate_did_data


def run_validation(seeds: Iterable[int] = range(20)) -> pd.DataFrame:
    """Estimate DID and DDD across seeds and return bias/coverage inputs."""

    records: list[dict[str, float | int | str | bool]] = []
    for seed in seeds:
        did_true = 2.0
        did = generate_did_data(n_entities=160, effect=did_true, seed=seed)
        did_result = fit_model(
            did,
            ModelConfig(
                preset="did_twfe",
                outcome="outcome",
                treatment="treated",
                post="post",
                entity="unit_id",
                time="year",
                controls=("covariate",),
                cluster="unit_id",
            ),
        )
        lower, upper = did_result.confidence_interval
        records.append(
            {
                "model": "DID",
                "seed": seed,
                "true_effect": did_true,
                "estimate": did_result.estimate,
                "std_error": did_result.standard_error,
                "ci_lower": lower,
                "ci_upper": upper,
                "covered": lower <= did_true <= upper,
            }
        )

        ddd_true = 1.5
        ddd = generate_ddd_data(n_entities=200, effect=ddd_true, seed=10_000 + seed)
        ddd_result = fit_model(
            ddd,
            ModelConfig(
                preset="ddd",
                outcome="outcome",
                treatment="treated",
                post="post",
                subgroup="eligible",
                fixed_effects=("unit_id", "year"),
                cluster="unit_id",
            ),
        )
        lower, upper = ddd_result.confidence_interval
        records.append(
            {
                "model": "DDD",
                "seed": seed,
                "true_effect": ddd_true,
                "estimate": ddd_result.estimate,
                "std_error": ddd_result.standard_error,
                "ci_lower": lower,
                "ci_upper": upper,
                "covered": lower <= ddd_true <= upper,
            }
        )
    return pd.DataFrame(records)


def validation_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model, group in results.groupby("model"):
        error = group["estimate"] - group["true_effect"]
        rows.append(
            {
                "model": str(model),
                "replications": len(group),
                "true_effect": float(group["true_effect"].iloc[0]),
                "mean_estimate": float(group["estimate"].mean()),
                "bias": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "coverage_95": float(group["covered"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_validation_figure(results: pd.DataFrame, path: str | Path) -> None:
    """Save estimates and 95% intervals as PDF, SVG, or PNG."""

    models = list(results["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(8.0, 3.5), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        group = results[results["model"] == model].reset_index(drop=True)
        x = np.arange(len(group))
        errors = np.vstack(
            [group["estimate"] - group["ci_lower"], group["ci_upper"] - group["estimate"]]
        )
        ax.errorbar(x, group["estimate"], yerr=errors, fmt="o", ms=3, alpha=0.75)
        ax.axhline(group["true_effect"].iloc[0], color="crimson", linestyle="--", label="True effect")
        ax.set_title(model)
        ax.set_xlabel("Simulation seed")
        ax.set_ylabel("Estimated effect")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Synthetic-data validation: estimates and 95% confidence intervals")
    fig.tight_layout()
    destination = Path(path)
    output_format = destination.suffix.lower().lstrip(".") or "pdf"
    fig.savefig(destination, format=output_format, dpi=180, bbox_inches="tight")
    plt.close(fig)
