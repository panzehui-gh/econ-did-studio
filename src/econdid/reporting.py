"""Reusable plots and export helpers."""

from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import dataframe_to_bytes
from .models import AnalysisResult


def _plot_group_trends(ax, data: pd.DataFrame, outcome: str, time_column: str) -> None:
    means = (
        data.groupby([time_column, "_treated"], observed=True)[outcome]
        .mean()
        .unstack("_treated")
        .sort_index()
    )
    labels = {0.0: "Control", 1.0: "Treated"}
    for value in means.columns:
        ax.plot(
            means.index,
            means[value],
            marker="o",
            linewidth=2,
            label=labels.get(float(value), str(value)),
        )
    ax.set_xlabel(str(time_column))
    ax.set_ylabel(outcome)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)


def trend_figure(result: AnalysisResult):
    """Plot descriptive DID trends, with separate subgroup panels for DDD."""

    config = result.config
    data = result.analysis_data
    time_column = config.time if config.time and config.time in data else "_post"
    if config.preset == "ddd" and "_subgroup" in data:
        subgroup_values = sorted(pd.unique(data["_subgroup"]))
        fig, axes = plt.subplots(
            1, len(subgroup_values), figsize=(7.8, 3.8), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, subgroup in zip(axes, subgroup_values):
            subset = data[data["_subgroup"] == subgroup]
            _plot_group_trends(ax, subset, config.outcome, time_column)
            label = "Target subgroup" if float(subgroup) == 1.0 else "Other subgroup"
            ax.set_title(label)
        fig.suptitle("Observed outcome trends by DDD subgroup (descriptive)")
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _plot_group_trends(ax, data, config.outcome, time_column)
        ax.set_title("Observed outcome trends (descriptive)")
    fig.tight_layout()
    return fig


def effect_figure(result: AnalysisResult):
    """Plot the target estimate and confidence interval against a zero line."""

    lower, upper = result.confidence_interval
    estimate = result.estimate
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.errorbar(
        estimate,
        0,
        xerr=np.array([[estimate - lower], [upper - estimate]]),
        fmt="o",
        markersize=7,
        capsize=5,
        color="#16697A",
        ecolor="#16697A",
        linewidth=2,
    )
    ax.axvline(0, color="#D1495B", linestyle="--", linewidth=1.3)
    ax.set_yticks([0], ["DID effect" if result.config.preset != "ddd" else "DDD effect"])
    ax.set_xlabel(f"Change in {result.config.outcome} (95% confidence interval)")
    ax.set_title("Estimated policy effect")
    ax.grid(axis="x", alpha=0.25)
    ax.annotate(
        f"{estimate:.3g}  [{lower:.3g}, {upper:.3g}]",
        xy=(estimate, 0),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
    )
    fig.tight_layout()
    return fig


def plain_language_interpretation(result: AnalysisResult) -> str:
    """Return a cautious, presentation-friendly description of the target effect."""

    lower, upper = result.confidence_interval
    direction = "increase" if result.estimate >= 0 else "decrease"
    design = "triple-difference" if result.config.preset == "ddd" else "difference-in-differences"
    significance = (
        "The 95% interval does not cross zero."
        if lower > 0 or upper < 0
        else "The 95% interval crosses zero, so the data are also compatible with no effect."
    )
    return (
        f"In this {design} specification, the estimated relative {direction} in "
        f"{result.config.outcome} is {abs(result.estimate):.4g} units "
        f"(95% CI {lower:.4g} to {upper:.4g}). {significance} "
        "This becomes a causal statement only if the research-design assumptions are credible."
    )


def figure_to_bytes(figure, output_format: str = "png") -> bytes:
    """Serialize a Matplotlib figure for app downloads and ZIP exports."""

    image = BytesIO()
    kwargs = {"format": output_format, "bbox_inches": "tight"}
    if output_format.lower() == "png":
        kwargs["dpi"] = 180
    figure.savefig(image, **kwargs)
    return image.getvalue()


def export_bundle(result: AnalysisResult, include_figures: bool = True) -> bytes:
    """Create a reproducibility ZIP with cleaned data, coefficients, and metadata."""

    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("analysis_data.csv", dataframe_to_bytes(result.analysis_data))
        archive.writestr(
            "coefficients.csv", result.coefficient_table().to_csv(index=False).encode("utf-8")
        )
        archive.writestr(
            "model_metadata.json",
            json.dumps(
                {"headline": result.headline(), "diagnostics": result.diagnostics},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
        archive.writestr("statsmodels_summary.txt", result.summary_text())
        archive.writestr("plain_language_interpretation.txt", plain_language_interpretation(result))
        if include_figures:
            figures = {
                "observed_trends": trend_figure(result),
                "treatment_effect": effect_figure(result),
            }
            for name, figure in figures.items():
                archive.writestr(f"{name}.png", figure_to_bytes(figure, "png"))
                archive.writestr(f"{name}.pdf", figure_to_bytes(figure, "pdf"))
                plt.close(figure)
    return buffer.getvalue()
