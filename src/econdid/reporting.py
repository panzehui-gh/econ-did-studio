"""Reusable plots and export helpers."""

from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .cleaning import dataframe_to_bytes
from .models import AnalysisResult


def trend_figure(result: AnalysisResult):
    """Plot observed mean outcomes by treatment group and time/post period."""

    config = result.config
    data = result.analysis_data
    time_column = config.time if config.time and config.time in data else "_post"
    means = (
        data.groupby([time_column, "_treated"], observed=True)[config.outcome]
        .mean()
        .unstack("_treated")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = {0.0: "Control", 1.0: "Treated"}
    for value in means.columns:
        ax.plot(
            means.index,
            means[value],
            marker="o",
            linewidth=2,
            label=labels.get(float(value), str(value)),
        )
    ax.set_title("Observed outcome trends (descriptive)")
    ax.set_xlabel(str(time_column))
    ax.set_ylabel(config.outcome)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def export_bundle(result: AnalysisResult) -> bytes:
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
        figure = trend_figure(result)
        image = BytesIO()
        figure.savefig(image, format="png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        archive.writestr("observed_trends.png", image.getvalue())
    return buffer.getvalue()
