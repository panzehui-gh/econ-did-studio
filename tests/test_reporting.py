from io import BytesIO
from zipfile import ZipFile

from econdid.models import ModelConfig, fit_model
from econdid.reporting import (
    effect_figure,
    export_bundle,
    plain_language_interpretation,
    trend_figure,
)
from econdid.synthetic import generate_ddd_data, generate_did_data


def _result():
    data = generate_did_data(n_entities=30, seed=5)
    return fit_model(
        data,
        ModelConfig(
            preset="did_twfe",
            outcome="outcome",
            treatment="treated",
            post="post",
            entity="unit_id",
            time="year",
        ),
    )


def test_export_bundle_contains_reproducibility_files():
    content = _result()
    with ZipFile(BytesIO(export_bundle(content))) as archive:
        names = set(archive.namelist())
    assert {
        "analysis_data.csv",
        "coefficients.csv",
        "model_metadata.json",
        "statsmodels_summary.txt",
        "plain_language_interpretation.txt",
        "observed_trends.png",
        "observed_trends.pdf",
        "treatment_effect.png",
        "treatment_effect.pdf",
    } <= names


def test_trend_figure_has_two_lines():
    figure = trend_figure(_result())
    assert len(figure.axes[0].lines) == 2


def test_effect_figure_and_interpretation_include_target_result():
    result = _result()
    figure = effect_figure(result)
    assert len(figure.axes[0].lines) >= 2
    text = plain_language_interpretation(result)
    assert "95% CI" in text
    assert result.config.outcome in text


def test_ddd_trends_use_separate_subgroup_panels():
    data = generate_ddd_data(n_entities=40, seed=8)
    result = fit_model(
        data,
        ModelConfig(
            preset="ddd",
            outcome="outcome",
            treatment="treated",
            post="post",
            subgroup="eligible",
            fixed_effects=("unit_id", "year"),
            cluster="unit_id",
            time="year",
        ),
    )
    figure = trend_figure(result)
    assert len(figure.axes) == 2
    assert all(len(axis.lines) == 2 for axis in figure.axes)
