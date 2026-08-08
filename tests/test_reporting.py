from io import BytesIO
from zipfile import ZipFile

from econdid.models import ModelConfig, fit_model
from econdid.reporting import export_bundle, trend_figure
from econdid.synthetic import generate_did_data


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
        "observed_trends.png",
    } <= names


def test_trend_figure_has_two_lines():
    figure = trend_figure(_result())
    assert len(figure.axes[0].lines) == 2

