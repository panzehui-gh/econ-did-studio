import numpy as np
import pandas as pd
import pytest

from econdid.models import ModelConfig, fit_model
from econdid.synthetic import generate_ddd_data, generate_did_data


def test_twfe_did_recovers_known_effect():
    data = generate_did_data(n_entities=260, effect=2.0, seed=7)
    result = fit_model(
        data,
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
    assert result.estimate == pytest.approx(2.0, abs=0.20)
    assert result.standard_error > 0
    assert result.p_value < 0.01
    assert int(result.model.nobs) == len(data)


def test_ddd_recovers_known_effect():
    data = generate_ddd_data(n_entities=360, effect=1.5, seed=11)
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
        ),
    )
    assert result.estimate == pytest.approx(1.5, abs=0.30)
    assert result.target_term == "_treated:_post:_subgroup"
    assert result.p_value < 0.01


def test_nonstandard_indicator_requires_direction_confirmation():
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 5.0, 6.0],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "period": ["before", "before", "after", "after"] * 2,
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        fit_model(
            data,
            ModelConfig(
                preset="did_basic", outcome="y", treatment="group", post="period"
            ),
        )
    result = fit_model(
        data,
        ModelConfig(
            preset="did_basic",
            outcome="y",
            treatment="group",
            post="period",
            treatment_value="B",
            post_value="after",
        ),
    )
    assert result.estimate == pytest.approx(2.0)


def test_missing_did_cell_is_rejected():
    data = pd.DataFrame({"y": [1, 2, 3], "t": [0, 0, 1], "post": [0, 1, 0]})
    with pytest.raises(ValueError, match="four"):
        fit_model(
            data,
            ModelConfig(preset="did_basic", outcome="y", treatment="t", post="post"),
        )


def test_analysis_does_not_mutate_input():
    data = generate_did_data(n_entities=20, seed=2)
    original = data.copy(deep=True)
    fit_model(
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
    pd.testing.assert_frame_equal(data, original)
