from io import BytesIO

import numpy as np
import pandas as pd

from econdid.cleaning import clean_dataframe, load_dataset


def test_clean_dataframe_is_non_destructive_and_audited():
    raw = pd.DataFrame(
        {
            " Outcome value ": ["1,200", " 2,500 ", "N/A", "1,200"],
            "Treatment group": ["Treated", "Control", "Control", "Treated"],
            "empty": ["", "", "", ""],
        }
    )
    original = raw.copy(deep=True)
    cleaned, report = clean_dataframe(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert list(cleaned.columns) == ["Outcome_value", "Treatment_group"]
    assert cleaned["Outcome_value"].iloc[0] == 1200
    assert cleaned["Outcome_value"].isna().sum() == 1
    assert report.dropped_empty_columns == ["empty"]
    assert report.dropped_duplicate_rows == 1
    assert "Outcome_value" in report.numeric_conversions


def test_load_dataset_infers_csv_separator():
    source = BytesIO("unit;outcome\nA;1\nB;2\n".encode("utf-8"))
    result = load_dataset(source, filename="data.csv")
    assert result.to_dict("list") == {"unit": ["A", "B"], "outcome": [1, 2]}


def test_percent_conversion():
    raw = pd.DataFrame({"rate": ["10%", "25%", np.nan]})
    cleaned, _ = clean_dataframe(raw)
    assert np.allclose(cleaned["rate"].dropna(), [0.10, 0.25])

