import pandas as pd

from econdid.mapping import suggest_roles


def test_suggest_roles_recognises_common_messy_names():
    df = pd.DataFrame(
        {
            "Outcome_value": [1.0, 2.0, 3.0, 4.0],
            "Treatment_group": [0, 0, 1, 1],
            "After_policy": [0, 1, 0, 1],
            "Unit_ID": ["A", "A", "B", "B"],
            "Year": [2020, 2021, 2020, 2021],
        }
    )
    suggestions = suggest_roles(df)
    assert suggestions["outcome"].column == "Outcome_value"
    assert suggestions["treatment"].column == "Treatment_group"
    assert suggestions["post"].column == "After_policy"
    assert suggestions["entity"].column == "Unit_ID"
    assert suggestions["time"].column == "Year"
    assert suggestions["outcome"].candidates


def test_suggest_roles_supports_japanese_names():
    df = pd.DataFrame(
        {
            "成果": [10.0, 11.0, 15.0, 16.0],
            "処理群": [0, 0, 1, 1],
            "政策後": [0, 1, 0, 1],
            "企業id": ["A", "A", "B", "B"],
            "年度": [2020, 2021, 2020, 2021],
        }
    )
    suggestions = suggest_roles(df)
    assert suggestions["outcome"].column == "成果"
    assert suggestions["treatment"].column == "処理群"
    assert suggestions["post"].column == "政策後"

