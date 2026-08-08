import json

from econdid.cli import main
from econdid.synthetic import generate_did_data


def test_cli_demo_and_inspect(tmp_path, capsys):
    demo = tmp_path / "messy.csv"
    assert main(["demo", "--messy", "--output", str(demo)]) == 0
    assert demo.exists()
    capsys.readouterr()
    assert main(["inspect", str(demo)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suggestions"]["outcome"]["column"] == "Outcome_value"
    assert payload["suggestions"]["treatment"]["column"] == "Treatment_group"


def test_cli_analyze_writes_reproducibility_outputs(tmp_path, capsys):
    source = tmp_path / "panel.csv"
    output = tmp_path / "result"
    generate_did_data(n_entities=40, seed=9).to_csv(source, index=False)
    exit_code = main(
        [
            "analyze",
            str(source),
            "--preset",
            "did_twfe",
            "--outcome",
            "outcome",
            "--treatment",
            "treated",
            "--post",
            "post",
            "--entity",
            "unit_id",
            "--time",
            "year",
            "--cluster",
            "unit_id",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert {
        "cleaned_data.csv",
        "analysis_data.csv",
        "coefficients.csv",
        "statsmodels_summary.txt",
        "metadata.json",
    } <= {path.name for path in output.iterdir()}
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["headline"]["target_term"] == "_did"
    assert metadata["headline"]["n_observations"] == 320
    assert "Saved reproducibility outputs" in capsys.readouterr().out

