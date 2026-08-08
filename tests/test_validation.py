from econdid.validation import run_validation, validation_summary


def test_small_validation_run_has_bounded_bias():
    results = run_validation(range(2))
    assert set(results["model"]) == {"DID", "DDD"}
    summary = validation_summary(results)
    assert len(summary) == 2
    assert (summary["bias"].abs() < 0.5).all()

