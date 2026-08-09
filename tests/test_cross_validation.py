from econdid.cross_validation import (
    cross_validation_summary,
    run_dirty_pipeline_validation,
    run_estimator_cross_validation,
)


def test_independent_manual_regression_matches_all_presets():
    results = run_estimator_cross_validation(range(1))
    assert set(results["preset"]) == {
        "did_basic",
        "did_twfe",
        "did_custom_fe",
        "ddd",
    }
    assert results["passed"].all()
    assert results["estimate_abs_difference"].max() < 1e-8
    assert results["std_error_abs_difference"].max() < 1e-8


def test_dirty_pipeline_maps_and_recovers_both_designs():
    results = run_dirty_pipeline_validation(range(1))
    assert set(results["model"]) == {"DID", "DDD"}
    assert results["passed"].all()
    assert results["mapping_passed"].all()


def test_cross_validation_summary_reports_full_pass_rate():
    estimator = run_estimator_cross_validation(range(1))
    dirty = run_dirty_pipeline_validation(range(1))
    summary = cross_validation_summary(estimator, dirty)
    assert len(summary) == 6
    assert (summary["pass_rate"] == 1.0).all()
