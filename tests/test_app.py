from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_demo_renders_without_exception():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "EconDID Studio"
    assert any(button.label == "Run model" for button in app.button)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any(metric.label == "Estimated effect" for metric in app.metric)


def test_ddd_demo_runs_end_to_end():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()
    app.radio[1].set_value("DDD").run()
    assert not app.exception
    assert app.selectbox[0].value == "ddd"
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any(metric.label == "Estimated effect" for metric in app.metric)


def test_visualizations_can_be_disabled_without_disabling_estimation():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()
    app.checkbox[0].set_value(False).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any(metric.label == "Estimated effect" for metric in app.metric)
    labels = {button.label for button in app.button}
    assert "Download trends (PNG)" not in labels
    assert "Download effect plot (vector PDF)" not in labels
