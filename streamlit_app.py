"""No-code Streamlit interface for EconDID Studio."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from econdid.cleaning import clean_dataframe, dataframe_to_bytes, load_dataset
from econdid.mapping import suggest_roles
from econdid.models import ModelConfig, PRESETS, fit_model
from econdid.reporting import export_bundle, trend_figure
from econdid.synthetic import generate_did_data, make_messy_example


st.set_page_config(page_title="EconDID Studio", page_icon="📊", layout="wide")

st.title("EconDID Studio")
st.caption("Guided difference-in-differences and triple-differences analysis — no coding required")

with st.sidebar:
    st.header("1 · Data")
    source_mode = st.radio("Data source", ["Try the synthetic demo", "Upload my file"])
    demo_model = "DID"
    if source_mode == "Try the synthetic demo":
        demo_model = st.radio("Synthetic design", ["DID", "DDD"], horizontal=True)
    uploaded = None
    if source_mode == "Upload my file":
        uploaded = st.file_uploader("CSV / TSV / Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])
    st.info(
        "Your source file is never overwritten. Every automatic change is listed in the cleaning audit."
    )

if source_mode == "Upload my file" and uploaded is None:
    st.subheader("Start with any tabular research dataset")
    st.write(
        "Upload a CSV, TSV, or Excel file. Column names and ordering may be untidy; "
        "you will confirm the inferred model roles before estimation."
    )
    st.stop()

try:
    if uploaded is None:
        if demo_model == "DDD":
            from econdid.synthetic import generate_ddd_data

            raw = make_messy_example(generate_ddd_data(n_entities=120, seed=43))
            source_label = "synthetic messy DDD demo"
        else:
            raw = make_messy_example(generate_did_data(n_entities=100, seed=42))
            source_label = "synthetic messy DID demo"
    else:
        raw = load_dataset(uploaded, filename=uploaded.name)
        source_label = uploaded.name
    cleaned, cleaning_report = clean_dataframe(raw)
except Exception as exc:
    st.error(f"Could not read or clean the data: {exc}")
    st.stop()

st.subheader("2 · Cleaning audit")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{cleaning_report.original_shape[0]} → {cleaning_report.cleaned_shape[0]}")
c2.metric("Columns", f"{cleaning_report.original_shape[1]} → {cleaning_report.cleaned_shape[1]}")
c3.metric("Duplicate rows removed", cleaning_report.dropped_duplicate_rows)
c4.metric("Missing markers standardized", cleaning_report.missing_values_standardized)
with st.expander("View all cleaning decisions and cleaned preview"):
    st.json(cleaning_report.to_dict())
    st.dataframe(cleaned.head(50), width="stretch")
    st.download_button(
        "Download cleaned data",
        dataframe_to_bytes(cleaned),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

st.subheader("3 · Choose a model and confirm what each column means")
preset = st.selectbox(
    "Model preset",
    list(PRESETS),
    format_func=lambda key: f"{PRESETS[key]['label']} — {PRESETS[key]['description']}",
    index=list(PRESETS).index("ddd") if demo_model == "DDD" else 1,
    key=f"preset_{demo_model}",
)
is_ddd = preset == "ddd"

with st.expander("Optional: describe your variables in plain language"):
    st.caption("These descriptions help ranking, but you always make the final column selection.")
    roles_for_description = ["outcome", "treatment", "post", "entity", "time"]
    if is_ddd:
        roles_for_description.append("subgroup")
    descriptions = {
        role: st.text_input(f"Meaning of {role}", key=f"description_{role}")
        for role in roles_for_description
    }

suggestions = suggest_roles(cleaned, descriptions=descriptions, require_subgroup=is_ddd)
columns = list(map(str, cleaned.columns))


def role_select(role: str, required: bool = True) -> str | None:
    suggestion = suggestions.get(role)
    default = suggestion.column if suggestion else None
    options = columns if required else ["— none —", *columns]
    if default in options:
        index = options.index(default)
    else:
        index = 0
    selected = st.selectbox(role.replace("_", " ").title(), options, index=index, key=f"role_{role}")
    if suggestion and suggestion.column == selected:
        confidence = int(round(100 * suggestion.confidence))
        st.caption(f"Suggested confidence: {confidence}% · " + "; ".join(suggestion.reasons))
    return None if selected == "— none —" else selected


left, middle, right = st.columns(3)
with left:
    outcome = role_select("outcome")
    treatment = role_select("treatment")
with middle:
    post = role_select("post")
    subgroup = role_select("subgroup") if is_ddd else None
with right:
    needs_panel = preset == "did_twfe"
    entity = role_select("entity", required=needs_panel)
    time = role_select("time", required=needs_panel)

selected_roles = {value for value in [outcome, treatment, post, subgroup, entity, time] if value}
remaining = [column for column in columns if column not in selected_roles]
controls = tuple(st.multiselect("Numeric control variables (optional)", remaining))
fixed_excluded = {value for value in (outcome, treatment, post, subgroup) if value}
fixed_excluded.update(controls)
if preset == "did_twfe":
    fixed_excluded.update(value for value in (entity, time) if value)
fixed_options = [column for column in columns if column not in fixed_excluded]
default_fe = tuple(value for value in (entity, time) if value and preset == "ddd")
fixed_effects = tuple(
    st.multiselect("Additional categorical fixed effects", fixed_options, default=list(default_fe))
)
cluster_options = ["— HC1 robust SE —", *columns]
cluster_default = cluster_options.index(entity) if entity in cluster_options else 0
cluster_choice = st.selectbox("Standard errors", cluster_options, index=cluster_default)
cluster = None if cluster_choice.startswith("—") else cluster_choice


def positive_selector(column: str | None, role: str):
    if not column:
        return None
    values = list(pd.unique(cleaned[column].dropna()))
    if len(values) != 2:
        st.warning(f"{role} currently has {len(values)} values; a binary column is required.")
        return None
    preferred = 0
    for index, value in enumerate(values):
        if value is True or str(value).strip().lower() in {
            "1", "treated", "post", "after", "yes", "true", "eligible", "対象", "政策後"
        }:
            preferred = index
            break
    return st.selectbox(
        f"Which {role} value means 1?",
        values,
        index=preferred,
        key=f"positive_{role}_{column}",
    )


st.markdown("**Confirm indicator direction**")
i1, i2, i3 = st.columns(3)
with i1:
    treatment_value = positive_selector(treatment, "treatment")
with i2:
    post_value = positive_selector(post, "post")
with i3:
    subgroup_value = positive_selector(subgroup, "subgroup") if is_ddd else None

with st.expander("See ranked alternatives for every role"):
    rows = []
    for role, suggestion in suggestions.items():
        for rank, candidate in enumerate(suggestion.candidates, start=1):
            rows.append(
                {
                    "role": role,
                    "rank": rank,
                    "column": candidate.column,
                    "score": round(candidate.score, 3),
                    "why": "; ".join(candidate.reasons),
                }
            )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.subheader("4 · Estimate and export")
st.warning(
    "DID/DDD estimates are causal only under research-design assumptions such as parallel trends. "
    "The software checks data structure, but it cannot prove those assumptions from column names."
)

if st.button("Run model", type="primary", width="stretch"):
    if not all([outcome, treatment, post]):
        st.error("Outcome, treatment, and post columns are required.")
        st.stop()
    try:
        config = ModelConfig(
            preset=preset,
            outcome=outcome,
            treatment=treatment,
            post=post,
            subgroup=subgroup,
            entity=entity,
            time=time,
            controls=controls,
            fixed_effects=fixed_effects,
            cluster=cluster,
            treatment_value=treatment_value,
            post_value=post_value,
            subgroup_value=subgroup_value,
        )
        result = fit_model(cleaned, config)
    except Exception as exc:
        st.error(f"The model was not estimated: {exc}")
        st.stop()

    headline = result.headline()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Estimated effect", f"{headline['estimate']:.4g}")
    r2.metric("Robust SE", f"{headline['standard_error']:.4g}")
    r3.metric("p-value", f"{headline['p_value']:.4g}")
    r4.metric("Observations", headline["n_observations"])
    st.code(headline["formula"], language="text")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    st.dataframe(result.coefficient_table(), width="stretch", hide_index=True)

    fig = trend_figure(result)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Diagnostics and full statsmodels output"):
        st.json(result.diagnostics)
        st.text(result.summary_text())

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download complete result bundle",
            export_bundle(result),
            file_name="econdid_results.zip",
            mime="application/zip",
        )
    with d2:
        st.download_button(
            "Download coefficient table",
            result.coefficient_table().to_csv(index=False),
            file_name="coefficients.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Download model metadata",
            json.dumps(
                {"source": source_label, "headline": headline, "diagnostics": result.diagnostics},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            file_name="model_metadata.json",
            mime="application/json",
        )
