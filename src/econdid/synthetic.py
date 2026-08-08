"""Synthetic data with known causal effects for examples and regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_did_data(
    n_entities: int = 240,
    n_periods: int = 8,
    effect: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced panel satisfying the parallel-trends construction."""

    if n_entities < 4 or n_periods < 4:
        raise ValueError("Use at least 4 entities and 4 periods.")
    rng = np.random.default_rng(seed)
    entity_fe = rng.normal(0, 1.5, n_entities)
    treated_by_entity = rng.binomial(1, 0.5, n_entities)
    if treated_by_entity.sum() in {0, n_entities}:
        treated_by_entity[: n_entities // 2] = 1
        treated_by_entity[n_entities // 2 :] = 0
    rows: list[dict[str, float | int | str]] = []
    policy_period = n_periods // 2
    for entity in range(n_entities):
        for period in range(n_periods):
            treated = int(treated_by_entity[entity])
            post = int(period >= policy_period)
            covariate = rng.normal()
            outcome = (
                5.0
                + entity_fe[entity]
                + 0.22 * period
                + 0.35 * covariate
                + effect * treated * post
                + rng.normal(0, 1.0)
            )
            rows.append(
                {
                    "unit_id": f"U{entity:04d}",
                    "year": 2015 + period,
                    "treated": treated,
                    "post": post,
                    "outcome": outcome,
                    "covariate": covariate,
                }
            )
    return pd.DataFrame(rows)


def generate_ddd_data(
    n_entities: int = 320,
    n_periods: int = 8,
    effect: float = 1.5,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate a balanced panel with a known triple-difference effect."""

    if n_entities < 8 or n_periods < 4:
        raise ValueError("Use at least 8 entities and 4 periods.")
    rng = np.random.default_rng(seed)
    entity_fe = rng.normal(0, 1.2, n_entities)
    pairs = np.tile(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        (int(np.ceil(n_entities / 4)), 1),
    )[:n_entities]
    pairs = pairs[rng.permutation(n_entities)]
    treated_by_entity = pairs[:, 0]
    subgroup_by_entity = pairs[:, 1]
    rows: list[dict[str, float | int | str]] = []
    policy_period = n_periods // 2
    for entity in range(n_entities):
        for period in range(n_periods):
            treated = int(treated_by_entity[entity])
            subgroup = int(subgroup_by_entity[entity])
            post = int(period >= policy_period)
            outcome = (
                4.0
                + entity_fe[entity]
                + 0.18 * period
                + 0.25 * treated * post
                - 0.2 * post * subgroup
                + 0.15 * treated * subgroup
                + effect * treated * post * subgroup
                + rng.normal(0, 1.0)
            )
            rows.append(
                {
                    "unit_id": f"D{entity:04d}",
                    "year": 2015 + period,
                    "treated": treated,
                    "post": post,
                    "eligible": subgroup,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def make_messy_example(df: pd.DataFrame) -> pd.DataFrame:
    """Create a realistically messy copy used to demonstrate cleaning/mapping."""

    result = df.copy()
    result = result.rename(
        columns={
            "unit_id": " Unit ID ",
            "year": "Year ",
            "treated": "Treatment group",
            "post": "After policy",
            "outcome": "Outcome value",
        }
    )
    if "Treatment group" in result:
        result["Treatment group"] = result["Treatment group"].map({1: "Treated", 0: "Control"})
    if "After policy" in result:
        result["After policy"] = result["After policy"].map({1: "Post", 0: "Pre"})
    result["Outcome value"] = result["Outcome value"].map(lambda value: f"{value:,.4f}")
    result["empty notes"] = ""
    return pd.concat([result, result.iloc[[0]]], ignore_index=True)
