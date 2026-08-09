"""Command-line interface for reproducible and batch analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .cleaning import clean_dataframe, dataframe_to_bytes, load_dataset
from .cross_validation import (
    cross_validation_summary,
    run_dirty_pipeline_validation,
    run_estimator_cross_validation,
    save_cross_validation_figure,
)
from .mapping import suggest_roles
from .models import ModelConfig, PRESETS, fit_model
from .synthetic import generate_ddd_data, generate_did_data, make_messy_example
from .validation import run_validation, save_validation_figure, validation_summary


def _csv_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--subgroup")
    parser.add_argument("--entity")
    parser.add_argument("--time")
    parser.add_argument("--controls", help="Comma-separated numeric control columns")
    parser.add_argument("--fixed-effects", help="Comma-separated categorical fixed effects")
    parser.add_argument("--cluster", help="Column used for cluster-robust standard errors")
    parser.add_argument("--treatment-value", help="Observed value that means treated=1")
    parser.add_argument("--post-value", help="Observed value that means post=1")
    parser.add_argument("--subgroup-value", help="Observed value that means subgroup=1")


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        preset=args.preset,
        outcome=args.outcome,
        treatment=args.treatment,
        post=args.post,
        subgroup=args.subgroup,
        entity=args.entity,
        time=args.time,
        controls=_csv_list(args.controls),
        fixed_effects=_csv_list(args.fixed_effects),
        cluster=args.cluster,
        treatment_value=args.treatment_value,
        post_value=args.post_value,
        subgroup_value=args.subgroup_value,
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    raw = load_dataset(args.data)
    cleaned, report = clean_dataframe(raw)
    suggestions = suggest_roles(cleaned, require_subgroup=args.ddd)
    payload = {
        "cleaning": report.to_dict(),
        "columns": {column: str(dtype) for column, dtype in cleaned.dtypes.items()},
        "suggestions": {
            role: {
                "column": suggestion.column,
                "confidence": round(suggestion.confidence, 3),
                "reasons": suggestion.reasons,
                "alternatives": [
                    {"column": candidate.column, "score": round(candidate.score, 3)}
                    for candidate in suggestion.candidates
                ],
            }
            for role, suggestion in suggestions.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    raw = load_dataset(args.data)
    cleaned, cleaning_report = clean_dataframe(raw)
    result = fit_model(cleaned, _config_from_args(args))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cleaned_data.csv").write_bytes(dataframe_to_bytes(cleaned))
    result.analysis_data.to_csv(output / "analysis_data.csv", index=False)
    result.coefficient_table().to_csv(output / "coefficients.csv", index=False)
    (output / "statsmodels_summary.txt").write_text(result.summary_text(), encoding="utf-8")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "cleaning": cleaning_report.to_dict(),
                "headline": result.headline(),
                "diagnostics": result.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result.headline(), ensure_ascii=False, indent=2, default=str))
    print(f"Saved reproducibility outputs to {output.resolve()}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    if args.model == "did":
        data = generate_did_data(effect=args.effect, seed=args.seed)
    else:
        data = generate_ddd_data(effect=args.effect, seed=args.seed)
    if args.messy:
        data = make_messy_example(data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(data)} synthetic rows to {output.resolve()}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = run_validation(range(args.replications))
    summary = validation_summary(results)
    results.to_csv(output / "validation_results.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    save_validation_figure(results, output / "validation_results.pdf")
    save_validation_figure(results, output / "validation_results.png")
    print(summary.to_string(index=False))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the independent estimator and dirty-pipeline verification suite."""

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    estimator = run_estimator_cross_validation(range(args.replications))
    dirty = run_dirty_pipeline_validation(range(args.dirty_replications))
    summary = cross_validation_summary(estimator, dirty)
    estimator.to_csv(output / "manual_vs_one_click.csv", index=False)
    dirty.to_csv(output / "dirty_pipeline_results.csv", index=False)
    summary.to_csv(output / "cross_validation_summary.csv", index=False)
    save_cross_validation_figure(estimator, output / "manual_vs_one_click.pdf")
    save_cross_validation_figure(estimator, output / "manual_vs_one_click.png")
    print(summary.to_string(index=False))
    if not bool(estimator["passed"].all() and dirty["passed"].all()):
        print("At least one independent verification failed.", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="econdid", description="Guided DID/DDD analysis with auditable data cleaning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Clean data and suggest semantic roles")
    inspect_parser.add_argument("data")
    inspect_parser.add_argument("--ddd", action="store_true", help="Also suggest a subgroup column")
    inspect_parser.set_defaults(func=cmd_inspect)

    analyze_parser = subparsers.add_parser("analyze", help="Run a reproducible DID/DDD model")
    analyze_parser.add_argument("data")
    analyze_parser.add_argument("--output", default="analysis_output")
    _add_model_arguments(analyze_parser)
    analyze_parser.set_defaults(func=cmd_analyze)

    demo_parser = subparsers.add_parser("demo", help="Generate data with a known causal effect")
    demo_parser.add_argument("--model", choices=("did", "ddd"), default="did")
    demo_parser.add_argument("--effect", type=float, default=2.0)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--messy", action="store_true")
    demo_parser.add_argument("--output", default="demo_data.csv")
    demo_parser.set_defaults(func=cmd_demo)

    validate_parser = subparsers.add_parser("validate", help="Run Monte Carlo validation")
    validate_parser.add_argument("--replications", type=int, default=20)
    validate_parser.add_argument("--output", default="docs/validation")
    validate_parser.set_defaults(func=cmd_validate)

    verify_parser = subparsers.add_parser(
        "verify", help="Compare one-click results with independent NumPy regression code"
    )
    verify_parser.add_argument("--replications", type=int, default=50)
    verify_parser.add_argument("--dirty-replications", type=int, default=30)
    verify_parser.add_argument("--output", default="docs/cross_validation")
    verify_parser.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
