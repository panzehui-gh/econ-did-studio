# Independent cross-validation evidence

This directory records two reproducible checks invoked with:

```bash
econdid verify --replications 50 --dirty-replications 30 --output docs/cross_validation
```

## Estimator cross-check

`manual_vs_one_click.csv` contains 200 randomized comparisons: 50 for each of
the four model presets. The comparator builds its own design matrix, solves OLS
with `numpy.linalg.pinv`, and implements HC1 or one-way cluster sandwich
covariance directly. It does not call statsmodels and does not reuse the fitted
model's design matrix.

All 200 comparisons passed at tolerance `1e-8`. The maximum absolute
coefficient difference was `1.321165e-14`; the maximum standard-error
difference was `5.384582e-15`.

## Dirty-data pipeline check

`dirty_pipeline_results.csv` contains 60 end-to-end runs (30 DID and 30 DDD).
For each run the input columns are reordered and transformed into the app's
messy demonstration format, including text indicators, formatted numeric
outcomes, an empty column, and a duplicate row. Cleaning, role suggestions,
duplicate removal, and estimation are then run. All 60 cases passed. The small
maximum estimate difference (`3.088247e-05`) is expected because the messy
generator deliberately rounds outcome strings to four decimal places.

`cross_validation_summary.csv` is the compact machine-readable summary.
`manual_vs_one_click.pdf` is the vector version of the comparison figure for
academic reports; the PNG is included for README display.
