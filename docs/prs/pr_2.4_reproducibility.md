# PR 2.4 — Reproducibility Regression Harness

## Context

Any refactor that changes HMM training, feature computation, or inference would
silently shift regime assignments. There is currently no tripwire for this.

## What this PR does

Adds `tests/regression/test_reproducibility.py`:
- Runs the full pipeline (features → train_hmm → viterbi_decode) on synthetic
  data with a fixed seed.
- On first run (no fixtures): generates `tests/regression/fixtures/`
  with one `.parquet` per asset-equivalent test case and exits with a PASS.
- On subsequent runs: loads fixtures, runs the same pipeline, asserts that the
  Viterbi path is bit-for-bit identical to the stored fixture.
- If a refactor changes outputs: test fails with a diff report.
  Re-generate fixtures with `uv run pytest --update-fixtures tests/regression/`.

## Implementation notes

- Uses **synthetic** data (not real yfinance data) so the test is offline and
  deterministic regardless of market data availability.
- Fixtures are stored as `.parquet` (not pickle) — schema-forward, human-readable
  with pandas, no deserialisation risk.
- Three test cases: 3-state Gaussian (small), 4-state Gaussian (medium), 5-state
  Gaussian (large) — covers model size variation.
- The `--update-fixtures` flag is implemented as a custom `pytest` option
  registered in `conftest.py`.

## Definition of done

- Fixtures generated and committed in `tests/regression/fixtures/`.
- Default `uv run pytest --tb=no -q` passes (regression tests compare to fixtures).
- Running with `--update-fixtures` regenerates fixtures and passes.
