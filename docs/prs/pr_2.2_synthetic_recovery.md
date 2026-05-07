# PR 2.2 — Synthetic-Data Falsification Suite

## Context

The engine has 1,297 tests but none verify that the HMM inference code recovers
known regimes from synthetic ground-truth data. If the Viterbi or Baum-Welch
implementations have a systematic bias, existing unit tests would not catch it
because they test API contracts, not recovery accuracy.

## What this PR does

Adds `tests/integration/test_synthetic_recovery.py` with three test classes,
all marked `@pytest.mark.slow` (excluded from default `pytest` runs):

1. **GaussianHMM recovery** — 3-regime sequence from a known transition matrix
   and Gaussian emissions (seed-controlled). Asserts:
   - ARI between Viterbi path and true labels ≥ 0.85
   - Recovered transition matrix within Frobenius distance 0.05 of true
   - Recovered means within 1 standard error of true

2. **StudentT-HMM recovery** — heavy-tailed emissions using `StudentTHMM`.
   Asserts ARI ≥ 0.80 (lower threshold: student-t is harder to fit than Gaussian).

3. **HSMM dwell recovery** — negative-binomial dwell injection via
   `apply_hsmm_correction`. Asserts that the corrected path's per-state empirical
   median dwell is within ±50% of the injected distribution's theoretical mean.

## Tradeoffs

- Tests are `@pytest.mark.slow` and excluded from default runs via a
  `pyproject.toml` marker filter. Run with `uv run pytest -m slow`.
- Synthetic sequences are length T=3,000 — long enough for reliable parameter
  recovery, short enough to run in < 30s per test.
- ARI threshold of 0.85 for Gaussian (0.80 for Student-t) is standard in the
  HMM recovery literature. Well-separated emission means give higher ARI; the
  test uses a 3-regime setup with clear separation.
- If any test fails, it is treated as a P0 bug per the brief's instructions.

## Definition of done

- `tests/integration/test_synthetic_recovery.py` exists and all three tests
  pass when run with `uv run pytest -m slow tests/integration/`.
- `uv run pytest --tb=no -q` (default run without slow) still shows 1,297 passed.
