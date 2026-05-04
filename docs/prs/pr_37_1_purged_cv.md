# PR 37.1 — Purged K-Fold CV with Embargo

## Context

The existing `WalkForwardHarness` (Phase 12) advances by one month, fits a model,
and labels the next month. It has no embargo, so returns on the day of the
recalibration boundary can leak through autocorrelation. It also produces a
single-path point estimate of performance — no variance, no tail.

Jane Street Kaggle competition post-mortems and de Prado (AFML ch. 7) both flag
this as the primary source of optimistic backtest bias in time-series strategy
evaluation. We adopt their methodology as the default going forward.

## What this PR does

Adds `src/rde/evaluation/purged_cv.py` with:

### `purged_k_fold_splits`
Rolling walk-forward generator with configurable `train_bars`, `test_bars`,
and `embargo_bars` (default 24 — one day for hourly data). Each fold's test
window is preceded by an embargo gap that excludes the boundary bars from
training, preventing autocorrelation leakage.

### `combinatorial_purged_splits`
Implements de Prado AFML ch. 7 combinatorial purged CV. Divides the time
series into N groups and generates all C(N, K) train/test combinations. Each
combination uses K groups as test, the remaining N-K groups as training (minus
embargo zones). Produces a **distribution** of fold outcomes rather than a
single path.

### `FoldResult`
Dataclass capturing per-fold metrics: Sharpe, Calmar, max drawdown, hit rate,
annualised return, vol, trade count, BIC, log-likelihood, n_params, and
regime-conditional mean returns. After all folds: Frobenius distances of each
fold's transition matrix and emission means from the cross-fold mean
(after state alignment by mean return rank).

### `run_purged_cv` / `run_combinatorial_purged_cv`
Orchestrators that:
1. Iterate over splits.
2. Fit `train_hmm` on each training window.
3. Build a default strategy (long in above-average-return regimes per model
   emission means, flat otherwise) using **causal** `OnlineDecoder` for test
   decoding — no within-fold look-ahead from Viterbi.
4. Run `run_backtest` on the test window.
5. Compute FoldResult and collect aligned transition matrices.
6. After all folds: compute and attach cross-fold parameter dispersion.
7. Save `results/{asset}/purged_cv_{date}.parquet` with one row per fold.

## Implementation notes

- **Causal decoding in strategy simulation**: uses `OnlineDecoder.batch_filter`
  (forward-only) rather than Viterbi, which is non-causal within the test
  window. Viterbi is used only for diagnostic regime-conditional return
  attribution on training data.
- **State alignment**: states are aligned across folds by sorting on the
  mean return emission. Frobenius distances are then computed on the aligned
  parameterisation to make fold-to-fold comparisons meaningful.
- **Default strategy**: long (+1.0) in regimes where `hmm.means_[k, ret_idx] > 0`
  (above-zero scaled mean return, i.e. above training-data-average return),
  flat otherwise. This is determined entirely from training data.
- **Transaction cost default**: 1bp (0.0001) per side. Pass `transaction_cost`
  to override.
- **ann_factor default**: 8760 for hourly crypto; pass 1638 (≈252×6.5) for
  hourly equities.

## Definition of done

- `run_purged_cv` and `run_combinatorial_purged_cv` exist and pass tests.
- `results/{asset}/purged_cv_{date}.parquet` written with correct schema.
- `PurgedCVResult` (summary) and `FoldResult` exported from
  `rde.evaluation`.
- Fast unit tests pass; slow integration tests run under `@pytest.mark.slow`.
