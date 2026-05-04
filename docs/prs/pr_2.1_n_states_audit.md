# PR 2.1 — Resolve the n_states=6 BIC Ceiling

## Context

Run 001 (2026-04-27) selected n=6 for BTC, which was the upper bound of `candidate_states: [2..6]`. This raised the question: is n=6 a genuine BIC minimum, or is the model constrained by the search range ceiling?

The status report (§6 Q1) found:
- `btc.yaml`, `eth.yaml`, `spy.yaml` already extended to `[2..8]`.
- BTC diagnostics: **n=6 selected from [2..8]** → confirmed minimum at 6 (BIC did not improve at 7 or 8).
- ETH diagnostics: **n=8 selected from [2..8]** → at ceiling; BIC may still decline.
- SPY diagnostics: **n=8 selected from [2..8]** → at ceiling; BIC may still decline.

## What this PR does

1. Extends `candidate_states` in all three configs to `[2,3,4,5,6,7,8,9,10]`.
2. Adds `scripts/run_bic_audit.py` — a one-shot script that:
   - Loads each asset from the parquet cache (no yfinance calls).
   - Runs `train_hmm` for each n_states ∈ [2..10] with n_restarts=5, n_iter=200.
   - Saves `results/{asset}/selection_table.csv` with columns `n_states, log_likelihood, aic, bic, n_restarts, best_seed`.
   - Saves `results/{asset}/bic_curve.png` — BIC vs n_states with minimum annotated.
3. Runs the script and saves all outputs to `results/`.
4. Writes `results/n_states_audit.md` interpreting findings:
   - Whether BIC has a clear minimum or is still declining at n=10.
   - Per-state dwell time comparison at n=6 vs n=8 for each asset (dwell < 2 bars is an overfitting signal).
   - Recommendation on final candidate_states setting for production configs.

## Tradeoffs

- Using n_restarts=5 (instead of the production default of 10) for speed. Five restarts is sufficient to identify BIC trend direction; it may not match the exact selected model from a full 10-restart run.
- n_iter=200 (instead of 1000) — adequate to confirm trend; converged restarts will be noted.
- Script is not wired into the CLI. It is a one-time analysis artifact under `scripts/`.
- `results/` outputs are gitignored by default; the CSV and PNG will be committed explicitly.

## Test evidence

No new pytest tests — this is a data/analysis artifact, not library code. The script uses only existing public API (`train_hmm`, `FeaturePipeline`, `YFinanceSource` cache path). The audit markdown documents the findings.

## Definition of done

- `results/BTC-USD/selection_table.csv`, `results/ETH-USD/selection_table.csv`, `results/SPY/selection_table.csv` exist.
- `results/BTC-USD/bic_curve.png`, `results/ETH-USD/bic_curve.png`, `results/SPY/bic_curve.png` exist.
- `results/n_states_audit.md` exists and contains: BIC minimum conclusions, dwell-time table for critical n values, and a production recommendation.
- `configs/btc.yaml`, `configs/eth.yaml`, `configs/spy.yaml` reflect the confirmed optimal search range.
