# Finding: BTC stored results used stale candidate_states range

Date: 2026-05-04  
Severity: Medium — incorrect model stored as production baseline for BTC

## What was found

`results/BTC-USD/diagnostics.txt` reports `n_states=6` and `BIC=79,070`, with
`configs/btc.yaml` having `candidate_states: [2,3,4,5,6,7,8]`.

The BIC audit run (`scripts/run_bic_audit.py`, 2026-05-04) found:

| n | BIC |
|---|-----|
| 6 | 78,938 |
| 7 | 75,078 |
| 8 | 72,158 |

A ~6,800 BIC gap between n=6 and n=8 is far beyond noise. If `candidate_states=[2..8]`
had been in effect when the stored BTC run was executed, BIC selection would have
chosen n=8, not n=6.

## Root cause (INFERENCE)

`git log --follow -- configs/btc.yaml` shows `candidate_states` was extended to
`[2,3,4,5,6,7,8]` in commit `8c099b9` (Phase 6). The stored `results/BTC-USD/`
were generated before Phase 6 (when the ceiling was `[2..6]` or `[2..5]`), and
`rde run` was not re-executed after the config was extended.

This is inferred — the results directory is gitignored and its creation date cannot
be confirmed from git history.

## Impact

- The BTC paper-trading backtest in Phase 35 used the 6-state model for online
  decoding. The 8-state model may produce materially different regime assignments.
- All analysis outputs under `results/BTC-USD/analysis/` were generated from the
  6-state model.
- `app/streamlit_app.py` displays BTC results from the 6-state model.

## Resolution

Re-run `uv run rde run --config configs/btc.yaml --save-model results/BTC-USD/model.pkl`
after setting `candidate_states: [2,3,4,5,6,7,8]` (already in btc.yaml). This will
select n=8 and regenerate all downstream outputs.

This is a data-regeneration task, not a code change. Not blocking for the current
audit pass, but should be done before the next live trading session.
