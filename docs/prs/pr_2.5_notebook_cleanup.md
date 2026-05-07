# PR 2.5 — Stale Prototype Removal + Notebook API Verification

## Context

After 35 phases of development the notebook directory may contain prototype
notebooks that reference removed APIs, or stale scratch files at the repository
root. Any such artefacts would mislead future readers about the current API surface.

## What this PR does

1. **Verifies all three notebooks against the current `src/rde/` API** using
   `uv run python` (not system Python) so the project's virtual environment
   and all dependencies are active.

2. **Documents the verified import surface** for each notebook:
   - `notebooks/01_btc_baseline.ipynb` — imports from `rde.config`, `rde.data`,
     `rde.features`, `rde.models`, `rde.inference`, `rde.evaluation`, `rde.labeling`,
     `rde.viz.interactive`. All symbols resolve cleanly.
   - `notebooks/02_model_selection.ipynb` — imports from `rde.config`, `rde.data`,
     `rde.features`, `rde.models`. All symbols resolve cleanly.
   - `notebooks/03_regime_analytics.ipynb` — imports from `rde.features.*`,
     `rde.models.hmm`, `rde.inference.viterbi`, `rde.inference.online`,
     `rde.evaluation.regime_analytics`, `rde.signals.regime_signal`. All symbols
     resolve cleanly.

3. **Confirms no stale artefacts exist:**
   - No `Untitled*.ipynb` files at repository root or in `notebooks/`.
   - No `scratch_*.py`, `tmp_*.py`, or `prototype*.py` files in `src/` or root.
   - `notebooks/03_regime_analytics.ipynb` is an undocumented but active notebook
     (not a prototype) — it exercises `OnlineDecoder` and `RegimeSignalGenerator`
     which are both present in the current codebase.

## Findings

- No code changes required. All notebooks use current, live API symbols.
- `notebooks/03_regime_analytics.ipynb` was not listed in CLAUDE.md section 4
  (repository layout). No action needed — the layout table is illustrative, not
  exhaustive.

## Definition of done

- All three notebooks' import cells verified to resolve in the project environment.
- No stale prototype files found or removed.
- This PR description committed to `docs/prs/`.
