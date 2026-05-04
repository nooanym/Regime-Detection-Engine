# PR 2.7 — Notion Workspace Update

## Context

The Notion workspace ("Regime Detection Engine") is the source of truth for
project management. After Phases 31–36 and the post-Phase-36 audit (items 2.1–2.6)
it needs to reflect completed work, open findings, and decision log entries.

## What this PR does

Updates the Notion workspace via MCP with:

1. **Roadmap & Milestones page** — Appends Phase 31–36 completion rows and the
   audit items 2.1–2.6.

2. **Regime Interpretation Diary** — Adds a Run 002+ entry documenting:
   - BIC ceiling audit finding: n=8 practical optimum for BTC/ETH/SPY
   - BTC anomaly: stored results used n=6 (stale pre-Phase-6 config)

3. **Decision Log** — Adds entries for:
   - `audit/2.1`: BIC ceiling at n=8 (dwell-time overfitting signal used as
     secondary stopping criterion when BIC did not show a clear minimum)
   - `audit/2.3`: No code changes to numerics (hmmlearn log-space confirmed safe)
   - `audit/2.4`: Parquet chosen over pickle for reproducibility fixtures
     (schema-forward, no deserialisation risk)

4. **Root page status line** — Updates "Current phase" to Phase 36 + audit complete.

## Definition of done

- Notion pages reflect Phase 36 completion and audit findings.
- All four update targets (roadmap, diary, decision log, root status) confirmed
  updated in Notion.
