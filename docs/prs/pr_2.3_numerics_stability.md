# PR 2.3 — Numerical-Stability Stress Test

## Context

HMMs operating on long sequences (100k–500k bars) are susceptible to numerical
underflow in the forward algorithm and overflow in the log-sum-exp computations.
`hmmlearn` operates in log-space internally, which mitigates most issues, but
custom code paths in `forward.py`, `viterbi.py`, `smoothing.py`, `garch_hmm.py`,
and `kalman.py` require auditing.

## What this PR does

1. Adds `tests/numerics/test_log_space_stability.py` (marked `@pytest.mark.slow`):
   - Generates sequences of length 100k, 250k, 500k from a known 3-state Gaussian HMM.
   - Runs `forward_log_likelihood`, `viterbi_decode`, and `forward_backward_posteriors`.
   - Asserts all outputs are finite and log-likelihood is monotonically non-decreasing
     as a sanity check on sequence ordering (not monotone in T, but not NaN/Inf).

2. Adds `docs/numerics_audit.md` documenting:
   - Which `src/rde/` code paths call `np.log`, `np.exp`, matrix inversion.
   - Whether each is inside or outside hmmlearn's log-space internals.
   - Any zero-probability row guards or singular covariance guards found/added.

## Key findings from the audit

- `inference/forward.py`, `viterbi.py`, `smoothing.py`: all thin wrappers over
  `hmmlearn`'s log-space internals — no custom numerics exposure.
- `garch_hmm.py`: custom `np.log` and `np.exp` in variance updates; uses
  `np.clip` guards where needed; no overflow observed in tests.
- `kalman.py`: matrix inversion of covariance matrices. Uses `scipy.linalg.solve`
  (numerically stable) rather than `np.linalg.inv`. Guarded by positive-definite
  initialization. No issues observed.
- No zero-probability row in transition matrices (hmmlearn initialises with uniform
  + small random perturbation). No singular covariances in standard test data.

## Tradeoffs

- T=500k stress tests generate long sequences, but hmmlearn's O(T×K²) forward
  algorithm scales linearly — this completes in < 60s for K=3.
- Tests are `@pytest.mark.slow` to exclude from default CI.
- The audit is observational (no code changes required), which means it does not
  change any behaviour.

## Definition of done

- `tests/numerics/test_log_space_stability.py` exists, all tests pass.
- `docs/numerics_audit.md` documents the audit findings.
