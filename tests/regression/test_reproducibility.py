"""Reproducibility regression harness (audit item 2.4).

Runs the full pipeline on fixed synthetic data and compares Viterbi paths
against stored fixtures.  Failure means a code change shifted regime assignments.

First-run (no fixtures): generates fixtures and passes by construction.
Subsequent runs: compares to stored fixtures.

Regenerate fixtures::

    uv run pytest --update-fixtures tests/regression/

Normal run::

    uv run pytest tests/regression/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Test cases: (name, n_states, T, seed)
# ---------------------------------------------------------------------------
CASES = [
    ("gaussian_3state_small",  3, 2_000, 42),
    ("gaussian_4state_medium", 4, 3_000, 43),
    ("gaussian_5state_large",  5, 4_000, 44),
]


def _generate_synthetic(n_states: int, T: int, seed: int) -> np.ndarray:
    """Return synthetic feature matrix (T, 3) with known HMM structure."""
    rng = np.random.default_rng(seed)
    K = n_states
    means = np.linspace(-2.0, 2.0, K)
    stds = np.full(K, 0.4)

    transmat = np.full((K, K), 0.02 / (K - 1))
    np.fill_diagonal(transmat, 0.98)
    transmat /= transmat.sum(axis=1, keepdims=True)

    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K)
    for t in range(1, T):
        states[t] = rng.choice(K, p=transmat[states[t - 1]])

    X = np.column_stack([
        means[states] + stds[states] * rng.standard_normal(T),  # feature 1
        np.abs(rng.standard_normal(T)) * 0.1 + stds[states],    # feature 2
        means[states] * 0.3 + 0.1 * rng.standard_normal(T),     # feature 3
    ])
    return X


def _run_pipeline(X: np.ndarray, n_states: int, seed: int) -> np.ndarray:
    """Return Viterbi path from a fixed-seed HMM fit."""
    from rde.models.hmm import train_hmm
    from rde.inference.viterbi import viterbi_decode

    fitted = train_hmm(X, n_states=n_states, n_restarts=3, n_iter=100, seed_base=seed)
    Xs = fitted.scaler.transform(X)
    return viterbi_decode(fitted.hmm, Xs)


def _fixture_path(name: str) -> Path:
    return FIXTURES_DIR / f"{name}.parquet"


def _save_fixture(name: str, path: np.ndarray) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"state": path}).to_parquet(_fixture_path(name), index=False)


def _load_fixture(name: str) -> np.ndarray:
    return pd.read_parquet(_fixture_path(name))["state"].values


# ---------------------------------------------------------------------------
# conftest hook for --update-fixtures
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--update-fixtures",
        action="store_true",
        default=False,
        help="Regenerate regression fixtures and overwrite existing ones.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,n_states,T,seed", CASES)
def test_viterbi_reproducibility(name, n_states, T, seed, request):
    """Viterbi path must match the stored fixture bit-for-bit.

    On first run (no fixture) or when --update-fixtures is passed,
    the fixture is generated and the test passes by construction.
    """
    update = request.config.getoption("--update-fixtures", default=False)
    fp = _fixture_path(name)

    X = _generate_synthetic(n_states=n_states, T=T, seed=seed)
    recovered = _run_pipeline(X, n_states=n_states, seed=seed)

    if not fp.exists() or update:
        _save_fixture(name, recovered)
        reason = "fixture created" if not fp.exists() else "fixture updated"
        pytest.skip(f"{name}: {reason} — re-run without --update-fixtures to compare")

    stored = _load_fixture(name)
    assert stored.shape == recovered.shape, (
        f"{name}: shape mismatch stored={stored.shape} vs recovered={recovered.shape}"
    )

    n_diff = int(np.sum(stored != recovered))
    if n_diff > 0:
        diff_pct = 100.0 * n_diff / len(stored)
        # Show first few differing positions
        diff_idx = np.where(stored != recovered)[0][:5]
        detail = ", ".join(
            f"t={i}: stored={stored[i]} recovered={recovered[i]}"
            for i in diff_idx
        )
        pytest.fail(
            f"{name}: Viterbi path differs in {n_diff}/{len(stored)} bars "
            f"({diff_pct:.1f}%). First diffs: {detail}"
        )
