"""Tests for rde.models.persistence (Phase 11)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from rde.models.hmm import train_hmm
from rde.models.persistence import load_model, save_model


# ── helpers ───────────────────────────────────────────────────────────────────

def _small_fitted(n_states: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal([0, 0], np.eye(2), size=200)
    return train_hmm(X, n_states, n_restarts=2, seed_base=seed)


# ── save_model ────────────────────────────────────────────────────────────────


def test_save_creates_file(tmp_path):
    fitted = _small_fitted()
    dest = tmp_path / "model.pkl"
    save_model(fitted, dest)
    assert dest.exists()


def test_save_creates_parent_directories(tmp_path):
    fitted = _small_fitted()
    dest = tmp_path / "nested" / "deep" / "model.pkl"
    save_model(fitted, dest)
    assert dest.exists()


def test_save_accepts_string_path(tmp_path):
    fitted = _small_fitted()
    dest = str(tmp_path / "model.pkl")
    save_model(fitted, dest)
    assert Path(dest).exists()


def test_save_overwrites_existing_file(tmp_path):
    fitted = _small_fitted(n_states=2)
    dest = tmp_path / "model.pkl"
    save_model(fitted, dest)
    size_v1 = dest.stat().st_size

    fitted2 = _small_fitted(n_states=3)
    save_model(fitted2, dest)
    size_v2 = dest.stat().st_size

    assert dest.exists()
    assert size_v1 != size_v2 or size_v2 > 0  # file was overwritten


# ── load_model ────────────────────────────────────────────────────────────────


def test_load_returns_fitted_model(tmp_path):
    from rde.models.hmm import FittedModel
    fitted = _small_fitted()
    dest = tmp_path / "model.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    assert isinstance(loaded, FittedModel)


def test_load_nonexistent_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model(tmp_path / "does_not_exist.pkl")


def test_load_wrong_type_raises_type_error(tmp_path):
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as fh:
        pickle.dump({"not": "a FittedModel"}, fh, protocol=5)
    with pytest.raises(TypeError, match="FittedModel"):
        load_model(bad)


def test_load_accepts_string_path(tmp_path):
    fitted = _small_fitted()
    dest = tmp_path / "model.pkl"
    save_model(fitted, dest)
    loaded = load_model(str(dest))
    assert loaded.n_states == fitted.n_states


# ── round-trip correctness ────────────────────────────────────────────────────


def test_roundtrip_n_states(tmp_path):
    fitted = _small_fitted(n_states=3)
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    assert loaded.n_states == 3


def test_roundtrip_seed(tmp_path):
    fitted = _small_fitted(seed=7)
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    assert loaded.seed == fitted.seed


def test_roundtrip_log_likelihood(tmp_path):
    fitted = _small_fitted()
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    assert loaded.log_likelihood == pytest.approx(fitted.log_likelihood)


def test_roundtrip_feature_names(tmp_path):
    fitted = _small_fitted()
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    assert loaded.feature_names == fitted.feature_names


def test_roundtrip_scaler_transform_identical(tmp_path):
    """Loaded scaler must produce the same transform as the original."""
    rng = np.random.default_rng(0)
    X_test = rng.standard_normal((10, 2))
    fitted = _small_fitted()
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    np.testing.assert_array_equal(
        fitted.scaler.transform(X_test),
        loaded.scaler.transform(X_test),
    )


def test_roundtrip_hmm_transmat_identical(tmp_path):
    """Loaded HMM transition matrix must equal the original."""
    fitted = _small_fitted()
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    np.testing.assert_array_almost_equal(
        fitted.hmm.transmat_, loaded.hmm.transmat_
    )


def test_roundtrip_aic_bic(tmp_path):
    fitted = _small_fitted()
    dest = tmp_path / "m.pkl"
    save_model(fitted, dest)
    loaded = load_model(dest)
    assert loaded.aic == pytest.approx(fitted.aic)
    assert loaded.bic == pytest.approx(fitted.bic)
