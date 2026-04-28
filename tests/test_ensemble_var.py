"""Tests for EnsembleHMM (rde.models.ensemble) and Regime VAR (rde.analysis.regime_var).

Covers:
  - align_states correctness (shape, valid permutation, identity, unscramble)
  - train_ensemble_hmm outputs (shapes, entropy, agreement, M=1, reference model)
  - fit_regime_var outputs (shapes, 1-state OLS equivalence, min_effective_n filter)
  - compute_irf outputs (shape, convergence, non-zero shock response)
  - granger_causality_table (columns, non-negative, all assets)
  - Single-asset AR degenerate case
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from rde.models.ensemble import (
    EnsembleConfig,
    EnsembleResult,
    align_states,
    train_ensemble_hmm,
)
from rde.analysis.regime_var import (
    RegimeVARConfig,
    RegimeVARResult,
    compute_irf,
    fit_regime_var,
    granger_causality_table,
)


# ═════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═════════════════════════════════════════════════════════════════════════════

RNG = np.random.default_rng(0)
T_SMALL = 200
N_STATES = 2
N_ASSETS = 3


def _make_synthetic_features(T: int = T_SMALL, d: int = 2, seed: int = 0) -> np.ndarray:
    """Return a (T, d) feature matrix drawn from a mixture of two Gaussians."""
    rng = np.random.default_rng(seed)
    half = T // 2
    X1 = rng.normal(loc=0.0, scale=1.0, size=(half, d))
    X2 = rng.normal(loc=2.0, scale=0.5, size=(T - half, d))
    return np.vstack([X1, X2]).astype(float)


def _make_returns(T: int = T_SMALL, N: int = N_ASSETS, seed: int = 1) -> pd.DataFrame:
    """Return a (T, N) DataFrame of synthetic log returns."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(T, N))
    cols = [f"asset_{i}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)


def _make_posteriors(T: int, K: int, seed: int = 2) -> np.ndarray:
    """Return a (T, K) array of valid posteriors (rows sum to 1)."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(K), size=T)
    return raw


def _posteriors_fn(model, X_scaled: np.ndarray) -> np.ndarray:
    """Thin wrapper around hmmlearn predict_proba for use in train_ensemble_hmm."""
    return model.predict_proba(X_scaled)


# ═════════════════════════════════════════════════════════════════════════════
# align_states tests  (tests 1-7)
# ═════════════════════════════════════════════════════════════════════════════

class TestAlignStates:
    def _make_clean_posteriors(self, T: int, K: int) -> np.ndarray:
        """Each row places almost all mass on one state, cycling through states."""
        arr = np.full((T, K), 0.01 / (K - 1))
        for t in range(T):
            arr[t, t % K] = 1.0 - 0.01
        return arr

    def test_output_shape_matches_input(self):
        """aligned_posteriors has the same shape as other_posteriors."""
        T, K = 50, 3
        ref = _make_posteriors(T, K, seed=10)
        other = _make_posteriors(T, K, seed=11)
        aligned, perm = align_states(ref, other)
        assert aligned.shape == (T, K)

    def test_permutation_is_valid_permutation(self):
        """permutation is a bijection on {0, ..., K-1}."""
        T, K = 50, 4
        ref = _make_posteriors(T, K, seed=20)
        other = _make_posteriors(T, K, seed=21)
        _, perm = align_states(ref, other)
        assert len(perm) == K
        assert set(perm.tolist()) == set(range(K)), f"perm is not a valid permutation: {perm}"

    def test_identity_input_gives_identity_permutation(self):
        """When ref == other the permutation should map each state to itself."""
        T, K = 60, 3
        ref = self._make_clean_posteriors(T, K)
        aligned, perm = align_states(ref, ref.copy())
        # Identity permutation: perm[k] == k for all k
        assert np.array_equal(perm, np.arange(K)), f"Expected identity perm, got {perm}"

    def test_identity_input_aligned_equals_other(self):
        """When ref == other, aligned posteriors should equal other (up to numerical noise)."""
        T, K = 60, 3
        ref = self._make_clean_posteriors(T, K)
        other = ref.copy()
        aligned, _ = align_states(ref, other)
        np.testing.assert_allclose(aligned, other, atol=1e-12)

    def test_scrambled_input_is_unscrambled(self):
        """A known column-permuted other should be restored to the reference order."""
        T, K = 80, 3
        rng = np.random.default_rng(30)
        ref = rng.dirichlet(np.ones(K), size=T)
        # Scramble: state 0→1, 1→2, 2→0
        scramble = np.array([1, 2, 0])
        other = ref[:, scramble]

        aligned, perm = align_states(ref, other)
        # After alignment, aligned should approximate ref
        np.testing.assert_allclose(aligned, ref, atol=0.15)

    def test_mismatched_shapes_raise(self):
        """align_states raises ValueError when shapes don't match."""
        ref = _make_posteriors(50, 3, seed=40)
        other = _make_posteriors(50, 4, seed=41)  # different K
        with pytest.raises(ValueError, match="shape"):
            align_states(ref, other)

    def test_permutation_length_equals_n_states(self):
        """permutation length equals K."""
        T, K = 40, 5
        ref = _make_posteriors(T, K, seed=50)
        other = _make_posteriors(T, K, seed=51)
        _, perm = align_states(ref, other)
        assert len(perm) == K


# ═════════════════════════════════════════════════════════════════════════════
# train_ensemble_hmm tests  (tests 8-22)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ensemble_result() -> EnsembleResult:
    """Fit a small ensemble for reuse across tests."""
    X = _make_synthetic_features(T=300, d=2, seed=0)
    config = EnsembleConfig(n_models=3, n_restarts_each=2, seed_step=50)
    return train_ensemble_hmm(
        X,
        n_states=2,
        posteriors_fn=_posteriors_fn,
        config=config,
        covariance_type="full",
        n_iter=50,
    )


class TestTrainEnsembleHMM:
    def test_returns_ensemble_result(self, ensemble_result):
        """train_ensemble_hmm returns an EnsembleResult instance."""
        assert isinstance(ensemble_result, EnsembleResult)

    def test_ensemble_posteriors_shape(self, ensemble_result):
        """ensemble_posteriors has shape (T, K)."""
        assert ensemble_result.ensemble_posteriors.shape == (300, 2)

    def test_ensemble_posteriors_rows_sum_to_one(self, ensemble_result):
        """Each row of ensemble_posteriors sums to 1 (within numerical tolerance)."""
        row_sums = ensemble_result.ensemble_posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_per_model_posteriors_shape(self, ensemble_result):
        """per_model_posteriors has shape (M, T, K)."""
        M, T, K = ensemble_result.per_model_posteriors.shape
        assert M == 3
        assert T == 300
        assert K == 2

    def test_entropy_shape(self, ensemble_result):
        """entropy has shape (T,)."""
        assert ensemble_result.entropy.shape == (300,)

    def test_entropy_non_negative(self, ensemble_result):
        """All entropy values are ≥ 0 (up to floating-point rounding)."""
        assert np.all(ensemble_result.entropy >= -1e-9)

    def test_agreement_shape(self, ensemble_result):
        """agreement has shape (T,)."""
        assert ensemble_result.agreement.shape == (300,)

    def test_agreement_in_unit_interval(self, ensemble_result):
        """All agreement values are in [0, 1]."""
        assert np.all(ensemble_result.agreement >= -1e-10)
        assert np.all(ensemble_result.agreement <= 1.0 + 1e-10)

    def test_n_models_matches_config(self, ensemble_result):
        """Number of models equals config.n_models."""
        assert len(ensemble_result.models) == 3

    def test_n_alignment_permutations_matches_n_models(self, ensemble_result):
        """Number of alignment permutations equals M."""
        assert len(ensemble_result.alignment_permutations) == 3

    def test_entropy_zero_when_one_state_dominates(self):
        """entropy is ~0 when the ensemble posterior has all mass on one state."""
        T, K = 100, 3
        # Degenerate posterior: all mass on state 0
        post = np.zeros((T, K))
        post[:, 0] = 1.0
        from rde.models.ensemble import _compute_entropy
        h = _compute_entropy(post)
        np.testing.assert_allclose(h, 0.0, atol=1e-6)

    def test_entropy_log_K_when_uniform(self):
        """entropy is log(K) when the posterior is uniform."""
        T, K = 100, 4
        post = np.full((T, K), 1.0 / K)
        from rde.models.ensemble import _compute_entropy
        h = _compute_entropy(post)
        np.testing.assert_allclose(h, np.log(K), atol=1e-5)

    def test_m1_agreement_all_ones(self):
        """When M=1, agreement is all-ones regardless of entropy."""
        X = _make_synthetic_features(T=150, d=2, seed=5)
        config = EnsembleConfig(n_models=1, n_restarts_each=2, seed_step=50)
        result = train_ensemble_hmm(
            X,
            n_states=2,
            posteriors_fn=_posteriors_fn,
            config=config,
            covariance_type="full",
            n_iter=50,
        )
        np.testing.assert_allclose(result.agreement, 1.0, atol=1e-10)

    def test_reference_model_has_highest_log_likelihood(self, ensemble_result):
        """reference_model has the highest log-likelihood among all models."""
        best_ll = max(m.log_likelihood for m in ensemble_result.models)
        assert ensemble_result.reference_model.log_likelihood == best_ll

    def test_forbidden_kwargs_raise(self):
        """Passing seed_base or n_restarts in train_kwargs raises ValueError."""
        X = _make_synthetic_features(T=100, d=2, seed=7)
        with pytest.raises(ValueError, match="seed_base"):
            train_ensemble_hmm(
                X, n_states=2, posteriors_fn=_posteriors_fn,
                seed_base=99, n_iter=10,
            )


# ═════════════════════════════════════════════════════════════════════════════
# fit_regime_var tests  (tests 23-37)
# ═════════════════════════════════════════════════════════════════════════════

class TestFitRegimeVar:
    def test_returns_list_of_regime_var_result(self):
        """fit_regime_var returns a list[RegimeVARResult]."""
        rets = _make_returns(T=150, N=2)
        posts = _make_posteriors(150, 2, seed=3)
        results = fit_regime_var(rets, posts)
        assert isinstance(results, list)
        assert all(isinstance(r, RegimeVARResult) for r in results)

    def test_correct_n_assets(self):
        """Each result has n_assets equal to number of columns in returns."""
        N = 3
        rets = _make_returns(T=150, N=N)
        posts = _make_posteriors(150, 2, seed=4)
        results = fit_regime_var(rets, posts)
        for r in results:
            assert r.n_assets == N

    def test_correct_lag_order(self):
        """Each result has the lag_order specified in config."""
        p = 2
        rets = _make_returns(T=200, N=2)
        posts = _make_posteriors(200, 2, seed=5)
        config = RegimeVARConfig(lag_order=p, min_effective_n=1.0)
        results = fit_regime_var(rets, posts, config=config)
        for r in results:
            assert r.lag_order == p

    def test_intercept_shape(self):
        """intercept has shape (N,)."""
        N = 3
        rets = _make_returns(T=150, N=N)
        posts = _make_posteriors(150, 2, seed=6)
        results = fit_regime_var(rets, posts)
        for r in results:
            assert r.intercept.shape == (N,)

    def test_coef_matrices_length_equals_p(self):
        """coef_matrices has length p."""
        p = 2
        rets = _make_returns(T=200, N=2)
        posts = _make_posteriors(200, 2, seed=7)
        config = RegimeVARConfig(lag_order=p, min_effective_n=1.0)
        results = fit_regime_var(rets, posts, config=config)
        for r in results:
            assert len(r.coef_matrices) == p

    def test_coef_matrices_shape(self):
        """Each coef_matrix has shape (N, N)."""
        N = 3
        p = 2
        rets = _make_returns(T=200, N=N)
        posts = _make_posteriors(200, 2, seed=8)
        config = RegimeVARConfig(lag_order=p, min_effective_n=1.0)
        results = fit_regime_var(rets, posts, config=config)
        for r in results:
            for A in r.coef_matrices:
                assert A.shape == (N, N)

    def test_residual_cov_symmetric(self):
        """residual_cov is symmetric."""
        rets = _make_returns(T=150, N=3)
        posts = _make_posteriors(150, 2, seed=9)
        results = fit_regime_var(rets, posts)
        for r in results:
            np.testing.assert_allclose(r.residual_cov, r.residual_cov.T, atol=1e-10)

    def test_residual_cov_positive_semi_definite(self):
        """residual_cov is positive semi-definite (all eigenvalues ≥ 0)."""
        rets = _make_returns(T=150, N=3)
        posts = _make_posteriors(150, 2, seed=10)
        results = fit_regime_var(rets, posts)
        for r in results:
            eigvals = np.linalg.eigvalsh(r.residual_cov)
            assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_single_state_equivalent_to_ols_var(self):
        """When all posteriors = 1 (1-state), result is the standard OLS VAR."""
        T, N = 200, 2
        p = 1
        rng = np.random.default_rng(11)
        rets = pd.DataFrame(rng.normal(0, 0.01, (T, N)), columns=["a", "b"])

        # 1-state posteriors (all ones)
        posts_1state = np.ones((T, 1))
        config = RegimeVARConfig(lag_order=p, min_effective_n=1.0)
        results = fit_regime_var(rets, posts_1state, config=config)
        assert len(results) == 1
        r = results[0]

        # Compare to numpy OLS
        ret_arr = rets.to_numpy()
        Z = np.hstack([np.ones((T - 1, 1)), ret_arr[:-1]])  # (T-1, 1+N)
        R = ret_arr[1:]
        B_ols, _, _, _ = np.linalg.lstsq(Z, R, rcond=None)

        # Intercept
        np.testing.assert_allclose(r.intercept, B_ols[0], atol=1e-6)
        # Lag-1 coefficient matrix (transposed because our API uses A[i,j]=effect on i)
        A_expected = B_ols[1:].T
        np.testing.assert_allclose(r.coef_matrices[0], A_expected, atol=1e-5)

    def test_low_effective_n_state_excluded(self):
        """States with Σγ < min_effective_n are not in the output."""
        T, N, K = 200, 2, 2
        # Give almost all weight to state 0
        posts = np.zeros((T, K))
        posts[:, 0] = 1.0  # state 0: effective_n = T
        posts[:, 1] = 0.0  # state 1: effective_n = 0

        rets = _make_returns(T=T, N=N)
        config = RegimeVARConfig(min_effective_n=10.0)
        results = fit_regime_var(rets, posts, config=config)

        returned_states = [r.state for r in results]
        assert 0 in returned_states
        assert 1 not in returned_states

    def test_mismatched_time_dimension_raises(self):
        """Mismatched T between returns and posteriors raises ValueError."""
        rets = _make_returns(T=100, N=2)
        posts = _make_posteriors(150, 2, seed=12)
        with pytest.raises(ValueError, match="rows"):
            fit_regime_var(rets, posts)

    def test_lag_order_zero_raises(self):
        """lag_order < 1 raises ValueError."""
        rets = _make_returns(T=100, N=2)
        posts = _make_posteriors(100, 2, seed=13)
        config = RegimeVARConfig(lag_order=0)
        with pytest.raises(ValueError, match="lag_order"):
            fit_regime_var(rets, posts, config=config)

    def test_asset_names_preserved(self):
        """asset_names in the result match the DataFrame columns."""
        cols = ["BTC", "ETH", "SPY"]
        rets = pd.DataFrame(
            np.random.default_rng(14).normal(0, 0.01, (150, 3)),
            columns=cols,
        )
        posts = _make_posteriors(150, 2, seed=15)
        results = fit_regime_var(rets, posts)
        for r in results:
            assert r.asset_names == cols

    def test_single_asset_ar1(self):
        """Single-asset case produces a scalar AR(1) coefficient matrix."""
        T = 200
        rng = np.random.default_rng(16)
        rets = pd.DataFrame(rng.normal(0, 0.01, (T, 1)), columns=["BTC"])
        posts = np.ones((T, 1))
        config = RegimeVARConfig(lag_order=1, min_effective_n=1.0)
        results = fit_regime_var(rets, posts, config=config)
        assert len(results) == 1
        r = results[0]
        assert r.n_assets == 1
        assert r.coef_matrices[0].shape == (1, 1)
        assert r.residual_cov.shape == (1, 1)


# ═════════════════════════════════════════════════════════════════════════════
# compute_irf tests  (tests 38-45)
# ═════════════════════════════════════════════════════════════════════════════

def _stationary_var_result(N: int = 2, p: int = 1, seed: int = 20) -> RegimeVARResult:
    """Build a RegimeVARResult with a guaranteed stationary VAR (small coefficients)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, (N, N))  # small → stationary
    intercept = rng.normal(0, 0.001, N)
    cov = np.eye(N) * 0.01
    return RegimeVARResult(
        state=0,
        n_assets=N,
        lag_order=p,
        intercept=intercept,
        coef_matrices=[A],
        residual_cov=cov,
        effective_n=100.0,
        asset_names=[f"asset_{i}" for i in range(N)],
    )


class TestComputeIRF:
    def test_irf_shape(self):
        """IRF has shape (horizons+1, N)."""
        N, horizons = 3, 10
        vr = _stationary_var_result(N=N)
        irf = compute_irf(vr, shock_asset=0, horizons=horizons)
        assert irf.shape == (horizons + 1, N)

    def test_irf_columns_are_asset_names(self):
        """IRF DataFrame columns match the asset names."""
        N = 3
        vr = _stationary_var_result(N=N)
        irf = compute_irf(vr, shock_asset=0, horizons=5)
        assert list(irf.columns) == vr.asset_names

    def test_contemporaneous_shock_asset_response_nonzero(self):
        """At horizon 0, the shocked asset has a non-zero response (Cholesky diagonal)."""
        N = 3
        vr = _stationary_var_result(N=N)
        irf = compute_irf(vr, shock_asset=0, horizons=5)
        # Cholesky L[0,0] > 0 for a PD matrix
        assert abs(irf.iloc[0, 0]) > 1e-10

    def test_irf_converges_toward_zero_for_stationary_var(self):
        """For a stationary VAR, the IRF magnitudes should diminish at large horizons."""
        N = 2
        vr = _stationary_var_result(N=N, seed=21)
        horizons = 30
        irf = compute_irf(vr, shock_asset=0, horizons=horizons)
        # Compare last 5 rows to first 5 rows: later responses should be smaller in norm
        early_norm = np.linalg.norm(irf.iloc[:5].values)
        late_norm = np.linalg.norm(irf.iloc[-5:].values)
        assert late_norm < early_norm, (
            f"IRF not decaying: early_norm={early_norm:.6f}  late_norm={late_norm:.6f}"
        )

    def test_irf_index_is_horizon_range(self):
        """IRF index runs from 0 to horizons inclusive."""
        vr = _stationary_var_result(N=2)
        horizons = 8
        irf = compute_irf(vr, shock_asset=0, horizons=horizons)
        assert list(irf.index) == list(range(horizons + 1))

    def test_out_of_range_shock_asset_raises(self):
        """shock_asset out of [0, N-1] raises ValueError."""
        vr = _stationary_var_result(N=2)
        with pytest.raises(ValueError, match="shock_asset"):
            compute_irf(vr, shock_asset=5, horizons=5)

    def test_irf_lag2_var(self):
        """compute_irf works for VAR(2) (two lag matrices)."""
        N, p = 2, 2
        rng = np.random.default_rng(22)
        A1 = rng.normal(0, 0.05, (N, N))
        A2 = rng.normal(0, 0.02, (N, N))
        cov = np.eye(N) * 0.01
        vr = RegimeVARResult(
            state=0,
            n_assets=N,
            lag_order=p,
            intercept=np.zeros(N),
            coef_matrices=[A1, A2],
            residual_cov=cov,
            effective_n=100.0,
            asset_names=["a", "b"],
        )
        irf = compute_irf(vr, shock_asset=0, horizons=5)
        assert irf.shape == (6, N)

    def test_single_asset_irf_scalar(self):
        """For N=1, IRF is a single-column DataFrame."""
        vr = RegimeVARResult(
            state=0,
            n_assets=1,
            lag_order=1,
            intercept=np.array([0.0]),
            coef_matrices=[np.array([[0.5]])],
            residual_cov=np.array([[0.01]]),
            effective_n=50.0,
            asset_names=["BTC"],
        )
        irf = compute_irf(vr, shock_asset=0, horizons=4)
        assert irf.shape == (5, 1)
        # At h=0 the response should be sqrt(0.01)=0.1
        np.testing.assert_allclose(abs(irf.iloc[0, 0]), 0.1, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# granger_causality_table tests  (tests 46-53)
# ═════════════════════════════════════════════════════════════════════════════

class TestGrangerCausalityTable:
    def _get_results(self, N: int = 3, K: int = 2, T: int = 200) -> list[RegimeVARResult]:
        rets = _make_returns(T=T, N=N)
        posts = _make_posteriors(T, K, seed=30)
        config = RegimeVARConfig(lag_order=1, min_effective_n=1.0)
        return fit_regime_var(rets, posts, config=config)

    def test_returns_dataframe(self):
        """granger_causality_table returns a DataFrame."""
        results = self._get_results()
        df = granger_causality_table(results)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        """DataFrame has columns: state, from_asset, to_asset, granger_strength."""
        results = self._get_results()
        df = granger_causality_table(results)
        for col in ("state", "from_asset", "to_asset", "granger_strength"):
            assert col in df.columns, f"Missing column: {col}"

    def test_granger_strength_non_negative(self):
        """All granger_strength values are ≥ 0."""
        results = self._get_results()
        df = granger_causality_table(results)
        assert (df["granger_strength"] >= 0).all()

    def test_all_assets_represented(self):
        """All asset names appear in both from_asset and to_asset."""
        N = 3
        results = self._get_results(N=N)
        df = granger_causality_table(results)
        asset_names = {f"asset_{i}" for i in range(N)}
        assert set(df["from_asset"].unique()) == asset_names
        assert set(df["to_asset"].unique()) == asset_names

    def test_all_states_represented(self):
        """All fitted state indices appear in the state column."""
        results = self._get_results()
        df = granger_causality_table(results)
        fitted_states = {r.state for r in results}
        assert set(df["state"].unique()) == fitted_states

    def test_n_rows_equals_states_times_n_squared(self):
        """Number of rows = Σ_k N^2 over all states."""
        N = 3
        results = self._get_results(N=N)
        expected_rows = len(results) * N * N
        df = granger_causality_table(results)
        assert len(df) == expected_rows

    def test_empty_input_returns_empty_dataframe(self):
        """Empty list of results returns an empty DataFrame with correct columns."""
        df = granger_causality_table([])
        assert len(df) == 0
        for col in ("state", "from_asset", "to_asset", "granger_strength"):
            assert col in df.columns

    def test_sorted_by_state_then_strength_descending(self):
        """Output is sorted by state asc, then granger_strength desc."""
        results = self._get_results(K=2)
        df = granger_causality_table(results)
        for state_val in df["state"].unique():
            sub = df[df["state"] == state_val]["granger_strength"].values
            # Check descending within state
            assert np.all(np.diff(sub) <= 1e-14), (
                f"State {state_val} granger_strength is not descending: {sub}"
            )
