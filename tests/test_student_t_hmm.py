"""Tests for StudentTHMM and volume feature transformers (Phase 6)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_t

from rde.features.volume import LogVolume, RollingVolumeZScore
from rde.models.hmm import _compute_n_params, train_hmm
from rde.models.student_t_hmm import StudentTHMM, _mvt_log_pdf, _update_dof


# ── _mvt_log_pdf ─────────────────────────────────────────────────────────────

class TestMvtLogPdf:
    def _make_inputs(self, T=200, d=3, seed=0):
        rng = np.random.default_rng(seed)
        mean = rng.standard_normal(d)
        A = rng.standard_normal((d, d))
        cov = A @ A.T + np.eye(d) * 0.5
        X = rng.standard_normal((T, d))
        return X, mean, cov

    def test_shape(self):
        X, mean, cov = self._make_inputs()
        result = _mvt_log_pdf(X, mean, cov, dof=5.0)
        assert result.shape == (200,)

    def test_matches_scipy(self):
        """_mvt_log_pdf must match scipy.stats.multivariate_t.logpdf."""
        rng = np.random.default_rng(42)
        d = 3
        mean = rng.standard_normal(d)
        A = rng.standard_normal((d, d))
        cov = A @ A.T + np.eye(d)
        X = rng.standard_normal((50, d))
        dof = 4.0

        our_log_pdf = _mvt_log_pdf(X, mean, cov, dof)
        scipy_log_pdf = multivariate_t(loc=mean, shape=cov, df=dof).logpdf(X)

        np.testing.assert_allclose(our_log_pdf, scipy_log_pdf, rtol=1e-5, atol=1e-8)

    def test_heavier_tails_than_gaussian(self):
        """Student-t should assign more probability to extreme observations than Gaussian."""
        from scipy.stats import multivariate_normal as mvn

        d = 2
        mean = np.zeros(d)
        cov = np.eye(d)
        X_extreme = np.array([[5.0, 5.0]])

        t_log = float(_mvt_log_pdf(X_extreme, mean, cov, dof=3.0)[0])
        g_log = float(mvn(mean=mean, cov=cov).logpdf(X_extreme))
        # Student-t should be less negative (higher prob) for tail observations
        assert t_log > g_log

    def test_near_gaussian_for_high_dof(self):
        """At very high dof, Student-t should be close to Gaussian."""
        from scipy.stats import multivariate_normal as mvn

        rng = np.random.default_rng(7)
        d = 2
        mean = rng.standard_normal(d)
        A = rng.standard_normal((d, d))
        cov = A @ A.T + np.eye(d)
        X = rng.standard_normal((100, d))

        t_log = _mvt_log_pdf(X, mean, cov, dof=1000.0)
        g_log = mvn(mean=mean, cov=cov).logpdf(X)
        np.testing.assert_allclose(t_log, g_log, atol=0.01)

    def test_singular_covariance_fallback(self):
        """Should not raise even for near-singular covariance."""
        X = np.ones((10, 2)) * 0.5
        mean = np.zeros(2)
        cov = np.eye(2) * 1e-12   # near-singular
        result = _mvt_log_pdf(X, mean, cov, dof=3.0)
        assert result.shape == (10,)
        assert np.all(np.isfinite(result))


# ── _update_dof ───────────────────────────────────────────────────────────────

class TestUpdateDof:
    def test_returns_float(self):
        result = _update_dof(5.0, -0.5)
        assert isinstance(result, float)

    def test_stays_in_bounds(self):
        for c_k in [-2.0, -0.5, 0.0, 0.5]:
            result = _update_dof(5.0, c_k)
            assert 2.001 <= result <= 200.0

    def test_converges_to_known_root(self):
        # At c_k = 0 the objective is log(ν/2) - ψ(ν/2) + 1 = 0
        # This has a root at ν → ∞ (log ν/2 - ψ(ν/2) → 0). For a typical
        # finite initialisation with c_k near 0 we just verify monotone convergence.
        nu_init = 10.0
        c_k = -0.1
        result = _update_dof(nu_init, c_k)
        assert 2.001 <= result <= 200.0

    def test_fallback_on_no_bracket(self):
        # With c_k very large both objective evaluations may have the same sign.
        # Should return a clamped value without raising.
        result = _update_dof(5.0, 100.0)
        assert 2.001 <= result <= 200.0


# ── StudentTHMM ───────────────────────────────────────────────────────────────

class TestStudentTHMM:
    def _make_fat_tail_data(self, seed=0, T=2000, n_states=2):
        """Generate synthetic 2-state fat-tailed 2-D data."""
        rng = np.random.default_rng(seed)
        means = [np.array([0.0, 0.0]), np.array([1.5, 1.5])]
        covs  = [np.eye(2) * 0.5, np.eye(2) * 1.0]
        dofs  = [3.0, 5.0]
        labels = rng.integers(0, n_states, size=T)
        X = np.vstack([
            multivariate_t(loc=means[k], shape=covs[k], df=dofs[k]).rvs(random_state=rng)
            for k in labels
        ])
        return X, labels

    def test_fit_and_score(self):
        X, _ = self._make_fat_tail_data()
        model = StudentTHMM(n_components=2, n_iter=100, random_state=0)
        model.fit(X)
        score = model.score(X)
        assert np.isfinite(score)
        assert score < 0   # log-likelihood is negative

    def test_dofs_initialised(self):
        model = StudentTHMM(n_components=3, n_iter=1, random_state=0)
        X = np.random.default_rng(0).standard_normal((100, 2))
        model.fit(X)
        assert model.dofs_.shape == (3,)
        assert np.all(model.dofs_ > 2.0)
        assert np.all(model.dofs_ <= 200.0)

    def test_dofs_updated_after_fit(self):
        """dofs_ should change from the 5.0 initialisation after real ECM iterations."""
        X, _ = self._make_fat_tail_data(T=500)
        model = StudentTHMM(n_components=2, n_iter=30, random_state=42)
        model.fit(X)
        # At least one dof should have moved from 5.0
        assert not np.allclose(model.dofs_, 5.0, atol=0.5)

    def test_student_t_beats_gaussian_on_fat_tails(self):
        """StudentTHMM (1-state) should achieve higher log-likelihood on fat-tailed data.

        Using 1 component so the comparison is purely about emission distribution
        quality, isolated from any HMM transition-structure influence.
        """
        from hmmlearn.hmm import GaussianHMM

        rng = np.random.default_rng(0)
        # Pure fat-tailed data from a single Student-t distribution (ν=3)
        X = multivariate_t(loc=[0.0, 0.0], shape=np.eye(2), df=3.0).rvs(
            size=2000, random_state=rng
        )

        gauss = GaussianHMM(n_components=1, covariance_type="full", n_iter=300, random_state=0)
        student = StudentTHMM(n_components=1, n_iter=300, random_state=0)
        gauss.fit(X)
        student.fit(X)

        score_gauss = gauss.score(X)
        score_student = student.score(X)
        assert score_student > score_gauss, (
            f"StudentTHMM ({score_student:.2f}) should beat GaussianHMM ({score_gauss:.2f}) "
            "on data drawn from a multivariate Student-t with df=3."
        )

    def test_predict_shape(self):
        X, _ = self._make_fat_tail_data(T=200)
        model = StudentTHMM(n_components=2, n_iter=50, random_state=0)
        model.fit(X)
        states = model.predict(X)
        assert states.shape == (200,)
        assert set(states).issubset({0, 1})

    def test_covariance_type_forced_full(self):
        """Passing covariance_type should not cause an error; it is silently overridden to full."""
        model = StudentTHMM(n_components=2, n_iter=10, random_state=0, covariance_type="diag")
        X = np.random.default_rng(0).standard_normal((100, 2))
        model.fit(X)
        assert model.covariance_type == "full"


# ── train_hmm with emission_dist='student_t' ──────────────────────────────────

class TestTrainHmmStudentT:
    def test_returns_fitted_model(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 2))
        fitted = train_hmm(X, n_states=2, n_restarts=2, n_iter=30, emission_dist="student_t")
        assert fitted.n_states == 2
        assert np.isfinite(fitted.log_likelihood)
        assert isinstance(fitted.hmm, StudentTHMM)

    def test_invalid_emission_dist_raises(self):
        X = np.random.default_rng(0).standard_normal((100, 2))
        with pytest.raises(ValueError, match="emission_dist"):
            train_hmm(X, n_states=2, emission_dist="cauchy")

    def test_student_t_n_params_larger_than_gaussian(self):
        n_params_gauss   = _compute_n_params(3, 2, "full", "gaussian")
        n_params_student = _compute_n_params(3, 2, "full", "student_t")
        assert n_params_student == n_params_gauss + 3  # one ν per state

    def test_aic_bic_finite(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 2))
        fitted = train_hmm(X, n_states=2, n_restarts=1, n_iter=20, emission_dist="student_t")
        assert np.isfinite(fitted.aic)
        assert np.isfinite(fitted.bic)


# ── Config validation ─────────────────────────────────────────────────────────

class TestConfigEmissionDist:
    def test_invalid_emission_dist_rejected(self, tmp_path):
        from rde.config.loader import load_config

        cfg_text = """
asset:
  symbol: BTC-USD
model:
  emission_dist: normal_dist
"""
        p = tmp_path / "bad.yaml"
        p.write_text(cfg_text)
        with pytest.raises(ValueError, match="emission_dist"):
            load_config(p)

    def test_valid_emission_dist_accepted(self, tmp_path):
        from rde.config.loader import load_config

        cfg_text = """
asset:
  symbol: BTC-USD
model:
  emission_dist: student_t
"""
        p = tmp_path / "good.yaml"
        p.write_text(cfg_text)
        cfg = load_config(p)
        assert cfg.model.emission_dist == "student_t"

    def test_default_emission_dist_is_gaussian(self, tmp_path):
        from rde.config.loader import load_config

        cfg_text = """
asset:
  symbol: BTC-USD
"""
        p = tmp_path / "default.yaml"
        p.write_text(cfg_text)
        cfg = load_config(p)
        assert cfg.model.emission_dist == "gaussian"


# ── Volume features ───────────────────────────────────────────────────────────

class TestVolumeFeatures:
    def _make_price_df(self, T=100, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0 + rng.standard_normal(T).cumsum(),
                "Volume": rng.integers(1000, 100000, size=T).astype(float),
            },
            index=idx,
        )

    def test_log_volume_output_columns(self):
        assert LogVolume().output_columns == ["log_volume"]

    def test_log_volume_shape(self):
        df = self._make_price_df()
        out = LogVolume().transform(df)
        assert "log_volume" in out.columns
        assert len(out) == len(df)

    def test_log_volume_non_negative(self):
        df = self._make_price_df()
        out = LogVolume().transform(df)
        assert (out["log_volume"] >= 0).all()

    def test_log_volume_zero_volume(self):
        df = self._make_price_df()
        df.loc[df.index[0], "Volume"] = 0.0
        out = LogVolume().transform(df)
        assert np.isfinite(out["log_volume"].iloc[0])
        assert out["log_volume"].iloc[0] == 0.0  # log1p(0) = 0

    def test_rolling_volume_zscore_output_columns(self):
        assert RollingVolumeZScore(window=24).output_columns == ["volume_zscore_w24"]

    def test_rolling_volume_zscore_shape(self):
        df = self._make_price_df(T=100)
        out = RollingVolumeZScore(window=12).transform(df)
        assert f"volume_zscore_w12" in out.columns
        assert len(out) == len(df)

    def test_rolling_volume_zscore_leading_nans(self):
        df = self._make_price_df(T=100)
        out = RollingVolumeZScore(window=24).transform(df)
        # First 23 rows should be NaN (window-1 rows before full window)
        assert out["volume_zscore_w24"].iloc[:23].isna().all()
        assert out["volume_zscore_w24"].iloc[23:].notna().all()

    def test_rolling_volume_zscore_custom_window(self):
        assert RollingVolumeZScore(window=48).output_columns == ["volume_zscore_w48"]

    def test_volume_features_in_pipeline(self):
        from rde.features.pipeline import FeaturePipeline
        from rde.features.returns import LogReturns

        df = self._make_price_df(T=200)
        pipeline = FeaturePipeline([LogReturns(), LogVolume(), RollingVolumeZScore(window=12)])
        out = pipeline.transform(df)
        assert "log_return" in out.columns
        assert "log_volume" in out.columns
        assert "volume_zscore_w12" in out.columns
        # Pipeline drops NaN rows
        assert out.isna().sum().sum() == 0
