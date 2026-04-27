"""Phase 4 tests: EvaluationConfig, _write_diagnostics, and CLI integration."""
from __future__ import annotations

import io
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rde.cli import _write_diagnostics, main
from rde.config.loader import load_config
from rde.config.schema import (
    AssetConfig,
    Config,
    EvaluationConfig,
    FeatureConfig,
    ModelConfig,
    RunConfig,
    SelectionConfig,
)
from rde.labeling.ranking import rank_states
from rde.models.hmm import FittedModel, train_hmm


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_synthetic_df(n: int = 500) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(0)
    prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.997,
            "Close": prices,
            "Volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )


def _make_fitted_model(n_states: int = 2) -> FittedModel:
    """Train a tiny FittedModel on synthetic data for test fixtures."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    return train_hmm(
        X,
        n_states,
        n_restarts=2,
        n_iter=50,
        seed_base=0,
        feature_names=["log_return", "volatility_w24", "smoothed_return_w12"],
    )


# ── EvaluationConfig defaults ──────────────────────────────────────────────────

class TestEvaluationConfigDefaults:
    def test_run_stability_default_false(self):
        cfg = EvaluationConfig()
        assert cfg.run_stability is False

    def test_run_walk_forward_default_false(self):
        cfg = EvaluationConfig()
        assert cfg.run_walk_forward is False

    def test_stability_n_runs_default(self):
        cfg = EvaluationConfig()
        assert cfg.stability_n_runs == 3

    def test_walk_forward_train_window_default(self):
        cfg = EvaluationConfig()
        assert cfg.walk_forward_train_window == "180d"

    def test_walk_forward_recalibration_default(self):
        cfg = EvaluationConfig()
        assert cfg.walk_forward_recalibration == "monthly"

    def test_walk_forward_n_restarts_default(self):
        cfg = EvaluationConfig()
        assert cfg.walk_forward_n_restarts == 3

    def test_config_evaluation_field_is_evaluation_config(self):
        cfg = Config(
            asset=AssetConfig(symbol="BTC-USD"),
            features=[FeatureConfig(name="LogReturns")],
            model=ModelConfig(),
            selection=SelectionConfig(),
            run=RunConfig(),
        )
        assert isinstance(cfg.evaluation, EvaluationConfig)

    def test_config_evaluation_defaults_via_factory(self):
        c1 = Config(
            asset=AssetConfig(symbol="BTC-USD"),
            features=[FeatureConfig(name="LogReturns")],
            model=ModelConfig(),
            selection=SelectionConfig(),
            run=RunConfig(),
        )
        c2 = Config(
            asset=AssetConfig(symbol="ETH-USD"),
            features=[FeatureConfig(name="LogReturns")],
            model=ModelConfig(),
            selection=SelectionConfig(),
            run=RunConfig(),
        )
        assert c1.evaluation is not c2.evaluation  # separate instances


# ── YAML config loads EvaluationConfig ────────────────────────────────────────

class TestEvaluationConfigFromYaml:
    @pytest.fixture
    def btc_cfg(self):
        return load_config(
            Path(__file__).parents[1] / "configs" / "btc.yaml"
        )

    def test_evaluation_section_loaded(self, btc_cfg):
        assert isinstance(btc_cfg.evaluation, EvaluationConfig)

    def test_run_stability_false_in_yaml(self, btc_cfg):
        assert btc_cfg.evaluation.run_stability is False

    def test_run_walk_forward_false_in_yaml(self, btc_cfg):
        assert btc_cfg.evaluation.run_walk_forward is False

    def test_stability_n_runs_in_yaml(self, btc_cfg):
        assert btc_cfg.evaluation.stability_n_runs == 3

    def test_walk_forward_train_window_in_yaml(self, btc_cfg):
        assert btc_cfg.evaluation.walk_forward_train_window == "180d"

    def test_walk_forward_recalibration_in_yaml(self, btc_cfg):
        assert btc_cfg.evaluation.walk_forward_recalibration == "monthly"

    def test_walk_forward_n_restarts_in_yaml(self, btc_cfg):
        assert btc_cfg.evaluation.walk_forward_n_restarts == 3


# ── _write_diagnostics: basic sections always present ────────────────────────

class TestWriteDiagnosticsBasic:
    @pytest.fixture(scope="class")
    def fixture(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("diag")
        fitted = _make_fitted_model(n_states=2)

        rng = np.random.default_rng(2)
        X = fitted.scaler.transform(rng.normal(size=(200, 3)))
        from hmmlearn.hmm import GaussianHMM
        states = fitted.hmm.predict(X)

        idx = pd.date_range("2023-01-01", periods=200, freq="1h", tz="UTC")
        df_features = pd.DataFrame(
            {
                "log_return": rng.normal(size=200),
                "volatility_w24": np.abs(rng.normal(size=200)),
                "smoothed_return_w12": rng.normal(size=200),
            },
            index=idx,
        )

        from rde.evaluation.transition import stationary_distribution
        stationary = stationary_distribution(fitted.hmm.transmat_)
        labelled = rank_states(fitted)

        path = tmp / "diagnostics.txt"
        _write_diagnostics(
            path,
            symbol="TEST-USD",
            fitted=fitted,
            states=states,
            df_features=df_features,
            labelled_states=labelled,
            scores_df=None,
            stationary=stationary,
            stability_result=None,
            wf_result=None,
            wf_cfg=None,
        )
        return path.read_text()

    def test_file_created(self, fixture):
        assert len(fixture) > 0

    def test_has_symbol(self, fixture):
        assert "TEST-USD" in fixture

    def test_has_transition_matrix_section(self, fixture):
        assert "TRANSITION MATRIX" in fixture

    def test_has_stationary_distribution_section(self, fixture):
        assert "STATIONARY DISTRIBUTION" in fixture

    def test_has_state_means_section(self, fixture):
        assert "STATE MEANS" in fixture

    def test_has_transition_entropy_section(self, fixture):
        assert "TRANSITION ENTROPY" in fixture

    def test_has_persistence_section(self, fixture):
        assert "PERSISTENCE" in fixture

    def test_has_heuristic_labels_section(self, fixture):
        assert "HEURISTIC LABELS" in fixture

    def test_no_stability_section_when_none(self, fixture):
        assert "STABILITY ANALYSIS" not in fixture

    def test_no_walk_forward_section_when_none(self, fixture):
        assert "WALK-FORWARD VALIDATION" not in fixture


# ── _write_diagnostics: stability section ─────────────────────────────────────

class TestWriteDiagnosticsStability:
    @pytest.fixture(scope="class")
    def diag_text(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("diag_stab")
        fitted = _make_fitted_model(n_states=2)
        rng = np.random.default_rng(3)
        X_scaled = fitted.scaler.transform(rng.normal(size=(200, 3)))
        states = fitted.hmm.predict(X_scaled)
        idx = pd.date_range("2023-01-01", periods=200, freq="1h", tz="UTC")
        df_features = pd.DataFrame(
            {
                "log_return": rng.normal(size=200),
                "volatility_w24": np.abs(rng.normal(size=200)),
                "smoothed_return_w12": rng.normal(size=200),
            },
            index=idx,
        )
        from rde.evaluation.transition import stationary_distribution
        stationary = stationary_distribution(fitted.hmm.transmat_)
        labelled = rank_states(fitted)

        # Minimal synthetic stability result
        n_runs = 3
        ari_matrix = np.array([
            [1.0, 0.85, 0.90],
            [0.85, 1.0, 0.88],
            [0.90, 0.88, 1.0],
        ])
        stability_result = {
            "models": [fitted] * n_runs,
            "state_sequences": [states] * n_runs,
            "ari_matrix": ari_matrix,
            "mean_ari": 0.876,
            "min_ari": 0.85,
            "transmat_std": 0.012,
            "means_std": 0.034,
        }

        path = tmp / "diagnostics.txt"
        _write_diagnostics(
            path,
            symbol="TEST-USD",
            fitted=fitted,
            states=states,
            df_features=df_features,
            labelled_states=labelled,
            scores_df=None,
            stationary=stationary,
            stability_result=stability_result,
            wf_result=None,
            wf_cfg=None,
        )
        return path.read_text()

    def test_stability_section_present(self, diag_text):
        assert "STABILITY ANALYSIS" in diag_text

    def test_n_runs_shown(self, diag_text):
        assert "3 independent runs" in diag_text

    def test_mean_ari_shown(self, diag_text):
        assert "mean_ARI" in diag_text
        assert "0.876" in diag_text

    def test_min_ari_shown(self, diag_text):
        assert "min_ARI" in diag_text
        assert "0.85" in diag_text

    def test_transmat_std_shown(self, diag_text):
        assert "transmat_std" in diag_text

    def test_ari_matrix_header(self, diag_text):
        assert "ARI matrix" in diag_text

    def test_ari_matrix_rows(self, diag_text):
        # Each run row should appear (r00, r01, r02)
        assert "r00" in diag_text
        assert "r01" in diag_text
        assert "r02" in diag_text


# ── _write_diagnostics: walk-forward section ──────────────────────────────────

class TestWriteDiagnosticsWalkForward:
    @pytest.fixture(scope="class")
    def diag_text(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("diag_wf")
        fitted = _make_fitted_model(n_states=2)
        rng = np.random.default_rng(4)
        X_scaled = fitted.scaler.transform(rng.normal(size=(200, 3)))
        states = fitted.hmm.predict(X_scaled)
        idx = pd.date_range("2023-01-01", periods=200, freq="1h", tz="UTC")
        df_features = pd.DataFrame(
            {
                "log_return": rng.normal(size=200),
                "volatility_w24": np.abs(rng.normal(size=200)),
                "smoothed_return_w12": rng.normal(size=200),
            },
            index=idx,
        )
        from rde.evaluation.transition import stationary_distribution
        stationary = stationary_distribution(fitted.hmm.transmat_)
        labelled = rank_states(fitted)

        # Synthetic walk-forward result: first 100 bars are warm-up (NaN),
        # last 100 bars have regime labels.
        wf_regime = np.concatenate([np.full(100, np.nan), (states[:100] % 2).astype(float)])
        wf_result = pd.DataFrame(
            {"regime": wf_regime, "regime_proba_0": rng.uniform(size=200)},
            index=idx,
        )

        wf_cfg = EvaluationConfig(
            walk_forward_train_window="90d",
            walk_forward_recalibration="monthly",
        )

        path = tmp / "diagnostics.txt"
        _write_diagnostics(
            path,
            symbol="TEST-USD",
            fitted=fitted,
            states=states,
            df_features=df_features,
            labelled_states=labelled,
            scores_df=None,
            stationary=stationary,
            stability_result=None,
            wf_result=wf_result,
            wf_cfg=wf_cfg,
        )
        return path.read_text()

    def test_walk_forward_section_present(self, diag_text):
        assert "WALK-FORWARD VALIDATION" in diag_text

    def test_recalibration_shown(self, diag_text):
        assert "monthly" in diag_text

    def test_train_window_shown(self, diag_text):
        assert "90d" in diag_text

    def test_oos_bars_count(self, diag_text):
        assert "OOS bars" in diag_text
        assert "100" in diag_text

    def test_warm_up_count(self, diag_text):
        assert "Warm-up" in diag_text

    def test_oos_percentage_shown(self, diag_text):
        assert "50.0%" in diag_text

    def test_first_oos_timestamp(self, diag_text):
        assert "First OOS" in diag_text

    def test_heuristic_disclaimer_in_wf(self, diag_text):
        assert "heuristic" in diag_text.lower()

    def test_regime_distribution_shown(self, diag_text):
        assert "OOS regime distribution" in diag_text


# ── CLI integration (mocked yfinance) ─────────────────────────────────────────

def _synthetic_yfinance_df(n: int = 600) -> pd.DataFrame:
    """Synthetic DataFrame shaped like yfinance output, single-level columns."""
    rng = np.random.default_rng(99)
    prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.997,
            "Close": prices,
            "Volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )


class TestCLIBasicRun:
    """CLI runs end-to-end with mocked yfinance and a fast-config."""

    @pytest.fixture(scope="class")
    def fast_config(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("cfg")
        p = tmp / "test.yaml"
        p.write_text(textwrap.dedent("""\
            asset:
              symbol: TEST-USD
              period: 730d
              interval: 1h

            features:
              - name: LogReturns
              - name: RollingVolatility
                params:
                  window: 24
              - name: SmoothedReturns
                params:
                  window: 12

            model:
              candidate_states: [2, 3]
              covariance_type: full
              n_restarts: 2
              n_iter: 50
              init_strategy: kmeans
              seed_base: 42

            selection:
              criterion: bic

            run:
              output_dir: {output_dir}/results/{{symbol}}/

            evaluation:
              run_stability: false
              stability_n_runs: 2
              run_walk_forward: false
              walk_forward_train_window: 60d
              walk_forward_recalibration: monthly
              walk_forward_n_restarts: 2
        """.format(output_dir=str(tmp))))
        return str(p), tmp

    @pytest.fixture(scope="class")
    def cli_result(self, fast_config):
        config_path, tmp = fast_config
        runner = CliRunner()
        with patch("rde.data.yfinance_source.yf") as mock_yf:
            mock_yf.download.return_value = _synthetic_yfinance_df()
            result = runner.invoke(
                main,
                ["run", "--config", config_path, "--no-cache"],
                catch_exceptions=False,
            )
        return result, tmp

    def test_exit_code_zero(self, cli_result):
        result, _ = cli_result
        assert result.exit_code == 0, result.output

    def test_output_mentions_symbol(self, cli_result):
        result, _ = cli_result
        assert "TEST-USD" in result.output

    def test_output_mentions_n_states(self, cli_result):
        result, _ = cli_result
        assert "n_states" in result.output

    def test_summary_section_present(self, cli_result):
        result, _ = cli_result
        assert "SUMMARY" in result.output

    def test_diagnostics_txt_created(self, cli_result):
        _, tmp = cli_result
        files = list(tmp.rglob("diagnostics.txt"))
        assert len(files) == 1

    def test_regimes_parquet_created(self, cli_result):
        _, tmp = cli_result
        files = list(tmp.rglob("regimes.parquet"))
        assert len(files) == 1

    def test_price_with_regimes_png_created(self, cli_result):
        _, tmp = cli_result
        files = list(tmp.rglob("price_with_regimes.png"))
        assert len(files) == 1

    def test_all_four_plots_created(self, cli_result):
        _, tmp = cli_result
        plot_names = {p.stem for p in tmp.rglob("*.png")}
        assert "price_with_regimes" in plot_names
        assert "transition_heatmap" in plot_names
        assert "regime_timeline" in plot_names
        assert "per_regime_returns" in plot_names

    def test_regimes_parquet_has_regime_column(self, cli_result):
        _, tmp = cli_result
        parquet = next(tmp.rglob("regimes.parquet"))
        df = pd.read_parquet(parquet)
        assert "regime" in df.columns

    def test_regimes_parquet_has_regime_label_column(self, cli_result):
        _, tmp = cli_result
        parquet = next(tmp.rglob("regimes.parquet"))
        df = pd.read_parquet(parquet)
        assert "regime_label" in df.columns

    def test_regimes_parquet_has_proba_columns(self, cli_result):
        _, tmp = cli_result
        parquet = next(tmp.rglob("regimes.parquet"))
        df = pd.read_parquet(parquet)
        proba_cols = [c for c in df.columns if c.startswith("regime_proba_")]
        assert len(proba_cols) > 0


class TestCLINStatesOverride:
    """--n-states flag bypasses AIC/BIC selection."""

    @pytest.fixture(scope="class")
    def fast_config(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("cfg_ns")
        p = tmp / "test.yaml"
        p.write_text(textwrap.dedent("""\
            asset:
              symbol: TEST-USD
              period: 730d
              interval: 1h
            features:
              - name: LogReturns
              - name: RollingVolatility
                params:
                  window: 24
              - name: SmoothedReturns
                params:
                  window: 12
            model:
              candidate_states: [2, 3]
              covariance_type: full
              n_restarts: 2
              n_iter: 50
              seed_base: 42
            selection:
              criterion: bic
            run:
              output_dir: {output_dir}/results/{{symbol}}/
            evaluation:
              run_stability: false
              run_walk_forward: false
        """.format(output_dir=str(tmp))))
        return str(p), tmp

    def test_fixed_n_states_in_output(self, fast_config):
        config_path, tmp = fast_config
        runner = CliRunner()
        with patch("rde.data.yfinance_source.yf") as mock_yf:
            mock_yf.download.return_value = _synthetic_yfinance_df()
            result = runner.invoke(
                main,
                ["run", "--config", config_path, "--no-cache", "--n-states", "3"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert "n_states=3 (fixed)" in result.output


class TestCLIStabilityFlag:
    """--stability flag triggers stability analysis."""

    @pytest.fixture(scope="class")
    def config_and_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("cfg_stab")
        p = tmp / "test.yaml"
        p.write_text(textwrap.dedent("""\
            asset:
              symbol: TEST-USD
              period: 730d
              interval: 1h
            features:
              - name: LogReturns
              - name: RollingVolatility
                params:
                  window: 24
              - name: SmoothedReturns
                params:
                  window: 12
            model:
              candidate_states: [2, 3]
              covariance_type: full
              n_restarts: 2
              n_iter: 50
              seed_base: 42
            selection:
              criterion: bic
            run:
              output_dir: {output_dir}/results/{{symbol}}/
            evaluation:
              run_stability: false
              stability_n_runs: 2
              run_walk_forward: false
        """.format(output_dir=str(tmp))))
        runner = CliRunner()
        with patch("rde.data.yfinance_source.yf") as mock_yf:
            mock_yf.download.return_value = _synthetic_yfinance_df()
            result = runner.invoke(
                main,
                ["run", "--config", str(p), "--no-cache", "--stability",
                 "--stability-runs", "2"],
                catch_exceptions=False,
            )
        return result, tmp

    def test_exit_code_zero(self, config_and_result):
        result, _ = config_and_result
        assert result.exit_code == 0, result.output

    def test_stability_output_in_summary(self, config_and_result):
        result, _ = config_and_result
        assert "Stability" in result.output
        assert "mean_ARI" in result.output

    def test_diagnostics_has_stability_section(self, config_and_result):
        _, tmp = config_and_result
        diag = next(tmp.rglob("diagnostics.txt")).read_text()
        assert "STABILITY ANALYSIS" in diag

    def test_diagnostics_has_ari_matrix(self, config_and_result):
        _, tmp = config_and_result
        diag = next(tmp.rglob("diagnostics.txt")).read_text()
        assert "ARI matrix" in diag


class TestCLIWalkForwardFlag:
    """--walk-forward flag triggers walk-forward validation."""

    @pytest.fixture(scope="class")
    def config_and_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("cfg_wf")
        p = tmp / "test.yaml"
        # Use a large enough dataset: need train_window + at least 1 recalibration
        p.write_text(textwrap.dedent("""\
            asset:
              symbol: TEST-USD
              period: 730d
              interval: 1h
            features:
              - name: LogReturns
              - name: RollingVolatility
                params:
                  window: 24
              - name: SmoothedReturns
                params:
                  window: 12
            model:
              candidate_states: [2, 3]
              covariance_type: full
              n_restarts: 2
              n_iter: 50
              seed_base: 42
            selection:
              criterion: bic
            run:
              output_dir: {output_dir}/results/{{symbol}}/
            evaluation:
              run_stability: false
              run_walk_forward: false
              walk_forward_train_window: 30d
              walk_forward_recalibration: monthly
              walk_forward_n_restarts: 1
        """.format(output_dir=str(tmp))))
        runner = CliRunner()
        # Need a large enough df: 30d train + several months of data = ~2000+ rows
        rng = np.random.default_rng(77)
        n = 3000
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
        idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.002,
                "Low": prices * 0.997,
                "Close": prices,
                "Volume": rng.integers(1000, 5000, n).astype(float),
            },
            index=idx,
        )
        with patch("rde.data.yfinance_source.yf") as mock_yf:
            mock_yf.download.return_value = df
            result = runner.invoke(
                main,
                ["run", "--config", str(p), "--no-cache", "--walk-forward"],
                catch_exceptions=False,
            )
        return result, tmp

    def test_exit_code_zero(self, config_and_result):
        result, _ = config_and_result
        assert result.exit_code == 0, result.output

    def test_walk_forward_in_summary(self, config_and_result):
        result, _ = config_and_result
        assert "Walk-forward" in result.output
        assert "OOS bars" in result.output

    def test_walk_forward_parquet_created(self, config_and_result):
        _, tmp = config_and_result
        files = list(tmp.rglob("walk_forward_regimes.parquet"))
        assert len(files) == 1

    def test_walk_forward_parquet_has_regime_column(self, config_and_result):
        _, tmp = config_and_result
        parquet = next(tmp.rglob("walk_forward_regimes.parquet"))
        df = pd.read_parquet(parquet)
        assert "regime" in df.columns

    def test_walk_forward_parquet_has_nan_warmup(self, config_and_result):
        _, tmp = config_and_result
        parquet = next(tmp.rglob("walk_forward_regimes.parquet"))
        df = pd.read_parquet(parquet)
        # Some rows should be NaN (warm-up period with no fitted model)
        assert df["regime"].isna().any()

    def test_walk_forward_parquet_has_oos_labels(self, config_and_result):
        _, tmp = config_and_result
        parquet = next(tmp.rglob("walk_forward_regimes.parquet"))
        df = pd.read_parquet(parquet)
        assert df["regime"].notna().any()

    def test_diagnostics_has_walk_forward_section(self, config_and_result):
        _, tmp = config_and_result
        diag = next(tmp.rglob("diagnostics.txt")).read_text()
        assert "WALK-FORWARD VALIDATION" in diag

    def test_diagnostics_walk_forward_oos_coverage(self, config_and_result):
        _, tmp = config_and_result
        diag = next(tmp.rglob("diagnostics.txt")).read_text()
        assert "OOS bars" in diag


class TestCLIInteractiveFlag:
    """--interactive flag saves HTML plots alongside PNGs."""

    def test_html_plots_created(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(textwrap.dedent("""\
            asset:
              symbol: TEST-USD
              period: 730d
              interval: 1h
            features:
              - name: LogReturns
              - name: RollingVolatility
                params:
                  window: 24
              - name: SmoothedReturns
                params:
                  window: 12
            model:
              candidate_states: [2, 3]
              covariance_type: full
              n_restarts: 2
              n_iter: 50
              seed_base: 42
            selection:
              criterion: bic
            run:
              output_dir: {output_dir}/results/{{symbol}}/
            evaluation:
              run_stability: false
              run_walk_forward: false
        """.format(output_dir=str(tmp_path))))
        runner = CliRunner()
        with patch("rde.data.yfinance_source.yf") as mock_yf:
            mock_yf.download.return_value = _synthetic_yfinance_df()
            result = runner.invoke(
                main,
                ["run", "--config", str(p), "--no-cache", "--interactive"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        html_files = list(tmp_path.rglob("*.html"))
        assert len(html_files) == 4  # one per plot


class TestCLIConfigValidation:
    """CLI surfaces config validation errors clearly."""

    def test_invalid_criterion_rejected(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(textwrap.dedent("""\
            asset:
              symbol: BTC-USD
            features:
              - name: LogReturns
            model:
              candidate_states: [2, 3]
            selection:
              criterion: rmse
            run:
              output_dir: /tmp/
        """))
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", str(p)])
        assert result.exit_code != 0 or "rmse" in (result.output + str(result.exception)).lower()

    def test_missing_config_file_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code != 0
