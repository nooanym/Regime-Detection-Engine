import sys
from pathlib import Path
import pytest

_APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(_APP_DIR))

try:
    from panels_analysis import (
        _build_slippage_figure,
        _build_factor_figure,
        _build_tail_risk_figure,
        _build_transition_duration_figure,
        _build_transition_entropy_figure,
        _build_corr_heatmap_figure,
        _build_cointegration_figure,
        _build_portfolio_figure,
    )
    import plotly.graph_objects as go
    _PANELS_AVAILABLE = True
except ImportError:
    _PANELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PANELS_AVAILABLE, reason="panels_analysis.py not found"
)


# ---------------------------------------------------------------------------
# Synthetic test data helpers
# ---------------------------------------------------------------------------

def _slippage_data():
    return {"slippage_by_regime": {"0": 0.001, "1": 0.002, "2": 0.0005}}


def _factor_data():
    return {
        "n_factors_per_regime": [3, 3],
        "explained_variance_ratios": [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
    }


def _tail_data():
    return [
        {"regime": 0, "xi": 0.1, "sigma": 0.005, "var_95": 0.02, "es_95": 0.03},
        {"regime": 1, "xi": None, "sigma": None, "var_95": 0.04, "es_95": None},
    ]


def _corr_matrix(d: int = 3):
    import numpy as np
    C = np.eye(d)
    C[0, 1] = C[1, 0] = 0.3
    return C.tolist()


def _feature_names(d: int = 3):
    return [f"feat_{i}" for i in range(d)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildSlippageFigure:
    def test_returns_figure(self):
        fig = _build_slippage_figure(_slippage_data(), n_regimes=3)
        assert isinstance(fig, go.Figure)

    def test_has_one_trace(self):
        fig = _build_slippage_figure(_slippage_data(), n_regimes=3)
        assert len(fig.data) >= 1

    def test_empty_slippage(self):
        fig = _build_slippage_figure({"slippage_by_regime": {}}, n_regimes=0)
        assert isinstance(fig, go.Figure)


class TestBuildFactorFigure:
    def test_returns_figure(self):
        fig = _build_factor_figure(_factor_data(), _feature_names())
        assert isinstance(fig, go.Figure)

    def test_n_traces_equals_n_regimes(self):
        K = 3
        data = {
            "n_factors_per_regime": [3] * K,
            "explained_variance_ratios": [[0.7, 0.2, 0.1]] * K,
        }
        fig = _build_factor_figure(data, _feature_names())
        assert len(fig.data) == K

    def test_empty_factor_result(self):
        fig = _build_factor_figure(
            {"n_factors_per_regime": [], "explained_variance_ratios": []}, []
        )
        assert isinstance(fig, go.Figure)


class TestBuildTailRiskFigure:
    def test_returns_figure_var(self):
        fig = _build_tail_risk_figure(_tail_data(), metric="var_95")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_es(self):
        fig = _build_tail_risk_figure(_tail_data(), metric="es_95")
        assert isinstance(fig, go.Figure)

    def test_skips_none_values(self):
        # es_95 for regime 1 is None — should not raise
        fig = _build_tail_risk_figure(_tail_data(), metric="es_95")
        assert isinstance(fig, go.Figure)

    def test_empty_list(self):
        fig = _build_tail_risk_figure([], metric="var_95")
        assert isinstance(fig, go.Figure)


class TestBuildTransitionDurationFigure:
    def test_returns_figure(self):
        fig = _build_transition_duration_figure([4.0, 8.0, 5.5], n_states=3)
        assert isinstance(fig, go.Figure)

    def test_single_regime(self):
        fig = _build_transition_duration_figure([10.0], n_states=1)
        assert isinstance(fig, go.Figure)


class TestBuildTransitionEntropyFigure:
    def test_returns_figure(self):
        import math
        fig = _build_transition_entropy_figure(math.log(2) * 0.8, n_states=2)
        assert isinstance(fig, go.Figure)

    def test_zero_entropy(self):
        fig = _build_transition_entropy_figure(0.0, n_states=3)
        assert isinstance(fig, go.Figure)


class TestBuildCorrHeatmapFigure:
    def test_returns_figure(self):
        fig = _build_corr_heatmap_figure(_corr_matrix(3), "Test Corr", _feature_names(3))
        assert isinstance(fig, go.Figure)

    def test_title_set(self):
        fig = _build_corr_heatmap_figure(_corr_matrix(2), "My Title", _feature_names(2))
        assert "My Title" in str(fig.layout.title)

    def test_2x2_matrix(self):
        fig = _build_corr_heatmap_figure([[1.0, 0.5], [0.5, 1.0]], "2x2", ["a", "b"])
        assert isinstance(fig, go.Figure)


class TestBuildCointegrationFigure:
    def test_returns_figure(self):
        fig = _build_cointegration_figure(
            {"0": 10.5, "1": None, "2": 8.2}, global_half_life=9.1
        )
        assert isinstance(fig, go.Figure)

    def test_all_none_half_lives(self):
        fig = _build_cointegration_figure(
            {"0": None, "1": None}, global_half_life=None
        )
        assert isinstance(fig, go.Figure)

    def test_global_only(self):
        fig = _build_cointegration_figure({}, global_half_life=5.0)
        assert isinstance(fig, go.Figure)


class TestBuildPortfolioFigure:
    def test_returns_figure(self):
        fig = _build_portfolio_figure(
            [0.4, 0.35, 0.25],
            [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2]],
            _feature_names(3),
        )
        assert isinstance(fig, go.Figure)

    def test_n_traces_equals_1_plus_n_regimes(self):
        K = 3
        fig = _build_portfolio_figure(
            [0.33, 0.33, 0.34],
            [[0.5, 0.3, 0.2]] * K,
            _feature_names(3),
        )
        # 1 blended + K regime traces
        assert len(fig.data) == K + 1

    def test_empty_weights(self):
        fig = _build_portfolio_figure([], [], [])
        assert isinstance(fig, go.Figure)
