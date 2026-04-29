"""Tests for rde.analysis.reporting — Phase 31 serialisation layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rde.analysis.pipeline import AnalysisReport
from rde.analysis.reporting import (
    report_to_dict,
    report_to_json,
    report_to_markdown,
    save_report,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_report(**kwargs) -> AnalysisReport:
    defaults = dict(
        symbol="TEST",
        run_timestamp="2026-04-28T12:00:00Z",
        n_obs=100,
        n_states=2,
        n_features=3,
        feature_names=["log_return", "volatility", "smoothed"],
        execution={"n_regimes": 2, "slippage_by_regime": {"0": 0.001, "1": 0.002}, "twap_example": [50.0, 50.0]},
        factor_result={"n_factors_per_regime": [2, 2], "explained_variance_ratios": [[0.7, 0.3], [0.6, 0.4]]},
        cointegration={"pair": "feat_0_vs_feat_1", "regime_half_lives": {"0": 10.5, "1": 8.2}, "global_half_life": 9.1},
        portfolio={"blended_weights": [0.5, 0.5, 0.0], "regime_weights": [[0.6, 0.4, 0.0], [0.3, 0.7, 0.0]]},
        signal_filtering={"ema_mean": 0.001, "ema_std": 0.01, "hp_mean": 0.0, "kalman_mean": 0.001},
        transition={"expected_durations": [4.0, 5.0], "5step_transmat": [[0.5, 0.5], [0.5, 0.5]], "mean_transition_entropy": 0.693},
        tail_risk={"regime_tails": [
            {"regime": 0, "xi": 0.1, "sigma": 0.005, "var_95": 0.02, "es_95": 0.03},
            {"regime": 1, "xi": 0.2, "sigma": 0.008, "var_95": 0.04, "es_95": None},
        ]},
        backtest={"tearsheet": {"total_return": 0.05, "sharpe": 0.8, "max_drawdown": 0.1, "ann_vol": 0.15}, "attribution": []},
        correlation={"stability": 0.15, "global_corr": [[1.0, 0.3], [0.3, 1.0]], "regime_corrs": [], "tail_dependence": []},
    )
    defaults.update(kwargs)
    return AnalysisReport(**defaults)


# ---------------------------------------------------------------------------
# report_to_dict
# ---------------------------------------------------------------------------


class TestReportToDict:
    def test_returns_dict(self) -> None:
        d = report_to_dict(_make_report())
        assert isinstance(d, dict)

    def test_top_level_keys(self) -> None:
        d = report_to_dict(_make_report())
        for key in ["symbol", "run_timestamp", "n_obs", "n_states", "n_features",
                    "feature_names", "execution", "factor_result", "cointegration",
                    "portfolio", "signal_filtering", "transition", "tail_risk",
                    "backtest", "correlation"]:
            assert key in d, f"Missing key: {key}"

    def test_none_fields_preserved(self) -> None:
        d = report_to_dict(_make_report(execution=None, cointegration=None))
        assert d["execution"] is None
        assert d["cointegration"] is None

    def test_symbol_matches(self) -> None:
        d = report_to_dict(_make_report(symbol="BTC-USD"))
        assert d["symbol"] == "BTC-USD"

    def test_feature_names_preserved(self) -> None:
        names = ["a", "b", "c"]
        d = report_to_dict(_make_report(feature_names=names))
        assert d["feature_names"] == names


# ---------------------------------------------------------------------------
# report_to_json
# ---------------------------------------------------------------------------


class TestReportToJson:
    def test_returns_string(self) -> None:
        j = report_to_json(_make_report())
        assert isinstance(j, str)

    def test_valid_json(self) -> None:
        parsed = json.loads(report_to_json(_make_report()))
        assert isinstance(parsed, dict)

    def test_json_has_symbol(self) -> None:
        parsed = json.loads(report_to_json(_make_report(symbol="ETH-USD")))
        assert parsed["symbol"] == "ETH-USD"

    def test_none_fields_are_json_null(self) -> None:
        parsed = json.loads(report_to_json(_make_report(execution=None, portfolio=None)))
        assert parsed["execution"] is None
        assert parsed["portfolio"] is None

    def test_writes_to_path(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "report.json"
        report_to_json(_make_report(), path=out)
        assert out.exists()
        assert "symbol" in out.read_text()

    def test_handles_inf_nan(self) -> None:
        r = _make_report(
            tail_risk={"regime_tails": [{"xi": float("inf"), "es_95": float("nan")}]}
        )
        parsed = json.loads(report_to_json(r))
        tail = parsed["tail_risk"]["regime_tails"][0]
        assert tail["xi"] is None
        assert tail["es_95"] is None

    def test_n_obs_correct(self) -> None:
        parsed = json.loads(report_to_json(_make_report(n_obs=9999)))
        assert parsed["n_obs"] == 9999


# ---------------------------------------------------------------------------
# report_to_markdown
# ---------------------------------------------------------------------------


class TestReportToMarkdown:
    def test_returns_string(self) -> None:
        assert isinstance(report_to_markdown(_make_report()), str)

    def test_contains_symbol(self) -> None:
        md = report_to_markdown(_make_report(symbol="BTC-USD"))
        assert "BTC-USD" in md

    def test_contains_all_section_headers(self) -> None:
        md = report_to_markdown(_make_report())
        for section in ["Execution", "Factor", "Cointegration", "Portfolio",
                        "Signal", "Transition", "Tail Risk", "Backtest", "Correlation"]:
            assert section in md, f"Missing section: {section}"

    def test_none_module_shows_not_run(self) -> None:
        md = report_to_markdown(_make_report(execution=None, cointegration=None))
        assert "not run" in md.lower()

    def test_includes_n_obs(self) -> None:
        md = report_to_markdown(_make_report(n_obs=12345))
        assert "12345" in md

    def test_tail_risk_shows_var(self) -> None:
        md = report_to_markdown(_make_report())
        assert "0.02" in md or "var" in md.lower()

    def test_markdown_has_header(self) -> None:
        md = report_to_markdown(_make_report())
        assert md.startswith("#")

    def test_footer_present(self) -> None:
        md = report_to_markdown(_make_report())
        assert "Regime Detection Engine" in md


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_creates_files(self, tmp_path: Path) -> None:
        save_report(_make_report(), tmp_path)
        assert (tmp_path / "analysis_report.json").exists()
        assert (tmp_path / "analysis.md").exists()

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "new" / "nested" / "dir"
        save_report(_make_report(), out)
        assert (out / "analysis_report.json").exists()

    def test_json_file_is_valid(self, tmp_path: Path) -> None:
        save_report(_make_report(), tmp_path)
        content = json.loads((tmp_path / "analysis_report.json").read_text())
        assert content["symbol"] == "TEST"

    def test_md_file_has_headers(self, tmp_path: Path) -> None:
        save_report(_make_report(), tmp_path)
        md = (tmp_path / "analysis.md").read_text()
        assert "#" in md

    def test_symbol_override_does_not_crash(self, tmp_path: Path) -> None:
        save_report(_make_report(symbol="DEFAULT"), tmp_path, symbol="OVERRIDE")
        assert (tmp_path / "analysis_report.json").exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        r = _make_report()
        save_report(r, tmp_path)
        save_report(r, tmp_path)  # second call should overwrite cleanly
        assert (tmp_path / "analysis_report.json").exists()
