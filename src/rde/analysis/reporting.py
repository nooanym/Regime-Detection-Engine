"""Reporting utilities for Phase 31 analysis pipeline.

Serialises an :class:`AnalysisReport` (produced by
``rde.analysis.pipeline``) to JSON, Markdown, and disk.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from rde.analysis.pipeline import AnalysisReport, AnalysisConfig  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitise(obj: Any) -> Any:
    """Recursively replace non-finite floats with None and numpy scalars with float."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    # numpy scalar duck-typing
    try:
        f = float(obj)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    except (TypeError, ValueError):
        pass
    return obj


def _bullet_list(d: dict) -> str:
    """Render a dict as a Markdown bullet list.

    Parameters
    ----------
    d:
        Mapping of key → value pairs.

    Returns
    -------
    str
        Newline-separated bullet lines.
    """
    lines = []
    for key, value in d.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def report_to_dict(report: AnalysisReport) -> dict:
    """Convert an :class:`AnalysisReport` to a plain Python dict.

    All fields in ``AnalysisReport`` are already plain Python types (the
    pipeline is responsible for normalising numpy/pandas values before
    building the report), so ``dataclasses.asdict`` is sufficient.

    Parameters
    ----------
    report:
        The analysis report to serialise.

    Returns
    -------
    dict
        A fully JSON-serialisable dict (assuming the pipeline has done its
        normalisation job).
    """
    return dataclasses.asdict(report)


def report_to_json(report: AnalysisReport, path: Path | None = None) -> str:
    """Serialise an :class:`AnalysisReport` to a JSON string.

    Parameters
    ----------
    report:
        The report to serialise.
    path:
        If given, the JSON string is also written to this file.  Parent
        directories are created automatically.

    Returns
    -------
    str
        The JSON-encoded report.
    """
    json_str = json.dumps(_sanitise(report_to_dict(report)), indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str, encoding="utf-8")
        logger.info("Wrote analysis report JSON to %s", path)
    return json_str


def report_to_markdown(report: AnalysisReport) -> str:
    """Render an :class:`AnalysisReport` as a Markdown document.

    Parameters
    ----------
    report:
        The report to render.

    Returns
    -------
    str
        A well-formatted Markdown string with one section per analysis
        module.  Modules that were skipped (``None``) are noted explicitly.
    """
    feature_str = ", ".join(report.feature_names)

    lines: list[str] = [
        f"# Regime Analysis Report — {report.symbol}",
        "",
        f"**Generated:** {report.run_timestamp}  ",
        f"**Observations:** {report.n_obs}  ",
        f"**Regimes:** {report.n_states}  ",
        f"**Features:** {feature_str}",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Helper: append a section with either a bullet list or a skip note
    # ------------------------------------------------------------------
    def _section(heading: str, data: dict | None, skip_note: str = "_Module not run._") -> None:
        lines.append(f"## {heading}")
        lines.append("")
        if data is None:
            lines.append(skip_note)
        else:
            lines.append(_bullet_list(data))
        lines.append("")

    # --- Execution & Market Impact ---
    _section("Execution & Market Impact", report.execution)

    # --- Factor Analysis ---
    _section("Factor Analysis", report.factor_result)

    # --- Cointegration & Spread Analysis ---
    _section(
        "Cointegration & Spread Analysis",
        report.cointegration,
        skip_note="_Module not run or skipped (requires d ≥ 2)._",
    )

    # --- Portfolio Optimisation ---
    _section(
        "Portfolio Optimisation",
        report.portfolio,
        skip_note="_Module not run or skipped (requires d ≥ 2)._",
    )

    # --- Signal Filtering ---
    _section("Signal Filtering", report.signal_filtering)

    # --- Regime Transition Analysis ---
    lines.append("## Regime Transition Analysis")
    lines.append("")
    if report.transition is None:
        lines.append("_Module not run._")
        lines.append("")
    else:
        # Top-level scalars / lists as bullet list, then a table for durations
        transition_display = {
            k: v
            for k, v in report.transition.items()
            if k != "expected_durations"
        }
        if transition_display:
            lines.append(_bullet_list(transition_display))
            lines.append("")

        durations = report.transition.get("expected_durations")
        if durations is not None:
            lines.append("### Expected Durations per Regime")
            lines.append("")
            lines.append("| Regime | Expected Duration |")
            lines.append("|--------|-------------------|")
            if isinstance(durations, (list, tuple)):
                for i, dur in enumerate(durations):
                    lines.append(f"| {i} | {dur} |")
            elif isinstance(durations, dict):
                for regime, dur in durations.items():
                    lines.append(f"| {regime} | {dur} |")
            lines.append("")

    # --- Tail Risk (EVT) ---
    lines.append("## Tail Risk (EVT)")
    lines.append("")
    if report.tail_risk is None:
        lines.append("_Module not run._")
        lines.append("")
    else:
        regime_tails = report.tail_risk.get("regime_tails")
        if regime_tails:
            lines.append("| regime | xi | sigma | VaR 95% | ES 95% |")
            lines.append("|--------|----|-------|---------|--------|")
            for row in regime_tails:
                regime = row.get("regime", "")
                xi = row.get("xi", "")
                sigma = row.get("sigma", "")
                var95 = row.get("VaR 95%", row.get("var_95", row.get("var95", "")))
                es95 = row.get("ES 95%", row.get("es_95", row.get("es95", "")))
                lines.append(f"| {regime} | {xi} | {sigma} | {var95} | {es95} |")
            lines.append("")
        # Remaining top-level keys (excluding regime_tails) as bullets
        other = {k: v for k, v in report.tail_risk.items() if k != "regime_tails"}
        if other:
            lines.append(_bullet_list(other))
            lines.append("")

    # --- Backtest ---
    lines.append("## Backtest (Long Regime 0 / Flat Otherwise)")
    lines.append("")
    if report.backtest is None:
        lines.append("_Module not run._")
        lines.append("")
    else:
        tearsheet = report.backtest.get("tearsheet")
        if tearsheet:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in tearsheet.items():
                lines.append(f"| {metric} | {value} |")
            lines.append("")
        other = {k: v for k, v in report.backtest.items() if k != "tearsheet"}
        if other:
            lines.append(_bullet_list(other))
            lines.append("")

    # --- Correlation Structure ---
    lines.append("## Correlation Structure")
    lines.append("")
    if report.correlation is None:
        lines.append("_Module not run._")
        lines.append("")
    else:
        stability = report.correlation.get("stability")
        if stability is not None:
            lines.append(f"- **stability**: {stability}")
        lines.append(
            "- **global_correlation_matrix**: see full JSON report for the complete matrix."
        )
        other = {
            k: v
            for k, v in report.correlation.items()
            if k not in ("stability", "global_correlation_matrix")
        }
        if other:
            lines.append(_bullet_list(other))
        lines.append("")

    lines.append("---")
    lines.append("*Generated by Regime Detection Engine — Phase 31*")
    lines.append("")

    return "\n".join(lines)


def save_report(
    report: AnalysisReport,
    output_dir: Path,
    symbol: str | None = None,
) -> None:
    """Persist an :class:`AnalysisReport` as both JSON and Markdown.

    Parameters
    ----------
    report:
        The analysis report to save.
    output_dir:
        Directory into which the files are written.  Created automatically
        if it does not exist.
    symbol:
        Optional symbol override used only for naming purposes.  Falls back
        to ``report.symbol`` when not provided.

    Notes
    -----
    The symbol parameter is accepted for API consistency but is not currently
    used in the output filenames; filenames are fixed as
    ``analysis_report.json`` and ``analysis.md``.
    """
    sym = symbol or report.symbol  # noqa: F841 — kept for potential future use

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analysis_report.json"
    report_to_json(report, path=json_path)
    logger.info("Wrote analysis report JSON to %s", json_path)

    md_path = output_dir / "analysis.md"
    md_content = report_to_markdown(report)
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Wrote analysis report Markdown to %s", md_path)
