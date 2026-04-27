"""Visualisation: static matplotlib plots and interactive Plotly plots."""

from rde.viz import interactive, static_plots
from rde.viz.interactive import (
    plot_per_regime_returns as plot_per_regime_returns_interactive,
    plot_price_with_regimes as plot_price_with_regimes_interactive,
    plot_regime_timeline as plot_regime_timeline_interactive,
    plot_transition_heatmap as plot_transition_heatmap_interactive,
)
from rde.viz.static_plots import (
    plot_per_regime_returns,
    plot_price_with_regimes,
    plot_regime_timeline,
    plot_transition_heatmap,
)

__all__ = [
    # static
    "plot_per_regime_returns",
    "plot_price_with_regimes",
    "plot_regime_timeline",
    "plot_transition_heatmap",
    # interactive
    "plot_per_regime_returns_interactive",
    "plot_price_with_regimes_interactive",
    "plot_regime_timeline_interactive",
    "plot_transition_heatmap_interactive",
    # modules
    "static_plots",
    "interactive",
]
