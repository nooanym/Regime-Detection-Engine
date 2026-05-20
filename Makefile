.PHONY: test test-all test-slow lint format validate-daily validate-hourly validate-rtmv backtest-rtmv backtest-rtmv-joint run-btc dashboard clean help monitor carry-backtest carry-signal carry-phase69 carry-phase75 carry-live-filtered carry-phase76 carry-phase78 carry-live-full multistrat-signal multistrat-backtest letf-pair-signal letf-pair-monitor live-rtmv upload-multistrat upload-rtmv

# ── Test targets ──────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -x --tb=short -q \
		--ignore=tests/integration \
		--ignore=tests/numerics \
		--ignore=tests/regression

test-all:
	uv run pytest tests/ -x --tb=short -q

test-slow:
	uv run pytest tests/integration tests/numerics tests/regression -x --tb=short -q -m slow

test-edge:
	uv run pytest tests/edge_validation/ -x --tb=short -q

# ── Lint / format ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

lint-fix:
	uv run ruff check --fix src/ tests/ scripts/

# ── Validation pipeline ───────────────────────────────────────────────────────

validate-daily:
	uv run python scripts/run_phase37_validation.py \
		--config configs/btc_daily.yaml \
		--n-states 8 \
		--ann-factor 252 \
		--train-bars 750 \
		--test-bars 125 \
		--embargo-bars 5 \
		--period-robustness-window-bars 504 \
		--period-robustness-step-bars 126

validate-hourly:
	uv run python scripts/run_phase37_validation.py \
		--config configs/btc.yaml \
		--n-states 8 \
		--ann-factor 8760 \
		--train-bars 4000 \
		--test-bars 500 \
		--embargo-bars 24 \
		--period-robustness-window-bars 4383 \
		--period-robustness-step-bars 730

validate-daily-fast:
	uv run python scripts/run_phase37_validation.py \
		--config configs/btc_daily.yaml \
		--n-states 8 \
		--ann-factor 252 \
		--train-bars 750 \
		--test-bars 125 \
		--embargo-bars 5 \
		--period-robustness-window-bars 504 \
		--period-robustness-step-bars 126 \
		--n-restarts 1 \
		--n-sims 20 \
		--n-permutations 3

validate-rtmv:
	uv run python scripts/run_rtmv_cv_validation.py \
		--assets SPY,GLD,TLT,IEF \
		--n-states 3 \
		--n-restarts 3 \
		--lookback-bars 504 \
		--lambda-tilt 0.30 \
		--n-folds 5 \
		--n-shuffle 100

validate-rtmv-fast:
	uv run python scripts/run_rtmv_cv_validation.py \
		--assets SPY,GLD,TLT,IEF \
		--n-states 3 \
		--n-restarts 1 \
		--lookback-bars 504 \
		--lambda-tilt 0.30 \
		--n-folds 5 \
		--n-shuffle 10

# ── Phase 48 live rebalancer ─────────────────────────────────────────────────

backtest-rtmv:
	uv run python scripts/run_rtmv_live.py \
		--assets SPY,GLD,TLT,IEF \
		--lambda-tilt 0.05 \
		--n-states 3 \
		--n-restarts 3 \
		--lookback-bars 504 \
		--rebalance-bars 21 \
		--output-dir results/rtmv_live \
		--mode backtest

live-rtmv:
	uv run python scripts/run_rtmv_live.py \
		--assets SPY,GLD,SHY,IEF,TLT \
		--lambda-tilt 0.05 \
		--lambda-by-state-rank 0.02,0.05,0.10 \
		--proxy-asset SPY \
		--n-states 3 \
		--n-restarts 3 \
		--momentum-tilt-scale 0.03 \
		--output-dir results/rtmv_live \
		--mode live \
		--poll-interval 3600

# 5-asset backtest (Phase 52a + Phase 57 momentum: SPY/GLD/SHY/IEF/TLT + spy_rank_bull + mom_03)
backtest-rtmv-5asset:
	uv run python scripts/run_rtmv_live.py \
		--assets SPY,GLD,SHY,IEF,TLT \
		--lambda-tilt 0.05 \
		--lambda-by-state-rank 0.02,0.05,0.10 \
		--proxy-asset SPY \
		--n-states 3 \
		--n-restarts 3 \
		--lookback-bars 504 \
		--rebalance-bars 21 \
		--momentum-tilt-scale 0.03 \
		--output-dir results/rtmv_live_5asset \
		--mode backtest

# Phase 54: joint 5D HMM backtest (SPY/GLD/SHY/IEF/TLT, single joint model)
backtest-rtmv-joint:
	uv run python scripts/run_rtmv_live.py \
		--assets SPY,GLD,SHY,IEF,TLT \
		--joint-hmm \
		--joint-hmm-cov-type diag \
		--lambda-tilt 0.05 \
		--lambda-by-state-rank 0.02,0.05,0.10 \
		--proxy-asset SPY \
		--n-states 3 \
		--n-restarts 3 \
		--lookback-bars 504 \
		--rebalance-bars 21 \
		--output-dir results/phase54_joint_hmm \
		--mode backtest

monitor:
	uv run python scripts/monitor_live_performance.py

# ── Funding carry strategy ────────────────────────────────────────────────────

carry-backtest:
	uv run python scripts/funding_carry_backtest.py \
		--symbols BTCUSDT ETHUSDT \
		--start-year 2020

carry-signal:
	uv run python scripts/funding_carry_backtest.py --live

# Phase 78 ROBUST: --momentum-filter --spot-spread-weight confirmed via half-period (A=22.60, B=28.86)
# and permutation test (p=0.0000). Near-zero MDD is genuine, not path-specific.
carry-live:
	uv run python scripts/run_carry_live.py --mode live --momentum-filter --spot-spread-weight

# Phase 78 GO: full Phase 75+76 live carry (explicit alias)
carry-live-full:
	uv run python scripts/run_carry_live.py --mode live --momentum-filter --spot-spread-weight --regime-scale bull-only

carry-live-backtest:
	uv run python scripts/run_carry_live.py --mode backtest --start-year 2020

# Phase 55: regime-scaled carry (bull-only 1x/1x/1.5x — best variant)
carry-phase55:
	uv run python scripts/run_phase55_regime_carry.py

# Phase 60: carry-weighted BTC+ETH allocation (STRONG GO: +0.630 Sharpe)
carry-phase60:
	uv run python scripts/run_phase60_carry_weight.py

# Phase 62: carry-weighted + bull-only regime scale (definitive carry config)
carry-phase62:
	uv run python scripts/run_phase62_regime_carry_weight.py

# Phase 69: regime-adaptive carry entry thresholds
carry-phase69:
	uv run python scripts/run_phase69_adaptive_threshold.py

# Phase 75: intraday UTC window filtering + lag-1 momentum filter for carry (GO: Sharpe 18.573)
carry-phase75:
	uv run python scripts/run_phase75_intraday_carry.py

# Phase 75 momentum filter deployed: live carry with lag-1 filter enabled
carry-live-filtered:
	uv run python scripts/run_carry_live.py --mode live --momentum-filter

# Phase 76: ETH-BTC carry spread signal for relative weighting
carry-phase76:
	uv run python scripts/run_phase76_carry_spread.py

# Phase 78: carry MDD robustness stress test (ROBUST: both halves >15.0, p=0.0000)
carry-phase78:
	uv run python scripts/run_phase78_carry_stress.py

# Phase 73: SOL perpetual funding carry re-test (2024-2026 data)
carry-phase73:
	uv run python scripts/run_phase73_sol_carry.py

# Phase 61: full multi-alpha portfolio synthesis
portfolio-phase61:
	uv run python scripts/run_phase61_full_portfolio.py

# Phase 65: delta-neutral LETF pair (SHORT TQQQ + LONG 3x QQQ, STRONG GO: Sharpe 4.887)
letf-pair:
	uv run python scripts/run_phase65_letf_pair.py

# Phase 66: portfolio frontier (carry + LETF pair + RTMV)
portfolio-frontier:
	uv run python scripts/run_phase66_portfolio_frontier.py

# Phase 67: LETF pair live paper-trading monitor
letf-pair-signal:
	uv run python scripts/run_letf_pair_live.py --mode signal

letf-pair-monitor:
	uv run python scripts/run_letf_pair_live.py --mode monitor

# Phase 79 (deployment hardened): unified multi-strategy live executor
# 80% carry (Phase 75+76+78: momentum_filter + spot_spread_weight, ROBUST p=0.0000)
# + 15% LETF 3-pair (Phase 70: TQQQ/QQQ + SOXL/SOXX + UPRO/SPY)
# + 5% RTMV (Phase 57+63: spy_rank_bull + mom_tilt=0.03, CV STRONG GO)
# Carry Sharpe:    18.13 (Phase 60) → 18.57 (Phase 75) → 19.73 (Phase 76) — Phase 78 ROBUST
# Combined Sharpe: 11.0483 (Phase 68) → 11.5636 (Phase 71) → 11.9359 (Phase 77), MDD −0.10%
# Signal output shows: per-leg signals + Phase 76 spot-spread weight + Phase 75 momentum filter status
multistrat-backtest:
	uv run python scripts/run_phase68_multistrat_live.py --mode backtest

multistrat-signal:
	uv run python scripts/run_phase68_multistrat_live.py --mode signal

# Phase 70: LETF pair universe expansion
letf-universe:
	uv run python scripts/run_phase70_letf_universe.py

# Phase 72: SPY HMM regime filter on 3-pair LETF combo (NO-GO: bear earns more from vol)
letf-regime-filter:
	uv run python scripts/run_phase72_letf_regime_filter.py

# Phase 74: portfolio frontier re-optimisation with upgraded 3-pair LETF leg (NO-GO: 80/15/5 remains optimal)
portfolio-reopt:
	uv run python scripts/run_phase74_portfolio_reopt.py

# ── Engine runs ───────────────────────────────────────────────────────────────

run-btc:
	uv run rde run --config configs/btc.yaml

run-eth:
	uv run rde run --config configs/eth.yaml

run-spy:
	uv run rde run --config configs/spy.yaml

analyse-btc:
	uv run rde analyse --config configs/btc.yaml

# ── Dashboard ─────────────────────────────────────────────────────────────────

dashboard:
	uv run streamlit run app/streamlit_app.py

# ── Supabase upload ───────────────────────────────────────────────────────────

upload-multistrat:
	uv run python scripts/upload_multistrat_to_supabase.py

upload-rtmv:
	uv run python scripts/upload_rtmv_to_supabase.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-cache:
	rm -f results/cache/*.parquet

clean-results:
	find results/ -name "*.parquet" -not -path "*/cache/*" -delete
	find results/ -name "*.md" -delete

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Regime Detection Engine — common targets"
	@echo ""
	@echo "  Testing"
	@echo "    make test               Fast suite (excludes integration/numerics/regression)"
	@echo "    make test-all           Full suite including slow tests"
	@echo "    make test-edge          Edge validation tests only"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint               Ruff check"
	@echo "    make format             Ruff format"
	@echo "    make lint-fix           Ruff check + auto-fix"
	@echo ""
	@echo "  Validation"
	@echo "    make validate-daily     Full Phase 37 validation on BTC daily (ann=252)"
	@echo "    make validate-hourly    Full Phase 37 validation on BTC hourly (ann=8760)"
	@echo "    make validate-daily-fast  Same but n_restarts=1, n_sims=20 for quick checks"
	@echo "    make validate-rtmv      Phase 47 RTMV purged CV (fold/cost/shuffle)"
	@echo "    make validate-rtmv-fast  Same but n_restarts=1, n_shuffle=10 for quick checks"
	@echo "    make backtest-rtmv      4-asset RTMV backtest (SPY/GLD/TLT/IEF, λ=0.05)"
	@echo "    make backtest-rtmv-5asset  5-asset RTMV backtest (+ SHY, Phase 52a winner)"
	@echo "    make live-rtmv          5-asset RTMV live paper trading (polls hourly)"
	@echo "    make monitor            Check live paper trading status (Sharpe, DD, fills)"
	@echo ""
	@echo "  Engine"
	@echo "    make run-btc            rde run on BTC"
	@echo "    make analyse-btc        rde analyse on BTC"
	@echo "    make dashboard          Streamlit dashboard"
	@echo ""
	@echo "  Funding carry alpha"
	@echo "    make carry-backtest     BTC+ETH funding carry backtest 2020-present"
	@echo "    make carry-signal       Current live funding rate + entry/exit signal"
	@echo ""
	@echo "  Supabase uploads"
	@echo "    make upload-multistrat  Upload Phase 68+ multistrat results to Supabase"
	@echo "    make upload-rtmv        Upload RTMV 5-asset backtest results to Supabase"
	@echo "    (both require SUPABASE_SERVICE_KEY env var or --service-key arg)"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean-cache        Delete cached parquet downloads"
	@echo "    make clean-results      Delete generated result files"
