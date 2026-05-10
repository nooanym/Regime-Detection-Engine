.PHONY: test test-all test-slow lint format validate-daily validate-hourly validate-rtmv backtest-rtmv run-btc dashboard clean help

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
		--assets SPY,GLD,TLT,IEF \
		--lambda-tilt 0.05 \
		--n-states 3 \
		--n-restarts 3 \
		--output-dir results/rtmv_live \
		--mode live \
		--poll-interval 3600

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
	@echo "    make backtest-rtmv      Phase 48 RTMV backtest (SPY/GLD/TLT/IEF, λ=0.05)"
	@echo "    make live-rtmv          Phase 48 RTMV live paper trading (polls hourly)"
	@echo ""
	@echo "  Engine"
	@echo "    make run-btc            rde run on BTC"
	@echo "    make analyse-btc        rde analyse on BTC"
	@echo "    make dashboard          Streamlit dashboard"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean-cache        Delete cached parquet downloads"
	@echo "    make clean-results      Delete generated result files"
