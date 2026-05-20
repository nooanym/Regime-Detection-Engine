-- Migration: 20260520_multistrat_snapshots
-- Multi-strategy portfolio tables for Phase 68+ combined portfolio
-- (80% carry + 15% LETF pair + 5% RTMV)
--
-- Apply via: supabase db push  OR  paste into Supabase SQL editor

-- ---------------------------------------------------------------------------
-- multistrat_runs: one row per backtest / live run
-- ---------------------------------------------------------------------------

create table if not exists multistrat_runs (
  id                   text primary key,
  started_at           timestamptz not null,
  ended_at             timestamptz,
  carry_weight         numeric default 0.80,
  letf_weight          numeric default 0.15,
  rtmv_weight          numeric default 0.05,
  carry_symbols        text[] default array['BTCUSDT', 'ETHUSDT'],
  letf_pairs           text[] default array['TQQQ/QQQ', 'SOXL/SOXX', 'UPRO/SPY'],
  rtmv_assets          text[] default array['SPY', 'GLD', 'SHY', 'IEF', 'TLT'],
  final_sharpe         numeric,
  final_ann_return     numeric,
  final_mdd            numeric,
  n_bars               integer,
  notes                text,
  created_at           timestamptz default now()
);

alter table multistrat_runs enable row level security;

create policy "Public read access"
  on multistrat_runs
  for select
  using (true);

comment on table multistrat_runs is
  'One row per multistrat backtest/live run. Populated by upload_multistrat_to_supabase.py.';

-- ---------------------------------------------------------------------------
-- multistrat_snapshots: daily portfolio snapshots
-- ---------------------------------------------------------------------------

create table if not exists multistrat_snapshots (
  id                   uuid primary key default gen_random_uuid(),
  run_id               text not null references multistrat_runs(id) on delete cascade,
  bar_date             date not null,
  -- equity curve
  equity               numeric not null,
  drawdown             numeric not null default 0,
  -- per-leg daily returns (decimal: 0.002 = +0.2%)
  r_carry              numeric,
  r_letf               numeric,
  r_rtmv               numeric,
  r_combined           numeric,
  -- allocation weights (constant for a given run, stored per-row for dashboard convenience)
  carry_weight         numeric default 0.80,
  letf_weight          numeric default 0.15,
  rtmv_weight          numeric default 0.05,
  -- rolling risk metric
  sharpe_rolling_252   numeric,
  -- cumulative returns per leg (decimal, e.g. 1.672 = +67.2% total since inception)
  cum_return_carry     numeric,
  cum_return_letf      numeric,
  cum_return_rtmv      numeric,
  cum_return_combined  numeric,
  created_at           timestamptz default now(),
  unique (run_id, bar_date)
);

create index if not exists multistrat_snapshots_run_date
  on multistrat_snapshots (run_id, bar_date);

alter table multistrat_snapshots enable row level security;

create policy "Public read access"
  on multistrat_snapshots
  for select
  using (true);

comment on table multistrat_snapshots is
  'Daily snapshots for the Phase 68+ multi-strategy portfolio (carry + LETF pair + RTMV). '
  'Populated by scripts/upload_multistrat_to_supabase.py.';

comment on column multistrat_snapshots.drawdown is
  'Drawdown from rolling peak as a positive fraction (0.021 = 2.1% drawdown).';

comment on column multistrat_snapshots.r_carry is
  'Carry leg (BTC+ETH funding carry) daily return as a decimal (0.002 = +0.2%).';

comment on column multistrat_snapshots.r_letf is
  'LETF pair leg (delta-neutral TQQQ/QQQ + SOXL/SOXX + UPRO/SPY) daily return.';

comment on column multistrat_snapshots.r_rtmv is
  'RTMV leg (SPY/GLD/SHY/IEF/TLT regime-tilted min-var) daily return.';

comment on column multistrat_snapshots.cum_return_combined is
  'Cumulative combined return since start of run (1.672 = portfolio has grown 67.2%).';
