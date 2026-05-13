# autoforecast — handoff

You (the next agent) are continuing work on `autoforecast`, a new
open-source repo. The previous session was on a laptop; the operator
(James, james@oraclebook.xyz) is moving to a cloud-hosted environment to
continue while flying. Build with that in mind: a fresh clone, no
existing volumes, no Modal account, no journal of prior runs.

This file is the single source of truth for picking up where things
were left.

---

## 1. Mission

Give the system a hypothesis like *"weather drives NSW peak demand"* and
have it run the full Kaggle-style modeling workflow autonomously, ending
with a promoted, journaled, reproducible forecasting model.

The killer feature is **auditable autonomy**: the LLM proposes and
explains, deterministic gates decide, every feature has a paired
backtest + DM p-value + regime-stratified Δrmse, every rejection has a
journaled reason.

Style: Karpathy's nanoGPT — minimal, narrative, every file readable cold,
runs on a single CPU in a coffee break for the demo.

---

## 2. Repo state at handoff

```
autoforecast/
├── README.md          ← exists, one-screen intro + target demo
├── HANDOFF.md         ← this file
├── .gitignore
├── docs/              ← empty (write the design docs here next)
└── .git/              ← initialized, branch=main
```

Remote: `git@github.com:jamhan/autoforecast.git` (or `https://github.com/jamhan/autoforecast.git`).

No source code yet. The README's `from autoforecast import run` example
is aspirational — your first deliverable is the skeleton that backs it.

---

## 3. The precursor: claudelphi

autoforecast is the open-source extraction of the autoresearch loop we
built inside `claudelphi` (a private trading project under
`~/thecleaners/pricingagent/claudelphi`). Do **not** copy code from
claudelphi — it's tangled with AEMO/BoM/oraclebook specifics. Use it as
a *reference implementation*: read the modules to understand the shape,
then write clean equivalents informed by what we learned.

Key claudelphi modules to read for reference (paths from
`pricingagent/claudelphi/src/claudelphi/`):

- `research/feature_lifecycle.py` — the orchestrator (propose → validate
  → backtest → judge → commit + bridge to cl_experiments)
- `research/feature_backtest.py` — paired backtest + DM + regime
  stratification
- `research/feature_validator.py` — the 5 cheap validation checks
- `research/feature_dsl.py` — strict-past feature mini-DSL
- `research/promote.py` — the 5-gate promotion check
- `research/leaderboard.py` — adjusted holdout RMSE ranking + overfit
  filter
- `research/drift.py` — Page-Hinkley + relative-RMSE drift detector
- `research/live_errors.py` — settlement → regime-stratified feedback
- `research/feature_critic.py` — LLM critic (explains, doesn't decide)
- `journal/schema.py` — DuckDB tables (cl_features, cl_experiments,
  cl_promotions, cl_baselines, cl_scores)

The skill markdown at
`pricingagent/claudelphi/skills/autoquant-modal-minimal.md` is the
condensed design rationale — read it first.

---

## 4. The Kaggle workflow → autoforecast mapping

A top Kaggle team's playbook on a time-series problem, with each step
mapped to an autoforecast module. **This is the spine of the system —
build it in roughly this order.**

| # | Top Kaggle team does | autoforecast automates via | Module |
|---|---|---|---|
| 0 | Read the brief, find the eval metric, note granularity | User writes `hypothesis.yaml`; parser validates | `hypothesis.py` |
| 1 | EDA — stationarity, ACF/PACF, seasonality, missingness, anomalies | Deterministic EDA: ADF + KPSS, FFT-peak seasonality, CUSUM regime detector, missingness map, outlier flag, correlation matrix at lags. Writes `eda_summary.json` for the LLM | `eda.py` |
| 2 | Build robust CV: walk-forward, embargo, locked holdout | Deterministic walk-forward CV builder. Holdout is one-and-only-one per row (enforced) | `splits.py` |
| 3 | Persistence + seasonal-naive baselines, then strong tabular ceiling (LightGBM + lag1-28 + calendar) | All baselines as families in `families.py`; baseline stage runs them; persistence anchors `cl_baselines` | `baselines.py`, `families.py` |
| 4 | Feature engineering — lags, rollings, calendar, interactions, target encoding | THE autoresearch loop. LLM proposes (using EDA + residuals + live errors), DSL validates strict-past, paired backtest evaluates, DM tests, regime stratifies, LLM critic explains, gate decides. Bridges to `cl_experiments` on keep | `lifecycle.py`, `propose.py`, `validate.py`, `backtest.py`, `critic.py`, `dsl.py` |
| 5 | Model family sweep — linear, RF, XGB, LightGBM, simple net | After feature loop converges (k consecutive rejects OR budget out), champion feature set is swept across all families. Paired protocol: fix features + split, vary family | `sweep.py` |
| 6 | HP tuning — Optuna on inner CV, multi-seed for variance | Time-boxed Optuna per family. Multi-seed variance estimate. Best HP per family becomes that family's champion | `tune.py` |
| 7 | Error analysis — plot residuals by regime, by feature, find worst days, propose targeted features | Deterministic regime breakdown of champion residuals + top-N worst days with feature values, fed into the next round's proposer prompt | `error_analysis.py` (feeds `propose.py`) |
| 8 | Stacking / blending — OOF predictions, fit stacker | Optional. Top-K configs' OOF preds → linear/rank/Nelder-Mead stacker. Adjusted holdout RMSE on the stacked output competes | `stack.py` |
| 9 | Validation discipline — if CV >> LB, CV is wrong | Phase E overfit guard + candidate-picker's val/holdout ratio filter (default 1.4). Every promotion attempt logs the gap | `promote.py`, `leaderboard.py` |
| 10 | Pick best validated model, submit | `maybe_promote()` runs 5 gates; pass → champion lives at `cl_promotions` latest. Reject → rationale journaled | `promote.py` |
| 11 | (post-Kaggle) Monitor live drift, retrain | Page-Hinkley + relative-RMSE on `cl_scores`. On trip → new `cl_baselines` row, reset gates, force next lifecycle to re-derive anchors | `drift.py` |
| 12 | (post-Kaggle) Continuous improvement from production errors | Live errors from `cl_scores` summarized by regime, injected into proposer prompt every round | `live_errors.py` |

---

## 5. Target repo layout

```
autoforecast/
├── README.md
├── HANDOFF.md
├── pyproject.toml
├── .gitignore
├── data/                            # synthetic generators + 1 public dataset
├── src/autoforecast/
│   ├── __init__.py
│   ├── hypothesis.py                # YAML parser + validator
│   ├── panel.py                     # Load + align time-series panels
│   ├── splits.py                    # Walk-forward CV + locked holdout
│   ├── eda.py                       # Stationarity, seasonality, regime, missingness, outliers
│   ├── dsl.py                       # Feature spec mini-DSL (strict-past)
│   ├── validate.py                  # 5 cheap pre-backtest checks
│   ├── families.py                  # Model registry
│   ├── baselines.py                 # Persistence + seasonal-naive + tabular ceiling
│   ├── backtest.py                  # Paired backtest + DM + regime stratification
│   ├── propose.py                   # LLM proposer + grid fallback
│   ├── critic.py                    # LLM judge — explains, never decides
│   ├── lifecycle.py                 # propose → validate → backtest → judge → commit → bridge
│   ├── sweep.py                     # Family sweep on champion features
│   ├── tune.py                      # Optuna HP tuning
│   ├── error_analysis.py            # Residual breakdown by regime
│   ├── stack.py                     # OOF stacking
│   ├── promote.py                   # 5-gate promotion check
│   ├── leaderboard.py               # Adjusted holdout RMSE + overfit filter
│   ├── drift.py                     # Page-Hinkley + re-baseline
│   ├── live_errors.py               # Settlement feedback into proposer
│   ├── journal.py                   # DuckDB schema + connect()
│   └── cli.py                       # `autoforecast <subcommand>`
├── tests/                           # one file per module
├── examples/
│   ├── 01_demo_synthetic.py         # baselines on synthetic data, no API key, <60s
│   ├── 02_one_feature.py            # paired backtest of one feature
│   ├── 03_loop_grid.py              # full inner loop, grid proposer
│   ├── 04_loop_llm.py               # full inner loop, LLM proposer
│   ├── 05_full_pipeline.py          # end-to-end on the headline benchmark
│   └── 06_replay.py                 # reproduce a frozen run from a shipped journal
├── notebooks/
│   └── walkthrough.ipynb            # Karpathy-style cell-by-cell build
└── docs/
    ├── kaggle-mapping.md            # (copy of §4 above, formatted)
    ├── architecture.md              # The pipeline diagram + module responsibilities
    ├── plan.md                      # Phase-by-phase build plan (§6 below)
    ├── pitfalls.md                  # Bugs we hit in claudelphi (§7 below)
    └── tradeoffs.md                 # Why DuckDB not Postgres, why Anthropic not local, etc
```

---

## 6. Build phases

Aim for one phase per week (assuming ~2-3h/day). Each phase produces a
runnable example.

### Phase 0 — skeleton + data + baselines (week 1)
**Goal**: `python examples/01_demo_synthetic.py` runs persistence and
LightGBM baselines on a 365-day synthetic panel in under 60 seconds.

- `pyproject.toml`, `.gitignore`, `src/autoforecast/__init__.py`
- `hypothesis.py` — YAML parser, dataclass result
- `panel.py` — `load_synthetic(n_days, seed)` + a real dataset loader
  (M4 monthly OR NSW peak demand from AEMO public CSV)
- `splits.py` — walk-forward CV + locked holdout
- `journal.py` — DuckDB tables: `cl_features`, `cl_experiments`,
  `cl_promotions`, `cl_baselines`, `cl_scores`. Schema in one screen.
- `families.py` — `persistence` + `seasonal_naive` + `lightgbm`
- `baselines.py` — run all baselines, write to `cl_experiments`, anchor
  persistence in `cl_baselines`
- `examples/01_demo_synthetic.py`

### Phase 1 — DSL + validate + paired backtest (week 2)
**Goal**: `python examples/02_one_feature.py` runs a paired backtest of a
single hand-coded feature and prints Δrmse + DM p-value + regime
breakdown.

- `dsl.py` — strict-past primitives: `lag`, `rolling_mean`,
  `rolling_max`, `log1p`, `diff`, `z_score`, `binary_threshold`,
  `interaction`. Compiler enforces `anchor - window, anchor)` half-open
  rolling.
- `validate.py` — 5 checks (shape, variance, determinism, leakage,
  multicollinearity)
- `backtest.py` — paired (champion ± feature), same family/HP/seed,
  regime-stratified Δrmse, Diebold-Mariano p-value with the 4 mandatory
  guards (n<5, mean ≤ 0, sd=0, non-finite t-stat → p=1.0)
- `examples/02_one_feature.py`

### Phase 2 — lifecycle + grid proposer + promote (week 3)
**Goal**: `python examples/03_loop_grid.py` runs 20 lifecycles on
synthetic data with a grid proposer, ends with a promoted champion.

- `propose.py` — grid mode (enumerate plausible DSL specs from EDA)
- `critic.py` — placeholder that always returns "let the gate decide"
- `lifecycle.py` — orchestrator: propose → validate → backtest → judge
  → commit. **MUST include the bridge to cl_experiments on keep** —
  this was the bug that killed claudelphi's gate for weeks. See
  pitfalls.md §1.
- `promote.py` — 5 gates: candidate exists, baseline exists, beats
  baseline by N%, beats current by hysteresis, domain check. Plus the
  Phase E overfit guard.
- `leaderboard.py` — rank by adjusted holdout RMSE, exclude val→holdout
  overfits
- `examples/03_loop_grid.py`

### Phase 3 — LLM proposer + critic (week 4)
**Goal**: replace grid with Anthropic-backed proposer. Same loop, but
the LLM picks features informed by EDA + residuals.

- `propose.py` — LLM mode, tool-use enforced schema
- `critic.py` — LLM judges with `{decision, rationale, confidence,
  failure_modes_observed}`. Decision is advisory; gate still decides.
- `examples/04_loop_llm.py`

### Phase 4 — sweep + tune + stack + error analysis (week 5)
**Goal**: full Kaggle chain. Champion features → family sweep → HP tune
→ optional stacking, with residual error analysis feeding back.

- `sweep.py`, `tune.py`, `stack.py`, `error_analysis.py`
- `examples/05_full_pipeline.py`

### Phase 5 — drift + live errors + re-baselining (week 6)
**Goal**: closed production loop. Drift detector trips on synthetic
distribution shift; re-baseline fires; proposer's prompt now references
the worst regime from live errors.

- `drift.py`, `live_errors.py`
- `examples/06_replay.py` (replay a frozen journal to reproduce)

### Phase 6 — headline benchmark (week 7)
**Goal**: one reproducible claim. "autoforecast achieves X% improvement
over persistence on Y benchmark, frozen journal included."

- Pick **one** public dataset that's not commercially sensitive (M4
  monthly is the safest; ERCOT load is great if you want energy-specific;
  NSW peak demand from AEMO public CSV is the obvious choice given the
  user's domain). Commit the frozen journal in `data/`.
- `notebooks/walkthrough.ipynb` — cell-by-cell build, narrative tone.
- Polish the README to reference the benchmark.

---

## 7. Pitfalls from claudelphi (don't repeat these)

### 7.1 The lifecycle → experiments bridge

**Bug**: in claudelphi, the feature lifecycle wrote accepted features to
`cl_features` (its own table) but **never** wrote a corresponding
`cl_experiments` row. The promotion gate reads from `cl_experiments`.
For weeks, 8 features got kept and 0 got promoted, because the two
tables were disconnected. The gate kept picking the same val-overfit
row off the leaderboard and rejecting it (32 consecutive identical
rejections).

**Fix**: every `decision == "keep"` MUST insert a `cl_experiments` row
with `status='done'`, `rmse_val=rmse_holdout=with_rmse`,
`holdout_evaluated_at=now`, `cv_signature` stamped, `feature_lifecycle_id`
linked, `parent_id` = champion's experiment_id. See claudelphi's
`research/feature_lifecycle.py::_emit_lifecycle_experiment`.

### 7.2 Ranking candidates by val RMSE

**Bug**: claudelphi's `_candidate_for` ranked the leaderboard by
`rmse_val`. The LLM proposer overfit to val (val=246, holdout=458 —
86% gap). The gate kept picking the same val-leader and rejecting it on
holdout. Same numbers, same rejection, every tick.

**Fix**: rank by **adjusted holdout RMSE** (`rmse_holdout + n_features ×
penalty`). Filter val→holdout overfits upfront (ratio > 1.4 excluded).
The Phase E in-gate overfit guard remains as a backstop. See claudelphi
`research/promote.py::_candidate_for` (the post-patch version).

### 7.3 cv_signature must match between candidate and baseline

**Bug**: when the code-level evaluation regime changes (we added real
public holidays + median imputation), old experiments are no longer
comparable to new ones. Without a regime tag, the gate compares apples
to oranges and ships nonsense.

**Fix**: every experiment row carries a `cv_signature` derived from its
config + a set of code-level toggles. The gate only considers rows
under the *current* signature. See claudelphi
`research/cv_signature.py`.

### 7.4 Diebold-Mariano returns garbage 5% of the time

**Bug**: vanilla DM blows up on small samples / zero variance / negative
mean improvement / non-finite t-stat. ~5% of paired backtests return
nonsense p-values, mostly p < 0.05 false positives.

**Fix**: 4 mandatory guards — return `p=1.0` when (a) n<5, (b) mean diff
≤ 0, (c) sd == 0, (d) t-stat non-finite.

### 7.5 Rolling ops must be strict-past

**Bug**: easy to write `rolling_mean(7)` that includes day-of as the
last element. Single most common silent leak in time-series feature
engineering.

**Fix**: the DSL compiler enforces `[anchor - window, anchor)`
half-open. Target day excluded. Spell this out in tests.

### 7.6 LLM critic deciding the verdict

**Bug**: easy to let the LLM pick keep/reject. The LLM's verdicts
correlate weakly with holdout improvement and accept noisy wins because
they "look interesting."

**Fix**: critic returns `{decision, rationale, confidence,
failure_modes_observed}` but the **gate decides** based on
deterministic thresholds (DM p-value, stability score, regime check).
Critic's freedom is in the rationale, not the verdict.

### 7.7 Persistence baseline isn't immortal

**Bug**: gate thresholds are calibrated against a persistence baseline
captured weeks ago. Seasonality changes; baseline drifts; thresholds
silently become wrong. Loop accepts noise (gates loose) and rejects
real wins (anchor moved).

**Fix**: `cl_baselines` is append-only. Drift detector (Page-Hinkley on
squared error vs anchor variance + 20% relative-RMSE divergence) cron'd
daily. On trip → write new `cl_baselines` row, force next lifecycle to
re-derive its anchor before any keep/reject.

### 7.8 Multiple-testing correction is missing

**Status**: not fixed in claudelphi. Running 100s of DM tests at p<0.05
guarantees false positives. autoforecast should ship with
Benjamini-Hochberg FDR over a rolling window of recent tests, or a
per-day acceptance budget.

---

## 8. First task for the cloud agent

Open the freshly-cloned repo. You'll see `README.md` + `HANDOFF.md` +
empty `docs/`. **Do not start coding yet.** Phase 0 is the right
starting point but it has prerequisites:

1. **Write `docs/architecture.md`** — the pipeline diagram from §4 and
   the module responsibilities from §5, formatted as a design doc. ~1
   page.
2. **Write `docs/pitfalls.md`** — copy §7 verbatim into its own file
   (it's referenced by other docs).
3. **Write `docs/plan.md`** — copy §6 verbatim, but expand Phase 0 into
   a concrete TODO list with file-by-file deliverables.
4. **Write `pyproject.toml`** — Python 3.11+, ruff + pytest, deps:
   `duckdb`, `numpy`, `pandas`, `pyyaml`, `lightgbm`, `scikit-learn`,
   `anthropic`, `scipy`. Dev deps: `pytest`, `ruff`.
5. **Write `.gitignore`** — Python defaults, `.venv/`, `*.duckdb`,
   `workspace/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`.

Once those are committed, **start Phase 0** with `journal.py` (schema)
because every other module depends on the table shapes.

Test discipline: write the test before the implementation when the
behaviour is non-obvious (DM guards, strict-past DSL, the lifecycle
bridge). Don't write tests for trivial getters.

---

## 9. Decisions already made

- **License**: Apache 2.0 (patent grant matters in finance).
- **Database**: DuckDB. Single file. No migrations framework — schema
  evolves via `ALTER TABLE … ADD COLUMN IF NOT EXISTS`.
- **LLM provider**: Anthropic via the official SDK. Tool-use for
  schema-enforced proposer/critic outputs.
- **Model families for v1**: `persistence`, `seasonal_naive`,
  `linear`, `random_forest`, `lightgbm`. Add neural later if the
  benchmark demands it.
- **Default DSL primitives**: `raw`, `lag`, `rolling_mean`,
  `rolling_max`, `log1p`, `diff`, `z_score`, `binary_threshold`,
  `interaction`. Cap composition depth at 4.
- **Active feature registry cap**: 20. Forces competition.
- **Adjusted RMSE penalty**: 0.5 RMSE units per feature (calibrated so a
  20-feature config needs to beat a 10-feature one by ~5 RMSE units on
  the adjusted score).
- **Overfit ratio cap**: holdout/val > 1.4 → excluded from candidate
  pool.
- **DM p-value thresholds (sample-size calibrated)**:
  - n < 50: p < 0.20
  - 50 ≤ n < 200: p < 0.10
  - n ≥ 200: p < 0.05

---

## 10. Decisions NOT yet made (do not guess — escalate)

- Headline benchmark dataset (M4 monthly vs ERCOT load vs NSW peak
  demand)
- Whether to support multi-target hypotheses in v1 (default: no, single
  target only)
- Whether the LLM critic is on by default in Phase 3 or always opt-in
- Naming: `autoforecast` is fine but check PyPI before claiming it
- Whether Modal/Replicate/Coiled/something-else is the cloud
  execution layer (Phase 5+ concern)

---

## 11. Contact / context

- Operator: James (james@oraclebook.xyz)
- Local precursor: `~/thecleaners/pricingagent/claudelphi/` (private)
- This repo: `https://github.com/jamhan/autoforecast`
- Today: 2026-05-13

If a decision blocks you, note it in `docs/decisions-pending.md` and
proceed with the most defensible default. Don't wait.
