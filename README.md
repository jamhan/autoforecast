# autoforecast

Production quant research is now accessible and available.

Give it a hypothesis — *"weather drives NSW peak demand"* — get a
Kaggle-grade forecasting model out the other side, with every decision
journaled and reproducible.

```
hypothesis.yaml  →  EDA  →  CV  →  baselines  →  feature loop  →  family sweep
                 →  HP tune  →  stack  →  error-analysis loop  →  promote  →  report
```

The LLM proposes and explains. Deterministic gates decide. Every accepted
feature has a paired backtest, a Diebold-Mariano p-value, and a
regime-stratified Δrmse behind it. Every rejected feature has a row
saying why.

## Status

Pre-alpha. The first usable core now exists: strict-past feature
generation, feature validation/selection helpers, experiment metrics,
SQLite journal, and a simple promotion gate.

Read in this order:

1. [docs/kaggle-mapping.md](docs/kaggle-mapping.md) — what a top Kaggle
   team does on a time-series problem, and how autoforecast automates
   each step.
2. [docs/architecture.md](docs/architecture.md) — the 12-stage pipeline,
   module layout, and where the LLM sits.
3. [docs/plan.md](docs/plan.md) — phased build plan, 7 weeks to a
   reproducible benchmark.

## The one-screen demo (target)

```python
from autoforecast import run

result = run("hypothesis.yaml")
print(result.champion)            # the promoted model
print(result.holdout_rmse)        # locked-window score
print(result.improvement_vs_persistence)
print(result.journal_path)        # reproduce with examples/replay.py
```

```yaml
# hypothesis.yaml
name: weather_nsw_peak_demand
target:
  source: aemo
  region: NSW1
  variable: peak_demand
  horizon_days: 1
hypothesized_drivers:
  - weather:tmax_d
  - weather:hdd_d
  - weather:cdd_d
  - calendar:dow
  - calendar:holidays
data_sources:
  - bom_nsw_observations
  - aemo_mmsdm
budget:
  llm_calls: 200
  wall_clock_hours: 24
```

## Why this exists

Most AutoML libraries:
- Let the LLM decide. (autoforecast: critic explains, gate decides.)
- Hand-wave rolling windows. (autoforecast: strict-past DSL with compiler.)
- Use feature importance. (autoforecast: paired backtest + DM test.)
- Skip drift detection. (autoforecast: Page-Hinkley + anchored baselines.)
- Don't close the production loop. (autoforecast: live errors feed back into the proposer.)

The bugs we hit building the precursor (claudelphi) are documented in
[docs/pitfalls.md](docs/pitfalls.md) so the next builder doesn't repeat
them.

## License

Apache 2.0 (planned). Patent grant matters in finance.
