# Energy Demand Example

This example is intentionally synthetic. It gives contributors a fast,
offline way to exercise the core `autoforecast` primitives before wiring
in real AEMO, ERCOT, utility, or internal load data.

It models daily peak demand with:

- weekly seasonality
- summer cooling load
- winter heating load
- weekend suppression
- persistence from recent demand

Run it from the repo root:

```bash
PYTHONPATH=src python examples/energy/energy_demo.py
```

The script prints:

- seasonal-naive baseline RMSE
- feature model RMSE
- promotion-gate decision
- journal path

