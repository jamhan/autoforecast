"""Small SQLite-backed experiment journal.

SQLite is used for the open-source core because it is in the standard
library. The schema stays close to the DuckDB tables used in the
precursor so it can be swapped later.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  target TEXT NOT NULL,
  family TEXT NOT NULL,
  config_json TEXT NOT NULL,
  feature_set_json TEXT NOT NULL,
  split_signature TEXT NOT NULL,
  status TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_val REAL,
  metric_holdout REAL,
  hypothesis TEXT,
  parent_id TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS feature_decisions (
  feature_id TEXT PRIMARY KEY,
  target TEXT NOT NULL,
  name TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  status TEXT NOT NULL,
  validator_json TEXT,
  metric_delta REAL,
  dm_p_value REAL,
  stability_score REAL,
  rationale TEXT,
  created_at TEXT NOT NULL,
  decided_at TEXT
);

CREATE TABLE IF NOT EXISTS promotions (
  promotion_id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL,
  target TEXT NOT NULL,
  metric_holdout REAL NOT NULL,
  baseline_metric REAL NOT NULL,
  improvement_pct REAL NOT NULL,
  rejected_reason TEXT,
  promoted_at TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Journal:
    path: Path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn

    def log_experiment(
        self,
        *,
        target: str,
        family: str,
        config: dict[str, Any],
        feature_set: list[str],
        split_signature: str,
        metric_name: str,
        status: str = "pending",
        hypothesis: str | None = None,
        parent_id: str | None = None,
    ) -> str:
        experiment_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO experiments (
                     experiment_id, target, family, config_json, feature_set_json,
                     split_signature, status, metric_name, hypothesis, parent_id, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    target,
                    family,
                    json.dumps(config, sort_keys=True),
                    json.dumps(feature_set),
                    split_signature,
                    status,
                    metric_name,
                    hypothesis,
                    parent_id,
                    utcnow(),
                ),
            )
        return experiment_id

    def finish_experiment(
        self,
        experiment_id: str,
        *,
        status: str,
        metric_val: float | None = None,
        metric_holdout: float | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE experiments
                   SET status = ?, metric_val = ?, metric_holdout = ?, finished_at = ?
                   WHERE experiment_id = ?""",
                (status, metric_val, metric_holdout, utcnow(), experiment_id),
            )

    def log_feature_decision(
        self,
        *,
        target: str,
        name: str,
        spec: dict[str, Any],
        status: str,
        validator: dict[str, Any] | None = None,
        metric_delta: float | None = None,
        dm_p_value: float | None = None,
        stability_score: float | None = None,
        rationale: str | None = None,
    ) -> str:
        feature_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO feature_decisions (
                     feature_id, target, name, spec_json, status, validator_json,
                     metric_delta, dm_p_value, stability_score, rationale, created_at,
                     decided_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    feature_id,
                    target,
                    name,
                    json.dumps(spec, sort_keys=True),
                    status,
                    json.dumps(validator or {}, sort_keys=True),
                    metric_delta,
                    dm_p_value,
                    stability_score,
                    rationale,
                    utcnow(),
                    utcnow(),
                ),
            )
        return feature_id
