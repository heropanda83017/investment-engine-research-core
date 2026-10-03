#!/usr/bin/env python3
"""Emit a fail-closed OOS readiness report before any performance claim.

A return/Sharpe/drawdown is intentionally absent until the engine has enough
versioned PIT decision dates to replay T+1-open transactions with costs,
limits, suspensions, a benchmark, and frozen source provenance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_DB = ROOT / "data" / "warehouse_live" / "warehouse.db"
DEFAULT_OUTPUT = ROOT / "reports" / "backtest" / "oos_readiness_latest.json"

EXECUTION_CONTRACT = {
    "timing": "signal_asof_close_to_next_bar_open",
    "commission": "required before performance metrics",
    "slippage": "required before performance metrics",
    "limit": "limit-up/limit-down execution must be modeled",
    "suspension": "suspension execution must be modeled",
    "benchmark": "benchmark series must be recorded",
}


def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def build_report(db_path: Path, *, minimum_periods: int = 36) -> dict:
    import duckdb
    from strategies.oos_readiness import assess_oos_readiness
    from strategies.pit_oos_replayer import replay_pit_oos

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        readiness = assess_oos_readiness(conn, minimum_periods=minimum_periods)
        replay = (
            replay_pit_oos(conn, minimum_periods=minimum_periods)
            if readiness["status"] == "READY_FOR_REPLAY" else None
        )
    finally:
        conn.close()
    performance = replay["performance"] if replay else None
    return {
        "report_type": "PIT_OOS_READINESS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "status": readiness["status"],
        "readiness": readiness,
        "execution_contract": EXECUTION_CONTRACT,
        "code_sha": _code_sha(),
        "replay": replay,
        "performance": performance,
        "performance_reason": (
            "NOT_READY: no strategy return, Sharpe, drawdown, or VaR is claimed "
            "until replayable PIT evidence reaches the minimum period gate."
            if replay is None else "Replay is read-only and transaction-level; no promotion is implied."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed PIT OOS readiness report")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-periods", type=int, default=36)
    args = parser.parse_args(argv)
    report = build_report(args.db, minimum_periods=args.minimum_periods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']} eligible_periods={report['readiness']['eligible_periods']}/"
          f"{report['readiness']['minimum_periods']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
