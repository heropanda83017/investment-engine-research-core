"""OOS performance reporting must fail closed until replayable evidence exists."""

from datetime import date, timedelta
import json

import duckdb
import pytest


def _schema(conn):
    conn.execute(
        """CREATE TABLE decision_evidence (
            pipeline_run_id VARCHAR, trade_date DATE, code VARCHAR, verdict VARCHAR,
            conviction VARCHAR, source_as_of VARCHAR, config_hash VARCHAR, code_sha VARCHAR
        )"""
    )


def test_oos_readiness_rejects_single_date_evidence_without_performance_metrics():
    from strategies.oos_readiness import assess_oos_readiness

    conn = duckdb.connect(":memory:")
    try:
        _schema(conn)
        conn.execute(
            "INSERT INTO decision_evidence VALUES "
            "('r1', '2026-08-27', '000001', 'BUY', 'MEDIUM', '2026-08-27', 'cfg', 'sha')"
        )
        result = assess_oos_readiness(conn, minimum_periods=36)
        assert result["status"] == "NOT_READY"
        assert result["eligible_periods"] == 1
        assert result["missing_periods"] == 35
        assert "performance" not in result
    finally:
        conn.close()


def test_oos_readiness_rejects_evidence_missing_replay_provenance():
    from strategies.oos_readiness import assess_oos_readiness

    conn = duckdb.connect(":memory:")
    try:
        _schema(conn)
        start = date(2026, 1, 1)
        conn.executemany(
            "INSERT INTO decision_evidence VALUES (?, ?, '000001', 'BUY', 'MEDIUM', ?, ?, ?)",
            [(f"r{i}", str(start + timedelta(days=i)), None, "cfg", "sha") for i in range(36)],
        )
        result = assess_oos_readiness(conn, minimum_periods=36)
        assert result["status"] == "NOT_READY"
        assert result["invalid_provenance_rows"] == 36
    finally:
        conn.close()


def test_oos_readiness_marks_only_replayable_36_period_evidence_ready():
    from strategies.oos_readiness import assess_oos_readiness

    conn = duckdb.connect(":memory:")
    try:
        _schema(conn)
        start = date(2026, 1, 1)
        rows = [
            (f"r{i}", str(start + timedelta(days=i)), "000001", "BUY", "MEDIUM",
             json.dumps({"stock_daily_core": str(start + timedelta(days=i))}), "cfg", "sha")
            for i in range(36)
        ]
        conn.executemany("INSERT INTO decision_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        result = assess_oos_readiness(conn, minimum_periods=36)
        assert result == {
            "status": "READY_FOR_REPLAY",
            "eligible_periods": 36,
            "minimum_periods": 36,
            "missing_periods": 0,
            "invalid_provenance_rows": 0,
        }
    finally:
        conn.close()


def test_oos_report_script_declares_next_open_cost_and_limit_contract():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "scripts" / "oos_backtest_report.py").read_text(encoding="utf-8")
    for text in ("signal_asof_close_to_next_bar_open", "commission", "slippage", "limit", "suspension", "code_sha"):
        assert text in source
    assert "NOT_READY" in source


def test_pit_replay_uses_next_actual_open_and_records_costed_trade_ledger():
    from strategies.pit_oos_replayer import replay_pit_oos

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            """CREATE TABLE decision_evidence (
                pipeline_run_id VARCHAR, code VARCHAR, name VARCHAR, trade_date DATE,
                verdict VARCHAR, conviction VARCHAR, evidence_json VARCHAR,
                evidence_hash VARCHAR, config_hash VARCHAR, code_sha VARCHAR
            )"""
        )
        conn.execute(
            """CREATE TABLE stock_daily_core (
                ts_code VARCHAR, trade_date DATE, open DOUBLE, close DOUBLE,
                up_limit DOUBLE, down_limit DOUBLE, vol DOUBLE, suspend_timing VARCHAR
            )"""
        )
        evidence = json.dumps({"source_as_of": {"stock_daily_core": "2026-01-02"}})
        conn.execute(
            "INSERT INTO decision_evidence VALUES ('r1', '000001', 'A', '2026-01-02', "
            "'BUY', 'HIGH', ?, 'hash', 'cfg', 'sha')", [evidence]
        )
        conn.executemany(
            "INSERT INTO stock_daily_core VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("000001.SZ", "2026-01-02", 10.0, 10.0, 11.0, 9.0, 1000.0, None),
                ("000001.SZ", "2026-01-05", 11.0, 12.0, 12.1, 9.9, 1000.0, None),
                ("000300.SH", "2026-01-02", 4.0, 4.0, 4.4, 3.6, 1000.0, None),
                ("000300.SH", "2026-01-05", 4.1, 4.2, 4.5, 3.7, 1000.0, None),
            ],
        )
        result = replay_pit_oos(conn, minimum_periods=1, holding_days=0)

        assert result["status"] == "READY"
        assert result["performance"] is not None
        fill = result["trade_ledger"][0]
        assert fill["code"] == "000001"
        assert fill["signal_date"] == "2026-01-02"
        assert fill["entry_date"] == "2026-01-05"
        assert fill["entry_open"] == 11.0
        assert fill["exit_close"] == 12.0
        assert fill["gross_return"] == 12.0 / 11.0 - 1
        assert fill["status"] == "FILLED"
        # provenance and cost are part of the replayable record
        assert fill["source_as_of"] == {"stock_daily_core": "2026-01-02"}
        assert fill["config_hash"] == "cfg"
        assert fill["code_sha"] == "sha"
        assert fill["cost_bps"] == 2 * (3.0 + 5.0) + 5.0
        assert result["trade_ledger"][0]["net_return"] < result["trade_ledger"][0]["gross_return"]
        assert result["benchmark"]["code"] == "000300"
    finally:
        conn.close()


def test_pit_replay_fails_closed_when_entry_open_is_at_limit_up():
    from strategies.pit_oos_replayer import replay_pit_oos

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            """CREATE TABLE decision_evidence (
                pipeline_run_id VARCHAR, code VARCHAR, name VARCHAR, trade_date DATE,
                verdict VARCHAR, conviction VARCHAR, evidence_json VARCHAR,
                evidence_hash VARCHAR, config_hash VARCHAR, code_sha VARCHAR
            )"""
        )
        conn.execute(
            """CREATE TABLE stock_daily_core (
                ts_code VARCHAR, trade_date DATE, open DOUBLE, close DOUBLE,
                up_limit DOUBLE, down_limit DOUBLE, vol DOUBLE, suspend_timing VARCHAR
            )"""
        )
        evidence = json.dumps({"source_as_of": {"stock_daily_core": "2026-01-02"}})
        conn.execute(
            "INSERT INTO decision_evidence VALUES ('r1', '000001', 'A', '2026-01-02', "
            "'BUY', 'HIGH', ?, 'hash', 'cfg', 'sha')", [evidence]
        )
        conn.executemany(
            "INSERT INTO stock_daily_core VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("000001.SZ", "2026-01-02", 10.0, 10.0, 11.0, 9.0, 1000.0, None),
                ("000001.SZ", "2026-01-05", 11.0, 11.0, 11.0, 9.0, 1000.0, None),
                ("000300.SH", "2026-01-02", 4.0, 4.0, 4.4, 3.6, 1000.0, None),
                ("000300.SH", "2026-01-05", 4.1, 4.2, 4.5, 3.7, 1000.0, None),
            ],
        )
        result = replay_pit_oos(conn, minimum_periods=1, holding_days=0)

        assert result["status"] == "READY"
        assert result["performance"] is None
        assert result["trade_ledger"][0]["status"] == "UNFILLED_LIMIT_UP"
    finally:
        conn.close()


def _seed_two_period_evidence(conn):
    """Two independently versioned decision dates with full stock + benchmark bars."""
    conn.execute(
        """CREATE TABLE decision_evidence (
            pipeline_run_id VARCHAR, code VARCHAR, name VARCHAR, trade_date DATE,
            verdict VARCHAR, conviction VARCHAR, evidence_json VARCHAR,
            evidence_hash VARCHAR, config_hash VARCHAR, code_sha VARCHAR
        )"""
    )
    conn.execute(
        """CREATE TABLE stock_daily_core (
            ts_code VARCHAR, trade_date DATE, open DOUBLE, close DOUBLE,
            up_limit DOUBLE, down_limit DOUBLE, vol DOUBLE, suspend_timing VARCHAR
        )"""
    )
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    conn.executemany(
        "INSERT INTO stock_daily_core VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # up_limit kept well above opens so both signals remain executable
            # (open at limit-up must fail closed; that path is covered elsewhere).
            ("000001.SZ", d, 10.0 + i, 10.5 + i, 30.0, 5.0, 1000.0, None)
            for i, d in enumerate(dates)
        ]
        + [("000300.SH", d, 4.0 + i * 0.1, 4.1 + i * 0.1, 30.0, 5.0, 1000.0, None)
           for i, d in enumerate(dates)],
    )
    ev1 = json.dumps({"source_as_of": {"stock_daily_core": "2026-01-02"}})
    ev2 = json.dumps({"source_as_of": {"stock_daily_core": "2026-01-05"}})
    conn.execute(
        "INSERT INTO decision_evidence (pipeline_run_id, code, name, trade_date, verdict, "
        "conviction, evidence_json, evidence_hash, config_hash, code_sha) "
        "VALUES ('r1', '000001', 'A', '2026-01-02', 'BUY', 'HIGH', ?, 'h', 'cfg1', 'sha1')", [ev1],
    )
    conn.execute(
        "INSERT INTO decision_evidence (pipeline_run_id, code, name, trade_date, verdict, "
        "conviction, evidence_json, evidence_hash, config_hash, code_sha) "
        "VALUES ('r2', '000001', 'A', '2026-01-05', 'BUY', 'HIGH', ?, 'h', 'cfg2', 'sha2')", [ev2],
    )


def test_pit_replay_records_provenance_and_nav_curve_with_spread_and_tax():
    from strategies.pit_oos_replayer import replay_pit_oos

    conn = duckdb.connect(":memory:")
    try:
        _seed_two_period_evidence(conn)
        result = replay_pit_oos(conn, minimum_periods=2, holding_days=2,
                                commission_bps=3.0, slippage_bps=5.0, stamp_tax_bps=5.0)
        assert result["status"] == "READY"
        assert result["performance"] is not None
        filled = [item for item in result["trade_ledger"] if item["status"] == "FILLED"]
        assert len(filled) == 2
        by_date = {item["signal_date"]: item for item in filled}
        # each signal keeps its own decision version and data as-of snapshot
        assert by_date["2026-01-02"]["config_hash"] == "cfg1"
        assert by_date["2026-01-02"]["code_sha"] == "sha1"
        assert by_date["2026-01-02"]["source_as_of"] == {"stock_daily_core": "2026-01-02"}
        assert by_date["2026-01-05"]["config_hash"] == "cfg2"
        assert by_date["2026-01-05"]["source_as_of"] == {"stock_daily_core": "2026-01-05"}
        # net return must include both-side commission + both-side slippage + sell-side stamp tax
        expected_cost_bps = 2 * 3.0 + 2 * 5.0 + 5.0  # 21 bps
        assert by_date["2026-01-02"]["net_return"] == pytest.approx(
            by_date["2026-01-02"]["gross_return"] - expected_cost_bps / 10000, abs=1e-12
        )
        # nav curve compounds each filled trade in chronological order
        nav = 1.0
        expected_nav = [1.0]
        for item in sorted(filled, key=lambda x: x["signal_date"]):
            nav *= 1 + item["net_return"]
            expected_nav.append(pytest.approx(nav))
        assert result["nav_curve"] == expected_nav
        assert "excess_return" in result["benchmark"]
    finally:
        conn.close()


def test_pit_replay_supersedes_evidences_loaded_outside_gate():
    """Each signal keeps its own config/code version; evidence is not shared."""
    from strategies.pit_oos_replayer import replay_pit_oos

    conn = duckdb.connect(":memory:")
    try:
        _seed_two_period_evidence(conn)
        result = replay_pit_oos(conn, minimum_periods=2, holding_days=2)
        filled = [i for i in result["trade_ledger"] if i["status"] == "FILLED"]
        # deduplicate by signal date; both periods should appear
        assert len({i["signal_date"] for i in filled}) == 2
        by_date = {i["signal_date"]: i for i in filled}
        assert by_date["2026-01-02"]["config_hash"] == "cfg1"
        assert by_date["2026-01-05"]["config_hash"] == "cfg2"
    finally:
        conn.close()


def test_pit_replay_treats_repeated_signal_date_as_same_period():
    """Two pipelines on the same trade_date are one independent period; the
    later run supersedes the earlier one, never double-counts."""
    from strategies.pit_oos_replayer import replay_pit_oos

    conn = duckdb.connect(":memory:")
    try:
        _seed_two_period_evidence(conn)
        ev = json.dumps({"source_as_of": {"stock_daily_core": "2026-01-02"}})
        conn.execute(
            "INSERT INTO decision_evidence (pipeline_run_id, code, name, trade_date, verdict, "
            "conviction, evidence_json, evidence_hash, config_hash, code_sha) "
            "VALUES ('r1b', '000001', 'B', '2026-01-02', 'BUY', 'HIGH', ?, 'h', 'cfgb', 'shab')",
            [ev],
        )
        result = replay_pit_oos(conn, minimum_periods=2, holding_days=2)
        filled = [i for i in result["trade_ledger"] if i["status"] == "FILLED"]
        # same signal date -> one period, but the later run (r1b) wins
        by_date = {i["signal_date"]: i for i in filled}
        assert len(by_date) == 2
        assert by_date["2026-01-02"]["config_hash"] == "cfgb"
    finally:
        conn.close()
