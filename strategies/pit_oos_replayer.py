"""Read-only PIT decision replay with T+1-open execution constraints.

This module is evidence plumbing, not a strategy-performance promotion path.
It emits performance only after the independent-period readiness gate passes.
"""
from __future__ import annotations

import json


def _date(value) -> str:
    return str(value)[:10]


def _evidence_rows(conn) -> list[dict]:
    """Load valid BUY evidence, superseding repeated signal dates by latest run.

    Each entry keeps its own decision version (config_hash/code_sha) and the
    data as-of snapshot captured at decision time; a later pipeline run on the
    same signal date supersedes the earlier one and is never double counted.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info('decision_evidence')").fetchall()}
    if "created_at" in columns:
        rows = conn.execute(
            """SELECT code, trade_date, evidence_json, config_hash, code_sha, created_at
               FROM decision_evidence WHERE verdict IN ('BUY', 'STRONG_BUY')
               ORDER BY trade_date, code, created_at ASC NULLS FIRST, pipeline_run_id ASC"""
        ).fetchall()
    else:
        # Minimal fixtures omit created_at; later pipeline_run_id wins on tie.
        rows = conn.execute(
            """SELECT code, trade_date, evidence_json, config_hash, code_sha, NULL
               FROM decision_evidence WHERE verdict IN ('BUY', 'STRONG_BUY')
               ORDER BY trade_date, code, pipeline_run_id ASC"""
        ).fetchall()
    latest: dict[tuple[str, str], dict] = {}
    from strategies.oos_readiness import _source_as_of_is_valid

    for code, trade_date, evidence_json, config_hash, code_sha, created_at in rows:
        try:
            payload = json.loads(evidence_json)
            source_as_of = payload.get("source_as_of")
        except (TypeError, json.JSONDecodeError):
            source_as_of = None
        # D07: 回放同样必须校验时点先后（源日不得晚于信号日），
        # 仅检查"非空 dict"会把未来信息当作可回放证据。
        if not (isinstance(source_as_of, dict) and source_as_of and config_hash and code_sha):
            continue
        if not _source_as_of_is_valid(source_as_of, trade_date):
            continue
        key = (str(code), _date(trade_date))
        latest[key] = {
            "code": str(code),
            "signal_date": _date(trade_date),
            "source_as_of": source_as_of,
            "config_hash": str(config_hash),
            "code_sha": str(code_sha),
        }
    return list(latest.values())


def _next_bar(conn, code: str, signal_date: str):
    return conn.execute(
        """SELECT trade_date, open, close, up_limit, down_limit, vol, suspend_timing
           FROM stock_daily_core
           WHERE SUBSTR(ts_code, 1, 6)=? AND trade_date > CAST(? AS DATE)
           ORDER BY trade_date LIMIT 1""",
        [code, signal_date],
    ).fetchone()


def _benchmark_return(conn, signal_date: str, exit_date: str, code: str):
    rows = conn.execute(
        """SELECT trade_date, close FROM stock_daily_core
           WHERE SUBSTR(ts_code, 1, 6)=? AND trade_date >= CAST(? AS DATE)
             AND trade_date <= CAST(? AS DATE) ORDER BY trade_date""",
        [code, signal_date, exit_date],
    ).fetchall()
    if len(rows) < 2 or not rows[0][1] or not rows[-1][1]:
        return None
    return float(rows[-1][1]) / float(rows[0][1]) - 1


def replay_pit_oos(
    conn, *, minimum_periods: int = 36, holding_days: int = 1,
    commission_bps: float = 3.0, slippage_bps: float = 5.0,
    stamp_tax_bps: float = 5.0, benchmark_code: str = "000300",
) -> dict:
    """Replay immutable BUY evidence without mutating the database.

    The next actual stock bar is used, never calendar/business-day approximation.
    Suspensions, missing opens and limit-up entries remain unfilled; no invented
    fill.  Net return subtracts both-side commission and slippage plus a
    sell-side stamp tax.  Each ledger row keeps its decision version and the
    data as-of snapshot frozen at decision time.
    """
    evidence = _evidence_rows(conn)
    periods = len({entry["signal_date"] for entry in evidence})
    base = {
        "execution_contract": "signal_asof_close_to_next_bar_open",
        "minimum_periods": minimum_periods,
        "eligible_periods": periods,
        "benchmark": {"code": benchmark_code, "return": None, "excess_return": None},
        "trade_ledger": [],
        "nav_curve": [],
    }
    if periods < minimum_periods:
        return {**base, "status": "NOT_READY", "performance": None,
                "performance_reason": "insufficient independently versioned PIT evidence"}

    cost_bps = 2 * (float(commission_bps) + float(slippage_bps)) + float(stamp_tax_bps)
    net_returns, benchmark_returns = [], []
    for entry in evidence:
        code, signal_date = entry["code"], entry["signal_date"]
        bar = _next_bar(conn, code, signal_date)
        if not bar:
            base["trade_ledger"].append({**entry, "status": "UNFILLED_NO_NEXT_BAR"})
            continue
        entry_date, entry_open, entry_close, up_limit, _, volume, suspension = bar
        entry_date = _date(entry_date)
        if suspension or not entry_open or not entry_close or not volume:
            base["trade_ledger"].append({**entry, "status": "UNFILLED_SUSPENDED"})
            continue
        if up_limit is not None and float(entry_open) >= float(up_limit):
            base["trade_ledger"].append({**entry, "status": "UNFILLED_LIMIT_UP"})
            continue
        # B04: 退出判定必须与入场同样严格。原实现只取 close，会在退出日停牌/跌停
        # 或零成交时伪造一笔无法执行的卖出。改为取该 bar 的完整字段并逐项校验。
        exit_row = conn.execute(
            """SELECT trade_date, close, down_limit, vol, suspend_timing FROM stock_daily_core
               WHERE SUBSTR(ts_code, 1, 6)=? AND trade_date >= CAST(? AS DATE)
               ORDER BY trade_date LIMIT 1 OFFSET ?""",
            [code, entry_date, max(0, int(holding_days))],
        ).fetchone()
        if not exit_row or not exit_row[1]:
            base["trade_ledger"].append({**entry, "status": "UNFILLED_NO_EXIT_BAR"})
            continue
        exit_date = _date(exit_row[0])
        exit_close = float(exit_row[1])
        exit_down_limit = exit_row[2]
        exit_volume = exit_row[3]
        exit_suspension = exit_row[4]
        if exit_suspension or not exit_volume:
            base["trade_ledger"].append(
                {**entry, "exit_date": exit_date, "status": "UNFILLED_EXIT_SUSPENDED"}
            )
            continue
        if exit_down_limit is not None and exit_close <= float(exit_down_limit):
            base["trade_ledger"].append(
                {**entry, "exit_date": exit_date, "status": "UNFILLED_EXIT_LIMIT_DOWN"}
            )
            continue
        gross_return = exit_close / float(entry_open) - 1
        net_return = gross_return - cost_bps / 10000
        item = {**entry, "entry_date": entry_date, "entry_open": float(entry_open),
                "exit_date": exit_date, "exit_close": exit_close,
                "gross_return": gross_return, "net_return": net_return,
                "cost_bps": cost_bps, "status": "FILLED"}
        base["trade_ledger"].append(item)
        net_returns.append(net_return)
        benchmark_return = _benchmark_return(conn, signal_date, exit_date, benchmark_code)
        if benchmark_return is not None:
            benchmark_returns.append(benchmark_return)
    if not net_returns:
        return {**base, "status": "READY", "performance": None,
                "performance_reason": "no executable fills under replay constraints"}

    mean_net = sum(net_returns) / len(net_returns)
    mean_benchmark = sum(benchmark_returns) / len(benchmark_returns) if benchmark_returns else None
    base["benchmark"]["return"] = mean_benchmark
    base["benchmark"]["excess_return"] = mean_net - mean_benchmark if mean_benchmark is not None else None

    # B03: 同一 signal_date 的多笔交易是**并行**建仓，不是串行复利。
    # 原实现逐笔 `nav *= 1 + net_return`，会把同日两只各涨 10% 串乘成 21%。
    # 正确做法：按 signal_date 分组，日内收益取等权平均，再跨日复利。
    #
    # ⚠️ R2（审计复核 2026-09-13）：以下序列是**事件研究（event-study）**口径，
    # 不是自融资组合净值——它没有现金占用、股数、逐日市值，也无法证明相邻信号
    # 不会复用尚未退出的资金。因此不再命名为 nav_curve/daily_returns，改为
    # event_* 前缀，避免被下游误当作组合 NAV 使用。
    # 真正的组合净值需先定义配置权重、现金、持仓与再平衡，按实际交易日估值。
    by_signal: dict[str, list[float]] = {}
    for item in base["trade_ledger"]:
        if item["status"] != "FILLED":
            continue
        by_signal.setdefault(str(item["signal_date"]), []).append(float(item["net_return"]))

    event_cum = 1.0
    event_curve = [1.0]
    event_returns: list[float] = []
    for signal_date in sorted(by_signal):
        day_returns = by_signal[signal_date]
        day_return = sum(day_returns) / len(day_returns)
        event_returns.append(day_return)
        event_cum *= 1 + day_return
        event_curve.append(round(event_cum, 12))

    base["event_study_curve"] = event_curve
    base["event_study_returns"] = event_returns
    base["event_study_method"] = "equal_weight_within_signal_date_then_compound"
    base["event_study_is_portfolio_nav"] = False  # 明确：这不是组合净值
    # 兼容字段：保留旧键名但标注为事件研究口径，防止下游 KeyError。
    base["nav_curve"] = event_curve
    base["daily_returns"] = event_returns
    base["nav_method"] = "EVENT_STUDY_ONLY__not_a_self_financing_portfolio_nav"
    return {**base, "status": "READY", "performance": {
        "mean_net_return": mean_net, "filled_trades": len(net_returns),
        "unfilled_trades": len(base["trade_ledger"]) - len(net_returns),
        "metric_scope": "event_study_not_portfolio_nav",
    }}
