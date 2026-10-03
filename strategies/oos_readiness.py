"""Fail-closed readiness check for replayable strategy OOS evidence."""
from __future__ import annotations

import json

_UNKNOWN_MARKERS = {"UNKNOWN", "ERROR", "NONE", "TODO", "PENDING", ""}


def _parse_as_of_date(value):
    """Parse a source-as-of date into ``datetime.date``; ``None`` if unusable."""
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # Accept both compact (YYYYMMDD) and ISO-ish (YYYY-MM-DD) forms.
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return _dt.datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text[:8], fmt).date()
        except ValueError:
            continue
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _source_as_of_is_valid(value, signal_date=None) -> bool:
    """Validate the source-as-of map retained inside immutable evidence JSON.

    D07: the original check only asked "is this a non-empty dict".  That accepts
    ``{"factor_scores": "TODO"}`` and, worse, a source timestamp *later* than the
    signal date — i.e. evidence built from data that did not exist yet at
    decision time.  Here every entry must be a real date, and when
    ``signal_date`` is supplied it must not be earlier than the source.

    An unparseable entry fails closed: it cannot be proven point-in-time.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if not isinstance(value, dict) or not value:
        return False
    signal = _parse_as_of_date(signal_date) if signal_date is not None else None
    for raw in value.values():
        if raw is None:
            return False
        text = str(raw).strip()
        if not text or text.upper() in _UNKNOWN_MARKERS:
            return False
        source_date = _parse_as_of_date(raw)
        if source_date is None:
            # Cannot prove this entry is a real point-in-time value.
            return False
        if signal is not None and source_date > signal:
            # Source data postdates the decision → future information.
            return False
    return True


def assess_oos_readiness(conn, *, minimum_periods: int = 36) -> dict:
    """Assess whether enough fully versioned PIT evidence exists to start replay.

    This deliberately reports no return, Sharpe, or drawdown. Those require a
    completed transaction-level replay with explicit execution assumptions.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info('decision_evidence')").fetchall()}
    if "source_as_of" in columns:
        rows = conn.execute(
            """SELECT trade_date, source_as_of, config_hash, code_sha
               FROM decision_evidence
               WHERE verdict IN ('BUY', 'STRONG_BUY')"""
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT trade_date, evidence_json, config_hash, code_sha
               FROM decision_evidence
               WHERE verdict IN ('BUY', 'STRONG_BUY')"""
        ).fetchall()
        rows = [
            (trade_date, (json.loads(evidence_json).get("source_as_of") if evidence_json else None),
             config_hash, code_sha)
            for trade_date, evidence_json, config_hash, code_sha in rows
        ]
    invalid = sum(
        1 for trade_date, source_as_of, config_hash, code_sha in rows
        # D07: 把 signal_date(trade_date) 传进去做时间先后校验，
        # 否则"未来源日"与"垃圾字符串"都会被当作有效证据。
        if not _source_as_of_is_valid(source_as_of, trade_date)
        or not config_hash
        or not code_sha
    )
    eligible_periods = len({trade_date for trade_date, _, _, _ in rows})
    return {
        "status": "READY_FOR_REPLAY" if invalid == 0 and eligible_periods >= minimum_periods else "NOT_READY",
        "eligible_periods": eligible_periods,
        "minimum_periods": minimum_periods,
        "missing_periods": max(minimum_periods - eligible_periods, 0),
        "invalid_provenance_rows": invalid,
    }
