"""Research-only cross-sectional factor neutralization helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _r2_against(x: pd.Series, y: pd.Series) -> float:
    values = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2 or values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return 0.0
    if float(np.var(values["x"].to_numpy(dtype=float))) < 1e-20:
        return 0.0
    corr = values["x"].corr(values["y"])
    return float(corr * corr) if pd.notna(corr) else 0.0


def neutralize_cross_section(
    frame: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    market_cap_column: str = "log_mv",
    industry_column: str = "industry",
) -> dict:
    """Remove market-cap and industry exposure from factor columns.

    A separate OLS regression is fitted for each factor using an intercept,
    market-cap control, and industry fixed effects.  The returned residuals
    are research outputs only; the input frame is never mutated.
    """
    if not factor_columns:
        return {"status": "blocked", "reason_code": "EMPTY_FACTOR_COLUMNS"}
    missing = [
        column
        for column in [*factor_columns, market_cap_column, industry_column]
        if column not in frame.columns
    ]
    if missing:
        reason = "MISSING_INDUSTRY_COLUMN" if industry_column in missing else "MISSING_COLUMN"
        return {"status": "blocked", "reason_code": reason, "missing_columns": missing}

    work = frame[[*factor_columns, market_cap_column, industry_column]].copy()
    work[market_cap_column] = pd.to_numeric(work[market_cap_column], errors="coerce")
    valid = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return {"status": "blocked", "reason_code": "MIN_SAMPLES", "samples": len(valid)}

    industry_dummies = pd.get_dummies(valid[industry_column].astype(str), drop_first=True, dtype=float)
    design = np.column_stack([
        np.ones(len(valid)),
        valid[market_cap_column].to_numpy(dtype=float),
        industry_dummies.to_numpy(dtype=float),
    ])
    output = frame.copy()
    diagnostics = {"market_cap_r2": {}}
    for factor in factor_columns:
        values = valid[factor].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coefficients
        output[f"{factor}_neutralized"] = pd.Series(residual, index=valid.index)
        diagnostics["market_cap_r2"][factor] = _r2_against(
            pd.Series(residual, index=valid.index), valid[market_cap_column]
        )

    return {
        "status": "evaluated",
        "reason_code": None,
        "samples": len(valid),
        "frame": output,
        "diagnostics": diagnostics,
    }
