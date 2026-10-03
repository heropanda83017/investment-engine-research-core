"""Tests for cross-sectional market-cap and industry neutralization."""

import pandas as pd
import pytest

from strategies.factor_neutralization import neutralize_cross_section


def test_neutralization_removes_industry_means_and_market_cap_exposure():
    frame = pd.DataFrame(
        {
            "factor": [10.0, 20.0, 30.0, 40.0],
            "forward_return": [1.0, 2.0, 3.0, 4.0],
            "log_mv": [1.0, 2.0, 3.0, 4.0],
            "industry": ["A", "A", "B", "B"],
        }
    )

    result = neutralize_cross_section(
        frame,
        factor_columns=["factor"],
        market_cap_column="log_mv",
        industry_column="industry",
    )

    assert result["status"] == "evaluated"
    residual = result["frame"]["factor_neutralized"]
    assert residual.groupby(frame["industry"]).mean().abs().max() == pytest.approx(0.0)
    assert result["diagnostics"]["market_cap_r2"]["factor"] < 1e-12


def test_neutralization_fail_closes_missing_group_column():
    frame = pd.DataFrame({"factor": [1.0, 2.0], "log_mv": [1.0, 2.0]})

    result = neutralize_cross_section(
        frame,
        factor_columns=["factor"],
        market_cap_column="log_mv",
        industry_column="industry",
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == "MISSING_INDUSTRY_COLUMN"
