"""test_factor_validation_pipeline.py"""
import pytest, numpy as np
from strategies.factor_validation_pipeline import (
    stage1_ic_screen, stage2_correlation_check, stage4_walk_forward, validate_factor
)

class TestValidation:
    def test_strong_factor(self):
        np.random.seed(42); n=500
        scores = np.random.randn(n)
        fwd = scores * 0.3 + np.random.randn(n) * 0.1
        r = validate_factor("strong", scores, fwd, {"mom": np.random.randn(n)})
        assert r.overall_verdict in ("PASS", "FAIL_STAGE2_REDUNDANT")

    def test_weak_factor(self):
        scores = np.random.randn(500)
        fwd = np.random.randn(500)
        r = validate_factor("weak", scores, fwd)
        assert r.overall_verdict == "FAIL_STAGE1"

    def test_insufficient_data(self):
        rank, sig = stage1_ic_screen(np.array([1,2,3]), np.array([1,2,3]))
        assert not sig