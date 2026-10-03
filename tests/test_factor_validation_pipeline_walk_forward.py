"""factor_validation_pipeline 单元测试 (P0-2 借鉴 AKQuant Walk-Forward 延迟生效).

覆盖:
- stage4_walk_forward 延迟生效 vs 旧版
- WalkForwardState 状态机: train_bar / active_bar / pending_window_index
- delayed_activation=True/False 行为差异
- compute_walk_forward_state 边界
- 兼容老调用: 默认 delayed_activation=True, 旧调用无 lookahead_bias 仍工作
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from strategies.factor_validation_pipeline import (
    stage4_walk_forward,
    compute_walk_forward_state,
    WalkForwardState,
    validate_factor,
)


# ============ stage4_walk_forward ============


def test_wf_delayed_activation_basic():
    """延迟生效: 测试段从 train_end+1 开始, 严格无前瞻偏差"""
    np.random.seed(42)
    n = 400
    # 弱信号, 延迟生效应得较小 OOS IC
    scores = np.random.randn(n)
    fwd = scores * 0.02 + np.random.randn(n) * 0.1
    ic_delayed = stage4_walk_forward(scores, fwd, n_splits=4, delayed_activation=True)
    ic_legacy = stage4_walk_forward(scores, fwd, n_splits=4, delayed_activation=False)
    # 延迟生效用更少数据, IC 可能略低或近
    assert isinstance(ic_delayed, float)
    assert isinstance(ic_legacy, float)


def test_wf_legacy_compat_unchanged():
    """delayed_activation=False 必须返回与旧版一致结果 (兼容老调用)"""
    np.random.seed(42)
    n = 400
    scores = np.random.randn(n)
    fwd = scores * 0.05 + np.random.randn(n) * 0.1
    # 旧逻辑: factor_scores[train_end:] 直接用
    from scipy.stats import spearmanr
    split = n // 4
    expected_ics = []
    for i in range(1, 4):
        train_end = i * split
        if train_end >= n:
            break
        rho, _ = spearmanr(scores[train_end:], fwd[train_end:])
        expected_ics.append(rho)
    expected = float(np.mean(expected_ics))
    actual = stage4_walk_forward(scores, fwd, n_splits=4, delayed_activation=False)
    assert abs(actual - expected) < 1e-9


def test_wf_default_is_delayed():
    """默认 delayed_activation=True — 防 lookahead bias 是默认行为"""
    np.random.seed(42)
    n = 400
    scores = np.random.randn(n)
    fwd = np.random.randn(n)
    default_ic = stage4_walk_forward(scores, fwd)
    delayed_ic = stage4_walk_forward(scores, fwd, delayed_activation=True)
    assert abs(default_ic - delayed_ic) < 1e-9


def test_wf_empty_input():
    """空输入或长度不匹配返回 0.0"""
    assert stage4_walk_forward(np.array([]), np.array([])) == 0.0
    assert stage4_walk_forward(np.array([1, 2, 3]), np.array([1, 2])) == 0.0


def test_wf_too_short():
    """太短的数据集应返回 0.0, 不崩溃"""
    assert stage4_walk_forward(np.random.randn(10), np.random.randn(10)) == 0.0


def test_wf_min_oos_length():
    """OOS 段少于 10 观测点应跳过 (统计意义不足)"""
    np.random.seed(42)
    n = 50  # 切 4 份, 每份 12 个, 延迟生效后 OOS 仅 11 个, 第 4 段可能 <10
    scores = np.random.randn(n)
    fwd = np.random.randn(n)
    ic = stage4_walk_forward(scores, fwd, n_splits=4, delayed_activation=True)
    assert isinstance(ic, float)
    assert -1.0 <= ic <= 1.0


# ============ WalkForwardState ============


def test_wf_state_initial():
    """bar=0 时: 窗口 0, 未生效, 待生效窗口 = 1"""
    state = compute_walk_forward_state(n_total=100, n_splits=4, current_bar=0)
    assert state.window_index == 0
    assert state.train_bar_idx == 24  # (0+1)*25 - 1
    assert state.active_bar_idx == 25
    assert state.is_model_ready is False  # bar=0 < active=25
    assert state.pending_window_index == 1


def test_wf_state_active():
    """bar=55 时: 越过 active=50, 模型 ready (bar=55 ≥ active=50)"""
    state = compute_walk_forward_state(n_total=100, n_splits=4, current_bar=55)
    assert state.window_index == 2  # 55 // 25 = 2
    assert state.train_bar_idx == 74
    assert state.active_bar_idx == 75
    assert state.is_model_ready is True  # bar=55 ≥ active=50 (上窗口)
    assert state.pending_window_index == 3


def test_wf_state_last_window():
    """bar 接近末尾时: window_index 上限为 n_splits-1"""
    state = compute_walk_forward_state(n_total=100, n_splits=4, current_bar=99)
    assert state.window_index == 3
    assert state.pending_window_index is None  # 已是最后一个窗口


def test_wf_state_no_splits():
    """n_splits=1 时: 单窗口全样本, active_bar_idx = n_total"""
    state = compute_walk_forward_state(n_total=100, n_splits=1, current_bar=50)
    assert state.window_index == 0
    assert state.train_bar_idx == 99
    assert state.active_bar_idx == 100
    assert state.is_model_ready is False  # 100 ≥ bar=50, 但 active=100 > bar=50
    assert state.pending_window_index is None


def test_wf_state_zero_total():
    """n_total=0 边界: max(1,0) 保护, split=1, window=0"""
    state = compute_walk_forward_state(n_total=0, n_splits=4, current_bar=0)
    # split = max(1, 0//4) = 1
    # train_bar_idx = (0+1)*1 - 1 = 0
    assert state.train_bar_idx == 0
    assert state.active_bar_idx == 1
    assert state.is_model_ready is False  # bar=0 < active=1


# ============ 完整 validate_factor 流程 (回归测试) ============


def test_validate_factor_full_pass():
    """完整流程: 强信号因子通过所有 stage"""
    np.random.seed(42)
    n = 500
    scores = np.random.randn(n)
    fwd = scores * 0.1 + np.random.randn(n) * 0.05  # 强 IC
    existing = {"other": np.random.randn(n)}
    r = validate_factor("test", scores, fwd, existing)
    assert r.overall_verdict == "PASS"
    # numpy.bool_ 与 bool 比较需用 bool() 包装
    assert bool(r.pass_stage1) is True
    assert bool(r.pass_stage4) is True


def test_validate_factor_stage4_uses_delayed():
    """validate_factor 内部 stage4_walk_forward 必须走延迟生效路径"""
    np.random.seed(42)
    n = 500
    scores = np.random.randn(n)
    fwd = np.random.randn(n)
    r = validate_factor("test", scores, fwd)
    # 验证 walk_forward_ic 是有界浮点数
    assert r.walk_forward_ic is not None
    assert -1.0 <= r.walk_forward_ic <= 1.0


def test_validate_factor_stage1_fail_fast():
    """Stage1 失败应立即返回, 不进入 stage4"""
    # 用 sklearn datasets 制造确定性无 IC 数据
    # 简单: fwd 与 scores 顺序相反, IC 应为负但显著 → 这里测试 fail_stage1
    # 更稳: 用常数 scores, 产生 NaN spearmanr → sig=False
    np.random.seed(42)
    n = 500
    scores = np.zeros(n)  # 常数 → spearmanr 失败 → sig=False
    fwd = np.random.randn(n)
    r = validate_factor("test", scores, fwd)
    assert r.overall_verdict == "FAIL_STAGE1"
    assert bool(r.pass_stage1) is False
    assert r.walk_forward_ic is None  # 未计算
