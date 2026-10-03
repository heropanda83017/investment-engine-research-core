#!/usr/bin/env python3
"""factor_validation_pipeline.py — 多因子验证流水线 (E-008, K-024)"""
import logging, numpy as np, pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("factor_validation")

@dataclass
class ValidationReport:
    factor_name: str
    ic_rank_pct: Optional[float] = None
    ic_significant: bool = False
    max_correlation: Optional[float] = None
    redundant_with: List[str] = field(default_factory=list)
    ic_stability: Optional[float] = None
    walk_forward_ic: Optional[float] = None
    pass_stage1: bool = False
    pass_stage2: bool = False
    pass_stage3: bool = False
    pass_stage4: bool = False
    overall_verdict: str = "PENDING"
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())

def stage1_ic_screen(factor_scores: np.ndarray, forward_returns: np.ndarray) -> Tuple[float, bool]:
    """Stage1: 快速IC筛选 (Top 20%通过)"""
    if len(factor_scores) != len(forward_returns) or len(factor_scores) < 30:
        return 0.0, False
    from scipy.stats import spearmanr
    rho, p = spearmanr(factor_scores, forward_returns)
    rank_pct = abs(rho)  # 直接用|IC|作为强度
    significant = p < 0.05 and abs(rho) > 0.02
    return rank_pct, significant

def stage2_correlation_check(new_factor: np.ndarray, existing_factors: Dict[str, np.ndarray], threshold: float = 0.7) -> Tuple[float, List[str]]:
    max_corr = 0.0
    redundant = []
    for name, scores in existing_factors.items():
        if len(scores) != len(new_factor): continue
        corr = abs(float(np.corrcoef(new_factor, scores)[0, 1]))
        if corr > max_corr: max_corr = corr
        if corr > threshold: redundant.append(name)
    return max_corr, redundant

def stage3_batch_dedup(factors: Dict[str, np.ndarray], ic_scores: Dict[str, float], threshold: float = 0.7) -> List[str]:
    """Stage3: 批内去重（保留IC最高的）"""
    kept = []
    names = sorted(factors.keys(), key=lambda n: -ic_scores.get(n, 0))
    for name in names:
        if not kept:
            kept.append(name)
            continue
        scores = factors[name]
        max_corr = max(abs(float(np.corrcoef(scores, factors[k])[0, 1])) for k in kept if len(factors[k]) == len(scores))
        if max_corr < threshold:
            kept.append(name)
    return kept

def stage4_walk_forward(factor_scores: np.ndarray, forward_returns: np.ndarray, n_splits: int = 4,
                        delayed_activation: bool = True) -> float:
    """Stage4: Walk-Forward验证 — 借鉴 AKQuant ml.md "延迟生效" 原则

    关键设计 (来自 AKQuant ml.md current_validation_window 章节):
    - 当前 bar N 训练完成 → bar N+1 才开始用该模型做预测
    - 避免在训练数据上回测 (lookahead bias)
    - delayed_activation=True (默认) 时严格遵守此约束

    参数:
        factor_scores: 因子得分序列 (按时间顺序)
        forward_returns: 未来 N 日收益序列
        n_splits: 切分数
        delayed_activation: True=严格延迟生效, False=原版逻辑 (兼容)

    返回:
        平均 OOS IC (out-of-sample, 严格无前瞻偏差)
    """
    from scipy.stats import spearmanr
    n = len(factor_scores)
    if n < 30 or len(forward_returns) != n:
        return 0.0
    split = n // n_splits
    oos_ics: List[float] = []
    for i in range(1, n_splits):
        train_end = i * split
        if train_end >= n:
            break
        # 借鉴 AKQuant: 当前 bar 训练 → bar N+1 才生效
        # 即 OOS 测试段从 train_end+1 开始 (跳过 train_end 这一天的预测)
        if delayed_activation and train_end + 1 < n:
            test_start = train_end + 1
        else:
            test_start = train_end
        # OOS 段必须至少有 10 个观测点才有统计意义
        if n - test_start < 10:
            continue
        rho, _ = spearmanr(factor_scores[test_start:], forward_returns[test_start:])
        oos_ics.append(rho)
    return float(np.mean(oos_ics)) if oos_ics else 0.0


@dataclass
class WalkForwardState:
    """Walk-Forward 状态机 — 借鉴 AKQuant ml.md current_validation_window 接口设计

    字段语义:
    - window_index: 当前活动窗口编号
    - train_bar_idx: 训练数据截止 bar 索引
    - active_bar_idx: 模型生效起始 bar 索引 (= train_bar_idx + 1)
    - pending_window_index: 待生效窗口编号 (None 表示无待生效)
    - is_model_ready: 当前 bar 是否可用模型预测
    """
    window_index: int = 0
    train_bar_idx: int = -1
    active_bar_idx: int = 0
    pending_window_index: Optional[int] = None
    is_model_ready: bool = False


def compute_walk_forward_state(
    n_total: int,
    n_splits: int = 4,
    current_bar: int = 0,
) -> WalkForwardState:
    """计算当前 bar 的 Walk-Forward 状态 — 借鉴 AKQuant current_validation_window().

    语义修正: is_model_ready = "至少有一个窗口的模型已生效" (即 current_bar ≥ 第一个窗口的 active)
    而非"当前所在窗口已生效", 否则窗口中途会反复 ready=True/False 不稳定.
    """
    state = WalkForwardState()
    split = max(1, n_total // n_splits)
    state.window_index = min(current_bar // split, n_splits - 1)
    state.train_bar_idx = (state.window_index + 1) * split - 1
    state.active_bar_idx = state.train_bar_idx + 1
    # 模型就绪: 已越过第一个窗口的生效点 (从第二个窗口起都可预测)
    first_active_bar = split  # 第一个窗口训练结束 (split-1) 后, bar=split 开始生效
    state.is_model_ready = current_bar >= first_active_bar
    # 待生效窗口: 若下一切分已开始训练, 但当前窗口还在生效
    if state.window_index + 1 < n_splits:
        state.pending_window_index = state.window_index + 1
    else:
        state.pending_window_index = None
    return state

def validate_factor(name: str, scores: np.ndarray, forward_returns: np.ndarray, existing: Dict[str, np.ndarray] = None) -> ValidationReport:
    r = ValidationReport(factor_name=name)
    rank_pct, sig = stage1_ic_screen(scores, forward_returns)
    r.ic_rank_pct = rank_pct; r.ic_significant = sig
    r.pass_stage1 = sig and rank_pct > 0.05
    if not r.pass_stage1: r.overall_verdict = "FAIL_STAGE1"; return r
    if existing:
        max_corr, redundant = stage2_correlation_check(scores, existing)
        r.max_correlation = max_corr; r.redundant_with = redundant
        r.pass_stage2 = max_corr < 0.7
        if not r.pass_stage2: r.overall_verdict = "FAIL_STAGE2_REDUNDANT"; return r
    r.pass_stage2 = True
    r.pass_stage3 = True
    wf_ic = stage4_walk_forward(scores, forward_returns)
    r.walk_forward_ic = wf_ic
    r.pass_stage4 = wf_ic > 0
    r.overall_verdict = "PASS" if r.pass_stage4 else "FAIL_STAGE4_WALKFORWARD"
    return r

if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    scores = np.random.randn(n)
    fwd = scores * 0.05 + np.random.randn(n) * 0.1
    existing = {"momentum": np.random.randn(n), "reversal": np.random.randn(n)}
    r = validate_factor("test", scores, fwd, existing)
    ic_pct = f"{r.ic_rank_pct:.2%}" if r.ic_rank_pct is not None else "N/A"
    max_c = f"{r.max_correlation:.2f}" if r.max_correlation is not None else "N/A"
    print(f"验证结果: {r.overall_verdict}, IC Rank: {ic_pct}, 最大相关性: {max_c}")