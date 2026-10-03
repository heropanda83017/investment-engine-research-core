#!/usr/bin/env python3
"""cpcv.py — 组合式分块IC稳定性诊断

对因子IC序列做组合式分块交叉验证，用purge/embargo控制训练-测试集的时间隔离。
**不训练模型**，只验证IC在各分块间的稳定性。

核心逻辑:
  1. 按时间顺序将IC序列分成 n_groups 个连续块
  2. 取 n_test_groups 个块作为测试组合，其余为训练
  3. purge: 剔除训练集中与测试集时间重叠的样本（交易日口径）
  4. embargo: 剔除测试集之后指定交易日内的训练样本（信息泄漏期）
  5. 用训练集到测试集的IC均值变化评估稳定性

差异（vs López de Prado 2018）:
  - 原版需训练模型并比较train/test预测误差
  - 本实现只计算测试折IC均值，purge/embargo只决定分割是否保留

# 交易日口径:
  purge/embargo 使用真实交易日历计算。
  传入 trading_calendar（全部A股交易日列表）后，每个日期被映射到交易日历位置。
  purge=N = 剔除训练集中与测试集相距 N 个交易日以内的样本。
  **若未传入 trading_calendar 或覆盖不足，生产诊断返回 CALENDAR_UNCOVERED，
  不降级计算。** 观测位置口径仅作为显式实验模式可用。

用法:
  from strategies.cpcv import CPCV, cpv_validate

  cv = CPCV(n_groups=6, n_test_groups=2, embargo=5)
  for train_idx, test_idx in cv.split(X, dates):
      ...

  result = cpv_validate(ics, dates, trading_calendar=trading_calendar)
  # {"passed": bool, "mean_ic": float, "icir": float, ...}
"""

import logging
import numpy as np
from itertools import combinations
from typing import Generator, List, Optional, Tuple

log = logging.getLogger("cpcv")


class CPCV:
    """Combinatorial Purged Cross-Validation"""

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 purge: int = 5, embargo: int = 5):
        if n_groups <= n_test_groups:
            raise ValueError(f"n_groups({n_groups}) must > n_test_groups({n_test_groups})")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge = purge
        self.embargo = embargo

    def split(self, X: np.ndarray, dates: np.ndarray,
              y: np.ndarray = None,
              trading_calendar: Optional[np.ndarray] = None
              ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """生成训练/测试分割

        Args:
            X: 特征数组（未使用，仅保持API兼容）
            dates: 日期数组（datetime64）
            y: 标签数组（可选）
            trading_calendar: 全部交易日列表（datetime64），用于精确交易日位置映射
        """
        n = len(X)
        if n < self.n_groups * 2:
            split = n // 2
            yield np.arange(split), np.arange(split, n)
            return

        sort_idx = np.argsort(dates)
        sorted_dates = dates[sort_idx]

        # 交易日位置映射
        if trading_calendar is not None:
            _pos_of_date = {str(d)[:10]: i for i, d in enumerate(trading_calendar)}
            pos_of_idx = np.array([
                _pos_of_date.get(str(d)[:10], -1) for d in dates
            ], dtype=np.int64)
            # 任何日期未映射 → 无法使用交易日历，不产生有效分割
            if (pos_of_idx < 0).any():
                log.warning(f"交易日历覆盖不足: {int((pos_of_idx < 0).sum())}/{n} 个日期未映射，不产生有效分割")
                return
        else:
            # 回退：观测位置 = 排序后位置
            pos_of_idx = np.argsort(sort_idx)

        group_size = n // self.n_groups
        groups = []
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n
            groups.append(sort_idx[start:end])

        test_combos = list(combinations(range(self.n_groups), self.n_test_groups))
        max_combos = 20
        if len(test_combos) > max_combos:
            step = len(test_combos) // max_combos
            test_combos = test_combos[::step][:max_combos]

        for test_groups in test_combos:
            test_set = set(test_groups)
            train_groups = [g for g in range(self.n_groups) if g not in test_set]
            test_idx = np.concatenate([groups[g] for g in test_groups])
            train_idx = np.concatenate([groups[g] for g in train_groups])

            # 识别连续测试区间
            # 有交易日历时按位置差>1判断；无日历时按自然日差>1判断
            sorted_test = test_idx[np.argsort(dates[test_idx])]
            test_segments = []
            if len(sorted_test) > 0:
                seg_start = 0
                for i in range(1, len(sorted_test)):
                    if trading_calendar is not None:
                        # 交易日历位置差 > 1 表示非连续交易日
                        is_break = (pos_of_idx[sorted_test[i]] -
                                    pos_of_idx[sorted_test[i - 1]]) > 1
                    else:
                        # 自然日差 > 1 表示非连续日
                        is_break = (dates[sorted_test[i]] -
                                    dates[sorted_test[i - 1]]) > np.timedelta64(1, 'D')
                    if is_break:
                        test_segments.append(sorted_test[seg_start:i])
                        seg_start = i
                test_segments.append(sorted_test[seg_start:])

            # 对每个连续测试区间分别应用purge/embargo
            # 使用AND逻辑：只有所有区间都允许保留的样本才保留
            keep_mask = np.ones(len(train_idx), dtype=bool)
            train_pos = pos_of_idx[train_idx]

            for seg in test_segments:
                seg_pos = pos_of_idx[seg]
                seg_min_pos = seg_pos.min()
                seg_max_pos = seg_pos.max()

                # Purge: 剔除seg之前purge个交易日内的训练样本
                if self.purge > 0:
                    pre_purge = train_pos < (seg_min_pos - self.purge)
                else:
                    pre_purge = train_pos < seg_min_pos

                # Embargo: 剔除seg之后embargo个交易日内的训练样本
                if self.embargo > 0:
                    post_embargo = train_pos > (seg_max_pos + self.embargo)
                else:
                    post_embargo = train_pos > seg_max_pos

                # 保留: pre-test + post-test+embargo
                keep_mask &= (pre_purge | post_embargo)

            train_idx = train_idx[keep_mask]

            if len(train_idx) < 10 or len(test_idx) < 10:
                continue
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        combos = list(combinations(range(self.n_groups), self.n_test_groups))
        return min(len(combos), 20)


def cpv_validate(ics: List[float], dates: List[str],
                 n_groups: int = 6, n_test_groups: int = 2,
                 min_ic: float = 0.03, purge: int = 5, embargo: int = 5,
                 trading_calendar: Optional[List[str]] = None,
                 require_calendar: bool = True) -> dict:
    """用CPCV验证因子IC的稳定性

    Args:
        ics: IC值列表
        dates: 日期字符串列表（YYYY-MM-DD格式）
        n_groups: 分组数
        n_test_groups: 每组测试分组数
        min_ic: 最小IC阈值
        purge: 训练集与测试集之间的清洗交易日数（默认5）
        embargo: 训练集与测试集之间的隔离交易日数（默认5）
        trading_calendar: 全部交易日列表（YYYY-MM-DD），用于精确交易日位置映射。
                          必需提供，否则返回 CALENDAR_UNCOVERED。
        require_calendar: 默认为True。为True且trading_calendar为None时返回CALENDAR_UNCOVERED。
                          设为False时使用观测位置口径（仅用于实验模式）。
    """
    if len(ics) < 20:
        return {"passed": False, "mean_ic": 0, "ic_std": 0,
                "icir": 0, "n_splits": 0, "positive_ratio": 0,
                "reason": f"样本不足({len(ics)}天)", "reason_code": "INSUFFICIENT_DATA"}

    if require_calendar and trading_calendar is None:
        return {"passed": False, "mean_ic": 0, "ic_std": 0,
                "icir": 0, "n_splits": 0, "positive_ratio": 0,
                "reason": "交易日历缺失，无法计算交易日口径CPCV",
                "reason_code": "CALENDAR_UNCOVERED"}

    if trading_calendar is not None:
        # 验证日历覆盖所有输入日期
        cal_set = set(trading_calendar)
        missing = [d for d in dates if d not in cal_set]
        if missing:
            return {"passed": False, "mean_ic": 0, "ic_std": 0,
                    "icir": 0, "n_splits": 0, "positive_ratio": 0,
                    "reason": f"交易日历覆盖不足: {len(missing)}/{len(dates)} 日期未映射",
                    "reason_code": "CALENDAR_UNCOVERED"}

    X = np.zeros((len(ics), 1))
    dates_np = np.array(dates, dtype='datetime64[D]')
    y = np.array(ics, dtype=np.float32)
    cv = CPCV(n_groups=n_groups, n_test_groups=n_test_groups, purge=purge, embargo=embargo)

    cal_np = None
    if trading_calendar is not None:
        cal_np = np.array(trading_calendar, dtype='datetime64[D]')

    fold_ics = []
    for train_idx, test_idx in cv.split(X, dates_np, y, trading_calendar=cal_np):
        if len(test_idx) == 0:
            continue
        test_ic = float(y[test_idx].mean())
        fold_ics.append(test_ic)

    if not fold_ics:
        return {"passed": False, "mean_ic": 0, "ic_std": 0,
                "icir": 0, "n_splits": 0, "positive_ratio": 0,
                "reason": "CPCV无有效分割",
                "reason_code": "NO_VALID_SPLITS"}

    mean_ic = float(np.mean(fold_ics))
    ic_std = float(np.std(fold_ics))
    icir = mean_ic / max(ic_std, 0.001) * (len(fold_ics) ** 0.5)
    positive_ratio = float(np.mean([1.0 if ic > 0 else 0.0 for ic in fold_ics]))
    passed = mean_ic > min_ic and positive_ratio >= 0.6

    log.info(f"  [CPCV] n_splits={len(fold_ics)}, mean_IC={mean_ic:+.4f}, "
             f"ICIR={icir:.2f}, positive_ratio={positive_ratio:.0%}, passed={passed}")
    return {"passed": passed, "mean_ic": mean_ic, "ic_std": ic_std,
            "icir": icir, "n_splits": len(fold_ics), "positive_ratio": positive_ratio,
            "reason_code": "EVALUATED"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    ics = np.random.randn(500) * 0.05 + 0.02
    dates = [f"2024-01-{d%31+1:02d}" for d in range(500)]
    r = cpv_validate(ics.tolist(), dates)
    print(f"CPCV: passed={r['passed']}, ICIR={r['icir']:.2f}, pos_ratio={r['positive_ratio']:.0%}")
    print("✅ CPCV模块加载完成")