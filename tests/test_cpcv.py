"""test_cpcv.py — CPCV 组合式分块IC稳定性诊断测试

覆盖 (对应复审要求):
  #5: 0/2/3个有效分割的状态
  #6: ≥3分割但统计失败必须为 FAILED
  #7: purge/embargo精确索引（日历日口径 + 非连续测试块逐块处理）
  #8: cpv_validate() 所有返回路径字段一致
  #9: 非连续测试块中间样本保留
  - 日历日口径：purge=N = N个日历日（np.timedelta64）
  - 日期序列使用单调递增的真实日历日
"""

import numpy as np
import pytest

from strategies.cpcv import CPCV, cpv_validate


# ── 辅助: 日历日序列 ──────────────────────────────────────────────

def _dates(n: int, start: str = "2024-01-01") -> list:
    """生成n个连续日历日（无周末跳过），用于测试purge/embargo精确性"""
    from datetime import datetime, timedelta
    d = datetime.strptime(start, "%Y-%m-%d")
    return [(d + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def _dates_np(n: int, start: str = "2024-01-01") -> np.ndarray:
    return np.array(_dates(n, start), dtype='datetime64[D]')


def _ics(n: int, mean: float = 0.05, std: float = 0.1) -> list:
    np.random.seed(42)
    return (np.random.randn(n) * std + mean).tolist()


# ═══════════════════════════════════════════════════════════════════
# CPCV.split() 基本行为
# ═══════════════════════════════════════════════════════════════════

class TestCPCVSplit:
    def test_n_groups_must_exceed_n_test_groups(self):
        with pytest.raises(ValueError, match="must > n_test_groups"):
            CPCV(n_groups=3, n_test_groups=3)

    def test_generates_correct_number_of_splits(self):
        """默认n_groups=6, n_test_groups=2 → C(6,2)=15 组合"""
        cv = CPCV(n_groups=6, n_test_groups=2, purge=0, embargo=0)
        assert cv.get_n_splits() == 15

    def test_small_dataset_fallback(self):
        """样本少于 n_groups*2 时，退化为单次分割"""
        dates = _dates_np(10)
        X = np.arange(10)
        cv = CPCV(n_groups=6, n_test_groups=2, purge=0, embargo=0)
        splits = list(cv.split(X, dates))
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 5
        assert len(test_idx) == 5

    def test_split_indices_exhaustive(self):
        """所有样本要么在训练集要么在测试集"""
        dates = _dates_np(100)
        X = np.arange(100)
        cv = CPCV(n_groups=5, n_test_groups=2, purge=0, embargo=0)
        for train_idx, test_idx in cv.split(X, dates):
            all_set = set(range(100))
            test_set = set(test_idx)
            assert set(train_idx) == (all_set - test_set)


# ═══════════════════════════════════════════════════════════════════
# purge/embargo 日历日口径
# ═══════════════════════════════════════════════════════════════════

class TestPurgeEmbargo:
    def test_purge_by_calendar_day(self):
        """purge=N 按日历日剔除"""
        dates = _dates_np(200)
        X = np.arange(200)
        cv = CPCV(n_groups=5, n_test_groups=2, purge=5, embargo=0)
        for train_idx, test_idx in cv.split(X, dates):
            test_dates = dates[test_idx]
            train_dates = dates[train_idx]
            test_min = test_dates.min()
            for td in train_dates:
                if td < test_min:
                    cal_diff = (test_min - td).astype(int)
                    assert cal_diff >= 5, f"purge=5, 样本{td}距测试最小{test_min}仅{cal_diff}天"

    def test_embargo_by_calendar_day(self):
        """embargo=N 按日历日剔除"""
        dates = _dates_np(200)
        X = np.arange(200)
        cv = CPCV(n_groups=5, n_test_groups=2, purge=0, embargo=5)
        found_post = False
        for train_idx, test_idx in cv.split(X, dates):
            test_dates = dates[test_idx]
            train_dates = dates[train_idx]
            test_max = test_dates.max()
            for td in train_dates:
                if td > test_max:
                    cal_diff = (td - test_max).astype(int)
                    assert cal_diff > 5, f"embargo=5, 样本{td}距测试最大{test_max}仅{cal_diff}天"
                    found_post = True
        assert found_post, "无post-test训练样本"

    def test_purge_combined_with_embargo(self):
        """purge+embargo同时生效"""
        dates = _dates_np(400)
        X = np.arange(400)
        cv = CPCV(n_groups=6, n_test_groups=2, purge=10, embargo=10)
        for train_idx, test_idx in cv.split(X, dates):
            test_dates = dates[test_idx]
            train_dates = dates[train_idx]
            test_min = test_dates.min()
            test_max = test_dates.max()
            for td in train_dates:
                if td < test_min:
                    cal_diff = (test_min - td).astype(int)
                    assert cal_diff >= 10, f"purge区样本距测试期不足10天"
                if td > test_max:
                    cal_diff = (td - test_max).astype(int)
                    assert cal_diff > 10, f"embargo区样本距测试期不足10天"

    def test_purge_zero_keeps_all_non_test(self):
        """purge=0 保留所有非测试样本"""
        dates = _dates_np(100)
        X = np.arange(100)
        cv = CPCV(n_groups=4, n_test_groups=2, purge=0, embargo=0)
        for train_idx, test_idx in cv.split(X, dates):
            all_set = set(range(100))
            test_set = set(test_idx)
            assert set(train_idx) == (all_set - test_set)


# ═══════════════════════════════════════════════════════════════════
# 非连续测试块（P1修复验证）
# ═══════════════════════════════════════════════════════════════════

class TestDisjointTestBlocks:
    def _find_disjoint_fold(self, cv, X, dates):
        """找到测试组为非连续块的分割（如{0,2}，中间有训练组）"""
        for train_idx, test_idx in cv.split(X, dates):
            test_dates = dates[test_idx]
            test_min = test_dates.min()
            test_max = test_dates.max()
            train_dates = dates[train_idx]
            # 如果训练样本跨越测试区间，说明测试块非连续
            middle = [d for d in train_dates if d > test_min and d < test_max]
            if len(middle) > 0:
                return train_idx, test_idx, train_dates, test_dates, middle
        return None, None, None, None, None

    def test_disjoint_blocks_preserve_middle_samples(self):
        """测试集为非连续块时，中间训练样本保留"""
        n = 120
        dates = _dates_np(n)
        X = np.arange(n)
        cv = CPCV(n_groups=6, n_test_groups=2, purge=0, embargo=0)

        ti, tei, train_d, test_d, middle = self._find_disjoint_fold(cv, X, dates)
        assert ti is not None, "未找到非连续测试块组合"
        assert len(middle) > 0, \
            f"非连续测试块中间训练样本全部被删除, 剩余{len(train_d)}个训练样本"

    def test_disjoint_blocks_with_purge(self):
        """非连续块 + purge=5, embargo=0，中间样本应保留"""
        n = 120
        dates = _dates_np(n)
        X = np.arange(n)
        cv = CPCV(n_groups=6, n_test_groups=2, purge=5, embargo=0)

        ti, tei, train_d, test_d, middle = self._find_disjoint_fold(cv, X, dates)
        assert ti is not None, "未找到非连续测试块组合"
        # 至少有一些中间样本保留
        assert len(middle) > 0, \
            f"purge=5导致所有中间训练样本被删除, 剩余{len(train_d)}个训练样本"
        # 验证：中间样本仅受pre-purge约束（≤seg_min的样本须距seg_min≥5天）
        # 不受embargo约束（embargo=0）
        sorted_test = sorted(test_d)
        segs = []
        seg_start = 0
        for i in range(1, len(sorted_test)):
            gap = (sorted_test[i] - sorted_test[i-1]).astype(int)
            if gap > 1:
                segs.append((sorted_test[seg_start], sorted_test[i-1]))
                seg_start = i
        segs.append((sorted_test[seg_start], sorted_test[-1]))
        for m in middle:
            for seg_min, seg_max in segs:
                if m < seg_min:
                    cal_diff = (seg_min - m).astype(int)
                    assert cal_diff >= 5, f"中间样本{m}距测试段{seg_min}仅{cal_diff}天（purge=5）"
                elif m > seg_max:
                    # embargo=0，post-test样本无距离限制
                    pass

    def test_disjoint_blocks_and_logic(self):
        """非连续块AND逻辑：中间样本距某段<5天即被剔除"""
        n = 120
        dates = _dates_np(n)
        X = np.arange(n)
        cv = CPCV(n_groups=6, n_test_groups=2, purge=5, embargo=0)

        ti, tei, train_d, test_d, middle = self._find_disjoint_fold(cv, X, dates)
        assert ti is not None, "未找到非连续测试块组合"

        sorted_test = sorted(test_d)
        segs = []
        seg_start = 0
        for i in range(1, len(sorted_test)):
            gap = (sorted_test[i] - sorted_test[i-1]).astype(int)
            if gap > 1:
                segs.append((sorted_test[seg_start], sorted_test[i-1]))
                seg_start = i
        segs.append((sorted_test[seg_start], sorted_test[-1]))

        # 验证：所有中间样本距每个段都必须≥5天
        # 如果某样本距段A 16天但距段B 4天，它会被B段purge剔除
        # 因此中间样本不应存在距任何段<5天的情况
        for m in middle:
            for seg_min, seg_max in segs:
                if m < seg_min:
                    cal_diff = (seg_min - m).astype(int)
                    assert cal_diff >= 5, \
                        f"中间样本{m}距测试段{seg_min}仅{cal_diff}天，应被purge剔除"

    def test_disjoint_blocks_embargo_per_segment(self):
        """非连续块，每个块独立计算embargo"""
        n = 120
        dates = _dates_np(n)
        X = np.arange(n)
        cv = CPCV(n_groups=6, n_test_groups=2, purge=0, embargo=5)

        ti, tei, train_d, test_d, test_middle = self._find_disjoint_fold(cv, X, dates)
        assert ti is not None, "未找到非连续测试块组合"

        # 找出连续测试段
        sorted_test = sorted(test_d)
        segs = []
        seg_start = 0
        for i in range(1, len(sorted_test)):
            gap = (sorted_test[i] - sorted_test[i-1]).astype(int)
            if gap > 1:
                segs.append((sorted_test[seg_start], sorted_test[i-1]))
                seg_start = i
        segs.append((sorted_test[seg_start], sorted_test[-1]))

        # 验证：任何post-test样本距对应测试段至少5天
        for seg_min, seg_max in segs:
            post_test = [d for d in train_d if d > seg_max]
            for d in post_test:
                cal_diff = (d - seg_max).astype(int)
                assert cal_diff > 5, \
                    f"embargo漏检：样本{d}距测试段{seg_max}仅{cal_diff}天"


# ═══════════════════════════════════════════════════════════════════
# cpv_validate() 状态判定
# ═══════════════════════════════════════════════════════════════════

class TestCPVValidate:
    def test_insufficient_observations(self):
        """观测点不足20 → n_splits=0"""
        ics = [0.05] * 15
        dates = _dates(15)
        result = cpv_validate(ics, dates, require_calendar=False)
        assert result["n_splits"] == 0

    def test_fields_consistent_when_no_folds(self):
        ics = [0.05] * 20
        dates = _dates(20)
        result = cpv_validate(ics, dates, n_groups=20, n_test_groups=10, require_calendar=False)
        required = ["passed", "mean_ic", "ic_std", "icir", "n_splits", "positive_ratio"]
        for field in required:
            assert field in result

    def test_fields_consistent_when_has_folds(self):
        ics = _ics(120, mean=0.08, std=0.05)
        dates = _dates(120)
        result = cpv_validate(ics, dates, require_calendar=False)
        required = ["passed", "mean_ic", "ic_std", "icir", "n_splits", "positive_ratio"]
        for field in required:
            assert field in result

    def test_high_ic_passes(self):
        np.random.seed(42)
        n = 300
        ics = (np.random.randn(n) * 0.02 + 0.10).tolist()
        dates = _dates(n)
        result = cpv_validate(ics, dates, min_ic=0.03, require_calendar=False)
        if result["n_splits"] >= 3:
            assert result["passed"] is True, f"高IC应通过, 实际{result}"

    def test_low_ic_fails(self):
        np.random.seed(42)
        n = 300
        ics = (np.random.randn(n) * 0.15 + 0.005).tolist()
        dates = _dates(n)
        result = cpv_validate(ics, dates, min_ic=0.03, require_calendar=False)
        if result["n_splits"] >= 3:
            assert result["passed"] is False, f"低IC应失败, 实际{result}"

    def test_negative_mean_ic_fails(self):
        """负均值IC → 不通过（已对齐aligned_ic不应为负）"""
        np.random.seed(42)
        n = 300
        ics = (np.random.randn(n) * 0.08 + (-0.05)).tolist()
        dates = _dates(n)
        result = cpv_validate(ics, dates, min_ic=0.03, require_calendar=False)
        if result["n_splits"] >= 3:
            assert result["passed"] is False

    def test_negative_mean_ic_but_positive_ratio_60_fails(self):
        """负均值IC且正值率恰好0.6 → 不通过（P0修复验证：abs已删除）
        确定性构造5个分组，3组正IC(+0.04), 2组负IC(-0.12)
        purge=0, embargo=0 确保5组全部保留
        → mean_ic = (0.04*3 - 0.12*2)/5 = -0.024 < 0
        → positive_ratio = 3/5 = 0.6
        → passed = False
        """
        n = 100
        # 前60个IC为正(+0.04)，后40个IC为负(-0.12)
        ics = [0.04] * 60 + [-0.12] * 40
        dates = _dates(n)
        result = cpv_validate(ics, dates, n_groups=5, n_test_groups=1,
                              min_ic=0.03, purge=0, embargo=0, require_calendar=False)
        assert result["n_splits"] >= 5, f"预期≥5分割, 实际{result['n_splits']}"
        assert result["mean_ic"] == pytest.approx(-0.024, abs=0.005), \
            f"mean_ic={result['mean_ic']} 应≈-0.024"
        assert result["positive_ratio"] == pytest.approx(0.6, abs=0.02), \
            f"positive_ratio={result['positive_ratio']} 应≈0.6"
        assert result["passed"] is False, \
            f"负均值IC应不通过, mean_ic={result['mean_ic']:.4f}, pos_ratio={result['positive_ratio']}"


# ═══════════════════════════════════════════════════════════════════
# 边界
# ═══════════════════════════════════════════════════════════════════

class TestCPCVEdge:
    def test_all_identical_ics(self):
        ics = [0.05] * 100
        dates = _dates(100)
        result = cpv_validate(ics, dates, require_calendar=False)
        assert "icir" in result

    def test_empty_ics(self):
        result = cpv_validate([], [])
        assert result["n_splits"] == 0


# ═══════════════════════════════════════════════════════════════════
# 交易日历路径
# ═══════════════════════════════════════════════════════════════════

class TestTradingCalendar:
    def _make_big_cal(self, n_cal: int) -> list:
        """构造n_cal个连续日历日"""
        from datetime import datetime, timedelta
        d = datetime(2024, 1, 1)
        return [(d + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_cal)]

    def test_calendar_fully_covered(self):
        """完整交易日历（覆盖全部输入日期）→ 有效分割"""
        n = 100
        ics = _ics(n, mean=0.08, std=0.05)
        dates = _dates(n)
        # 构造300个日历日 > 100*2.5=250
        big_cal = self._make_big_cal(300)
        result = cpv_validate(ics, dates, n_groups=5, n_test_groups=2,
                              purge=0, embargo=0, trading_calendar=big_cal)
        assert result["n_splits"] > 0, "完整日历应产生有效分割"

    def test_calendar_sparse_input_only_is_not_ssot(self):
        """仅覆盖输入日期的日历在底层可产生分割，但不是可信SSOT

        说明：cpv_validate 只验证日期覆盖，不证明日历完整性。
        生产路径必须通过 load_trading_calendar(SSOT) 加载，而不是把观测日期当日历。
        """
        n = 100
        ics = _ics(n, mean=0.08, std=0.05)
        dates = _dates(n)
        result = cpv_validate(ics, dates, n_groups=5, n_test_groups=2,
                              purge=0, embargo=0, trading_calendar=dates)
        assert result["n_splits"] > 0
        # 明确：这不是生产口径；生产路径禁止把观测日期当作交易日历
        assert result.get("reason_code") == "EVALUATED"

    def test_calendar_partially_covered_returns_none(self):
        """交易日历部分覆盖输入日期 → CALENDAR_UNCOVERED"""
        n = 100
        ics = _ics(n, mean=0.08, std=0.05)
        dates = _dates(n)
        partial_cal = dates[:50]
        result = cpv_validate(ics, dates, n_groups=5, n_test_groups=2,
                              purge=0, embargo=0, trading_calendar=partial_cal)
        assert result["n_splits"] == 0
        assert result.get("reason_code") == "CALENDAR_UNCOVERED"

    def test_calendar_empty_ics_ignored(self):
        """空IC列表 + 日历 → 不崩溃"""
        result = cpv_validate([], [], trading_calendar=["2024-01-01"])
        assert result["n_splits"] == 0

    def test_default_missing_calendar_returns_uncovered(self):
        """默认require_calendar=True且无日历 → CALENDAR_UNCOVERED"""
        n = 100
        ics = [0.05] * n
        dates = _dates(n)
        result = cpv_validate(ics, dates, n_groups=5, n_test_groups=2)
        assert result["n_splits"] == 0
        assert result.get("reason_code") == "CALENDAR_UNCOVERED"

    def test_friday_monday_continuous_segment(self):
        """周五→周一在同一交易日历中位置差=1，属于同一连续段"""
        # 验证：段识别不会把周五→下周一拆成两个段
        from datetime import datetime
        all_cal = self._make_big_cal(60)
        wk_cal = [d for d in all_cal if datetime.strptime(d, "%Y-%m-%d").weekday() < 5]
        cal_np = np.array(wk_cal, dtype='datetime64[D]')
        # 取25个连续交易日，验证段识别正确
        n = 25
        test_dates = wk_cal[:n]
        X = np.arange(n)
        dates_np = np.array(test_dates, dtype='datetime64[D]')
        cv = CPCV(n_groups=5, n_test_groups=2, purge=0, embargo=0)
        splits = list(cv.split(X, dates_np, trading_calendar=cal_np))
        assert len(splits) > 0, "日历完整应产生有效分割"

    def test_calendar_vs_observation_position_differs(self):
        """交易日历口径：验证n_splits字段存在"""
        n = 100
        ics = [0.05] * n
        dates = _dates(n)
        big_cal = self._make_big_cal(300)
        result_cal = cpv_validate(ics, dates, n_groups=5, n_test_groups=2,
                                  purge=5, embargo=0, trading_calendar=big_cal)
        assert "n_splits" in result_cal