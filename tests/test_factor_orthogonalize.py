"""Unit tests for strategies/factor_orthogonalize.py — 因子正交化"""
import numpy as np
import pandas as pd
import pytest
from strategies.factor_orthogonalize import (
    orthogonalize, OrthogonalizeResult, report, _count_high_pairs
)


# ═══════════════════════════════════════════════════════════════════
# 合成测试数据
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def highly_correlated():
    """3个因子，trend~alpha191 高度相关 (r=0.9)"""
    np.random.seed(42)
    n = 100
    base = np.random.normal(0, 1, n)
    trend = base * 10 + 50
    alpha191 = base * 9 + 50 + np.random.normal(0, 0.5, n)  # r≈0.99
    volume = np.random.normal(50, 15, n)
    return pd.DataFrame({
        "trend": trend,
        "alpha191": alpha191,
        "volume": volume,
    })


@pytest.fixture
def three_correlated():
    """3个因子两两相关 (r≈0.6-0.8)"""
    np.random.seed(42)
    n = 200
    f1 = np.random.normal(0, 1, n)
    f2 = f1 * 0.7 + np.random.normal(0, 0.7, n)
    f3 = f1 * 0.3 + f2 * 0.5 + np.random.normal(0, 0.8, n)
    return pd.DataFrame({
        "momentum": (f1 * 15 + 50).clip(0, 100),
        "alpha": (f2 * 15 + 50).clip(0, 100),
        "volume": (f3 * 15 + 50).clip(0, 100),
    })


# ═══════════════════════════════════════════════════════════════════
# Gram-Schmidt
# ═══════════════════════════════════════════════════════════════════

class TestGramSchmidt:
    def test_reduces_correlation(self, highly_correlated):
        """正交化后 trend~alpha191 相关性 ~0"""
        result = orthogonalize(highly_correlated, method="gram_schmidt")
        r = result.corr_after.loc["trend", "alpha191"]
        assert abs(r) < 0.05, f"残余相关: {r}"

    def test_preserves_first_factor(self, highly_correlated):
        """第一个因子（trend）保持不变"""
        result = orthogonalize(highly_correlated, method="gram_schmidt")
        original = highly_correlated["trend"].values
        orthogonalized = result.scores["trend"].values
        # 允许微小数值误差
        assert np.allclose(original, orthogonalized, atol=1e-10)

    def test_orthogonal_pairs_have_low_corr(self, three_correlated):
        """三因子两两正交后 |r| < 0.1"""
        result = orthogonalize(three_correlated, method="gram_schmidt")
        corr = result.corr_after
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                assert abs(corr.iloc[i, j]) < 0.1, \
                    f"{corr.columns[i]} ~ {corr.columns[j]}: r={corr.iloc[i, j]:.3f}"

    def test_reduces_high_pair_count(self, highly_correlated):
        """正交后高相关对数应减少"""
        result = orthogonalize(highly_correlated, method="gram_schmidt")
        assert result.n_high_after < result.n_high_before

    def test_single_factor_unchanged(self):
        """单因子 → 不变"""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = orthogonalize(df, method="gram_schmidt")
        assert np.allclose(df["a"].values, result.scores["a"].values)

    def test_order_affects_result(self, three_correlated):
        """不同顺序 → 结果不同（第一个因子不变，后续解相关）"""
        result_a = orthogonalize(three_correlated, method="gram_schmidt",
                                 order=["momentum", "alpha", "volume"])
        result_b = orthogonalize(three_correlated, method="gram_schmidt",
                                 order=["volume", "alpha", "momentum"])
        # momentum 在 result_a 是第一个（不变），在 result_b 是最后一个（变化大）
        diff = np.abs(result_a.scores["momentum"] - result_b.scores["momentum"]).max()
        assert diff > 0.01


# ═══════════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════════

class TestPCA:
    def test_pca_reduces_correlation(self, highly_correlated):
        """PCA 正交化后所有因子间相关性 ~0"""
        result = orthogonalize(highly_correlated, method="pca")
        corr = result.corr_after
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                assert abs(r) < 0.05, f"PCA残余: {corr.columns[i]}~{corr.columns[j]}: r={r:.4f}"

    def test_pca_reduces_high_pairs(self, three_correlated):
        """PCA 正交后高相关对 ≈ 0"""
        result = orthogonalize(three_correlated, method="pca")
        assert result.n_high_after == 0

    def test_pca_preserves_variance_ranking(self, three_correlated):
        """PCA 后第一主成分与原始评分的高相关"""
        result = orthogonalize(three_correlated, method="pca")
        before_total = three_correlated.sum(axis=1)
        after_first_pc = result.scores.iloc[:, 0]  # 第一主成分
        rank_corr = abs(before_total.corr(after_first_pc, method="spearman"))
        assert rank_corr > 0.5, f"排序相关: {rank_corr:.3f}"


# ═══════════════════════════════════════════════════════════════════
# 通用
# ═══════════════════════════════════════════════════════════════════

class TestGeneral:
    def test_unknown_method_raises(self, three_correlated):
        """未知方法 → ValueError"""
        with pytest.raises(ValueError):
            orthogonalize(three_correlated, method="unknown")

    def test_result_dataclass_fields(self, three_correlated):
        """返回值包含所有必要字段"""
        result = orthogonalize(three_correlated, method="gram_schmidt")
        assert isinstance(result, OrthogonalizeResult)
        assert hasattr(result, "scores")
        assert hasattr(result, "corr_before")
        assert hasattr(result, "corr_after")
        assert hasattr(result, "method")
        assert result.method == "gram_schmidt"

    def test_same_shape(self, three_correlated):
        """输入输出形状一致"""
        result = orthogonalize(three_correlated, method="gram_schmidt")
        assert result.scores.shape == three_correlated.shape
        assert list(result.scores.columns) == list(three_correlated.columns)

    def test_no_nan_in_output(self, three_correlated):
        """正交化后无 NaN"""
        result = orthogonalize(three_correlated, method="gram_schmidt")
        assert not result.scores.isna().any().any()

    def test_different_methods_produce_different_results(self, three_correlated):
        """Gram-Schmidt 和 PCA 给出不同结果"""
        gs = orthogonalize(three_correlated, method="gram_schmidt")
        pca = orthogonalize(three_correlated, method="pca")
        diff = np.abs(gs.scores.values - pca.scores.values).max()
        assert diff > 0.01


# ═══════════════════════════════════════════════════════════════════
# _count_high_pairs
# ═══════════════════════════════════════════════════════════════════

class TestCountHighPairs:
    def test_all_low_corr(self):
        """无高相关 → 0"""
        corr = pd.DataFrame([[1.0, 0.1, 0.2],
                             [0.1, 1.0, 0.3],
                             [0.2, 0.3, 1.0]],
                            index=["a", "b", "c"],
                            columns=["a", "b", "c"])
        assert _count_high_pairs(corr) == 0

    def test_one_high_pair(self):
        """一对高相关 → 1"""
        corr = pd.DataFrame([[1.0, 0.85, 0.1],
                             [0.85, 1.0, 0.2],
                             [0.1, 0.2, 1.0]],
                            index=["a", "b", "c"],
                            columns=["a", "b", "c"])
        assert _count_high_pairs(corr) == 1

    def test_all_high(self):
        """全部高相关 → C(n,2)"""
        corr = pd.DataFrame([[1.0, 0.9, 0.8],
                             [0.9, 1.0, 0.85],
                             [0.8, 0.85, 1.0]],
                            index=["a", "b", "c"],
                            columns=["a", "b", "c"])
        assert _count_high_pairs(corr) == 3  # 3 choose 2 = 3


# ═══════════════════════════════════════════════════════════════════
# report — 输出格式
# ═══════════════════════════════════════════════════════════════════

class TestReport:
    def test_report_contains_method(self, three_correlated):
        """报告包含方法名"""
        result = orthogonalize(three_correlated, method="gram_schmidt")
        rpt = report(result)
        assert "gram_schmidt" in rpt
        assert "正交前" in rpt
        assert "正交后" in rpt
