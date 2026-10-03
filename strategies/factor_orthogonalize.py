"""factor_orthogonalize.py — 因子正交化

去除因子间的冗余信号，使因子评分保持相互独立。

方法:
1. Gram-Schmidt: 按给定顺序，每个因子减去与之前所有因子的投影
   → 适合已知优先级的情况（如：高IC因子放前面）
2. PCA: 主成分分析，转为不相关的主成分
   → 适合无优先级但需保持维数的情况

用法:
    from strategies.factor_orthogonalize import orthogonalize

    scores = pd.DataFrame({"trend": [...], "volume": [...], ...})
    result = orthogonalize(scores, method="gram_schmidt", order=["trend", "alpha191", ...])
    # result.score  = 正交化后的评分
    # result.corr_before, result.corr_after 用于验证效果
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("factor_orthogonalize")

# 正交化后允许的残余最大相关系数
_MAX_RESIDUAL_CORR = 0.10


@dataclass
class OrthogonalizeResult:
    scores: pd.DataFrame            # 正交化后评分 (同形状)
    corr_before: pd.DataFrame       # 原始相关性矩阵
    corr_after: pd.DataFrame        # 正交化后相关性矩阵
    method: str                     # 使用的方法
    n_high_before: int = 0          # 正交前 |r|>0.7 的因子对数
    n_high_after: int = 0           # 正交后 |r|>0.7 的因子对数
    col_order: List[str] = field(default_factory=list)


def orthogonalize(
    df: pd.DataFrame,
    method: str = "gram_schmidt",
    order: Optional[List[str]] = None,
    n_components: Optional[int] = None,
) -> OrthogonalizeResult:
    """主入口：对因子评分 DataFrame 做正交化

    Args:
        df: 因子评分 DataFrame (每列一个因子)
        method: "gram_schmidt" | "pca"
        order: Gram-Schmidt 的因子顺序（高IC/有信心放前面）
        n_components: PCA 保留的主成分数（默认 = 因子数）

    Returns:
        OrthogonalizeResult
    """
    cols = [c for c in df.columns if c != "code"] if "code" in df.columns else list(df.columns)
    data = df[cols].copy()

    corr_before = data.corr()
    n_high_before = _count_high_pairs(corr_before)

    if method == "gram_schmidt":
        result_df = _gram_schmidt(data, order or cols)
    elif method == "pca":
        result_df = _pca_orthogonalize(data, n_components or len(cols))
    else:
        raise ValueError(f"未知正交化方法: {method}")

    # 修正均值偏移：保持原始均值（仅去相关，不改变因子水平）
    for col in cols:
        result_df[col] = result_df[col] - result_df[col].mean() + data[col].mean()

    corr_after = result_df.corr()
    n_high_after = _count_high_pairs(corr_after)

    return OrthogonalizeResult(
        scores=result_df,
        corr_before=corr_before,
        corr_after=corr_after,
        method=method,
        n_high_before=n_high_before,
        n_high_after=n_high_after,
        col_order=order or cols,
    )


# ═══════════════════════════════════════════════════════════════════
# Gram-Schmidt
# ═══════════════════════════════════════════════════════════════════

def _gram_schmidt(data: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """Gram-Schmidt 正交化

    第一个因子不变，后续因子减去其在前面所有因子上的投影。
    返回与原 DataFrame 相同形状的结果。
    """
    result = pd.DataFrame(index=data.index)
    orthogonal_basis = []

    for col in order:
        if col not in data.columns:
            log.warning("因子 %s 不在数据中，跳过", col)
            continue

        v = data[col].values.astype(float)
        v_centered = v - v.mean()

        # 减去与之前所有正交基的投影
        for q in orthogonal_basis:
            # projection = <v, q> / <q, q> * q
            dot_vq = np.dot(v_centered, q)
            dot_qq = np.dot(q, q)
            if abs(dot_qq) > 1e-12:
                proj = (dot_vq / dot_qq) * q
                v_centered = v_centered - proj

        # 归一化（可选：保留方差信息）
        # 这里不归一化，只去相关；让方差保持
        result[col] = v_centered + data[col].mean()
        orthogonal_basis.append(v_centered / (np.std(v_centered) + 1e-12))

    return result


# ═══════════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════════

def _pca_orthogonalize(data: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """PCA 正交化

    通过 SVD 分解得到不相关的主成分。
    n_components ≤ 因子数。
    """
    cols = list(data.columns)
    n_components = min(n_components, len(cols))

    # 中心化
    means = data[cols].mean()
    centered = data[cols] - means

    # SVD
    U, S, Vt = np.linalg.svd(centered.values, full_matrices=False)

    # 取前 n_components 个主成分
    components = U[:, :n_components] * S[:n_components]

    # 映射回原始列名的 DataFrame
    result = pd.DataFrame(
        components[:, :len(cols)] if components.shape[1] >= len(cols) else
        np.pad(components, ((0, 0), (0, len(cols) - components.shape[1])), mode='constant'),
        index=data.index,
        columns=cols,
    )

    # 恢复原始均值
    result = result + means.values

    return result


# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════

def _count_high_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> int:
    """统计 |r| > threshold 的因子对数 (不包括自相关)"""
    cols = corr.columns
    count = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) > threshold:
                count += 1
    return count


def report(result: OrthogonalizeResult) -> str:
    """生成简洁的正交化效果报告"""
    lines = [
        f"因子正交化报告 — 方法: {result.method}",
        f"  正交前 |r|>0.7: {result.n_high_before} 对",
        f"  正交后 |r|>0.7: {result.n_high_after} 对",
        "",
        "高相关对变化:",
    ]
    before = set()
    cols = result.corr_before.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_before = result.corr_before.iloc[i, j]
            r_after = result.corr_after.iloc[i, j]
            if abs(r_before) > 0.7 or abs(r_after) > 0.7:
                icon = "✅" if abs(r_after) < 0.15 else "🟡" if abs(r_after) < 0.3 else "🔴"
                lines.append(f"  {icon} {cols[i]} ~ {cols[j]}: {r_before:+.3f} → {r_after:+.3f}")

    return "\n".join(lines)


# ── 兼容封装：与 factor_ortho.orthogonalize_factors() 接口一致 ──
# 返回 {col}_ortho 格式的 DataFrame，用于现有调用方平滑迁移

def orthogonalize_factors(df: pd.DataFrame,
                          order: list = None,
                          suffix: str = "_ortho") -> pd.DataFrame:
    """对因子做 Gram-Schmidt 正交化（兼容 factor_ortho 接口）

    Args:
        df: 因子 DataFrame，每列一个因子
        order: 正交化顺序（前列先保留）
        suffix: 正交化列后缀

    Returns:
        带 {col}{suffix} 列的 DataFrame
    """
    if df.empty or len(df.columns) < 2:
        log.warning(f"正交化需要 >=2 因子列, 当前 {len(df.columns)}")
        return pd.DataFrame(index=df.index)

    # 准备所有列的默认 order（以 df 列序为准）
    result = orthogonalize(df, method="gram_schmidt", order=order or list(df.columns))
    renamed = {c: f"{c}{suffix}" for c in result.scores.columns}
    return result.scores.rename(columns=renamed)
