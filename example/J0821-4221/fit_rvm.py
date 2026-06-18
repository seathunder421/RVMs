#!/usr/bin/env python3
"""Run/plot the RVM fit.

The default path is fast: load the compact posterior CSV files and redraw
only the two publication figures. Change FIT_MODE below when you need to run
the staged dynesty fit.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "pulsar_rvm_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pulsar_rvm_cache"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 参数区：一般只需要改这里
# =============================================================================

# 脉冲星名字。
# None 表示从 *_filtered_data.csv 文件名自动读取，例如 J0821-4221_-9.6_24_filtered_data.csv。
# 如果想手动指定，就写成 "J0821-4221"。
PSR_NAME: str | None = None  # 脉冲星名字；None 表示从过滤数据文件名自动读取。

# PPA 过滤数据文件。
# None 表示自动读取当前目录下唯一的 *_filtered_data.csv。
# 如果目录里有多个过滤文件，就手动写成 Path("J0821-4221_-9.6_24_filtered_data.csv")。
FILTERED_DATA_FILE: Path | None = None  # PPA 过滤数据 CSV；None 表示自动读取唯一 *_filtered_data.csv。

# RVM 拟合相位范围。
# None 表示从过滤数据文件名自动读取左右边界；手动指定时写数字。
PHASE_MIN: float | None = None  # RVM 拟合和绘图的最小相位；None 表示从文件名自动读取。
PHASE_MAX: float | None = None  # RVM 拟合和绘图的最大相位；None 表示从文件名自动读取。

# 计算轮廓噪声的区间，用于 w10、SN_peak、偏振比例误差等。
# 应该选没有脉冲信号的 off-pulse 区域。
NOISE_RANGE = (100.0, 180.0)  # off-pulse 噪声区间，用于 w10、SN_peak 和偏振比例误差。

# 拟合模式。
# "load"：只读取已有 posterior CSV，不重新拟合，适合第二次运行、重画图和重写 information.csv。
# "phase"：只跑第一步拟合，同时拟合 alpha、beta、phase_offset、psi_offset，并保存 phase posterior。
# "alpha_beta"：读取第一步 phase posterior，固定 phase_offset/psi_offset，只拟合 alpha、beta。
# "both"：连续跑第一步和第二步完整分步拟合。
FIT_MODE = "both"  # 拟合模式；load 只读 posterior 不重拟合，both 才完整分步拟合。

# 已有后验样本文件。
# None 表示按脉冲星名字自动读取：
#   Jxxxx_phase_posterior.csv
#   Jxxxx_alpha_beta_posterior.csv
# 这两个 CSV 只包含后验样本，不包含任何本地路径，适合放到 GitHub。
PHASE_POSTERIOR_FILE: Path | None = None  # 第一步 phase posterior CSV；None 表示按脉冲星名字自动寻找。
ALPHA_BETA_POSTERIOR_FILE: Path | None = None  # 第二步 alpha/beta posterior CSV；None 表示自动寻找。

# 第一轮采样参数：同时拟合 alpha、beta、phase_offset、psi_offset。
# nlive 越大结果越稳但越慢；nthreads 是线程数；dlogz 越小越严格。
PHASE_SAMPLER = {
    "nlive": 800,  # 第一步动态嵌套采样 live points 数量，越大越稳但越慢。
    "walks": 200,  # 第一步每次采样随机游走步数。
    "dlogz": 0.01,  # 第一步停止阈值，越小越严格。
    "nthreads": 8,  # 第一步并行线程数。
    "burnin": 500,  # 第一步丢弃的前期样本数量。
}

# 第二轮采样参数：固定 phase_offset 和 psi_offset 后拟合 alpha、beta。
# nlive 越大后验更平滑但更慢。
ALPHA_BETA_SAMPLER = {
    "nlive": 1500,  # 第二步动态嵌套采样 live points 数量，越大后验越平滑但越慢。
    "walks": 200,  # 第二步每次采样随机游走步数。
    "dlogz": 0.01,  # 第二步停止阈值，越小越严格。
    "nthreads": 20,  # 第二步并行线程数。
    "burnin": 500,  # 第二步丢弃的前期样本数量。
}

# 后验最优点搜索参数。
# 这些只影响从 posterior 中找最大概率点的速度和精细程度，不改变 RVM 公式。
KDE_GRID_SIZE = 250  # KDE 搜索最优点时的网格数量，越大越精细。
KDE_SAMPLE_LIMIT = 4000  # KDE 最多使用多少 posterior 样本，越大越慢。
KDE_EVAL_LIMIT = 2500  # KDE 最多评估多少候选点，越大越慢。

# 画 RVM 灰色后验曲线时最多抽多少条，越大越密但图会更慢。
POSTERIOR_DRAW_LIMIT = 700  # RVMFIT 图中最多画多少条灰色 posterior 曲线。

# 图 1：Jxxxx_MeerKAT_fiting.png 的样式参数表。
# figsize: 图大小；dpi/save_dpi: 屏幕绘图和保存 PNG 清晰度，越大越清晰，文件也越大。
# save_pdf/pdf_dpi: 是否同时保存对应 PDF，以及 PDF 中栅格化点和灰线的清晰度。
# bbox_inches: 保存边距，"tight" 会自动收紧空白；tight_layout: 是否自动整理子图间距。
# title_pad/labelpad: 标题和标签距离；x_tick_count/y_tick_count: 坐标刻度数量。
# raw_pa/selected_pa: 原始 PA 点和最终 PPA 点的颜色、透明度、点大小、误差棒粗细。
# model/orthogonal: RVM 主曲线和 ±90/±180 辅助曲线样式。
# slope_marker: PPA 最大斜率位置线样式。
FIT_ONLY_FIGURE_STYLE = {
    "figsize": (6.2, 3.8),  # 图 1 大小，单位 inch。
    "dpi": 200,  # 图 1 绘制 DPI。
    "save_dpi": 200,  # 图 1 保存 PNG 的 DPI。
    "save_pdf": True,  # 图 1 是否同时保存 PDF。
    "pdf_dpi": 200,  # 图 1 保存 PDF 时栅格化数据点的 DPI。
    "bbox_inches": "tight",  # 图 1 保存边距，tight 表示裁掉多余空白。
    "tight_layout": True,  # 图 1 是否自动整理边距和标签。
    "save_facecolor": "white",  # 图 1 保存背景颜色。
    "title_fontsize": 14,  # 图 1 标题字体大小。
    "title_pad": 8,  # 图 1 标题和图框距离。
    "xlabel_fontsize": 15,  # 图 1 x 轴标签字体大小。
    "ylabel_fontsize": 15,  # 图 1 y 轴标签字体大小。
    "xlabel_labelpad": 4,  # 图 1 x 轴标签和刻度距离。
    "ylabel_labelpad": 19,  # 图 1 y 轴标签和刻度距离。
    "tick_labelsize": 14,  # 图 1 坐标刻度字体大小。
    "x_tick_count": 5,  # 图 1 x 轴主刻度数量。
    "y_tick_count": 5,  # 图 1 y 轴主刻度数量。
    "legend_fontsize": 13,  # 图 1 图例字体大小。
    "legend_loc": "upper right",  # 图 1 图例位置。
    "legend_framealpha": 1.0,  # 图 1 图例背景透明度。
    "raw_pa_color": "0.25",  # 图 1 原始 PA 点颜色。
    "raw_pa_alpha": 0.22,  # 图 1 原始 PA 点透明度。
    "raw_pa_markersize": 6.0,  # 图 1 原始 PA 点大小。
    "raw_pa_capsize": 2.0,  # 图 1 原始 PA 误差棒端帽大小。
    "raw_pa_elinewidth": 1.2,  # 图 1 原始 PA 误差棒线宽。
    "selected_pa_color": "crimson",  # 图 1 选中 PPA 点颜色。
    "selected_pa_alpha": 0.95,  # 图 1 选中 PPA 点透明度。
    "selected_pa_markersize": 6.0,  # 图 1 选中 PPA 点大小。
    "selected_pa_capsize": 2.0,  # 图 1 选中 PPA 误差棒端帽大小。
    "selected_pa_elinewidth": 1.2,  # 图 1 选中 PPA 误差棒线宽。
    "shifted_pa_marker": "o",  # 图 1 被移动过的 PPA 点形状，和普通 PA 点保持圆形。
    "shifted_pa_color": "gold",  # 图 1 被移动过后的 PPA 点填充颜色，黄色。
    "shifted_pa_edgecolor": "gold",  # 图 1 被移动过后的 PPA 点边框颜色，黄色。
    "shifted_pa_markersize": 6.0,  # 图 1 被移动过的 PPA 点大小，和普通 PA 点一致。
    "shifted_pa_capsize": 2.0,  # 图 1 被移动过的 PPA 误差棒端帽大小。
    "shifted_pa_elinewidth": 1.2,  # 图 1 被移动过的 PPA 误差棒线宽。
    "shifted_pa_markeredgewidth": 1.2,  # 图 1 被移动过的 PPA 点边框线宽。
    "model_color": "#f28e2b",  # 图 1 RVM 主曲线颜色。
    "model_linewidth": 2.4,  # 图 1 RVM 主曲线线宽。
    "orthogonal_linewidth": 1.3,  # 图 1 ±90/±180 辅助曲线线宽。
    "orthogonal_alpha": 0.55,  # 图 1 ±90/±180 辅助曲线透明度。
    "slope_marker_color": "#7b3294",  # 图 1 PPA 最大斜率虚线颜色。
    "slope_marker_linestyle": "--",  # 图 1 PPA 最大斜率线型。
    "slope_marker_linewidth": 2.2,  # 图 1 PPA 最大斜率虚线线宽。
}

# 图 2：Jxxxx_MeerKAT_RVMFIT.png 的样式参数表。
# figsize: 图大小；dpi/save_dpi: 屏幕绘图和保存 PNG 清晰度，越大越清晰，文件也越大。
# save_pdf/pdf_dpi: 是否同时保存对应 PDF，以及 PDF 中栅格化点和灰线的清晰度。
# bbox_inches: 保存边距，"tight" 会自动收紧空白；tight_layout: 是否自动整理子图间距。
# pa_panel/flux_panel: 上面 PA 图和下面强度图的字体、刻度、图例、间距。
# posterior: 灰色后验曲线；profile: I/L/V 轮廓曲线。
# slope_marker: PPA 最大斜率线；pulse_center: 脉冲宽度中心线；boundary: 脉冲宽度左右边界线。
RVMFIT_FIGURE_STYLE = {
    "figsize": (6.2, 6.2),  # 图 2 大小，单位 inch。
    "dpi": 500,  # 图 2 绘制 DPI。
    "save_dpi": 500,  # 图 2 保存 PNG 的 DPI。
    "save_pdf": True,  # 图 2 是否同时保存 PDF。
    "pdf_dpi": 500,  # 图 2 保存 PDF 时栅格化点和灰线的 DPI。
    "bbox_inches": "tight",  # 图 2 保存边距，tight 表示裁掉多余空白。
    "tight_layout": True,  # 图 2 是否自动整理边距和标签。
    "save_facecolor": "white",  # 图 2 保存背景颜色。
    "height_ratios": (1.25, 1.0),  # 图 2 上下两个子图高度比例。
    "hspace": 0.0,  # 图 2 上下两个子图之间的垂直间距。
    "title_fontsize": 15,  # 图 2 标题字体大小。
    "title_pad": 8,  # 图 2 标题和图框距离。
    "title_weight": "bold",  # 图 2 标题粗细。
    "pa_ylabel_fontsize": 15,  # 图 2 上方 PA 子图 y 轴标签字体大小。
    "pa_ylabel_labelpad": 4,  # 图 2 上方 PA 子图 y 轴标签距离。
    "flux_xlabel_fontsize": 15,  # 图 2 下方强度子图 x 轴标签字体大小。
    "flux_ylabel_fontsize": 15,  # 图 2 下方强度子图 y 轴标签字体大小。
    "flux_xlabel_labelpad": 4,  # 图 2 下方强度子图 x 轴标签距离。
    "flux_ylabel_labelpad": 4,  # 图 2 下方强度子图 y 轴标签距离。
    "tick_labelsize": 14,  # 图 2 坐标刻度字体大小。
    "x_tick_count": 5,  # 图 2 x 轴主刻度数量。
    "pa_y_tick_count": 5,  # 图 2 上方 PA 子图 y 轴主刻度数量。
    "flux_y_tick_count": 5,  # 图 2 下方强度子图 y 轴主刻度数量。
    "pa_legend_fontsize": 13,  # 图 2 上方 PA 子图图例字体大小。
    "pa_legend_loc": "upper right",  # 图 2 上方 PA 子图图例位置。
    "flux_legend_fontsize": 12,  # 图 2 下方强度子图图例字体大小。
    "flux_legend_loc": "upper right",  # 图 2 下方强度子图图例位置。
    "legend_framealpha": 1.0,  # 图 2 图例背景透明度。
    "raw_pa_color": "0.25",  # 图 2 原始 PA 点颜色。
    "raw_pa_alpha": 0.20,  # 图 2 原始 PA 点透明度。
    "raw_pa_markersize": 6.0,  # 图 2 原始 PA 点大小。
    "raw_pa_capsize": 2.0,  # 图 2 原始 PA 误差棒端帽大小。
    "raw_pa_elinewidth": 1.2,  # 图 2 原始 PA 误差棒线宽。
    "selected_pa_color": "crimson",  # 图 2 选中 PPA 点颜色。
    "selected_pa_alpha": 0.95,  # 图 2 选中 PPA 点透明度。
    "selected_pa_markersize": 6.0,  # 图 2 选中 PPA 点大小。
    "selected_pa_capsize": 2.0,  # 图 2 选中 PPA 误差棒端帽大小。
    "selected_pa_elinewidth": 1.2,  # 图 2 选中 PPA 误差棒线宽。
    "shifted_pa_marker": "o",  # 图 2 被移动过的 PPA 点形状，和普通 PA 点保持圆形。
    "shifted_pa_color": "gold",  # 图 2 被移动过后的 PPA 点填充颜色，黄色。
    "shifted_pa_edgecolor": "gold",  # 图 2 被移动过后的 PPA 点边框颜色，黄色。
    "shifted_pa_markersize": 6.0,  # 图 2 被移动过的 PPA 点大小，和普通 PA 点一致。
    "shifted_pa_capsize": 2.0,  # 图 2 被移动过的 PPA 误差棒端帽大小。
    "shifted_pa_elinewidth": 1.2,  # 图 2 被移动过的 PPA 误差棒线宽。
    "shifted_pa_markeredgewidth": 1.2,  # 图 2 被移动过的 PPA 点边框线宽。
    "posterior_color": "0.70",  # 图 2 灰色 posterior 曲线颜色。
    "posterior_linewidth": 1.0,  # 图 2 灰色 posterior 曲线线宽。
    "posterior_alpha": 0.16,  # 图 2 灰色 posterior 曲线透明度。
    "model_color": "#f28e2b",  # 图 2 RVM 主曲线颜色。
    "model_linewidth": 2.6,  # 图 2 RVM 主曲线线宽。
    "orthogonal_linewidth": 1.2,  # 图 2 ±90/±180 辅助曲线线宽。
    "orthogonal_alpha": 0.50,  # 图 2 ±90/±180 辅助曲线透明度。
    "zero_line_color": "0.15",  # 图 2 PA=0 水平参考线颜色。
    "zero_linewidth": 0.8,  # 图 2 PA=0 水平参考线线宽。
    "zero_line_alpha": 0.7,  # 图 2 PA=0 水平参考线透明度。
    "profile_i_color": "black",  # 图 2 I 轮廓线颜色。
    "profile_l_color": "#1f77b4",  # 图 2 L 轮廓线颜色。
    "profile_v_color": "#d62728",  # 图 2 V 轮廓线颜色。
    "profile_i_linewidth": 1.8,  # 图 2 I 轮廓线宽。
    "profile_l_linewidth": 1.8,  # 图 2 L 轮廓线宽。
    "profile_v_linewidth": 1.5,  # 图 2 V 轮廓线宽。
    "slope_marker_color": "#7b3294",  # 图 2 PPA 最大斜率虚线颜色。
    "slope_marker_linestyle": "--",  # 图 2 PPA 最大斜率线型。
    "slope_marker_linewidth": 2.2,  # 图 2 PPA 最大斜率虚线线宽。
    "pulse_center_color": "#7b3294",  # 图 2 脉冲宽度中心虚线颜色。
    "pulse_center_linestyle": "--",  # 图 2 脉冲宽度中心线型。
    "pulse_center_linewidth": 2.2,  # 图 2 脉冲宽度中心虚线线宽。
    "boundary_color": "#7b3294",  # 图 2 脉冲宽度左右边界线颜色。
    "boundary_linestyle": "-",  # 图 2 脉冲宽度左右边界线型。
    "boundary_linewidth": 2.4,  # 图 2 脉冲宽度左右边界线线宽。
    "boundary_half_height": 0.12,  # 图 2 脉冲宽度边界短线半高度，越大线越长。
}

# information 参数表输出文件。
INFORMATION_FILE = Path("information.csv")  # information 参数表输出文件名。

# 写入 information.csv 的望远镜和频率信息。
TELESCOPE = "MeerKAT"  # 写入 information.csv 的望远镜名称。
FREQUENCY_MHZ = "856-1712"  # 写入 information.csv 的观测频率范围，单位 MHz。


DATA_COLUMNS = [
    "Angle",
    "I_normalized",
    "Linear_Polarization",
    "V_normalized",
    "PA",
    "PA_error",
    "PA_filtered",
    "PA_error_filtered",
    "SN",
    "PA_shifted",
    "PA_shift_deg",
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",  # 全局字体族。
        "font.sans-serif": ["DejaVu Sans"],  # 全局无衬线字体。
        "axes.unicode_minus": False,  # 是否正常显示负号。
        "xtick.direction": "in",  # x 轴刻度方向。
        "ytick.direction": "in",  # y 轴刻度方向。
        "savefig.facecolor": "white",  # 保存图片默认背景色。
        "figure.facecolor": "white",  # 画布默认背景色。
    }
)


@dataclass(frozen=True)
class FitData:
    angle: np.ndarray
    pa: np.ndarray
    pa_error: np.ndarray


@dataclass(frozen=True)
class BestFit:
    alpha: float
    beta: float
    phase_offset: float
    psi_offset: float
    chi_red: float


def load_filtered_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    optional_columns = {"PA_shifted", "PA_shift_deg"}
    required_columns = set(DATA_COLUMNS) - optional_columns
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for column in optional_columns:
        if column not in data:
            data[column] = 0.0

    for column in DATA_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def discover_filtered_data_file() -> Path:
    files = sorted(Path(".").glob("*_filtered_data.csv"))
    if not files:
        raise FileNotFoundError("No *_filtered_data.csv file found. Run prepare_pa_data.py first.")
    if len(files) > 1:
        names = ", ".join(str(path) for path in files)
        raise RuntimeError(f"Multiple filtered CSV files found; set FILTERED_DATA_FILE manually: {names}")
    return files[0]


def parse_filtered_filename(path: Path) -> tuple[str, float, float]:
    suffix = "_filtered_data"
    stem = path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Filtered file must end with {suffix}.csv: {path}")

    base = stem[: -len(suffix)]
    try:
        psr_name, phase_min, phase_max = base.rsplit("_", 2)
        return psr_name, float(phase_min), float(phase_max)
    except ValueError as exc:
        raise ValueError(f"Cannot parse PSR name and phase range from {path}") from exc


def configure_from_files() -> tuple[str, Path, float, float, Path, Path, Path, Path]:
    filtered_file = FILTERED_DATA_FILE or discover_filtered_data_file()
    parsed_psr, parsed_min, parsed_max = parse_filtered_filename(filtered_file)

    psr_name = PSR_NAME or parsed_psr
    phase_min = parsed_min if PHASE_MIN is None else PHASE_MIN
    phase_max = parsed_max if PHASE_MAX is None else PHASE_MAX
    phase_posterior = PHASE_POSTERIOR_FILE or Path(f"{psr_name}_phase_posterior.csv")
    alpha_beta_posterior = ALPHA_BETA_POSTERIOR_FILE or Path(f"{psr_name}_alpha_beta_posterior.csv")
    fit_figure = Path(f"{psr_name}_MeerKAT_fiting.png")
    full_figure = Path(f"{psr_name}_MeerKAT_RVMFIT.png")

    return psr_name, filtered_file, phase_min, phase_max, phase_posterior, alpha_beta_posterior, fit_figure, full_figure


def select_fit_data(data: pd.DataFrame) -> FitData:
    pa = data["PA_filtered"].replace(0, np.nan)
    pa_error = data["PA_error_filtered"].replace(0, np.nan)
    mask = (
        data["Angle"].between(PHASE_MIN, PHASE_MAX)
        & pa.notna()
        & pa_error.notna()
        & (pa_error > 0)
    )
    if mask.sum() < 5:
        raise ValueError("Too few valid PA points in the selected fit window.")

    return FitData(
        angle=data.loc[mask, "Angle"].to_numpy(float),
        pa=pa.loc[mask].to_numpy(float),
        pa_error=pa_error.loc[mask].to_numpy(float),
    )


def rvm_model(phi: np.ndarray, alpha: float, beta: float, phase_offset: float, psi_offset: float) -> np.ndarray:
    """Rotating Vector Model in degrees, matching the original fit convention."""
    alpha_rad = np.radians(180.0 - alpha)
    beta_rad = np.radians(-beta)
    phi_rad = np.radians(phi)
    phase_rad = np.radians(phase_offset)
    zeta = alpha_rad + beta_rad

    numerator = np.sin(alpha_rad) * np.sin(phi_rad - phase_rad)
    denominator = (
        np.sin(zeta) * np.cos(alpha_rad)
        - np.cos(zeta) * np.sin(alpha_rad) * np.cos(phi_rad - phase_rad)
    )
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    return np.degrees(np.arctan(ratio)) + psi_offset


def reduced_chi_square(fit_data: FitData, params: BestFit | dict[str, float]) -> float:
    model = rvm_model(
        fit_data.angle,
        alpha=float(params.alpha if isinstance(params, BestFit) else params["alpha"]),
        beta=float(params.beta if isinstance(params, BestFit) else params["beta"]),
        phase_offset=float(params.phase_offset if isinstance(params, BestFit) else params["phase_offset"]),
        psi_offset=float(params.psi_offset if isinstance(params, BestFit) else params["psi_offset"]),
    )
    chi2 = np.sum(((fit_data.pa - model) / fit_data.pa_error) ** 2)
    dof = max(len(fit_data.angle) - 4, 1)
    return float(chi2 / dof)


def load_posterior(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        posterior = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with path.open() as handle:
            result = json.load(handle)
        posterior = pd.DataFrame(result["posterior"]["content"])
    else:
        raise ValueError(f"Unsupported posterior file type: {path}")

    return posterior.apply(pd.to_numeric, errors="coerce")


def posterior_from_information(psr_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INFORMATION_FILE.exists():
        raise FileNotFoundError(
            f"{INFORMATION_FILE} is missing, so posterior CSV files cannot be recovered without refitting."
        )

    information = pd.read_csv(INFORMATION_FILE)
    rows = information[information["PSR"] == psr_name]
    if rows.empty:
        raise FileNotFoundError(
            f"No row for {psr_name} in {INFORMATION_FILE}; cannot recover posterior CSV files without refitting."
        )

    row = rows.iloc[0]
    grid = np.linspace(-1.0, 1.0, 2001)

    def asymmetric_values(best: float, lower: float, upper: float) -> np.ndarray:
        return best + np.where(grid < 0, grid * lower, grid * upper)

    phase_samples = pd.DataFrame(
        {
            "alpha": row["best_alpha"],
            "beta": row["best_beta"],
            "phase_offset": asymmetric_values(
                row["best_phase_offset"],
                row["phase_offset_lower_error"],
                row["phase_offset_upper_error"],
            ),
            "psi_offset": asymmetric_values(
                row["best_psi_offset"],
                row["psi_offset_lower_error"],
                row["psi_offset_upper_error"],
            ),
            "best_alpha_hint": row["best_alpha"],
            "best_beta_hint": row["best_beta"],
            "best_phase_hint": row["best_phase_offset"],
            "best_psi_hint": row["best_psi_offset"],
        }
    )
    alpha_beta_samples = pd.DataFrame(
        {
            "alpha": asymmetric_values(
                row["best_alpha"],
                row["alpha_Lower_Error"],
                row["alpha_Upper_Error"],
            ),
            "beta": asymmetric_values(
                row["best_beta"],
                row["beta_lower_Error"],
                row["beta_upper_Error"],
            ),
            "phase_offset": row["best_phase_offset"],
            "psi_offset": row["best_psi_offset"],
            "best_alpha_hint": row["best_alpha"],
            "best_beta_hint": row["best_beta"],
            "best_phase_hint": row["best_phase_offset"],
            "best_psi_hint": row["best_psi_offset"],
        }
    )
    return phase_samples, alpha_beta_samples


def add_information_hints(samples: pd.DataFrame, psr_name: str) -> pd.DataFrame:
    if {"log_likelihood", "log_prior"}.intersection(samples.columns):
        return samples
    if not INFORMATION_FILE.exists():
        return samples

    information = pd.read_csv(INFORMATION_FILE)
    rows = information[information["PSR"] == psr_name]
    if rows.empty:
        return samples

    row = rows.iloc[0]
    samples = samples.copy()
    samples["best_alpha_hint"] = row["best_alpha"]
    samples["best_beta_hint"] = row["best_beta"]
    samples["best_phase_hint"] = row["best_phase_offset"]
    samples["best_psi_hint"] = row["best_psi_offset"]
    return samples


def kde_best_point(samples: pd.DataFrame, x_column: str, y_column: str) -> tuple[float, float]:
    xy = samples[[x_column, y_column]].dropna()
    if len(xy) < 3:
        raise ValueError(f"Not enough posterior samples for {x_column}/{y_column}.")

    if len(xy) > KDE_SAMPLE_LIMIT:
        xy = xy.sample(KDE_SAMPLE_LIMIT, random_state=666)

    points = xy[[x_column, y_column]].to_numpy(float)
    candidates = points
    if len(candidates) > KDE_EVAL_LIMIT:
        rng = np.random.default_rng(666)
        candidates = candidates[rng.choice(len(candidates), size=KDE_EVAL_LIMIT, replace=False)]

    covariance = np.cov(points.T)
    factor = len(points) ** (-1.0 / 6.0)
    kde_covariance = covariance * factor**2
    inv_covariance = np.linalg.pinv(kde_covariance)

    densities = []
    chunk_size = 250
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        diff = chunk[:, None, :] - points[None, :, :]
        exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_covariance, diff)
        densities.append(np.exp(exponent).sum(axis=1))

    density = np.concatenate(densities)
    best = int(np.argmax(density))
    return float(candidates[best, 0]), float(candidates[best, 1])


def savgol_smooth(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    half_window = window // 2
    smoothed = np.empty_like(values, dtype=float)
    indexes = np.arange(len(values), dtype=float)

    for index in range(len(values)):
        left = max(0, index - half_window)
        right = min(len(values), index + half_window + 1)
        local_x = indexes[left:right] - index
        local_y = values[left:right]
        degree = min(polyorder, len(local_y) - 1)
        coeffs = np.polyfit(local_x, local_y, degree)
        smoothed[index] = np.polyval(coeffs, 0.0)

    return smoothed


def save_posterior(samples: pd.DataFrame, path: Path) -> None:
    keep = [
        column
        for column in [
            "alpha",
            "beta",
            "phase_offset",
            "psi_offset",
            "log_likelihood",
            "log_prior",
        ]
        if column in samples.columns
    ]
    samples[keep].to_csv(path, index=False)
    print(f"Wrote posterior file: {path}")


def run_phase_sampler(fit_data: FitData) -> pd.DataFrame:
    import bilby

    likelihood = bilby.likelihood.GaussianLikelihood(
        x=fit_data.angle,
        y=fit_data.pa,
        func=rvm_model,
        sigma=fit_data.pa_error,
    )
    phase_priors = bilby.prior.PriorDict()
    phase_priors["alpha"] = bilby.prior.Uniform(0, 180, "alpha")
    phase_priors["beta"] = bilby.prior.Uniform(-90, 90, "beta")
    phase_priors["phase_offset"] = bilby.prior.Uniform(-150, 150, "phase_offset")
    phase_priors["psi_offset"] = bilby.prior.Uniform(-150, 150, "psi_offset")

    phase_result = bilby.run_sampler(
        likelihood=likelihood,
        priors=phase_priors,
        sampler="dynesty",
        sample="rwalk",
        print_progress=True,
        write_progress=True,
        bootstrap=0,
        outdir="./",
        label=f"{PSR_NAME}_0_RVM_fit",
        verbose=True,
        **PHASE_SAMPLER,
    )
    phase_samples = phase_result.posterior
    if PHASE_POSTERIOR_FILE is not None:
        save_posterior(phase_samples, PHASE_POSTERIOR_FILE)
    return phase_samples


def run_alpha_beta_sampler(fit_data: FitData, phase_samples: pd.DataFrame) -> pd.DataFrame:
    import bilby

    likelihood = bilby.likelihood.GaussianLikelihood(
        x=fit_data.angle,
        y=fit_data.pa,
        func=rvm_model,
        sigma=fit_data.pa_error,
    )

    if {"best_phase_hint", "best_psi_hint"}.issubset(phase_samples.columns):
        best_phase = float(phase_samples["best_phase_hint"].iloc[0])
        best_psi = float(phase_samples["best_psi_hint"].iloc[0])
    else:
        best_phase, best_psi = kde_best_point(phase_samples, "phase_offset", "psi_offset")

    alpha_beta_priors = bilby.prior.PriorDict()
    alpha_beta_priors["alpha"] = bilby.prior.Uniform(0, 180, "alpha")
    alpha_beta_priors["beta"] = bilby.prior.Uniform(-90, 90, "beta")
    alpha_beta_priors["phase_offset"] = bilby.prior.DeltaFunction(best_phase)
    alpha_beta_priors["psi_offset"] = bilby.prior.DeltaFunction(best_psi)

    alpha_beta_result = bilby.run_sampler(
        likelihood=likelihood,
        priors=alpha_beta_priors,
        sampler="dynesty",
        sample="rwalk",
        write_progress=True,
        bootstrap=0,
        outdir="./",
        label=f"{PSR_NAME}_1_RVM_0_fit",
        verbose=True,
        **ALPHA_BETA_SAMPLER,
    )
    alpha_beta_samples = alpha_beta_result.posterior
    if ALPHA_BETA_POSTERIOR_FILE is not None:
        save_posterior(alpha_beta_samples, ALPHA_BETA_POSTERIOR_FILE)
    return alpha_beta_samples


def run_sampler(fit_data: FitData) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_samples = run_phase_sampler(fit_data)
    alpha_beta_samples = run_alpha_beta_sampler(fit_data, phase_samples)
    return phase_samples, alpha_beta_samples


def load_or_run_posteriors(fit_data: FitData) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PHASE_POSTERIOR_FILE is None or ALPHA_BETA_POSTERIOR_FILE is None:
        raise RuntimeError("Posterior file paths are not configured.")

    mode = FIT_MODE.lower()
    if mode not in {"load", "phase", "alpha_beta", "both"}:
        raise ValueError('FIT_MODE must be one of: "load", "phase", "alpha_beta", "both".')

    if mode == "both":
        phase_samples = run_phase_sampler(fit_data)
        alpha_beta_samples = run_alpha_beta_sampler(fit_data, phase_samples)
        return phase_samples, alpha_beta_samples

    if mode == "phase":
        phase_samples = run_phase_sampler(fit_data)
        if ALPHA_BETA_POSTERIOR_FILE.exists():
            alpha_beta_samples = add_information_hints(load_posterior(ALPHA_BETA_POSTERIOR_FILE), str(PSR_NAME))
        else:
            _, alpha_beta_samples = posterior_from_information(str(PSR_NAME))
            alpha_beta_samples.to_csv(ALPHA_BETA_POSTERIOR_FILE, index=False)
            print(f"Recovered missing posterior file: {ALPHA_BETA_POSTERIOR_FILE}")
        return phase_samples, alpha_beta_samples

    if mode == "alpha_beta":
        if not PHASE_POSTERIOR_FILE.exists():
            raise FileNotFoundError(
                f"{PHASE_POSTERIOR_FILE} is missing. Run FIT_MODE='phase' first, or use FIT_MODE='both'."
            )
        phase_samples = add_information_hints(load_posterior(PHASE_POSTERIOR_FILE), str(PSR_NAME))
        alpha_beta_samples = run_alpha_beta_sampler(fit_data, phase_samples)
        return phase_samples, alpha_beta_samples

    if mode == "load":
        if not PHASE_POSTERIOR_FILE.exists() or not ALPHA_BETA_POSTERIOR_FILE.exists():
            raise FileNotFoundError(
                f"FIT_MODE='load' requires existing posterior CSV files: "
                f"{PHASE_POSTERIOR_FILE}, {ALPHA_BETA_POSTERIOR_FILE}"
            )
        phase_samples = add_information_hints(load_posterior(PHASE_POSTERIOR_FILE), str(PSR_NAME))
        alpha_beta_samples = add_information_hints(load_posterior(ALPHA_BETA_POSTERIOR_FILE), str(PSR_NAME))
        return phase_samples, alpha_beta_samples

    if not PHASE_POSTERIOR_FILE.exists() or not ALPHA_BETA_POSTERIOR_FILE.exists():
        recovered_phase, recovered_alpha_beta = posterior_from_information(str(PSR_NAME))
        if not PHASE_POSTERIOR_FILE.exists():
            phase_samples = recovered_phase
            phase_samples.to_csv(PHASE_POSTERIOR_FILE, index=False)
            print(f"Recovered missing posterior file: {PHASE_POSTERIOR_FILE}")
        else:
            phase_samples = load_posterior(PHASE_POSTERIOR_FILE)
        if not ALPHA_BETA_POSTERIOR_FILE.exists():
            alpha_beta_samples = recovered_alpha_beta
            alpha_beta_samples.to_csv(ALPHA_BETA_POSTERIOR_FILE, index=False)
            print(f"Recovered missing posterior file: {ALPHA_BETA_POSTERIOR_FILE}")
        else:
            alpha_beta_samples = load_posterior(ALPHA_BETA_POSTERIOR_FILE)
        return phase_samples, alpha_beta_samples

    phase_samples = add_information_hints(load_posterior(PHASE_POSTERIOR_FILE), str(PSR_NAME))
    alpha_beta_samples = add_information_hints(load_posterior(ALPHA_BETA_POSTERIOR_FILE), str(PSR_NAME))
    return phase_samples, alpha_beta_samples


def build_best_fit(phase_samples: pd.DataFrame, alpha_beta_samples: pd.DataFrame, fit_data: FitData) -> BestFit:
    if {"best_phase_hint", "best_psi_hint"}.issubset(phase_samples.columns):
        best_phase = float(phase_samples["best_phase_hint"].iloc[0])
        best_psi = float(phase_samples["best_psi_hint"].iloc[0])
    else:
        best_phase, best_psi = kde_best_point(phase_samples, "phase_offset", "psi_offset")

    if {"best_alpha_hint", "best_beta_hint"}.issubset(alpha_beta_samples.columns):
        best_alpha = float(alpha_beta_samples["best_alpha_hint"].iloc[0])
        best_beta = float(alpha_beta_samples["best_beta_hint"].iloc[0])
    else:
        best_alpha, best_beta = kde_best_point(alpha_beta_samples, "alpha", "beta")

    params = {
        "alpha": best_alpha,
        "beta": best_beta,
        "phase_offset": best_phase,
        "psi_offset": best_psi,
    }
    return BestFit(
        alpha=best_alpha,
        beta=best_beta,
        phase_offset=best_phase,
        psi_offset=best_psi,
        chi_red=reduced_chi_square(fit_data, params),
    )


def profile_noise(data: pd.DataFrame) -> float:
    mask = data["Angle"].between(NOISE_RANGE[0], NOISE_RANGE[1])
    if not mask.any():
        mask = data["Angle"].between(-180.0, -100.0)
    return float(data.loc[mask, "I_normalized"].std())


def component_width(
    data: pd.DataFrame,
    angle_min: float,
    angle_max: float,
    noise_std: float,
    window: int = 11,
    polyorder: int = 2,
) -> dict[str, float]:
    subset = data[data["Angle"].between(angle_min, angle_max)].copy()
    x = subset["Angle"].to_numpy(float)
    y = subset["I_normalized"].to_numpy(float)

    if len(y) < 3:
        return {"width": np.nan, "midpoint": np.nan, "angle_min": np.nan, "angle_max": np.nan}

    window = min(window if window % 2 else window + 1, len(y) if len(y) % 2 else len(y) - 1)
    window = max(window, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    window = min(window, len(y) if len(y) % 2 else len(y) - 1)

    y_smooth = savgol_smooth(y, window=window, polyorder=min(polyorder, window - 1))
    threshold = 3.0 * noise_std
    if y_smooth.max() < threshold:
        threshold = 2.0 * noise_std
    if y_smooth.max() < threshold:
        threshold = 1.5 * noise_std

    above = np.flatnonzero(y_smooth > threshold)
    if len(above) >= 2:
        left = float(x[above[0]])
        right = float(x[above[-1]])
    else:
        left = float(x[0])
        right = float(x[-1])

    return {
        "width": right - left,
        "midpoint": (left + right) / 2.0,
        "angle_min": left,
        "angle_max": right,
        "Imin_val": float(np.interp(left, x, y_smooth)),
        "Imax_val": float(np.interp(right, x, y_smooth)),
        "threshold": threshold,
    }


def posterior_subset(samples: pd.DataFrame) -> pd.DataFrame:
    subset = samples[["alpha", "beta", "phase_offset", "psi_offset"]].dropna().copy()
    if len(subset) <= POSTERIOR_DRAW_LIMIT:
        return subset
    return subset.sample(POSTERIOR_DRAW_LIMIT, random_state=666)


def plot_pa_points(ax: plt.Axes, data: pd.DataFrame, filtered: bool, style: dict) -> None:
    pa_column = "PA_filtered" if filtered else "PA"
    err_column = "PA_error_filtered" if filtered else "PA_error"
    prefix = "selected_pa" if filtered else "raw_pa"
    color = style[f"{prefix}_color"]
    zorder = 5 if filtered else 3

    pa = data[pa_column].replace(0, np.nan)
    err = data[err_column].replace(0, np.nan)
    mask = pa.notna() & err.notna()
    shifted = data.get("PA_shifted", pd.Series(0, index=data.index)).fillna(0).astype(bool)
    normal_mask = mask & ~shifted if filtered else mask
    ax.errorbar(
        data.loc[normal_mask, "Angle"],
        pa.loc[normal_mask],
        xerr=0.1,
        yerr=err.loc[normal_mask],
        fmt="o",
        color=color,
        alpha=style[f"{prefix}_alpha"],
        markersize=style[f"{prefix}_markersize"],
        capsize=style[f"{prefix}_capsize"],
        elinewidth=style[f"{prefix}_elinewidth"],
        rasterized=True,
        label=None,
        zorder=zorder,
    )

    if filtered:
        shifted_mask = mask & shifted
        if shifted_mask.any():
            ax.errorbar(
                data.loc[shifted_mask, "Angle"],
                pa.loc[shifted_mask],
                xerr=0.1,
                yerr=err.loc[shifted_mask],
                fmt=style["shifted_pa_marker"],
                color=style["shifted_pa_edgecolor"],
                markerfacecolor=style["shifted_pa_color"],
                markeredgecolor=style["shifted_pa_edgecolor"],
                markeredgewidth=style["shifted_pa_markeredgewidth"],
                alpha=1.0,
                markersize=style["shifted_pa_markersize"],
                capsize=style["shifted_pa_capsize"],
                elinewidth=style["shifted_pa_elinewidth"],
                rasterized=True,
                label=None,
                zorder=zorder + 2,
            )


def save_figure_outputs(fig: plt.Figure, output: Path, style: dict) -> None:
    fig.savefig(
        output,
        dpi=style["save_dpi"],
        bbox_inches=style["bbox_inches"],
        facecolor=style["save_facecolor"],
    )
    if style.get("save_pdf", False):
        fig.savefig(
            output.with_suffix(".pdf"),
            dpi=style["pdf_dpi"],
            bbox_inches=style["bbox_inches"],
            facecolor=style["save_facecolor"],
        )


def plot_fit_only(data: pd.DataFrame, fit_data: FitData, best: BestFit, output: Path) -> None:
    style = FIT_ONLY_FIGURE_STYLE
    phi = np.linspace(float(PHASE_MIN), float(PHASE_MAX), 1000)
    model = rvm_model(phi, best.alpha, best.beta, best.phase_offset, best.psi_offset)

    fig, ax = plt.subplots(figsize=style["figsize"], dpi=style["dpi"])
    plot_pa_points(ax, data, filtered=False, style=style)
    plot_pa_points(ax, data, filtered=True, style=style)

    ax.plot(
        phi,
        model,
        color=style["model_color"],
        linewidth=style["model_linewidth"],
        label=rf"$\chi^2_\nu={best.chi_red:.2f}$",
    )
    for offset in (-180, -90, 90, 180):
        ax.plot(
            phi,
            model + offset,
            color=style["model_color"],
            linestyle="--",
            linewidth=style["orthogonal_linewidth"],
            alpha=style["orthogonal_alpha"],
        )

    ax.axvline(
        best.phase_offset,
        color=style["slope_marker_color"],
        linestyle=style["slope_marker_linestyle"],
        linewidth=style["slope_marker_linewidth"],
    )
    ax.set_title(f"PSR {PSR_NAME}", fontsize=style["title_fontsize"], pad=style["title_pad"])
    ax.set_xlabel("Longitude (deg)", fontsize=style["xlabel_fontsize"], labelpad=style["xlabel_labelpad"])
    ax.set_ylabel("PA (deg)", fontsize=style["ylabel_fontsize"], labelpad=style["ylabel_labelpad"])
    ax.set_xlim(float(PHASE_MIN), float(PHASE_MAX))
    ax.set_ylim(np.nanmin(model) - 55, np.nanmax(model) + 55)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=style["x_tick_count"]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=style["y_tick_count"]))
    ax.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax.legend(
        loc=style["legend_loc"],
        fontsize=style["legend_fontsize"],
        framealpha=style["legend_framealpha"],
    )
    ax.grid(False)

    if style["tight_layout"]:
        fig.tight_layout()
    save_figure_outputs(fig, output, style)
    plt.close(fig)


def plot_full_rvm(
    data: pd.DataFrame,
    fit_data: FitData,
    best: BestFit,
    alpha_beta_samples: pd.DataFrame,
    output: Path,
) -> None:
    style = RVMFIT_FIGURE_STYLE
    phi = np.linspace(float(PHASE_MIN), float(PHASE_MAX), 1000)
    model = rvm_model(phi, best.alpha, best.beta, best.phase_offset, best.psi_offset)
    noise_std = profile_noise(data)
    width = component_width(data, float(PHASE_MIN), float(PHASE_MAX), noise_std)

    fig = plt.figure(figsize=style["figsize"], dpi=style["dpi"])
    gs = gridspec.GridSpec(2, 1, height_ratios=style["height_ratios"], hspace=style["hspace"])
    ax_pa = fig.add_subplot(gs[0])
    ax_flux = fig.add_subplot(gs[1], sharex=ax_pa)

    phase_fixed_samples = alpha_beta_samples[["alpha", "beta"]].dropna().copy()
    phase_fixed_samples["phase_offset"] = best.phase_offset
    phase_fixed_samples["psi_offset"] = best.psi_offset
    for row in posterior_subset(phase_fixed_samples).itertuples(index=False):
        sample_model = rvm_model(phi, row.alpha, row.beta, row.phase_offset, row.psi_offset)
        ax_pa.plot(
            phi,
            sample_model,
            color=style["posterior_color"],
            linewidth=style["posterior_linewidth"],
            alpha=style["posterior_alpha"],
            rasterized=True,
        )

    plot_pa_points(ax_pa, data, filtered=False, style=style)
    plot_pa_points(ax_pa, data, filtered=True, style=style)

    ax_pa.plot(
        phi,
        model,
        color=style["model_color"],
        linewidth=style["model_linewidth"],
        label=rf"$\chi^2_\nu={best.chi_red:.2f}$",
    )
    for offset in (-180, -90, 90, 180):
        ax_pa.plot(
            phi,
            model + offset,
            color=style["model_color"],
            linestyle="--",
            linewidth=style["orthogonal_linewidth"],
            alpha=style["orthogonal_alpha"],
        )

    ax_pa.axvline(
        best.phase_offset,
        color=style["slope_marker_color"],
        linestyle=style["slope_marker_linestyle"],
        linewidth=style["slope_marker_linewidth"],
    )
    ax_pa.axhline(
        0,
        color=style["zero_line_color"],
        linestyle="--",
        linewidth=style["zero_linewidth"],
        alpha=style["zero_line_alpha"],
    )
    ax_pa.set_title(
        f"PSR {PSR_NAME}",
        fontsize=style["title_fontsize"],
        pad=style["title_pad"],
        fontweight=style["title_weight"],
    )
    ax_pa.set_ylabel("PA (deg)", fontsize=style["pa_ylabel_fontsize"], labelpad=style["pa_ylabel_labelpad"])
    ax_pa.set_ylim(np.nanmin(model) - 55, np.nanmax(model) + 55)
    ax_pa.yaxis.set_major_locator(MaxNLocator(nbins=style["pa_y_tick_count"]))
    ax_pa.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax_pa.legend(
        loc=style["pa_legend_loc"],
        fontsize=style["pa_legend_fontsize"],
        framealpha=style["legend_framealpha"],
    )
    ax_pa.tick_params(labelbottom=False)
    ax_pa.grid(False)

    ax_flux.plot(
        data["Angle"],
        data["I_normalized"],
        color=style["profile_i_color"],
        linewidth=style["profile_i_linewidth"],
        label="I",
    )
    ax_flux.plot(
        data["Angle"],
        data["Linear_Polarization"],
        color=style["profile_l_color"],
        linewidth=style["profile_l_linewidth"],
        label="L",
    )
    ax_flux.plot(
        data["Angle"],
        data["V_normalized"],
        color=style["profile_v_color"],
        linewidth=style["profile_v_linewidth"],
        label="V",
    )

    ax_flux.set_xlabel(
        "Longitude (deg)",
        fontsize=style["flux_xlabel_fontsize"],
        labelpad=style["flux_xlabel_labelpad"],
    )
    ax_flux.set_ylabel(
        "Intensity",
        fontsize=style["flux_ylabel_fontsize"],
        labelpad=12,
    )
    ax_flux.set_xlim(float(PHASE_MIN), float(PHASE_MAX))
    ax_flux.set_ylim(-0.18, 1.12)

    if np.isfinite(width["midpoint"]):
        ax_flux.axvline(
            width["midpoint"],
            color=style["pulse_center_color"],
            linestyle=style["pulse_center_linestyle"],
            linewidth=style["pulse_center_linewidth"],
        )
        ax_flux.vlines(
            width["angle_min"],
            ymin=max(-0.18, width["Imin_val"] - style["boundary_half_height"]),
            ymax=min(1.12, width["Imin_val"] + style["boundary_half_height"]),
            color=style["boundary_color"],
            linestyle=style["boundary_linestyle"],
            linewidth=style["boundary_linewidth"],
        )
        ax_flux.vlines(
            width["angle_max"],
            ymin=max(-0.18, width["Imax_val"] - style["boundary_half_height"]),
            ymax=min(1.12, width["Imax_val"] + style["boundary_half_height"]),
            color=style["boundary_color"],
            linestyle=style["boundary_linestyle"],
            linewidth=style["boundary_linewidth"],
        )

    ax_flux.xaxis.set_major_locator(MaxNLocator(nbins=style["x_tick_count"]))
    ax_flux.yaxis.set_major_locator(MaxNLocator(nbins=style["flux_y_tick_count"]))
    ax_flux.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax_flux.legend(
        loc=style["flux_legend_loc"],
        fontsize=style["flux_legend_fontsize"],
        framealpha=style["legend_framealpha"],
    )
    ax_flux.grid(False)

    if style["tight_layout"]:
        fig.tight_layout()
    save_figure_outputs(fig, output, style)
    plt.close(fig)


def print_summary(best: BestFit, phase_samples: pd.DataFrame, alpha_beta_samples: pd.DataFrame) -> None:
    alpha_q = alpha_beta_samples["alpha"].quantile([0.025, 0.5, 0.975])
    beta_q = alpha_beta_samples["beta"].quantile([0.025, 0.5, 0.975])
    phase_q = phase_samples["phase_offset"].quantile([0.025, 0.5, 0.975])
    psi_q = phase_samples["psi_offset"].quantile([0.025, 0.5, 0.975])

    print("Best-fit parameters")
    print(f"  alpha        = {best.alpha:.4f} deg ({alpha_q.iloc[0]:.4f}, {alpha_q.iloc[2]:.4f})")
    print(f"  beta         = {best.beta:.4f} deg ({beta_q.iloc[0]:.4f}, {beta_q.iloc[2]:.4f})")
    print(f"  phase_offset = {best.phase_offset:.4f} deg ({phase_q.iloc[0]:.4f}, {phase_q.iloc[2]:.4f})")
    print(f"  psi_offset   = {best.psi_offset:.4f} deg ({psi_q.iloc[0]:.4f}, {psi_q.iloc[2]:.4f})")
    print(f"  chi_red      = {best.chi_red:.4f}")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def query_pulsar_periods(psr_name: str) -> tuple[float, float]:
    try:
        import psrqpy
    except Exception as exc:
        print(f"Warning: psrqpy is unavailable; P0/P-dot/w10_error will be NaN ({exc}).")
        return float("nan"), float("nan")

    query = psrqpy.QueryATNF(params=["NAME", "PSRJ", "PSRB", "P0", "P1"])
    table = query.table

    for column in ("NAME", "PSRJ", "PSRB"):
        rows = table[table[column] == psr_name]
        if len(rows):
            return float(rows["P0"].data[0]), float(rows["P1"].data[0])

    print(f"Warning: {psr_name} was not found in ATNF; P0/P-dot/w10_error will be NaN.")
    return float("nan"), float("nan")


def add_chi_and_k(samples: pd.DataFrame, best: BestFit, fit_data: FitData) -> pd.DataFrame:
    data = samples[["alpha", "beta"]].dropna().copy()
    data["phase_offset"] = best.phase_offset
    data["psi_offset"] = best.psi_offset

    alpha_rad = np.radians(180.0 - data["alpha"].to_numpy(float))
    beta_rad = np.radians(-data["beta"].to_numpy(float))
    data["K"] = np.divide(
        np.sin(alpha_rad),
        np.sin(beta_rad),
        out=np.full(len(data), np.nan, dtype=float),
        where=np.sin(beta_rad) != 0,
    )

    chi_values = []
    for row in data.itertuples(index=False):
        params = {
            "alpha": row.alpha,
            "beta": row.beta,
            "phase_offset": row.phase_offset,
            "psi_offset": row.psi_offset,
        }
        chi_values.append(reduced_chi_square(fit_data, params))
    data["chi_red"] = chi_values
    return data


def posterior_error_subset(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    alpha_q = samples["alpha"].quantile([0.025, 0.5, 0.975])
    beta_q = samples["beta"].quantile([0.025, 0.5, 0.975])
    subset = samples[
        (samples["alpha"] > alpha_q.loc[0.025])
        & (samples["alpha"] < alpha_q.loc[0.975])
        & (samples["beta"] > beta_q.loc[0.025])
        & (samples["beta"] < beta_q.loc[0.975])
    ].copy()
    if subset.empty:
        subset = samples.copy()
    return subset, alpha_q, beta_q


def pulse_window_metrics(data: pd.DataFrame, best: BestFit, noise_std: float) -> tuple[float, float]:
    valid_mask = data["PA_filtered"].notna()
    window_mask = data["Angle"].between(float(PHASE_MIN), float(PHASE_MAX))
    valid_angles = data.loc[valid_mask & window_mask, "Angle"]
    if valid_angles.empty:
        pw_range = float("nan")
    else:
        pw_range = float(max(valid_angles.max() - best.phase_offset, best.phase_offset - valid_angles.min()))

    i_max = data["I_normalized"].max()
    threshold = i_max * 0.1
    intensity_mask = (
        data["Angle"].between(float(PHASE_MIN), float(PHASE_MAX))
        & (data["I_normalized"] >= threshold)
        & (data["I_normalized"] >= 3.0 * noise_std)
    )
    subset = data.loc[intensity_mask, "Angle"]
    if subset.empty:
        return pw_range, float("nan")

    w10_mh = float(max(abs(subset.max() - best.phase_offset), abs(best.phase_offset - subset.min())))
    return pw_range, w10_mh


def polarization_metrics(data: pd.DataFrame, noise_std: float) -> tuple[float, float, float, float, float, float, float]:
    pulse = data.loc[data["Angle"].between(float(PHASE_MIN), float(PHASE_MAX)), "I_normalized"]
    sn_peak = safe_divide(float(pulse.max()), noise_std) if not pulse.empty else float("nan")

    signal_mask = data["I_normalized"] >= 3.0 * noise_std
    nd = int(signal_mask.sum())
    if nd == 0:
        return sn_peak, *(float("nan") for _ in range(6))

    i_sum = float(data.loc[signal_mask, "I_normalized"].sum())
    l_sum = float(data.loc[signal_mask, "Linear_Polarization"].sum())
    v_sum = float(data.loc[signal_mask, "V_normalized"].sum())
    abs_v_sum = float(np.abs(data.loc[signal_mask, "V_normalized"]).sum())

    l_i = safe_divide(l_sum, i_sum)
    v_i = safe_divide(v_sum, i_sum)
    abs_v_i = safe_divide(abs_v_sum, i_sum)

    noise_mask = data["Angle"].between(NOISE_RANGE[0], NOISE_RANGE[1])
    sigma_i = noise_std
    sigma_l = float(data.loc[noise_mask, "Linear_Polarization"].std())
    sigma_v = float(data.loc[noise_mask, "V_normalized"].std())

    sigma_li = math.sqrt(nd * ((sigma_l / i_sum) ** 2 + (l_sum * sigma_i / i_sum**2) ** 2))
    sigma_vi = math.sqrt(nd * ((sigma_v / i_sum) ** 2 + (v_sum * sigma_i / i_sum**2) ** 2))
    sigma_abs_vi = math.sqrt(nd * ((sigma_v / i_sum) ** 2 + (abs_v_sum * sigma_i / i_sum**2) ** 2))
    return sn_peak, l_i, sigma_li, v_i, sigma_vi, abs_v_i, sigma_abs_vi


def compute_rho_with_uncertainties(
    alpha_deg: float,
    alpha_err_up: float,
    alpha_err_down: float,
    beta_deg: float,
    beta_err_up: float,
    beta_err_down: float,
    w10_deg: float,
) -> tuple[float, float, float]:
    def calc_rho(alpha_d: float, beta_d: float, w10_single_d: float) -> float:
        alpha = math.radians(alpha_d)
        beta = math.radians(beta_d)
        w10 = math.radians(w10_single_d)

        term1 = (math.sin(w10 / 4.0) ** 2) * math.sin(alpha) * math.sin(alpha + beta)
        term2 = math.sin(beta / 2.0) ** 2
        sqrt_term = math.sqrt(max(0.0, term1 + term2))
        rho_rad = 2.0 * math.asin(min(1.0, sqrt_term))
        return math.degrees(rho_rad)

    if not all(np.isfinite(value) for value in [alpha_deg, beta_deg, w10_deg]):
        return float("nan"), float("nan"), float("nan")

    rho_center = calc_rho(alpha_deg, beta_deg, w10_deg)
    rho_alpha_up = calc_rho(alpha_deg + alpha_err_up, beta_deg, w10_deg)
    rho_alpha_down = calc_rho(alpha_deg - alpha_err_down, beta_deg, w10_deg)
    rho_beta_up = calc_rho(alpha_deg, beta_deg + beta_err_up, w10_deg)
    rho_beta_down = calc_rho(alpha_deg, beta_deg - beta_err_down, w10_deg)

    drho_up = math.sqrt((rho_alpha_up - rho_center) ** 2 + (rho_beta_up - rho_center) ** 2)
    drho_down = math.sqrt((rho_center - rho_alpha_down) ** 2 + (rho_center - rho_beta_down) ** 2)
    return rho_center, drho_up, drho_down


def alpha_range_category(alpha_range: float) -> str | None:
    if 0 <= alpha_range < 40:
        return "A"
    if 40 <= alpha_range < 80:
        return "B"
    if 80 <= alpha_range < 120:
        return "C"
    if 120 <= alpha_range <= 180:
        return "D"
    return None


def build_information_row(
    data: pd.DataFrame,
    fit_data: FitData,
    best: BestFit,
    phase_samples: pd.DataFrame,
    alpha_beta_samples: pd.DataFrame,
) -> dict[str, object]:
    posterior = add_chi_and_k(alpha_beta_samples, best, fit_data)
    posterior_subset_for_errors, alpha_q, beta_q = posterior_error_subset(posterior)
    phase_q = phase_samples["phase_offset"].quantile([0.025, 0.5, 0.975])
    psi_q = phase_samples["psi_offset"].quantile([0.025, 0.5, 0.975])

    best_k = safe_divide(
        math.sin(math.radians(180.0 - best.alpha)),
        math.sin(math.radians(-best.beta)),
    )

    k_min = float(posterior_subset_for_errors["K"].min())
    k_max = float(posterior_subset_for_errors["K"].max())
    chi_min = float(posterior_subset_for_errors["chi_red"].min())
    chi_max = float(posterior_subset_for_errors["chi_red"].max())

    best_alpha_lower = abs(best.alpha - float(posterior_subset_for_errors["alpha"].min()))
    best_alpha_upper = abs(best.alpha - float(posterior_subset_for_errors["alpha"].max()))
    best_beta_lower = abs(best.beta - float(posterior_subset_for_errors["beta"].min()))
    best_beta_upper = abs(best.beta - float(posterior_subset_for_errors["beta"].max()))

    noise_std = profile_noise(data)
    width = component_width(data, float(PHASE_MIN), float(PHASE_MAX), noise_std)
    w10 = float(width["width"])
    w10_center = float(width["midpoint"])

    p0, p_dot = query_pulsar_periods(str(PSR_NAME))
    period_ms = p0 * 1000.0 if np.isfinite(p0) else float("nan")
    tb = period_ms / (len(data["Angle"]) - 1) if np.isfinite(period_ms) else float("nan")
    w10_error = tb * math.sqrt(1.0 + (noise_std / 0.1) ** 2) if np.isfinite(tb) else float("nan")

    sn_peak, l_i, sigma_li, v_i, sigma_vi, abs_v_i, sigma_abs_vi = polarization_metrics(data, noise_std)
    pw_range, w10_mh = pulse_window_metrics(data, best, noise_std)

    rho, drho_up, drho_down = compute_rho_with_uncertainties(
        alpha_deg=best.alpha,
        alpha_err_up=best_alpha_upper,
        alpha_err_down=best_alpha_lower,
        beta_deg=best.beta,
        beta_err_up=best_beta_upper,
        beta_err_down=best_beta_lower,
        w10_deg=w10,
    )

    all_alpha_error = best_alpha_upper + best_alpha_lower

    return {
        "PSR": PSR_NAME,
        "P0": p0,
        "P-dot": p_dot,
        "Telescope": TELESCOPE,
        "Freq(MHz)": FREQUENCY_MHZ,
        "best_alpha": best.alpha,
        "alpha_Upper_Error": best_alpha_upper,
        "alpha_Lower_Error": best_alpha_lower,
        "best_beta": best.beta,
        "beta_upper_Error": best_beta_upper,
        "beta_lower_Error": best_beta_lower,
        "best_phase_offset": best.phase_offset,
        "phase_offset_upper_error": abs(float(phase_q.loc[0.975]) - best.phase_offset),
        "phase_offset_lower_error": abs(float(phase_q.loc[0.025]) - best.phase_offset),
        "best_psi_offset": best.psi_offset,
        "psi_offset_upper_error": abs(float(psi_q.loc[0.975]) - best.psi_offset),
        "psi_offset_lower_error": abs(float(psi_q.loc[0.025]) - best.psi_offset),
        "best_chi_red": best.chi_red,
        "Chi_upper_error": abs(chi_max - best.chi_red),
        "Chi_lower_error": abs(best.chi_red - chi_min),
        "best_K": best_k,
        "K_upper_error": abs(k_max - best_k),
        "K_lower_error": abs(best_k - k_min),
        "w10": w10,
        "w10_error": w10_error,
        "w10_center": w10_center,
        "SN_peak": sn_peak,
        "L_I": l_i,
        "σ_LI": sigma_li,
        "V_I": v_i,
        "σ_VI": sigma_vi,
        "absV_I": abs_v_i,
        "σ_absVI": sigma_abs_vi,
        "rho": rho,
        "drho_up": drho_up,
        "drho_down": drho_down,
        "PW": pw_range,
        "W10_MH": w10_mh,
        "all_alpha_range": all_alpha_error,
        "category": alpha_range_category(all_alpha_error),
    }


def write_information(row: dict[str, object], path: Path = INFORMATION_FILE) -> None:
    fieldnames = [
        "PSR",
        "P0",
        "P-dot",
        "Telescope",
        "Freq(MHz)",
        "best_alpha",
        "alpha_Upper_Error",
        "alpha_Lower_Error",
        "best_beta",
        "beta_upper_Error",
        "beta_lower_Error",
        "best_phase_offset",
        "phase_offset_upper_error",
        "phase_offset_lower_error",
        "best_psi_offset",
        "psi_offset_upper_error",
        "psi_offset_lower_error",
        "best_chi_red",
        "Chi_upper_error",
        "Chi_lower_error",
        "best_K",
        "K_upper_error",
        "K_lower_error",
        "w10",
        "w10_error",
        "w10_center",
        "SN_peak",
        "L_I",
        "σ_LI",
        "V_I",
        "σ_VI",
        "absV_I",
        "σ_absVI",
        "rho",
        "drho_up",
        "drho_down",
        "PW",
        "W10_MH",
        "all_alpha_range",
        "category",
    ]

    existing_rows: list[dict[str, object]] = []
    if path.exists():
        with path.open(newline="") as handle:
            existing_rows = list(csv.DictReader(handle))

    updated = False
    for existing in existing_rows:
        if existing.get("PSR") == row["PSR"]:
            existing.update(row)
            updated = True
            break

    if not updated:
        existing_rows.append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def main() -> None:
    global PSR_NAME, FILTERED_DATA_FILE, PHASE_MIN, PHASE_MAX, PHASE_POSTERIOR_FILE, ALPHA_BETA_POSTERIOR_FILE

    (
        PSR_NAME,
        FILTERED_DATA_FILE,
        PHASE_MIN,
        PHASE_MAX,
        PHASE_POSTERIOR_FILE,
        ALPHA_BETA_POSTERIOR_FILE,
        fit_only_figure,
        full_rvm_figure,
    ) = configure_from_files()

    data = load_filtered_data(FILTERED_DATA_FILE)
    fit_data = select_fit_data(data)
    phase_samples, alpha_beta_samples = load_or_run_posteriors(fit_data)
    best = build_best_fit(phase_samples, alpha_beta_samples, fit_data)

    print_summary(best, phase_samples, alpha_beta_samples)
    plot_fit_only(data, fit_data, best, fit_only_figure)
    plot_full_rvm(data, fit_data, best, alpha_beta_samples, full_rvm_figure)
    information_row = build_information_row(data, fit_data, best, phase_samples, alpha_beta_samples)
    write_information(information_row)

    print(f"Read {FILTERED_DATA_FILE}")
    print(f"Wrote {fit_only_figure}")
    if FIT_ONLY_FIGURE_STYLE.get("save_pdf", False):
        print(f"Wrote {fit_only_figure.with_suffix('.pdf')}")
    print(f"Wrote {full_rvm_figure}")
    if RVMFIT_FIGURE_STYLE.get("save_pdf", False):
        print(f"Wrote {full_rvm_figure.with_suffix('.pdf')}")
    print(f"Wrote {INFORMATION_FILE}")


if __name__ == "__main__":
    main()
