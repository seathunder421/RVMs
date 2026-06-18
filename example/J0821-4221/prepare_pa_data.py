#!/usr/bin/env python3
"""Prepare PA data for the RVM fit.

This script reads the raw pulsar CSV, computes normalized Stokes profiles,
estimates PA errors, applies the small hand-tuned PA wraps, and writes the
filtered CSV used by the fitting script.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
import os
from pathlib import Path
import tempfile

os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "pulsar_rvm_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pulsar_rvm_cache"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# =============================================================================
# 参数区：一般只需要改这里
# =============================================================================

# 原始数据文件。
# None 表示自动读取当前目录下唯一的 9 列原始 CSV，例如 J0821-4221.csv。
# 如果目录里有多个原始 CSV，就手动写成 Path("J0821-4221.csv")。
RAW_DATA_FILE: Path | None = None  # 原始数据 CSV；None 表示自动寻找当前目录唯一原始 CSV。

# 显示/输出相位范围，也是输出过滤数据文件名里的左右边界。
# PPA 检查图横轴会显示 PHASE_MIN 到 PHASE_MAX。
# fit_rvm.py 默认也会读取这个范围内的 PPA 点。
PHASE_MIN = -9.6  # 图中显示和输出数据保留的最小相位，经度单位 deg。
PHASE_MAX = 24.0  # 图中显示和输出数据保留的最大相位，经度单位 deg。

# 噪声区间，用来计算 I/Q/U/V 的标准差。
# 应该选没有脉冲信号的 off-pulse 区域，保持和原程序逻辑一致。
NOISE_RANGE = (-180.0, -100.0)  # off-pulse 噪声区间，用来估计 I/Q/U/V 噪声标准差。

# PPA 检测阈值：只有 L > PA_DETECTION_SIGMA * sigma_I 的点才计算 PA。
PA_DETECTION_SIGMA = 2.0  # 计算 PA 的线偏振检测阈值，L 必须大于该倍数的噪声。

# PPA 信噪比截断：SN < PA_SN_CUTOFF 的点会从 PA_filtered 中去掉。
PA_SN_CUTOFF = 2.0  # PPA 信噪比下限，低于这个值的点会从 PA_filtered 去掉。

# 是否使用原来的 scipy quad + brentq 计算 PA_error。
# True：严格走原始 scipy 逻辑；False：用同公式的无 scipy 数值积分。
# 如果运行环境没有 scipy，或 scipy 安装不可用，保持 False 更通用。
USE_SCIPY_PA_ERROR = False  # 是否使用 scipy 原始积分算法计算 PA_error；False 更通用。

# 自定义 PPA 选择区域，只控制哪些 PA_filtered 保留进入 RVM 拟合。
# 注意：检查图里不画黄色区域，只用红点显示最终保留的 PPA。
# 可以写多个区间，例如 [(-9.6, -2.0), (3.0, 24.0)]。
PPA_SELECTION_REGIONS = [  # 自定义进入 RVM 拟合的 PPA 相位区间，可写多个区间。
    (PHASE_MIN, PHASE_MAX),  # 默认保留整个显示相位范围。
]

# 是否输出 PPA 选择检查图：Jxxxx_PPA_selection.png。
MAKE_PPA_SELECTION_PLOT = True  # 是否输出 PPA 选择检查图。

# PPA 选择检查图大小和清晰度。
# figsize: 图大小；dpi/save_dpi: 屏幕绘图和保存 PNG 清晰度，越大越清晰，文件也越大。
# save_pdf/pdf_dpi: 是否同时保存对应 PDF，以及 PDF 中栅格化点的清晰度。
# bbox_inches: 保存边距，"tight" 会自动收紧空白；tight_layout: 是否自动整理子图间距。
PPA_SELECTION_FIGSIZE = (7.2, 5.4)  # PPA 选择图大小，单位 inch。
PPA_SELECTION_PLOT_DPI = 450  # PPA 选择图绘制 DPI。
PPA_SELECTION_SAVE_DPI = 450  # PPA 选择图保存 PNG 的 DPI。
PPA_SELECTION_SAVE_PDF = True  # 是否同时保存 PPA 选择图 PDF。
PPA_SELECTION_PDF_DPI = 450  # PPA 选择图保存 PDF 时栅格化点的 DPI。
PPA_SELECTION_BBOX_INCHES = "tight"  # 保存图片边距，tight 会自动裁掉多余空白。
PPA_SELECTION_TIGHT_LAYOUT = True  # 是否自动整理子图和标签间距。
PPA_SELECTION_SAVE_FACECOLOR = "white"  # 保存图片背景颜色。

# PPA 检查图线条粗细。
PPA_PROFILE_LINEWIDTH = 2.6  # PPA 检查图中 I/L/V 轮廓线宽。
PPA_ERRORBAR_LINEWIDTH = 1.8  # PPA 数据点误差棒线宽。
PPA_POINT_SIZE = 6.5  # PPA 数据点大小。
PPA_SHIFTED_POINT_MARKER = "o"  # 被手动移动过的 PPA 点标记形状，和普通 PA 点保持圆形。
PPA_SHIFTED_POINT_SIZE = 6.5  # 被手动移动过的 PPA 点标记大小，和普通 PA 点一致。
PPA_SHIFTED_POINT_FACE_COLOR = "gold"  # 被手动移动过后的 PPA 点填充颜色，黄色。
PPA_SHIFTED_POINT_EDGE_COLOR = "gold"  # 被手动移动过后的 PPA 点边框颜色，黄色。
PPA_SHIFTED_POINT_LINEWIDTH = 1.0  # 被手动移动过的 PPA 点边框线宽。

# PPA 检查图字体大小。
PPA_TITLE_FONTSIZE = 17  # PPA 检查图标题字体大小。
PPA_LABEL_FONTSIZE = 15  # PPA 检查图坐标轴标签字体大小。
PPA_TICK_LABELSIZE = 14  # PPA 检查图刻度数字字体大小。
PPA_LEGEND_FONTSIZE = 12  # PPA 检查图图例字体大小。
PPA_PA_LEGEND_LOC = "upper right"  # PPA 检查图上方 PA 面板图例位置。
PPA_FLUX_LEGEND_LOC = "upper right"  # PPA 检查图下方强度面板图例位置。

# 手动 PPA 调节规则，按顺序执行。
#
# 常用字段：
# enabled: 是否启用这条规则，False 表示只是模板，不执行。
# name: 规则说明，方便自己看。
# angle_min/angle_max: 角度在这个范围内才执行，使用 angle_min < Angle < angle_max。
# angle_lt/angle_gt: 也可以只写 Angle < 某值 或 Angle > 某值。
# pa_lt/pa_gt: PA 小于/大于某值才执行。
# pa_min/pa_max: PA 在某个范围内才执行。
# pa_is_nan: True 表示只选 PA 是 NaN 的点；False 表示只选 PA 不是 NaN 的点。
# action:
#   "shift"：移动 PPA，配合 offset_deg=90 或 180。
#   "set_nan"：把点设成 np.nan，相当于去掉这个 PPA 点。
#   "set_zero"：把点设成 0，后面画图和拟合也会忽略。
#   "set_value"：把 PPA 改成指定 value。
#
# 下面前两条是当前实际使用的规则；后面 8 条是常用模板，默认 enabled=False。
PA_WRAP_RULES = [
    # 当前规则1：-2 到 8 度内，PA<0 的点去掉。
    {"enabled": False, "name": "当前规则1：-2 到 8 度内，PA<0 的点去掉", "angle_min": -2.0, "angle_max": 8.0, "pa_lt": 0.0, "action": "set_nan"},  # 启用后删除 -2 到 8 度内 PA<0 的点。
    # 当前规则2：Angle<-3 且 PA<-50 的点加 180 度。
    {"enabled": True, "name": "当前规则2：Angle<-3 且 PA<-50 的点加 180 度", "angle_lt": -3.0, "pa_lt": -50.0, "action": "shift", "offset_deg": 180.0},  # 启用后把 Angle<-3 且 PA<-50 的点上移 180 度。
    # 模板1：某个角度范围内 PA<0 加 90 度。
    {"enabled": True,"name": "模板1：某个角度范围内 PA<0 加 90 度", "angle_min": -2.0, "angle_max": 8.0, "pa_lt": 0.0, "action": "shift", "offset_deg": 90.0},  # 改 enabled=True 后，把指定范围 PA<0 的点上移 90 度。
    # 模板2：某个角度范围内 PA>0 减 90 度。
    {"enabled": False, "name": "模板2：某个角度范围内 PA>0 减 90 度", "angle_min": -2.0, "angle_max": 8.0, "pa_gt": 0.0, "action": "shift", "offset_deg": -90.0},  # 改 enabled=True 后，把指定范围 PA>0 的点下移 90 度。
    # 模板3：Angle 小于某值且 PA 大于某值，加 90 度。
    {"enabled": False, "name": "模板3：Angle 小于某值且 PA 大于某值，加 90 度", "angle_lt": -3.0, "pa_gt": 20.0, "action": "shift", "offset_deg": 90.0},  # 改 enabled=True 后，把 Angle 小于阈值且 PA 大于阈值的点上移 90 度。
    # 模板4：Angle 大于某值且 PA 小于某值，加 180 度。
    {"enabled": False, "name": "模板4：Angle 大于某值且 PA 小于某值，加 180 度", "angle_gt": 8.0, "pa_lt": -60.0, "action": "shift", "offset_deg": 180.0},  # 改 enabled=True 后，把 Angle 大于阈值且 PA 小于阈值的点上移 180 度。
    # 模板5：某个角度范围内 PA 大于某值，设成 np.nan 去掉。
    {"enabled": False, "name": "模板5：某个角度范围内 PA 大于某值，设成 np.nan 去掉", "angle_min": -5.0, "angle_max": 5.0, "pa_gt": 60.0, "action": "set_nan"},  # 改 enabled=True 后，删除指定范围内 PA 大于阈值的点。
    # 模板6：某个角度范围内 PA 小于某值，设成 np.nan 去掉。
    {"enabled": False, "name": "模板6：某个角度范围内 PA 小于某值，设成 np.nan 去掉", "angle_min": -5.0, "angle_max": 5.0, "pa_lt": -60.0, "action": "set_nan"},  # 改 enabled=True 后，删除指定范围内 PA 小于阈值的点。
    # 模板7：PA 在某个范围内的点加 90 度。
    {"enabled": False, "name": "模板7：PA 在某个范围内的点加 90 度", "pa_min": -30.0, "pa_max": 30.0, "action": "shift", "offset_deg": 90.0},  # 改 enabled=True 后，把 PA 在指定范围内的点上移 90 度。
    # 模板8：PA 是 NaN 的点保持去掉。
    {"enabled": False, "name": "模板8：PA 是 NaN 的点保持去掉", "pa_is_nan": True, "action": "set_nan"},  # 改 enabled=True 后，保持 NaN 点不进入拟合。
    
    
    
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


RAW_COLUMNS = ["isub", "ichan", "Bin", "I", "Q", "U", "V", "PA", "PA_error"]
OUTPUT_COLUMNS = [
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

EXCLUDED_CSV_PATTERNS = (
    "*_filtered_data.csv",
    "*_data.csv",
    "combined_data.csv",
    "alpha_beta.csv",
    "PA_filtered_log.csv",
)


@dataclass(frozen=True)
class NoiseStats:
    i: float
    q: float
    u: float
    v: float


_SCIPY_PA_ERROR = None


def original_pa_error_with_scipy(noise_level: float, linear_pol: float) -> float | None:
    global _SCIPY_PA_ERROR

    if not USE_SCIPY_PA_ERROR:
        return None

    if _SCIPY_PA_ERROR is False:
        return None

    if _SCIPY_PA_ERROR is None:
        try:
            from scipy.integrate import quad
            from scipy.optimize import brentq
            from scipy.special import erf as scipy_erf

            _SCIPY_PA_ERROR = (quad, brentq, scipy_erf)
        except Exception:
            _SCIPY_PA_ERROR = False
            return None

    quad, brentq, scipy_erf = _SCIPY_PA_ERROR
    snr = linear_pol / noise_level

    def original_distribution(psi: float, p0: float) -> float:
        eta_0 = (p0 / np.sqrt(2)) * np.cos(2 * psi)
        term1 = 1 / np.sqrt(np.pi)
        term2 = term1 + eta_0 * np.exp(eta_0**2) * (1 + scipy_erf(eta_0))
        term3 = np.exp(-p0**2 / 2)
        return term1 * term2 * term3

    def original_integral(p0: float, psi_max: float) -> float:
        result, _ = quad(lambda psi: original_distribution(psi, p0), -psi_max, psi_max)
        return result

    try:
        psi_rad = brentq(lambda psi: original_integral(snr, psi) - 0.6826, 0, np.pi / 2)
    except ValueError:
        return 0.0

    return float(np.rad2deg(psi_rad))


def pa_error(noise_level: float, linear_pol: float) -> float:
    """Return the PA uncertainty in degrees."""
    if noise_level <= 0:
        return 0.0

    snr = linear_pol / noise_level
    if snr > 10:
        return 28.65 / snr
    if snr <= 2:
        return 0.0

    original_value = original_pa_error_with_scipy(noise_level, linear_pol)
    if original_value is not None:
        return original_value

    def distribution(psi: np.ndarray, p0: float) -> np.ndarray:
        eta_0 = (p0 / np.sqrt(2.0)) * np.cos(2.0 * psi)
        term1 = 1.0 / np.sqrt(np.pi)
        term2 = term1 + eta_0 * np.exp(eta_0**2) * (1.0 + np.vectorize(erf)(eta_0))
        term3 = np.exp(-(p0**2) / 2.0)
        return term1 * term2 * term3

    def enclosed_probability(psi_max: float) -> float:
        grid = np.linspace(-psi_max, psi_max, 801)
        return float(np.trapz(distribution(grid, snr), grid))

    try:
        low = 0.0
        high = np.pi / 2.0
        if enclosed_probability(high) < 0.6826:
            return 0.0
        for _ in range(60):
            mid = (low + high) / 2.0
            if enclosed_probability(mid) < 0.6826:
                low = mid
            else:
                high = mid
        psi_rad = (low + high) / 2.0
    except (ValueError, FloatingPointError, OverflowError):
        return 0.0

    return float(np.rad2deg(psi_rad))


def read_raw_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, delimiter=",", on_bad_lines="skip")
    if data.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"{path} must contain {len(RAW_COLUMNS)} columns; found {data.shape[1]}.")

    data.columns = RAW_COLUMNS
    data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return data


def is_excluded_csv(path: Path) -> bool:
    return any(path.match(pattern) for pattern in EXCLUDED_CSV_PATTERNS)


def discover_raw_data_file() -> Path:
    candidates = [path for path in sorted(Path(".").glob("*.csv")) if not is_excluded_csv(path)]
    valid_files: list[Path] = []

    for path in candidates:
        try:
            preview = pd.read_csv(path, header=None, nrows=5)
        except Exception:
            continue
        if preview.shape[1] == len(RAW_COLUMNS):
            valid_files.append(path)

    if not valid_files:
        raise FileNotFoundError("No raw 9-column pulsar CSV file found.")
    if len(valid_files) > 1:
        names = ", ".join(str(path) for path in valid_files)
        raise RuntimeError(f"Multiple raw CSV files found; set RAW_DATA_FILE manually: {names}")

    return valid_files[0]


def format_phase(value: float) -> str:
    return f"{value:g}"


def center_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Center the total-intensity peak at pulse phase 0."""
    data = data.copy()
    data["Bin"] = data["Bin"] / data["Bin"].max()

    peak_bin = data.loc[data["I"].idxmax(), "Bin"]
    shift_steps = int((0.5 - peak_bin) * len(data))

    for column in ["I", "Q", "U", "V"]:
        data[column] = np.roll(data[column].to_numpy(), shift_steps)

    data["Angle"] = (data["Bin"] - 0.5) * 360.0
    return data


def normalize_stokes(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    i_peak = data["I"].max()
    if i_peak == 0:
        raise ValueError("Cannot normalize Stokes parameters because max(I) is zero.")

    data["I_normalized"] = data["I"] / i_peak
    data["Q_normalized"] = data["Q"] / i_peak
    data["U_normalized"] = data["U"] / i_peak
    data["V_normalized"] = data["V"] / i_peak
    return data


def estimate_noise(data: pd.DataFrame) -> NoiseStats:
    mask = (data["Angle"] > NOISE_RANGE[0]) & (data["Angle"] <= NOISE_RANGE[1])
    if not mask.any():
        raise ValueError(f"No samples found in NOISE_RANGE={NOISE_RANGE}.")

    return NoiseStats(
        i=float(np.std(data.loc[mask, "I_normalized"])),
        q=float(np.std(data.loc[mask, "Q_normalized"])),
        u=float(np.std(data.loc[mask, "U_normalized"])),
        v=float(np.std(data.loc[mask, "V_normalized"])),
    )


def compute_pa_columns(data: pd.DataFrame, noise: NoiseStats) -> pd.DataFrame:
    data = data.copy()

    raw_l2 = data["Q_normalized"] ** 2 + data["U_normalized"] ** 2
    noise_l2 = noise.q**2 + noise.u**2
    debiased_l = np.sqrt(np.abs(raw_l2 - noise_l2))
    data["Linear_Polarization"] = np.where(raw_l2 >= noise_l2, debiased_l, -debiased_l)

    detected = data["Linear_Polarization"] > PA_DETECTION_SIGMA * noise.i
    data["PA"] = np.where(
        detected,
        0.5 * np.degrees(np.arctan2(data["U"], data["Q"])),
        np.nan,
    )
    data["PA_error"] = [pa_error(noise.i, value) for value in data["Linear_Polarization"]]
    data["PA_filtered"] = data["PA"]
    data["PA_error_filtered"] = data["PA_error"]
    data["PA_shifted"] = 0
    data["PA_shift_deg"] = 0.0
    return data


def build_pa_rule_mask(data: pd.DataFrame, rule: dict) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    angle = data["Angle"]
    pa_column = rule.get("pa_column", "PA")
    pa = data[pa_column]

    if "angle_min" in rule:
        mask &= angle > rule["angle_min"]
    if "angle_max" in rule:
        mask &= angle < rule["angle_max"]
    if "angle_gt" in rule:
        mask &= angle > rule["angle_gt"]
    if "angle_lt" in rule:
        mask &= angle < rule["angle_lt"]

    if "pa_lt" in rule:
        mask &= pa < rule["pa_lt"]
    if "pa_gt" in rule:
        mask &= pa > rule["pa_gt"]
    if "pa_min" in rule:
        mask &= pa > rule["pa_min"]
    if "pa_max" in rule:
        mask &= pa < rule["pa_max"]
    if "pa_is_nan" in rule:
        mask &= pa.isna() if rule["pa_is_nan"] else pa.notna()

    return mask


def apply_pa_wraps(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for rule in PA_WRAP_RULES:
        if not rule.get("enabled", True):
            continue

        mask = build_pa_rule_mask(data, rule)
        action = rule.get("action", "shift")

        if action == "shift":
            offset = rule.get("offset_deg", 0.0)
            if pd.isna(offset):
                data.loc[mask, ["PA_filtered", "PA_error_filtered"]] = np.nan
            else:
                data.loc[mask, "PA_filtered"] += offset
                data.loc[mask, "PA_shifted"] = 1
                data.loc[mask, "PA_shift_deg"] += offset
        elif action == "set_nan":
            data.loc[mask, ["PA_filtered", "PA_error_filtered"]] = np.nan
        elif action == "set_zero":
            data.loc[mask, ["PA_filtered", "PA_error_filtered"]] = 0.0
        elif action == "set_value":
            data.loc[mask, "PA_filtered"] = rule["value"]
        else:
            raise ValueError(f"Unknown PA_WRAP_RULES action: {action}")

    return data


def apply_sn_cut(data: pd.DataFrame, noise: NoiseStats) -> pd.DataFrame:
    data = data.copy()
    data["SN"] = data["Linear_Polarization"] / noise.i
    low_sn = data["SN"] < PA_SN_CUTOFF
    data.loc[low_sn, ["PA_filtered", "PA_error_filtered", "SN"]] = 0.0
    return data


def ppa_region_mask(angle: pd.Series) -> pd.Series:
    if not PPA_SELECTION_REGIONS:
        return pd.Series(True, index=angle.index)

    mask = pd.Series(False, index=angle.index)
    for left, right in PPA_SELECTION_REGIONS:
        low = min(left, right)
        high = max(left, right)
        mask |= angle.between(low, high)
    return mask


def apply_ppa_selection(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    outside_selection = ~ppa_region_mask(data["Angle"])
    data.loc[outside_selection, ["PA_filtered", "PA_error_filtered"]] = 0.0
    return data


def plot_pa_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    error_column: str,
    color: str,
    alpha: float,
    label: str,
    include_shifted: bool = True,
) -> None:
    pa = data[column].replace(0, np.nan)
    pa_error = data[error_column].replace(0, np.nan)
    mask = pa.notna() & pa_error.notna()
    if not include_shifted and "PA_shifted" in data:
        mask &= ~data["PA_shifted"].fillna(0).astype(bool)
    ax.errorbar(
        data.loc[mask, "Angle"],
        pa.loc[mask],
        xerr=0.1,
        yerr=pa_error.loc[mask],
        fmt="o",
        color=color,
        alpha=alpha,
        markersize=PPA_POINT_SIZE,
        capsize=2.2,
        elinewidth=PPA_ERRORBAR_LINEWIDTH,
        rasterized=True,
        label=label,
    )


def plot_shifted_pa_markers(ax: plt.Axes, data: pd.DataFrame) -> None:
    pa = data["PA_filtered"].replace(0, np.nan)
    pa_error = data["PA_error_filtered"].replace(0, np.nan)
    shifted = data["PA_shifted"].fillna(0).astype(bool)
    mask = shifted & pa.notna() & pa_error.notna()
    if not mask.any():
        return

    ax.errorbar(
        data.loc[mask, "Angle"],
        pa.loc[mask],
        xerr=0.1,
        yerr=pa_error.loc[mask],
        fmt=PPA_SHIFTED_POINT_MARKER,
        markersize=PPA_SHIFTED_POINT_SIZE,
        markerfacecolor=PPA_SHIFTED_POINT_FACE_COLOR,
        markeredgecolor=PPA_SHIFTED_POINT_EDGE_COLOR,
        markeredgewidth=PPA_SHIFTED_POINT_LINEWIDTH,
        ecolor=PPA_SHIFTED_POINT_EDGE_COLOR,
        capsize=2.2,
        elinewidth=PPA_ERRORBAR_LINEWIDTH,
        alpha=1.0,
        rasterized=True,
        label="Shifted PPA",
        zorder=8,
    )


def plot_ppa_selection(data: pd.DataFrame, psr_name: str, output_file: Path) -> None:
    fig = plt.figure(figsize=PPA_SELECTION_FIGSIZE, dpi=PPA_SELECTION_PLOT_DPI)
    grid = gridspec.GridSpec(2, 1, height_ratios=[1.25, 1.0], hspace=0.0)
    ax_pa = fig.add_subplot(grid[0])
    ax_flux = fig.add_subplot(grid[1], sharex=ax_pa)

    plot_pa_points(ax_pa, data, "PA", "PA_error", color="0.35", alpha=0.25, label="Raw PA")
    plot_pa_points(
        ax_pa,
        data,
        "PA_filtered",
        "PA_error_filtered",
        color="crimson",
        alpha=0.95,
        label="Unshifted PPA",
        include_shifted=False,
    )
    plot_shifted_pa_markers(ax_pa, data)

    ax_pa.set_title(f"PSR {psr_name} PPA selection", fontsize=PPA_TITLE_FONTSIZE, pad=8)
    ax_pa.set_ylabel("PA (deg)", fontsize=PPA_LABEL_FONTSIZE)
    ax_pa.set_ylim(-180, 180)
    ax_pa.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_pa.legend(loc=PPA_PA_LEGEND_LOC, fontsize=PPA_LEGEND_FONTSIZE, framealpha=1.0)
    ax_pa.tick_params(labelbottom=False, labelsize=PPA_TICK_LABELSIZE)
    ax_pa.grid(False)

    ax_flux.plot(data["Angle"], data["I_normalized"], color="black", linewidth=PPA_PROFILE_LINEWIDTH, label="I")
    ax_flux.plot(data["Angle"], data["Linear_Polarization"], color="#1f77b4", linewidth=PPA_PROFILE_LINEWIDTH, label="L")
    ax_flux.plot(data["Angle"], data["V_normalized"], color="#d62728", linewidth=PPA_PROFILE_LINEWIDTH, label="V")
    ax_flux.set_xlabel("Longitude (deg)", fontsize=PPA_LABEL_FONTSIZE)
    ax_flux.set_ylabel("Intensity", fontsize=PPA_LABEL_FONTSIZE)
    ax_flux.set_xlim(PHASE_MIN, PHASE_MAX)
    ax_flux.set_ylim(-0.18, 1.12)
    ax_flux.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_flux.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_flux.tick_params(axis="both", labelsize=PPA_TICK_LABELSIZE)
    ax_flux.legend(loc=PPA_FLUX_LEGEND_LOC, fontsize=PPA_LEGEND_FONTSIZE, framealpha=1.0)
    ax_flux.grid(False)

    if PPA_SELECTION_TIGHT_LAYOUT:
        fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=PPA_SELECTION_SAVE_DPI,
        bbox_inches=PPA_SELECTION_BBOX_INCHES,
        facecolor=PPA_SELECTION_SAVE_FACECOLOR,
    )
    if PPA_SELECTION_SAVE_PDF:
        fig.savefig(
            output_file.with_suffix(".pdf"),
            dpi=PPA_SELECTION_PDF_DPI,
            bbox_inches=PPA_SELECTION_BBOX_INCHES,
            facecolor=PPA_SELECTION_SAVE_FACECOLOR,
        )
    plt.close(fig)


def process_data(path: Path) -> pd.DataFrame:
    data = read_raw_data(path)
    data = center_profile(data)
    data = normalize_stokes(data)
    noise = estimate_noise(data)

    print(f"Noise std: I={noise.i:.6g}, Q={noise.q:.6g}, U={noise.u:.6g}, V={noise.v:.6g}")

    data = compute_pa_columns(data, noise)
    data = apply_pa_wraps(data)
    data = apply_sn_cut(data, noise)
    data = apply_ppa_selection(data)
    return data


def main() -> None:
    raw_file = RAW_DATA_FILE or discover_raw_data_file()
    psr_name = raw_file.stem
    output_file = Path(f"{psr_name}_{format_phase(PHASE_MIN)}_{format_phase(PHASE_MAX)}_filtered_data.csv")

    processed = process_data(raw_file)
    processed.to_csv(output_file, columns=OUTPUT_COLUMNS, index=False)
    if MAKE_PPA_SELECTION_PLOT:
        plot_file = Path(f"{psr_name}_PPA_selection.png")
        plot_ppa_selection(processed, psr_name, plot_file)
        print(f"Wrote {plot_file}")
        if PPA_SELECTION_SAVE_PDF:
            print(f"Wrote {plot_file.with_suffix('.pdf')}")
    print(f"Read {raw_file}")
    print(f"Wrote {output_file}")
    print(f"Selected fit window: {PHASE_MIN} to {PHASE_MAX} deg")
    print(f"Selected PPA regions: {PPA_SELECTION_REGIONS}")


if __name__ == "__main__":
    main()
