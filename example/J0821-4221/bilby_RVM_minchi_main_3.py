import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
import seaborn as sns
import csv
import json
import pickle
from PIL import Image
import math
import glob
import warnings
import psrqpy
import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
import bilby
import matplotlib.patches as patches
from scipy.signal import savgol_filter


# 设置 Pandas 配置
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)   # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列

# 设置 NumPy 配置
import matplotlib as mpl
mpl.use("Agg")  # 使用安全的渲染后端，防止 macOS PDF backend 崩溃

# 字体与PDF设置（不要求安装额外字体）
mpl.rcParams.update({
    'font.sans-serif': ['DejaVu Sans'],   # matplotlib 自带字体，跨平台安全
    'font.family': 'sans-serif',
    'pdf.fonttype': 3,                    # 使用 Type 3 字体，避免 TrueType 嵌入问题
    'ps.fonttype': 3,
    'pdf.use14corefonts': False,          # 不强制使用 Helvetica
    'axes.unicode_minus': False,          # 支持负号
})


np.random.seed(666)
# 设置 Matplotlib 配置
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 忽略警告
warnings.filterwarnings("ignore")


def process_data(file):
    data = pd.read_csv(file, header=None, delimiter=',', on_bad_lines='skip')
    data.columns = ['Angle','I_normalized','Linear_Polarization', 'V_normalized', 'PA','PA_error','PA_filtered',"PA_error_filtered","SN"]
    # 将列转换为数值类型，处理可能存在的非数值数据
    for col in ['Angle','I_normalized','Linear_Polarization', 'V_normalized', 'PA','PA_error','PA_filtered',"PA_error_filtered","SN"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data   



print("Current working directory:", os.getcwd())
# 匹配所有以 filtered_data.csv 结尾的文件
file = glob.glob("*_filtered_data.csv")[0]
print("读取当前文件：:",file)
print("===========================================================================================") 
print("===========================================================================================") 
psr = file.split("_")[0]
all_min, all_max = map(float, [file.split("_")[1], file.split("_")[2]])  # 转换为浮点数
data = process_data(file)   
# 打印分离结果
print("脉冲星：", psr)
print("左边界：", all_min)
print("右边界：", all_max)
print("===========================================================================================") 

Angle = data["Angle"]
PA = data["PA_filtered"]
PA_error = data["PA_error_filtered"]
PA[PA == 0] = np.nan  # 将 PA 中的 0 替换为 NaN
PA_error[PA_error == 0] = np.nan
mask = (Angle >= all_min) & (Angle <= all_max)
Angle_filtered = Angle[mask]
PA_filtered = PA[mask]
PA_error = PA_error[mask]
valid_mask = ~np.isnan(PA_filtered)  # 生成非 NaN 掩码

Angle_filtered = Angle_filtered[valid_mask]
PA_filtered = PA_filtered[valid_mask]
PA_err_filtered =  PA_error[valid_mask]


# 创建一个 DataFrame
df = pd.DataFrame({'Angle': Angle_filtered,'PA': PA_filtered,'PA_error': PA_err_filtered})
# 保存为 CSV 文件
df.to_csv(f'{psr}_data.csv', index=False)


print("======================================MCMC=================================================") 


def get_component_width(data, angle_min_search, angle_max_search, std_I,
                        window=11, polyorder=2,
                        plot=True, ax=None):

    # ===== 原始筛选 =====
    anglemask = (data['Angle'] >= angle_min_search) & (data['Angle'] <= angle_max_search)
    subset = data[anglemask].copy()

    x = subset['Angle'].values
    y = subset['I_normalized'].values

    if len(y) < window or len(y) < 3:
        return {
            "width": np.nan,
            "midpoint": np.nan
        }

    # ===== window 修正 =====
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window < polyorder + 2:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1

    if window < 3:
        return {
            "width": np.nan,
            "midpoint": np.nan
        }

    # ===== 平滑 =====
    y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)

    # ===== 峰值 =====
    idx = np.argmax(y_smooth)
    Imax = y_smooth[idx]
    Imaxbin = x[idx]

    # ===== 阈值 =====
    Ith = 3 * std_I
    if np.max(y_smooth) < Ith:
        Ith = 2 * std_I
    if np.max(y_smooth) < Ith:
        Ith = 1.5 * std_I

    # ===== crossings（保留用于 fallback）=====
    crossings = []
    for i in range(len(x) - 1):
        y1, y2 = y_smooth[i], y_smooth[i + 1]
        x1, x2 = x[i], x[i + 1]

        if y1 == Ith:
            crossings.append(x1)

        if (y1 - Ith) * (y2 - Ith) < 0:
            if y2 != y1:
                xc = x1 + (Ith - y1) / (y2 - y1) * (x2 - x1)
                crossings.append(xc)

    if len(x) > 0 and y_smooth[-1] == Ith:
        crossings.append(x[-1])

    crossings = np.array(sorted(crossings))

    # =========================================================
    # ⭐ 核心：用 > Ith 区域定义“整体宽度”（支持多峰）
    # =========================================================
    above = x[y_smooth > Ith]

    if len(above) >= 2:
        angle_min = above[0]
        angle_max = above[-1]

    else:
        print("[DEBUG] No continuous >3σ region → fallback")

        if len(crossings) >= 2:
            angle_min = crossings[0]
            angle_max = crossings[-1]

        elif len(crossings) == 1:
            if crossings[0] < Imaxbin:
                angle_min = crossings[0]
                angle_max = x[-1]
            else:
                angle_min = x[0]
                angle_max = crossings[0]

        else:
            print("[DEBUG] Final fallback → full range")
            angle_min = x[0]
            angle_max = x[-1]

    # ===== 顺序保护 =====
    if angle_min > angle_max:
        angle_min, angle_max = angle_max, angle_min

    # ===== 插值 =====
    Imin_val = np.interp(angle_min, x, y_smooth)
    Imax_val = np.interp(angle_max, x, y_smooth)

    width = angle_max - angle_min
    midpoint = (angle_min + angle_max) / 2

    # ===== 画图 =====
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
            created_fig = True
        else:
            created_fig = False

        ax.plot(x, y, '.', color='black', alpha=0.6, markersize=8)
        ax.plot(x, y_smooth, color='black', lw=2.5)

        ax.axhline(Ith, color='m', ls='--', lw=2.5)
        ax.scatter(Imaxbin, Imax, color='blue', s=60, zorder=12)

        ax.axvline(angle_min, color='m', ls='--', lw=1.5)
        ax.axvline(angle_max, color='m', ls='--', lw=1.5)

        ax.fill_betweenx([0, Imax], angle_min, angle_max,
                         color='orange', alpha=0.15)

        ax.text(midpoint, Imax * 0.6,
                f"$W_{{3\\sigma}} = {width:.2f}^\\circ$",
                ha='center', va='center',
                fontsize=26, weight='bold',
                color='#00CED1',
                bbox=dict(facecolor='grey', alpha=0.8, edgecolor='none'))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.tick_params(axis='both', labelsize=20)

        ax.set_title("Smoothed Profile", fontsize=26, pad=10, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

        if created_fig:
            plt.tight_layout()
            plt.savefig("3sigma_width.png", dpi=300)
            plt.close(fig)

    return {
        "Imax": Imax,
        "Imaxbin": Imaxbin,
        "subset": subset,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "Imin_val": Imin_val,
        "Imax_val": Imax_val,
        "width": width,
        "midpoint": midpoint,
        "threshold": Ith,
        "crossings": crossings
    }


    
def rvm_model(phi, alpha, beta, phase_offset, psi_offset):
    # 计算 RVM
    alpha = np.radians(180 - alpha)
    beta = np.radians(-beta)
    phi = np.radians(phi)
    phase_offset = np.radians(phase_offset)
    zeta = alpha + beta
    numerator = np.sin(alpha) * np.sin(phi - phase_offset)
    denominator = np.sin(zeta) * np.cos(alpha) - np.cos(zeta) * np.sin(alpha) * np.cos(phi - phase_offset)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.degrees(np.arctan(numerator / denominator)) + psi_offset
# 定义似然函数
likelihood = bilby.likelihood.GaussianLikelihood(x=Angle_filtered,y=PA_filtered,func=rvm_model,sigma=PA_err_filtered)
# 定义先验
priors = bilby.prior.PriorDict()
priors['alpha'] = bilby.prior.Uniform(0, 180, 'alpha')
priors['beta'] = bilby.prior.Uniform(-90, 90, 'beta')
priors['phase_offset'] = bilby.prior.Uniform(-150, 150, 'phase_offset')
priors['psi_offset'] = bilby.prior.Uniform(-150,150, 'psi_offset')

if __name__ == '__main__':   
    quantile = [0.025, 0.5, 0.975]            
    result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='dynesty',
                                     nlive = 800,
                                     sample="rwalk",
                                     dlogz =0.01,
                                     walks=200,
                                     print_progress=True,
                                     write_progress=True,
                                     bootstrap=0,
                                     nthreads=8,
                                     burnin=500,
                                     outdir='./',
                                     label=psr + f"_{0}_RVM_fit",
                                     verbose=True)   

    truths = [result.posterior['phase_offset'].mean(),result.posterior['psi_offset'].mean()]
    fig = result.plot_corner(
        parameters=['phase_offset', 'psi_offset'],  # 指定采样参数
        labels=[r'$\phi$', r'$\psi$'],  # 自定义轴标签
        show_titles=True,  # 显示每个参数的标题
        title_fmt='.2f',  # 标题的格式化方式
        quantiles=quantile,  # 显示分位点
        color='grey',  # 线和点的颜色
        smooth=1,  # 平滑度
        truths=truths,  # 设置真实值为拟合参数均值
        truth_color='red', )
    



    # 添加自定义分位点线条
    quantiles = [0.00135, 0.025, 0.16, 0.5, 0.84, 0.975, 0.99865]
    colors = ["red", "orange", "blue", "green", "blue", "orange", "red"]
    sigma_labels = ["-3σ", "-2σ", "-1σ", "Median", "+1σ", "+2σ", "+3σ"]
    # 获取角点图的所有子图
    axes = np.array(fig.axes).reshape(len(truths), len(truths))
    # 遍历对角线上的直方图，绘制参考线
    for i, ax in enumerate(axes.diagonal()):
        data1 = result.posterior[result.posterior.columns[i]].values  # 当前参数的采样值
        for q, color, label in zip(quantiles, colors, sigma_labels):
            quantile_value = np.percentile(data1, q * 100)  # 计算百分位数
            ax.axvline(quantile_value, color=color, linestyle="--", linewidth=0.8)
            ax.text(quantile_value, ax.get_ylim()[1] * 0.8,  label, color=color, fontsize=7, rotation=90, verticalalignment="center", horizontalalignment="right" )
   # 显示图像
    fig.suptitle("Corner Plot", fontsize=16, y=1.05)
    fig.savefig(psr + "_phase_psi_corner_plot.png")

    posterior_samples = result.posterior
    phase_samples = np.array(posterior_samples['phase_offset'])
    psi_samples = np.array(posterior_samples['psi_offset'])
    # 计算二维 KDE
    samples = np.vstack([phase_samples, psi_samples])
    kde = gaussian_kde(samples)
    kde.set_bandwidth(bw_method='scott')  # 自动调整带宽
    # 生成网格点
    phase_range = np.linspace(min(phase_samples), max(phase_samples), 400)
    psi_range = np.linspace(min(psi_samples), max(psi_samples), 400)
    phase_grid, psi_grid = np.meshgrid(phase_range, psi_range)
    grid_points = np.vstack([phase_grid.ravel(), psi_grid.ravel()])
    # 计算密度
    density = kde(grid_points)
    density_grid = density.reshape(phase_grid.shape)
    # 找到密度最大的位置
    max_density_idx = np.argmax(density)
    best_phase = phase_grid.ravel()[max_density_idx]
    best_psi = psi_grid.ravel()[max_density_idx]
    print(f"密度最大点: phase = {best_phase}, psi = {best_psi}")

    # 创建子图
    fig, ax = plt.subplots(figsize=(8, 7),dpi=600)
    # 假设 phase_grid, psi_grid, density_grid 已经被定义并且是合适的二维数组
    # 定义密度图的等高线级别
    levels = np.linspace(np.min(density_grid), np.max(density_grid), 20)  # 更高的等高线密度
    # 绘制 2D KDE 等高线图，使用原始 density_grid 数据
    contour = ax.contourf(phase_grid, psi_grid, density_grid, levels=levels, cmap="viridis")
    # 绘制最密集点
    ax.scatter(best_phase, best_psi, color="red", marker="*", s=150, label="Max Probability", edgecolor='black', linewidth=1.5)
    # 设置轴标签和标题，增强可读性
    ax.set_xlabel(r"$\varphi$ (deg)", fontsize=14)
    ax.set_ylabel(r"$\psi$ (deg)", fontsize=14)
    # 设置图例
    ax.legend(fontsize=10)
    # 创建归一化对象，将颜色条范围设置为 [0, 1]
    norm = Normalize(vmin=0, vmax=1)
    # 使用 ScalarMappable 创建颜色条，并将颜色条归一化
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  # 需要设置一个空数组
    # 添加颜色条，确保颜色条被归一化到 [0, 1]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Probability Density", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    # 调整图形布局
    plt.tight_layout()
    # 保存图形
    plt.savefig("2D_KDE_phase_psi_density_normalized.png")
    plt.savefig("2D_KDE_phase_psi_density_normalized.pdf")
    # 归一化密度到 [0,1]
    density_min, density_max = np.min(density), np.max(density)
    density_norm = (density - density_min) / (density_max - density_min)
    density_grid_norm = density_norm.reshape(phase_grid.shape)
    # 设置颜色归一化
    norm = mcolors.Normalize(vmin=0, vmax=1)
    levels = np.linspace(0, 1, 5)  # 10 个等间隔等级
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    im = ax.imshow(density_grid_norm, cmap="viridis", origin="lower", aspect="auto", 
                extent=[min(phase_samples), max(phase_samples), min(psi_samples), max(psi_samples)], 
                interpolation='bilinear', norm=norm)
    ax.scatter(best_phase, best_psi, color="red", marker="x", s=20, label="best parameter")
    ax.text(best_phase + 0.02, best_psi + 0.02, f"({best_phase:.3f}, {best_psi:.3f})", color="red", fontsize=8)
    # 设置坐标轴标签
    ax.set_ylabel(r"$\psi$ (deg)",fontsize=12)
    ax.set_xlabel(r"$\varphi$ (deg)",fontsize=12)
    ax.legend(fontsize=10)
    # 添加颜色条，归一化到 [0,1] 并保留一位小数
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, ticks=levels)
    cbar.set_label("Normalized Density")
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in levels])  # 保留 1 位小数
    # 保存图像
    fig.savefig("2D_phsae_psi_density_plot.png", dpi=300, bbox_inches='tight')

    phase_quantiles = posterior_samples["phase_offset"].quantile(quantile)
    phase_lower = phase_quantiles[quantile[0]]
    phase_upper = phase_quantiles[quantile[2]]
    # 计算 psi 的分位点
    psi_quantiles = posterior_samples["psi_offset"].quantile(quantiles)
    psi_lower = psi_quantiles[quantile[0]]
    psi_upper = psi_quantiles[quantile[2]]
    # 输出结果
    print(f"Phase : {phase_lower},  {phase_upper}")
    print(f"Psi : {psi_lower}, : {psi_upper}")

    likelihood = bilby.likelihood.GaussianLikelihood(x=Angle_filtered,y=PA_filtered,func=rvm_model,sigma=PA_err_filtered)
    priors = bilby.prior.PriorDict()
    priors['alpha'] = bilby.prior.Uniform(0, 180, 'alpha')
    priors['beta'] = bilby.prior.Uniform(-90, 90, 'beta')
    priors['phase_offset'] = bilby.prior.DeltaFunction(best_phase) 
    priors['psi_offset'] = bilby.prior.DeltaFunction(best_psi)   
    for i in range(1):
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   sampler='dynesty',
                                     nlive=1500,
                                     sample="rwalk",
                                     dlogz =0.01,
                                     walks=200,
                                     bootstrap=0,
                                     nthreads=20,
                                     burnin=500,
                                     write_progress=True,
                                     outdir='./',
                                    label=psr + f"_{1}_RVM_{i}_fit",
                                    verbose=True)   


    
    truths = [result.posterior['alpha'].mean(),result.posterior['beta'].mean()]
    fig = result.plot_corner(
        parameters=['alpha', 'beta'],  # 指定采样参数
        labels=[r'$\alpha$', r'$\beta$'],  # 自定义轴标签
        show_titles=True,  # 显示每个参数的标题
        title_fmt='.2f',  # 标题的格式化方式
        quantiles= quantile,  # 显示分位点
        color='grey',  # 线和点的颜色
        smooth=0,  # 平滑度
        truths=truths,  # 设置真实值为拟合参数均值
        truth_color='red', )
    # 添加自定义分位点线条
    quantiles = [0.00135, 0.025, 0.16, 0.5, 0.84, 0.975, 0.99865]
    colors = ["red", "orange", "blue", "green", "blue", "orange", "red"]
    sigma_labels = ["-3σ", "-2σ", "-1σ", "Median", "+1σ", "+2σ", "+3σ"]
    # 获取角点图的所有子图
    axes = np.array(fig.axes).reshape(len(truths), len(truths))
    # 遍历对角线上的直方图，绘制参考线
    for i, ax in enumerate(axes.diagonal()):
        data1 = result.posterior[result.posterior.columns[i]].values  # 当前参数的采样值
        for q, color, label in zip(quantiles, colors, sigma_labels):
            quantile_value = np.percentile(data1, q * 100)  # 计算百分位数
            ax.axvline(quantile_value, color=color, linestyle="--", linewidth=0.8)
            ax.text(quantile_value, ax.get_ylim()[1] * 0.8,  label, color=color, fontsize=7, rotation=90, verticalalignment="center", horizontalalignment="right" )
    # 显示图像
    fig.suptitle("Corner Plot", fontsize=16, y=1.05)
    fig.savefig(psr + "_alpha_beta_corner_plot.png")

    if os.path.exists("alpha_beta.csv"):
        os.remove("alpha_beta.csv")
        print("成功删除:alpha_beta.csv")
    # 生成 CSV 文件路径
    for i in range(1):
      #  df = pd.DataFrame(pickle.load(open(f"{psr}_{i}_RVM_fit_dynesty.pickle", "rb")).samples, columns=['alpha', 'beta'])   
        df = pd.DataFrame(json.load(open(f"{psr}_{1}_RVM_{i}_fit_result.json"))["posterior"]["content"])[['alpha', 'beta']]
        # 追加数据
        if os.path.exists("alpha_beta.csv"):
            df.to_csv("alpha_beta.csv", mode='a', header=False, index=False)  # 追加模式，不写表头
        else:
            df.to_csv("alpha_beta.csv", index=False)   # 写入模式，写表头
# 读取数据
    posterior_samples = pd.read_csv("alpha_beta.csv")
    alpha_samples = np.array(posterior_samples['alpha'])
    beta_samples = np.array(posterior_samples['beta'])
    # 计算二维 KDE
    samples = np.vstack([alpha_samples, beta_samples])
    kde = gaussian_kde(samples)
    kde.set_bandwidth(bw_method='scott')  # 自动调整带宽
    # 生成高分辨率网格
    alpha_range = np.linspace(min(alpha_samples), max(alpha_samples), 200)
    beta_range = np.linspace(min(beta_samples), max(beta_samples), 200)
    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
    grid_points = np.vstack([alpha_grid.ravel(), beta_grid.ravel()])
    # 计算二维密度
    density = kde(grid_points)
    density_grid = density.reshape(alpha_grid.shape)

    # 找到密度最高点
    max_idx = np.unravel_index(np.argmax(density_grid), density_grid.shape)
    best_alpha = alpha_range[max_idx[1]]
    best_beta = beta_range[max_idx[0]]
    print(f"最优 alpha = {best_alpha}, 最优 beta = {best_beta}")
    print(f"最优 alpha = {best_alpha}, 最优 beta = {best_beta}")
    # 设置 Seaborn 风格，并去掉网格
        # 归一化密度到 [0,1]
    density_min, density_max = np.min(density), np.max(density)
    density_norm = (density - density_min) / (density_max - density_min)
    density_grid_norm = density_norm.reshape(alpha_grid.shape)
    # 设置颜色归一化
    norm = mcolors.Normalize(vmin=0, vmax=1)
    levels = np.linspace(0, 1, 5)  # 10 个等间隔等级
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    im = ax.imshow(density_grid_norm, cmap="viridis", origin="lower", aspect="auto", 
                extent=[min(alpha_samples), max(alpha_samples), min(beta_samples), max(beta_samples)], 
                interpolation='bilinear', norm=norm)
    # 标出密度最大点
    ax.scatter(best_alpha, best_beta, color="red", marker="x", s=20, label="best parameter")
    ax.text(best_alpha + 0.02, best_beta + 0.02, f"({best_alpha:.3f}, {best_beta:.3f})", color="red", fontsize=8)
    # 设置坐标轴标签]

    print("================================================")



    N = len(Angle_filtered) 
    a_rad = np.radians(180 - best_alpha)
    b_rad = np.radians(-best_beta)
    best_K = np.sin(a_rad) / np.sin(b_rad)
    psi_fit_chi = rvm_model(Angle_filtered, alpha=best_alpha, beta=best_beta, phase_offset=best_phase, psi_offset=best_psi)
   
    squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2) 
    best_chi_red = squared / (N - 4) 
    print(best_chi_red)
    ax.text(0.02, 0.98, f"$\\chi_{{red}}$ = {best_chi_red:.2f}", 
            color="white", fontsize=8, fontweight="bold", ha="left", va="top", 
            transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.set_xlabel(r"$\alpha$ (deg)",fontsize=10)
    ax.set_ylabel(r"$\beta$ (deg)",fontsize=10)
    ax.legend(fontsize=10)
    # 添加颜色条，归一化到 [0,1] 并保留一位小数
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, ticks=levels)
    cbar.set_label("Probability Density", fontsize=14)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in levels])  # 保留 1 位小数
    cbar.ax.tick_params(labelsize=14)
    # 保存图像
    fig.savefig("2D_alpha_beta_density_plot.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 7),dpi=600)
    # 假设 alpha_grid, beta_grid, density_grid 已经被定义并且是合适的二维数组
    # 定义密度图的等高线级别
    levels = np.linspace(np.min(density_grid), np.max(density_grid), 20)  # 更高的等高线密度
    # 绘制 2D KDE 等高线图，使用原始 density_grid 数据
    contour = ax.contourf(alpha_grid, beta_grid, density_grid, levels=levels, cmap="viridis")
    # 绘制最密集点
   # ax.scatter(best_alpha, best_beta, color="red", marker="*", s=150, label=f"$\chi^2_{{\nu}}$ = {best_chi_red:.2f}", edgecolor='black', linewidth=1.5)
    ax.scatter(best_alpha, best_beta, color="red", marker="*", s=150, label="Max Probability", edgecolor='black', linewidth=1.5)
   # ax.scatter(best_alpha, best_beta, color="red", marker="*", s=150, label=f"$\\chi_{{red}}$ = {best_chi_red:.2f}", edgecolor='black', linewidth=1.5)
    # 设置轴标签和标题，增强可读性
    ax.set_xlabel(r"$\alpha$ (deg)", fontsize=18)
    ax.set_ylabel(r"$\beta$  (deg)", fontsize=18)
    # 设置图例
   # ax.set_ylim(1.7,2.4)
    ax.legend(fontsize=10)
    # 创建归一化对象，将颜色条范围设置为 [0, 1]
    norm = Normalize(vmin=0, vmax=1)
    # 使用 ScalarMappable 创建颜色条，并将颜色条归一化
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  # 需要设置一个空数组
    # 添加颜色条，确保颜色条被归一化到 [0, 1]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Probability Density', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    # 调整图形布局
    plt.tight_layout()
    plt.savefig("2D_KDE_alpha_beta_density_normalized.png")
    plt.savefig("2D_KDE_alpha_beta_density_normalized.pdf")



    # 过的alpha 和 beta 最优解 以及误差
    quantile = [0.025, 0.5, 0.975] 
    #quantile = [0.025, 0.5, 0.975]
    # 计算 alpha 的分位点
    alpha_quantiles = posterior_samples["alpha"].quantile(quantile)
    alpha_lower = alpha_quantiles[quantile[0]]
    alpha_median = alpha_quantiles[quantile[1]]
    alpha_upper = alpha_quantiles[quantile[2]]
    # 计算 beta 的分位点
    beta_quantiles = posterior_samples["beta"].quantile(quantile)
    beta_lower = beta_quantiles[quantile[0]]
    beta_median = beta_quantiles[quantile[1]]
    beta_upper = beta_quantiles[quantile[2]]
    # 输出结果
    print(f"Alpha 16%: {alpha_lower}, Median: {alpha_median}, 84%: {alpha_upper}")
    print(f"Beta 16%: {beta_lower}, Median: {beta_median}, 84%: {beta_upper}")

    # 设置 Seaborn 风格
    # 选择四个参数及其最优值
    params = ["alpha", "beta"]
    best_values = [best_alpha, best_beta]  # 请替换为你的最优值
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # 遍历参数并绘制直方图1
    for ax, param, best_val in zip(axes.flatten(), params, best_values):
        sns.histplot(posterior_samples[param], bins=200, kde=True, ax=ax)
        ax.axvline(best_val, color='r', linestyle='--', linewidth=2, label=f"Best {param}")  # 标注最优值
        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.legend()
    # 调整子图间距
    fig.suptitle(f"Parameter Distributions of {psr}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Distribution of parameter")


    # 计算 phase 的分位点
        # 上面把 位置定出来了 现在开始计算误差
    def compute_K_and_chi(posterior_samples, best_alpha=None, best_beta=None, best_phase=None, best_psi=None):
        K_slopes = []
        Chi = []
        N = len(Angle_filtered)  # 数据点数量

        for i in range(len(posterior_samples)):  # 只绘制符合条件的样本
            alpha = posterior_samples.iloc[i]['alpha'] if best_alpha is None else best_alpha
            beta = posterior_samples.iloc[i]['beta'] if best_beta is None else best_beta
            phase_offset = best_phase if best_phase is not None else posterior_samples.iloc[i]['phase_offset']
            psi_offset = best_psi if best_psi is not None else posterior_samples.iloc[i]['psi_offset']

            a_rad = np.radians(180 - alpha)
            b_rad = np.radians(-beta)
            K = np.sin(a_rad) / np.sin(b_rad)
            K_slopes.append(K)

            # 计算每组样本的拟合曲线
            psi_fit_chi = rvm_model(Angle_filtered, alpha=alpha, beta=beta, phase_offset=phase_offset, psi_offset=psi_offset)
            squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2)
            chi_red = squared / (N - 4)
            Chi.append(chi_red)

        return K_slopes, Chi

    # 处理 alpha_beta.csv
    alpha_beta = pd.read_csv("alpha_beta.csv")
    K_slopes, Chi = compute_K_and_chi(alpha_beta, best_alpha=None, best_beta=None, best_phase=best_phase, best_psi=best_psi)
    alpha_beta["phase_offset"] = best_phase
    alpha_beta["psi_offset"] = best_psi
    alpha_beta["chi_red"] = Chi
    alpha_beta["K"] = K_slopes
    alpha_beta.to_csv("alpha_beta.csv", index=False)
    # 合并后只保留 alpha_beta.csv
    combined_df = alpha_beta
    combined_df.to_csv("combined_data.csv", index=False)
    posterior_sample = pd.read_csv("combined_data.csv")
    posterior_sample = posterior_sample[(posterior_sample["alpha"] < alpha_upper) & (posterior_sample["alpha"] > alpha_lower) & \
                                        (posterior_sample["beta"] < beta_upper) & (posterior_sample["beta"] > beta_lower)] #
                                            
    print("===========================================================================================")
    K_max_value = max(posterior_sample["K"])
    K_min_value = min(posterior_sample["K"])
    print(K_max_value, K_min_value,best_K)
    K_lower_error = abs(best_K - K_min_value)
    K_upper_error = abs(K_max_value - best_K)
    # Calculate Chi_max_value and Chi_min_value
    Chi_max_value = max(posterior_sample["chi_red"])
    Chi_min_value = min(posterior_sample["chi_red"])
    # Calculate the Chi errors
    Chi_lower_error = abs(best_chi_red - Chi_min_value)
    Chi_upper_error = abs(Chi_max_value - best_chi_red)

    # Output the results
    print("Best K:", best_K)
    print("best chi",best_chi_red)
    print("K_max_value:", K_max_value, "K_min_value:", K_min_value)
    print("K_lower_error:", K_lower_error, "K_upper_error:", K_upper_error)
    print("Chi_max_value:", Chi_max_value, "Chi_min_value:", Chi_min_value)
    print("Chi_lower_error:", Chi_lower_error, "Chi_upper_error:", Chi_upper_error)

    # ========== 生成相位序列 ==========
    phi_fit = np.linspace(all_min, all_max, 1000)
    # ========== 建立图像 ==========
    fig, ax1 = plt.subplots(figsize=(5, 3), dpi=100)
    # ========== 后验样本曲线 ==========
    print(len(posterior_sample))
    for i in range(len(posterior_sample)):  # 只绘制符合条件的样本
        alpha = posterior_sample.iloc[i]['alpha']
        beta = posterior_sample.iloc[i]['beta']
        phase_offset = posterior_sample.iloc[i]['phase_offset']
        psi_offset = posterior_sample.iloc[i]['psi_offset']
        psi_fit = rvm_model(phi_fit, alpha=alpha, beta=beta, phase_offset=phase_offset, psi_offset=psi_offset)
        plt.plot(phi_fit, psi_fit, color='grey', linewidth=4, alpha=0.4)  # 透明度调整
    # ========== Best-fit 曲线 ==========
    psi_fit = rvm_model(phi_fit, alpha=best_alpha, beta=best_beta,
                       phase_offset=best_phase, psi_offset=best_psi)
    ax1.plot(phi_fit, psi_fit,
            label=f"$\\chi^2_{{\\nu}} = {best_chi_red:.1f}$",
            color='orange', linewidth=2)
    # ========== 对称曲线 ==========
    ax1.plot(phi_fit, psi_fit + 90, linestyle='--', color='orange', alpha=0.7, linewidth=2, rasterized=True)
    ax1.plot(phi_fit, psi_fit - 90, linestyle='--', color='orange', alpha=0.7, linewidth=2, rasterized=True)
    ax1.plot(phi_fit, psi_fit + 180, linestyle='--', color='orange', alpha=0.7, linewidth=2, rasterized=True)
    ax1.plot(phi_fit, psi_fit - 180, linestyle='--', color='orange', alpha=0.7, linewidth=2, rasterized=True)
    # ========== 偏振角误差条函数 ==========
    def plot_pa_with_error(ax, angle, pa, pa_error, color, pa_replace=None, alpha=1.0):
        if pa_replace is not None:
            replace_condition, replace_value = pa_replace
            pa = np.where(pa == replace_condition, replace_value, pa)
        pa = np.where(pa == 0, np.nan, pa)
        ax.errorbar(
            angle, pa, xerr=0.1, yerr=pa_error, fmt='o',
            alpha=alpha, color=color, markersize=5,
            capsize=3, elinewidth=2, rasterized=True
        )

    # ========== 绘制偏振角点 ==========
    plot_pa_with_error(ax1, data['Angle'], data['PA'], data['PA_error'],
                    color='black', alpha=0.2)   # 半透明
    plot_pa_with_error(ax1, data['Angle'], data['PA_filtered'], data['PA_error_filtered'],
                    color='red', alpha=1.0)     # 不透明

    # 重叠点
    overlap_mask = ~np.isnan(data['PA']) & ~np.isnan(data['PA_filtered']) & (data['PA'] == data['PA_filtered'])
    angle_overlap = data['Angle'][overlap_mask]
    pa_overlap = data['PA'][overlap_mask]
    if len(pa_overlap) != 0:
        pa_overlap[pa_overlap == 0] = np.nan
        pa_error_overlap = data['PA_error'][overlap_mask]
        ax1.errorbar(
            angle_overlap, pa_overlap, xerr=0.1, yerr=pa_error_overlap, fmt='o',
            color='black', markersize=5, capsize=3, elinewidth=2,
            rasterized=True   # ✅
        )
    # ========== 标记最佳相位 ==========
    ax1.axvline(best_phase, color='m', linestyle='--', linewidth=0.8)
    # ========== 图形设置 =========
    ax1.xaxis.set_visible(True)             # 如果你要显示下轴
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))  # 下轴 7 个刻度
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))  # 左轴 5 个刻度
    ax1.set_title(f"PSR {psr}", fontsize=15, pad=8)
    ax1.set_xlabel('Longitude(°)', fontsize=15)
    ax1.set_ylabel('PA(°)', fontsize=15)
    ax1.legend(fontsize=12, loc='upper right', framealpha=1.0)
    ax1.set_xlim(all_min, all_max)
    ax1.set_ylim(min(psi_fit) - 50, max(psi_fit) + 50)
    ax1.grid(False)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    # 设置图像光栅化阈值
    # 保存 PNG
    # ⭐ 强制白背景
    plt.gcf().patch.set_facecolor('white')
    plt.gca().set_facecolor('white')

    plt.savefig(f"{psr}_MeerKAT_fiting.png",
                dpi=100,
                bbox_inches="tight",
                facecolor='white')   # ⭐ 关键
    plt.close()

    # PNG -> PDF
    png_file = f"{psr}_MeerKAT_fiting.png"
    pdf_file = f"{psr}_MeerKAT_fiting.pdf"

    # 如果 PDF 已存在，先删除
    if os.path.exists(pdf_file):
        os.remove(pdf_file)

    # ⭐ 转换时去掉透明通道（非常关键）
    img = Image.open(png_file).convert("RGB")
    img.save(pdf_file, "PDF", resolution=100.0)

    # 读取数据
    data1 = pd.read_csv("combined_data.csv")
    alpha = data1["alpha"].values
    beta = data1["beta"].values
    chi_red = data1["chi_red"].values
    # 找到最小的卡方值
    # 过滤数据（仅保留 chi_red 在 best_chi_red+10 以内的点）
    mask = chi_red < best_chi_red + 10
    alpha_filtered = alpha[mask]
    beta_filtered = beta[mask]
    chi_red_filtered = chi_red[mask]
    print("有效数据点：", len(alpha_filtered))
    # 计算二维直方图
    bins = 300
    density, xedges, yedges = np.histogram2d(alpha_filtered, beta_filtered, bins=bins, density=True)
    # 生成像素左下角坐标
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    # 计算符合条件的点并累积到 density
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            # 计算卡方值（这里需要 rvm_model 和相关变量）
            psi_fit_chi = rvm_model(Angle_filtered, alpha=x, beta=y, 
                                    phase_offset=phase_offset, psi_offset=psi_offset)
            squared = np.sum(((PA_filtered - psi_fit_chi) / PA_err_filtered) ** 2)
            chi_red_value = squared / (N - 4)
            if Chi_min_value< chi_red_value < Chi_max_value :
                density[i, j] += 1  # 直接累加到 density
    # 处理密度数据，避免 log 归一化问题
    density[density == 0] = np.nan
    vmin, vmax = np.nanpercentile(density, [1, 100])
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    # 绘制图像
    fig, ax = plt.subplots(figsize=(6, 5), dpi=400)
    # 绘制密度图
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(density.T, cmap="viridis", origin="lower", 
                aspect="auto", extent=extent, norm=norm)
    # 标注最佳 (alpha, beta)
    ax.scatter(best_alpha, best_beta, color="red", marker="x", s=10)
    ax.text(best_alpha + 3, best_beta, f"({best_alpha:.3f}, {best_beta:.3f})", 
            color="red", fontsize=7, ha="left", va="bottom")
    # 显示最小卡方值
    ax.text(0.02, 0.98, f"$\\chi^2_{{\\nu}}$ = {best_chi_red:.2f}$^{{+{Chi_upper_error:.2f}}}_{{-{Chi_lower_error:.2f}}}$", 
        color="black", fontsize=8, fontweight="bold", ha="left", va="top", 
        transform=ax.transAxes, bbox=dict(facecolor='grey', alpha=0.35, edgecolor='none')) 
    # 设置坐标轴标签
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    ax.grid(False)
    # 保存图像
    fig.savefig("2D_1_plot_combined.png", dpi=300, bbox_inches='tight', transparent=True)
    print("图像已保存为 2D_1_plot_combined.png")       


    def plot_data(data, x_min, x_max, phi, model,psr, x_max_slope=None, y_max_slope=None):
        # 设置默认的刻度线方向为向内
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # 创建图形和网格布局
        fig = plt.figure(figsize=(6, 6), dpi=400)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.2,1], hspace=0)   # Set hspace to 0 to remove space between subplots

        ax1 = fig.add_subplot(gs[0])  # Top subplot for Polarization Angle
        ax2 = fig.add_subplot(gs[1])  # Bottom subplot for Flux
         # Bottom subplot for Flux
        # 强制设置刻度方向为向内
        ax1.tick_params(axis='x', direction='in')
        ax1.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='y', direction='in')

        ax1.plot(phi, model + 90,  linestyle='--', color='orange', linewidth=2, alpha=0.8)
        ax1.plot(phi, model - 90,  linestyle='--', color='orange', linewidth=2, alpha=0.8)
        ax1.plot(phi, model + 180, linestyle='--', color='orange', linewidth=2, alpha=0.8)
        ax1.plot(phi, model - 180, linestyle='--', color='orange', linewidth=2, alpha=0.8)

        def plot_pa_with_error(ax, angle, pa, pa_error, color, pa_replace=None, alpha=1.0):
            if pa_replace is not None:
                replace_condition, replace_value = pa_replace
                pa = np.where(pa == replace_condition, replace_value, pa)
            pa = np.where(pa == 0, np.nan, pa)
            ax.errorbar(
                angle, pa, xerr=0.1, yerr=pa_error, fmt='o',
                alpha=alpha, color=color, markersize=5,
                capsize=3, elinewidth=2, rasterized=True
            )

        plot_pa_with_error(ax1, data['Angle'], data['PA'], data['PA_error'], color='black', alpha=0.3)
        plot_pa_with_error(ax1, data['Angle'], data['PA_filtered'], data['PA_error_filtered'], color='red', alpha=1.0)

        overlap_mask = ~np.isnan(data['PA']) & ~np.isnan(data['PA_filtered']) & (data['PA'] == data['PA_filtered'])
        angle_overlap = data['Angle'][overlap_mask]
        pa_overlap = data['PA'][overlap_mask]
        if len(pa_overlap) != 0:
            pa_overlap[pa_overlap == 0] = np.nan
            pa_error_overlap = data['PA_error'][overlap_mask]
            ax1.errorbar(angle_overlap, pa_overlap, xerr=0.1, yerr=pa_error_overlap, fmt='o', color='black', markersize=5, capsize=3, elinewidth=2)
       
    
        ax1.plot(phi, model, linestyle='-', color='orange', linewidth=3, label=f"$\\chi^2_{{\\nu}}$ = {best_chi_red:.2f}", alpha=1)
        ax1.set_title(f"PSR {psr}", fontsize=20, pad=12, fontweight='bold')
        ax1.set_ylabel('PA (°)', fontsize=16)
        ax1.set_ylim(min(model)-50, max(model)+50) 
        ax1.set_xlim(x_min, x_max)
        ax1.axvline(best_phase, color='m', linestyle='--', linewidth=0.8)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(False)
        ax1.legend(loc='upper right', fontsize=14, framealpha=0.8)
       # ax1.legend(loc='best', fontsize=8, framealpha=1)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.tick_params(axis='both', which='minor', labelsize=15)

        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))    
        mask = (data['Angle'] >= x_min) & (data['Angle'] <= x_max)
        data_selected = data[mask]
        model = rvm_model(data_selected['Angle'],alpha=best_alpha,beta=best_beta,phase_offset=best_phase,psi_offset=best_psi)
        residuals = model - data_selected['PA_filtered']
    
        # 筛选有效（非 NaN）PA_filtered 对应的 Angle 值
        valid_mask = ~np.isnan(data['PA_filtered'])
        wask = (data['Angle'] >= x_min) & (data['Angle'] <= x_max)
        valid_angles = data['Angle'][valid_mask&wask]

        # 最小值、最大值及其绝对差值（非负）
        min_angle = valid_angles.min()
        max_angle = valid_angles.max()
        
        PW_range1 = max_angle - best_phase  # 保证非负
        print(PW_range1)
        PW_range2 = best_phase - min_angle # 保证非负
        print(PW_range2)
        PW_range = max(PW_range1, PW_range2)
        print(PW_range)
  
        print("最小 Angle:", min_angle)
        print("最大 Angle:", max_angle)
        print("最大和最小 Angle 之间的距离（非负）:", PW_range)
        angle_mask = (data['Angle'] >= 100) & (data['Angle'] <= 180)
        std_I = np.std(data["I_normalized"][angle_mask]) 
        # 构建一个 mask：总强度大于 2σ
        intensity_mask = data['I_normalized'] > 3 * std_I
        linear_pol_mask = data['Linear_Polarization'] >= 2* std_I
        non_negative_I = data['I_normalized'] >= 0

        V_pol_mask = abs(data['V_normalized']) >= 2* std_I
        final = intensity_mask & V_pol_mask  # 如果你希望也限定角度范围

        print("std_I =", std_I)
        print("2σ =", 2 * std_I)
        ax2.plot(data['Angle'], data['I_normalized'], color='black', linewidth=2, label='I')
        ax2.plot(data['Angle'], data['Linear_Polarization'], color='blue', linewidth=2, label='L')
        ax2.plot(data['Angle'], data['V_normalized'], color='red', linewidth=2, label='V')

        Imax = data['I_normalized'].max()
        threshold = Imax * 0.1
        # 先选出角度在 [-25, -15] 范围内的行
        anglemask = (data['Angle'] >= x_min) & (data['Angle'] <= x_max)
        # 再在这个子集里筛出 I_normalized >= threshold 的行
        mask = anglemask & (data['I_normalized'] >= threshold)& (data['I_normalized'] >= 3*std_I)
        subset = data[mask]
        # 输出这些点在 Angle 上的最大和最小值（或你需要的其他列）
        angle_min = subset['Angle'].min()
        angle_max = subset['Angle'].max()
        print(angle_min,angle_max)

        pulse = data["I_normalized"][(data['Angle'] >= x_min) & (data['Angle'] <= x_max)]
        snr_peak = pulse.max() / std_I
            # ---------------------------
        main_info = get_component_width(data, x_min,x_max, std_I)
        w10 = main_info["width"]
        ax2.axvline(main_info["midpoint"], color='m', linestyle='--', linewidth=0.8)
        ax2.errorbar(x=main_info["angle_min"], y=main_info["Imin_val"],
                    yerr=0.1, fmt='o', color='m', markersize=0.01,
                    elinewidth=2, capsize=0, linestyle='--', linewidth=0.8)
        ax2.errorbar(x=main_info["angle_max"], y=main_info["Imax_val"],
                    yerr=0.1, fmt='o', color='m', markersize=0.01,
                    elinewidth=2, capsize=0, linestyle='--', linewidth=0.8)
        # 全宽
        # 以 RVM 子午线为中心的最大半宽 (Max Half-width)
        # 从子午线到左右边界的最大距离
        W10_MH = max(abs(angle_max - best_phase),
                    abs(best_phase - angle_min))
    
        print("W10 (全宽) =", w10)
        print("左边宽度",abs(best_phase - angle_min))
        print("右边宽度",abs(angle_max - best_phase))
        print("W10_MH (以RVM子午线为中心的最大半宽) =", W10_MH) 

        w10_center = main_info["midpoint"]
        q = psrqpy.QueryATNF(params=["NAME","PSRJ","PSRB", "P0", "P1", "TYPE", "BINARY", "ASSOC", "P1_I"])  
        t = q.table
        psr_data = t[t['NAME'] == psr]
        if len(psr_data) == 0:
            psr_data = t[t['PSRJ'] == psr]
            if len(psr_data) == 0:
                psr_data = t[t['PSRB'] == psr]
        P= psr_data["P0"].data[0]*1000
        print("脉冲星周期：",P)
        tb  = P/(len(data["Angle"])-1)
        print("bin数",len(data["Angle"])-1)
        w10_error = tb * math.sqrt(1 + (std_I / 0.1) ** 2)
        print("w10误差",w10_error)

        # 假设你想画的矩形中心点为 (x, y)，宽度为 w，高度为 h
        x = x_min+2     # 例如角度
        y = 0.4       # 例如线性偏振值
        w = 360/(len(data["Angle"])-1) # 矩形宽度（x方向）
        h = 3*std_I      # 矩形高度（y方向）
        # Rectangle 需要左下角的坐标，因此需要从中心坐标计算
        lower_left_x = x - w/2
        lower_left_y = y - h/2
        rect = patches.Rectangle((lower_left_x, lower_left_y), w, h,linewidth=0.8, edgecolor='black', facecolor='none', linestyle='-')
        ax2.add_patch(rect)

        ax2.set_ylabel('Intensity', fontsize=16,labelpad=12)
        ax2.set_xlabel('Longitude(°)', fontsize=16)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(-0.18, 1.1)
        ax2.grid(False)
        ax2.xaxis.set_visible(True)             # 如果你要显示下轴
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))  # 下轴 7 个刻度
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))  # 左轴 5 个刻度
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='minor', labelsize=15)
        ax2.legend(loc='upper right', fontsize=8, framealpha=1)
 
        mask = data["I_normalized"] >= 3 * std_I
        Nd = mask.sum()
        print("I大于3sigma的点:",Nd)
        I_sum, L_sum, V_sum, absV_sum = data["I_normalized"][mask].sum(), data["Linear_Polarization"][mask].sum(), data["V_normalized"][mask].sum(), np.abs(data["V_normalized"][mask]).sum()
        L_I, V_I, absV_I = L_sum / I_sum, V_sum / I_sum, absV_sum / I_sum
        σI, σL, σV = std_I, np.std(data["Linear_Polarization"][angle_mask]), np.std(data["V_normalized"][angle_mask])
        σ_LI = np.sqrt(Nd * ((σL/I_sum)**2 + (L_sum*σI/I_sum**2)**2))
        σ_VI = np.sqrt(Nd * ((σV/I_sum)**2 + (V_sum*σI/I_sum**2)**2))
        σ_absVI = np.sqrt(Nd * ((σV/I_sum)**2 + (absV_sum*σI/I_sum**2)**2))
        print(f"L/I = {L_I:.4f} ± {σ_LI:.4f}")
        print(f"V/I = {V_I:.4f} ± {σ_VI:.4f}")
        print(f"|V|/I = {absV_I:.4f} ± {σ_absVI:.4f}")
        # 假设你已经绘图完毕
        png_file = psr + "_MeerKAT_RVMFIT.png"
        pdf_file = psr + "_MeerKAT_RVMFIT.pdf"
        # 保存高分辨率 PNG
        # ⭐ 强制白背景
        plt.gcf().patch.set_facecolor('white')
        plt.gca().set_facecolor('white')

        # 保存 PNG（关键：facecolor）
        plt.savefig(png_file, dpi=400, bbox_inches="tight", facecolor='white')
        plt.close()

        # 如果 PDF 已存在，先删除
        if os.path.exists(pdf_file):
            os.remove(pdf_file)

        # ⭐ 转 PDF 时再强制 RGB（防透明）
        img = Image.open(png_file).convert("RGB")
        img.save(pdf_file, "PDF", resolution=400.0)
        return w10, w10_error, w10_center,snr_peak,L_I,σ_LI,V_I,σ_VI,absV_I,σ_absVI,PW_range,W10_MH


    phi_fit = np.linspace(all_min, all_max, 1000) 
    model = rvm_model(phi_fit, alpha=best_alpha, beta=best_beta, phase_offset=best_phase, psi_offset=best_psi)
    w10, w10_error, w10_center,snr_peak,L_I,σ_LI,V_I,σ_VI,absV_I,σ_absVI,PW_range,W10_MH = plot_data(data=data, x_min=all_min, x_max=all_max, phi=phi_fit, model=model, psr=psr)



    best_alpha = best_alpha
    alpha_min = min(posterior_sample["alpha"])
    alpha_max = max(posterior_sample["alpha"])
    best_alpha_lower = abs(best_alpha - alpha_min)
    best_alpha_upper = abs(best_alpha - alpha_max)
    
    best_beta =  best_beta
    beta_min = min(posterior_sample["beta"])
    beta_max = max(posterior_sample["beta"]) 
    best_beta_lower =  abs(best_beta - beta_min)
    best_beta_upper =  abs(best_beta - beta_max)


   
    best_phase_offset = best_phase
    phase_offset_upper = abs(phase_upper - best_phase)
    phase_offset_lower = abs(phase_lower - best_phase)

    best_psi_offset = best_psi
    psi_offset_upper = abs(psi_upper - best_psi)
    psi_offset_lower = abs(psi_lower - best_psi)



 

    def compute_rho_with_uncertainties(alpha_deg, alpha_err_up, alpha_err_down,
                                        beta_deg, beta_err_up, beta_err_down,
                                        W10_deg):
        """
        计算束开角 rho 及其由 alpha 和 beta 引起的上下误差（忽略 W10 误差）。

        参数：
            alpha_deg       - alpha（度）
            alpha_err_up    - alpha 上误差（度）
            alpha_err_down  - alpha 下误差（度）
            beta_deg        - beta（度）
            beta_err_up     - beta 上误差（度）
            beta_err_down   - beta 下误差（度）
            W10_deg         - W10 脉冲宽度（度），固定值

        返回：
            rho, drho_up, drho_down - 束开角（度）及其上下误差
        """

        def calc_rho(alpha_d, beta_d, W10_single_d):
            """
            W10_single_d 是子午线到最宽轮廓边缘的距离（单边宽度）
            """
            alpha = math.radians(alpha_d)
            beta = math.radians(beta_d)
            W10 = math.radians(W10_single_d)

            sin_alpha = math.sin(alpha)
            sin_zeta = math.sin(alpha + beta)
            sin_W10_2 = math.sin(W10 / 4)   # ✅ 注意是 /2，因为输入是单边宽度
            sin_beta_2 = math.sin(beta / 2)

            term1 = (sin_W10_2 ** 2) * sin_alpha * sin_zeta
            term2 = sin_beta_2 ** 2
            sqrt_term = math.sqrt(term1 + term2)
            rho_rad =  2*math.asin(min(1.0, sqrt_term))
            return math.degrees(rho_rad)


        # 中心值
        rho_center = calc_rho(alpha_deg, beta_deg, W10_deg)

        # 扰动
        rho_alpha_up = calc_rho(alpha_deg + alpha_err_up, beta_deg, W10_deg)
        rho_alpha_down = calc_rho(alpha_deg - alpha_err_down, beta_deg, W10_deg)
        rho_beta_up = calc_rho(alpha_deg, beta_deg + beta_err_up, W10_deg)
        rho_beta_down = calc_rho(alpha_deg, beta_deg - beta_err_down, W10_deg)

        # 误差
        drho_alpha_up = abs(rho_alpha_up - rho_center)
        drho_alpha_down = abs(rho_center - rho_alpha_down)
        drho_beta_up = abs(rho_beta_up - rho_center)
        drho_beta_down = abs(rho_center - rho_beta_down)

        # 总合成误差
        drho_up = math.sqrt(drho_alpha_up**2 + drho_beta_up**2)
        drho_down = math.sqrt(drho_alpha_down**2 + drho_beta_down**2)

        return rho_center, drho_up, drho_down

    # 示例调用
    rho, drho_up, drho_down = compute_rho_with_uncertainties(
        alpha_deg=best_alpha, 
        alpha_err_up=best_alpha_upper, 
        alpha_err_down=best_alpha_lower,
        beta_deg=best_beta, 
        beta_err_up=best_beta_upper, 
        beta_err_down=best_beta_lower,
        W10_deg=w10  # 不使用误差
    )

    print(f"束开角 rho = {rho:.2f} +{drho_up:.2f}/-{drho_down:.2f} 度")


    all_alpha_error = best_alpha_upper + best_alpha_lower
    # 分类判断
    if 0 <= all_alpha_error < 40:
        category = "A"
    elif 40 <= all_alpha_error < 80:
        category = "B"
    elif 80 <= all_alpha_error < 120:
        category = "C"
    elif 120 <= all_alpha_error <= 180:
        category = "D"
    else:
        category = None  # 超出范围

    print("all_alpha_error =", all_alpha_error, "分类 =", category)

    # 初始化变量
    information = []  # 用于存储 information.csv 数据
    information_dict = {"PSR": psr}  # 用字典存储每个 psr 的参数估计值和误差
    q = psrqpy.QueryATNF(params=["NAME","PSRJ","PSRB", "P0", "P1", "TYPE", "BINARY", "ASSOC", "P1_I"])  
    t = q.table
    # 查找目标脉冲星的数据
    pulsarname = 'NAME'
    psr_data = t[t['NAME'] == psr]

    if len(psr_data) == 0:
        pulsarname = "PSRJ"
        psr_data = t[t['PSRJ'] == psr]
        if len(psr_data) == 0:
            pulsarname = "PSRB"
            psr_data = t[t['PSRB'] == psr]
        
    period0 = psr_data["P0"].data[0]
    period1 = psr_data["P1"].data[0]
    information_dict["P0"] = period0
    information_dict["P-dot"] = period1
    information_dict["Telescope"] = "MeerKAT"
    information_dict["Freq(MHz)"] = "856-1712"
    information_dict["best_alpha"] = best_alpha
    information_dict["alpha_Upper_Error"] = best_alpha_upper
    information_dict["alpha_Lower_Error"] = best_alpha_lower
    information_dict[f"best_beta"] = best_beta
    information_dict[f"beta_upper_Error"] = best_beta_upper
    information_dict[f"beta_lower_Error"] = best_beta_lower
    information_dict[f"best_phase_offset"] = best_phase
    information_dict[f"phase_offset_upper_error"] = phase_offset_upper
    information_dict[f"phase_offset_lower_error"] = phase_offset_lower
    information_dict[f"best_psi_offset"] = best_psi
    information_dict[f"psi_offset_upper_error"] = psi_offset_upper
    information_dict[f"psi_offset_lower_error"] = psi_offset_lower
    information_dict[f"best_chi_red"] = best_chi_red
    information_dict["Chi_upper_error"] = Chi_upper_error
    information_dict["Chi_lower_error"] =Chi_lower_error
    information_dict["best_K"] = best_K
    information_dict["K_upper_error"] = K_upper_error
    information_dict["K_lower_error"] = K_lower_error
    information_dict["w10"] = w10
    information_dict["w10_error"] = w10_error
    information_dict["w10_center"] = w10_center
    information_dict["SN_peak"] = snr_peak
    information_dict["L_I"] = L_I
    information_dict["σ_LI"] = σ_LI
    information_dict["V_I"] = V_I
    information_dict["σ_VI"] = σ_VI
    information_dict["absV_I"] = absV_I
    information_dict["σ_absVI"] = σ_absVI
    information_dict["rho"] = rho
    information_dict["drho_up"] = drho_up
    information_dict["drho_down"] = drho_down
    information_dict["PW"] = PW_range
    information_dict["W10_MH"] = W10_MH
    information_dict["all_alpha_range"] =all_alpha_error
    information_dict["category"] =category
    information.append(information_dict)
    print(information)
    
    # 动态生成表头，包含所有参数和 chi_squared_red
    fieldnames = [
        "PSR", "P0", "P-dot",
        "Telescope", "Freq(MHz)",
        "best_alpha", "alpha_Upper_Error", "alpha_Lower_Error",
        "best_beta", "beta_upper_Error", "beta_lower_Error",
        "best_phase_offset", "phase_offset_upper_error", "phase_offset_lower_error",
        "best_psi_offset", "psi_offset_upper_error", "psi_offset_lower_error",
        "best_chi_red", "Chi_upper_error", "Chi_lower_error",
        "best_K", "K_upper_error", "K_lower_error",
        "w10", "w10_error", "w10_center","SN_peak",
        "L_I", "σ_LI", "V_I", "σ_VI",
        "absV_I", "σ_absVI",
        "rho", "drho_up", "drho_down",
        "PW", "W10_MH","all_alpha_range",
        "category"
    ]

    information_filename = "../information.csv"
    # 读取现有的 information.csv 文件内容
    if os.path.exists(information_filename):
        with open(information_filename, mode='r', newline='') as info_file:
            reader = csv.DictReader(info_file)
            existing_data = [row for row in reader]
    else:
        existing_data = []
    # 查找是否已经存在对应的 psr
    psr_exists = False
    for row in existing_data:
        if row["PSR"] == information_dict["PSR"]:
            # 如果存在，更新现有行
            row.update(information_dict)
            psr_exists = True
            break
    # 如果 psr 不存在，添加新的一行
    if not psr_exists:
        existing_data.append(information_dict)
    # 将更新后的数据写回 information.csv
    with open(information_filename, mode='w', newline='') as info_file:
        writer = csv.DictWriter(info_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
    print("===========================================================================================")    
    print(f"Information saved to {information_filename}")







