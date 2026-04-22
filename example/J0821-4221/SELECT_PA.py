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
import glob
import warnings
import math
import psrqpy
import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
import bilby
from scipy.optimize import brentq  # 更快地查找积分目标
import matplotlib.patches as patches
from scipy.integrate import quad
from scipy.special import erf
import matplotlib.patches as patches
# 设置 Pandas 配置
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)   # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
# 设置 NumPy 配置
np.set_printoptions(suppress=True)
np.random.seed(666)
# 设置 Matplotlib 配置
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 忽略警告
warnings.filterwarnings("ignore")
current_file_path = os.path.realpath(__file__)
# 获取当前文件所在文件夹的路径
folder_path = os.path.dirname(current_file_path)
psr = os.path.basename(folder_path)

def PA_err(noise_level, L):
    """
    计算偏振角误差。
    :param noise_level: 噪声水平
    :param L: 线性偏振强度
    :return: 偏振角误差
    """
    SN = L / noise_level
    if SN > 10:
        return 28.65 / SN
    elif 2 < SN <= 10:
        def G(psi, P0):
            eta_0 = (P0 / np.sqrt(2)) * np.cos(2 * psi)
            term1 = 1 / np.sqrt(np.pi)
            term2 = term1 + eta_0 * np.exp(eta_0**2) * (1 + erf(eta_0))
            term3 = np.exp(-P0**2 / 2)
            return term1 * term2 * term3

        def integral_G(P0, psi_max):
            result, _ = quad(lambda psi: G(psi, P0), -psi_max, psi_max)
            return result
        
        try:
            psi_rad = brentq(lambda psi: integral_G(SN, psi) - 0.6826, 0, np.pi / 2)
            return np.rad2deg(psi_rad)
        except ValueError:
            return 0
    else:
        return 0


def plot_data(data,x_min, x_max,psr,x_max_slope=None,y_max_slope = None):
    # Create a figure and a gridspec
    fig = plt.figure(figsize=(6, 5), dpi=600)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1.5], hspace=0)  # Set hspace to 0 to remove space between subplots
    # Create subplots
    ax1 = fig.add_subplot(gs[0])  # Top subplot for Polarization Angle
    ax2 = fig.add_subplot(gs[1])  # Bottom subplot for Flux
    if x_max_slope is not None:
        print(y_max_slope)
        ax1.vlines(x=x_max_slope, ymin= y_max_slope-15 , ymax= y_max_slope+15 , color='blue', linestyle='-', label=f"Max Slope")

    def plot_pa_with_error2(ax, angle, pa, pa_error, label, color, pa_replace=None):
        if pa_replace is not None:
            replace_condition, replace_value = pa_replace
            pa = np.where(pa == replace_condition, replace_value, pa)
        pa = np.where(pa == 0, np.nan, pa)
        ax.errorbar(angle, pa, xerr=0.1, yerr=pa_error, fmt='.', markerfacecolor='none', color=color, markersize=4, capsize=0, elinewidth=1, label=label)
    
    #指定你的拟合数据
    pa = data['PA_filtered']
    # 调用函数
    plot_pa_with_error2(ax1, data['Angle'], data['PA'], data['PA_error'], 'Original Data', color='black')
    plot_pa_with_error2(ax1, data['Angle'], pa, data['PA_error_filtered'], 'Modified Data', color='purple', pa_replace=(data['PA'], np.nan))
   
    overlap_mask = ~np.isnan(data['PA']) & ~np.isnan(pa) & (data['PA'] == pa)
    angle_overlap = data['Angle'][overlap_mask]
    pa_overlap = data['PA'][overlap_mask]
    if len(pa_overlap) != 0:
        pa_overlap[pa_overlap==0]=np.nan
        pa_error_overlap = data['PA_error'][overlap_mask]
        ax1.errorbar(angle_overlap, pa_overlap, xerr=0.1, yerr=pa_error_overlap, fmt='.', color='#4B0082', markersize=4, label='Overlap')
    ax1.set_title(psr)
    ax1.set_ylabel('PA (degree)')
    ax1.set_ylim(-180, 180)
    ax1.set_xlim(x_min, x_max)
    ax1.grid()
    ax1.legend(loc='upper left')
    #y_ticks = [-145,-90, -45, 0, 45, 90, 145]
    #ax1.set_yticks(y_ticks)
    #ax1.set_yticklabels(y_ticks)
    ax1.xaxis.set_visible(False)
    ax1.legend(loc='upper left', fontsize=5)
    # Plot Normalized Flux I, Linear Polarization L, and Circular Polarization V
    # x = data['Angle'].values
    # y = data['I_normalized'].values
    # L = data['Linear_Polarization'].values
    # V = data['V_normalized'].values  # Ensure V is normalized
    # # Handling circular wraparound in the plot
    # ax2.plot(x, y, color='black', linewidth=1.5, label='Normalized Flux I')
    # ax2.plot(x, L, color='red', linewidth=1.5, label='Linear Polarization L')
    # ax2.plot(x, V, color='blue', linewidth=1, label='Circular Polarization V')




    ax2.plot(data['Angle'], data['I_normalized'], color='black', linewidth=2, label='I')
    ax2.plot(data['Angle'], data['Linear_Polarization'], color='blue', linewidth=2, label='L')
    ax2.plot(data['Angle'], data['V_normalized'], color='red', linewidth=2, label='V')



    
    Imax = data['I_normalized'].max()
    threshold = Imax * 0.1
    # 先选出角度在 [-25, -15] 范围内的行
    angle_mask = (data['Angle'] >= -25) & (data['Angle'] <= 25)
    # 再在这个子集里筛出 I_normalized >= threshold 的行
    mask = angle_mask & (data['I_normalized'] >= threshold)
    subset = data[mask]
    print(subset)
    # 输出这些点在 Angle 上的最大和最小值（或你需要的其他列）
    angle_min = subset['Angle'].min()
    angle_max = subset['Angle'].max()
    print(angle_min,angle_max)



    Imin = data.loc[data['Angle'] == angle_min, 'I_normalized'].values[0]
    Imax = data.loc[data['Angle'] == angle_max, 'I_normalized'].values[0]
    ax2.errorbar(x=angle_min,y=Imin,yerr=0.1,fmt='o', color='m',markersize=0.01,elinewidth=3,capsize=0)
    ax2.errorbar(x=angle_max,y=Imax,yerr=0.1,fmt='o', color='m',markersize=0.01,elinewidth=3,capsize=0)
    W10 = angle_max - angle_min
    print(W10)
    midpoint = (angle_min + angle_max) / 2
    print(midpoint)  # 输出 5.0
    ax2.axvline(midpoint, color='m', linestyle='--', linewidth=0.8)


    angle_mask = (data['Angle'] >= -180) & (data['Angle'] <= -100)
    std_I = np.std(data["I_normalized"][angle_mask]) 


    q = psrqpy.QueryATNF(params=["NAME","PSRJ","PSRB", "P0", "P1", "TYPE", "BINARY", "ASSOC", "P1_I"])  
    t = q.table
    psr_data = t[t['NAME'] == "J0821-4221"]
    P= psr_data["P0"].data[0]*1000
    print(P)
    tb  = P/len(data["Angle"])
    print(len(data["Angle"]))
    w_sigma = tb * math.sqrt(1 + (std_I / 0.1) ** 2)
    print(w_sigma)

    x = -5        # 例如角度
    y = 0.4       # 例如线性偏振值
    w = 0.5        # 矩形宽度（x方向）
    h = 0.2       # 矩形高度（y方向）

    # Rectangle 需要左下角的坐标，因此需要从中心坐标计算
    lower_left_x = x - w/2
    lower_left_y = y - h/2

    # 创建矩形对象
    rect = patches.Rectangle((lower_left_x, lower_left_y), w, h,
                            linewidth=1, edgecolor='green', facecolor='none', linestyle='--')

    # 添加到已有的 ax3 图上
    ax2.add_patch(rect)



    ax2.set_ylabel('Normalized Intensity')
    ax2.set_xlabel('Pulse Phase')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(-0.18, 1.1)
    ax2.legend(loc='upper left', fontsize=5)
    plt.savefig( psr + "_RVMFIT.png", dpi=700)


def process_data(file,x_min, x_max):
    data = pd.read_csv(file, header=None, delimiter=',', on_bad_lines='skip')
    # 检查数据列数是否符合预期
    if data.shape[1] != 9:
        raise ValueError("Expected 9 columns in the data.")
    # 指定列名
    data.columns = ['isub', 'ichan', 'Bin', 'I', 'Q', 'U', 'V', 'PA', 'PA_error']
    # 将列转换为数值类型，处理可能存在的非数值数据
    for col in ['isub', 'ichan', 'Bin', 'I', 'Q', 'U', 'V', 'PA', 'PA_error']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # 去掉任何包含NaN的行
    data = data.dropna()
    # 归一化 'Bin' 列
    data['Bin'] = data['Bin'] / data['Bin'].max()
    max_bin_index = data['I'].idxmax()
    max_bin = data['Bin'].iloc[max_bin_index]
    # 计算移位量并进行数据移位
    shift = 0.5 - max_bin
    shift_steps = int(shift * len(data))
    for col in ['I',"Q","U",'V']:
        data[col] = np.roll(data[col], shift_steps)
    # 调整 Bin 对应的角
    data['Angle'] = (data['Bin'] - 0.5) * 360  # 将 Bin 转换到 [-180, 180] 范围
    I_abs_max = data['I'].max()
    data["I_normalized"] = data["I"]/I_abs_max
    data["Q_normalized"] = data["Q"]/I_abs_max
    data["U_normalized"] = data["U"]/I_abs_max
    data["V_normalized"] = data["V"]/I_abs_max
    angle_mask = (data['Angle'] > -180 ) & (data['Angle'] <= -100)
    std_I = np.std(data["I_normalized"][angle_mask])  # 偏振 Q 的标准差
    print("I 的标准差:", std_I)
    std_Q = np.std(data['Q_normalized'][angle_mask])  # 偏振 Q 的标准差
    print("Q 的标准差:", std_Q)
    std_U = np.std(data['U_normalized'][angle_mask])  # 偏振 U 的标准差
    print("U 的标准差:", std_U)
    std_V = np.std(data['V_normalized'][angle_mask])  # 偏振 U 的标准差
    print("V 的标准差:", std_V)
    raw_L_squared = data['Q_normalized']**2 + data['U_normalized']**2
    noise_L_squared = std_Q**2 + std_U**2
    data['Linear_Polarization'] = np.where(raw_L_squared >= noise_L_squared, np.sqrt(abs(raw_L_squared - noise_L_squared)), -np.sqrt(abs(noise_L_squared - raw_L_squared)))
  #  data['V_normalized'] = np.where(data["V_normalized"] >= std_V, np.sqrt(abs(data["V_normalized"]**2 - std_V**2)), -np.sqrt(abs(data["V_normalized"]**2 - std_V**2)))
    data["PA"] = np.where((data['Linear_Polarization']  > 2*std_I), 0.5 * np.arctan2(data['U'], data['Q']) * 180 / np.pi, np.nan)
    data["PA_error"] = [PA_err(std_I, L) for L in data['Linear_Polarization']] 
    data['PA_error_filtered'] = data['PA_error']
    data['PA_filtered'] = data['PA']

    # data.loc[data['PA'] > 0, 'PA_filtered'] -= 90
    # data.loc[data['PA'] < 0, 'PA_filtered'] += 90
    data.loc[(data['Angle'] > -2)&(data['Angle'] < 8) & (data['PA'] < 0), 'PA_filtered'] += 90
    data.loc[(data['Angle'] <-3) & (data['PA'] < -50), 'PA_filtered'] += 180
    # data.loc[(data['Angle'] > -29.736070381231666) & (data['PA'].notna()) & (data['PA_filtered'] != 0), 'PA_filtered'] += 90
    #data.loc[(data['Angle'] > 0.16595307917888547) & (data['PA'] < 0), 'PA_filtered'] += 90
    # data.loc[(data['Angle'] > -15.32506106497308) &(data['Angle'] < -15.32506106497308) & (data['PA'].notna()) & (data['PA_filtered'] != 0), 'PA_filtered'] -= 90
    # data.loc[(data['Angle'] > -15.32506106497308) &(data['Angle'] < -15.32506106497308) & (data['PA'].notna()) & (data['PA_filtered'] != 0), 'PA_filtered'] += 90
    # data.loc[data['Angle'].isin([-19.53079178885631, -21.994134897360706]), 'PA_filtered'] += 90
    # data.loc[data['Angle'].isin([-19.53079178885631, -21.994134897360706]), 'PA_filtered'] += 180
    # data.loc[data['Angle'].isin([-19.53079178885631, -21.994134897360706]), 'PA_filtered'] = np.nan


    data['SN'] = data['Linear_Polarization'] / std_I
    data.loc[data['SN'] < 2, ['PA_filtered', 'PA_error_filtered','SN']] = 0 
    columns_to_write = ['Angle','I_normalized','Linear_Polarization', 'V_normalized', 'PA','PA_error','PA_filtered',"PA_error_filtered","SN"]     
    data.to_csv(data.to_csv(psr + f"_{x_min}_{x_max}_filtered_data.csv", columns=columns_to_write, index=False))
    return data


file = psr + ".csv"
all_min,all_max = -9.6,24
processed_data = process_data(file, x_min=all_min, x_max=all_max)
print(processed_data.columns)
plot_data(data=processed_data, x_min=all_min, x_max=all_max,psr=psr)    