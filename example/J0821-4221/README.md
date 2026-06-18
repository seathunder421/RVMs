# 使用步骤

1. 安装运行依赖。

   ```bash
   pip install numpy pandas matplotlib
   ```

   如果需要重新运行 RVM 拟合，再安装：

   ```bash
   pip install bilby dynesty
   ```

   如果要使用 `scipy` 版本的 PA 误差计算，或查询 ATNF 参数，可选安装：

   ```bash
   pip install scipy psrqpy
   ```

2. 准备原始数据。

   将原始 9 列 CSV 放在当前目录，例如：

   ```text
   J0821-4221.csv
   ```

   如果当前目录只有一个原始 CSV，`prepare_pa_data.py` 会自动读取；如果有多个原始 CSV，在脚本开头参数区手动设置：

   ```python
   RAW_DATA_FILE = Path("J0821-4221.csv")
   ```

3. 按需要修改 PPA 筛选参数。

   打开 `prepare_pa_data.py`，通常只需要改脚本开头的参数区：

   ```python
   PHASE_MIN = -9.6
   PHASE_MAX = 24.0
   NOISE_RANGE = (-180.0, -100.0)
   PA_DETECTION_SIGMA = 2.0
   PA_SN_CUTOFF = 2.0
   PPA_SELECTION_REGIONS = [
       (PHASE_MIN, PHASE_MAX),
   ]
   ```

4. 按需要手动调节 PPA 跳变。

   在 `prepare_pa_data.py` 的 `PA_WRAP_RULES` 中启用或关闭规则：

   ```python
   PA_WRAP_RULES = [
       {"enabled": True, "angle_lt": -3.0, "pa_lt": -50.0, "action": "shift", "offset_deg": 180.0},
       {"enabled": True, "angle_min": -2.0, "angle_max": 8.0, "pa_lt": 0.0, "action": "shift", "offset_deg": 90.0},
   ]
   ```

   常用操作：

   ```text
   action="shift"    移动 PPA，配合 offset_deg 使用
   action="set_nan"  删除该 PPA 点
   offset_deg=90     上移 90 度
   offset_deg=180    上移 180 度
   ```

5. 生成筛选后的 PPA 数据。

   ```bash
   python prepare_pa_data.py
   ```

   运行后会生成：

   ```text
   J0821-4221_-9.6_24_filtered_data.csv
   J0821-4221_PPA_selection.png
   J0821-4221_PPA_selection.pdf
   ```

6. 检查 `J0821-4221_PPA_selection.png`。

   如果 PPA 点选择或跳变调节不合适，回到第 3 步和第 4 步修改参数，然后重新运行：

   ```bash
   python prepare_pa_data.py
   ```

7. 设置 RVM 拟合模式。

   打开 `fit_rvm.py`，在脚本开头参数区设置 `FIT_MODE`。

   只读取已有 posterior 并重新出图：

   ```python
   FIT_MODE = "load"
   ```

   重新运行完整两步拟合：

   ```python
   FIT_MODE = "both"
   ```

   只运行第一步拟合：

   ```python
   FIT_MODE = "phase"
   ```

   读取第一步 posterior，只运行第二步拟合：

   ```python
   FIT_MODE = "alpha_beta"
   ```

8. 按需要修改 RVM 拟合参数。

   `fit_rvm.py` 中常用参数为：

   ```python
   PSR_NAME = None
   FILTERED_DATA_FILE = None
   PHASE_MIN = None
   PHASE_MAX = None
   NOISE_RANGE = (100.0, 180.0)
   PHASE_POSTERIOR_FILE = None
   ALPHA_BETA_POSTERIOR_FILE = None
   ```

   当目录中只有一个 `*_filtered_data.csv` 时，`FILTERED_DATA_FILE = None` 会自动读取该文件。

9. 运行 RVM 拟合或重新出图。

   ```bash
   python fit_rvm.py
   ```

   运行后会生成或更新：

   ```text
   J0821-4221_phase_posterior.csv
   J0821-4221_alpha_beta_posterior.csv
   J0821-4221_MeerKAT_fiting.png
   J0821-4221_MeerKAT_fiting.pdf
   J0821-4221_MeerKAT_RVMFIT.png
   J0821-4221_MeerKAT_RVMFIT.pdf
   information.csv
   ```

10. 如果只需要基于已有结果重新生成图片和 `information.csv`，保留以下文件，并将 `FIT_MODE` 设为 `"load"` 后运行：

    ```bash
    python fit_rvm.py
    ```

    需要保留的文件：

    ```text
    J0821-4221_-9.6_24_filtered_data.csv
    J0821-4221_phase_posterior.csv
    J0821-4221_alpha_beta_posterior.csv
    ```
