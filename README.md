# 2026 数维杯秋季赛 B 题

本项目解答车载双站观测的四个建模问题。先读 [数学建模解答](SOLUTION.md) 和 [逐步推导](details_for_solution.md)，其中包含公式推导、可行性与最优性证明、算法步骤、误差检验、调度方案和扰动结果。

| 问题 | 主要方法 | 结果 |
| --- | --- | --- |
| 1. 可观测区与单站选择 | 建筑阴影、连续截面积分、最优弧与就绪时间下界 | 可观测 5608.77 m²，V4 驶往 (83.3244, 62.5331) m，5.0645 s 完成准备 |
| 2. 双站定位和车辆调度 | 方位交会雅可比、可视图、复用链下界与自适应空间细分 | T2 最小一阶 DRMS 0.1436 m。题面三告警方案为 10.3521 s，最近法为 13.1392 s，最早就绪贪心为 10.9690 s；0.50 m 精度对照为 25.0019 s |
| 3. 指向误差 | 投影正态角度密度、概率积分、蒙特卡洛 | 最大累计里程 66.8373 m，80 m 时要求 $a+80b\le1.63675$ m |
| 4. 扰动重调度 | 事件回放、健康任务子集枚举与逐车续行 | 419 个扰动情景和 1 个无扰动对照；从中抽取 40 个情景，与 0.50 m 精度模型的结果配对比较 |

## 运行实验

建议使用 Python 3.10 或 3.11。在包含 `main.py` 和 `requirements.txt` 的项目根目录执行下列命令。题面数据已写入 `geometry.py`，仿真使用固定随机种子。

### 1. 安装依赖并运行常规实验

**Windows（PowerShell）**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B main.py --output results_local
```

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B main.py --output results_local
```

前两条命令只需在新环境中执行一次。常规运行求解四问，输出误差实验、调度对照、扰动汇总及逐车时间轴。`--output results_local` 会将结果写到独立目录；省略该参数会写入 `results/` 并覆盖其中的同名文件。程序会打印扰动情景进度，`summary.json` 在整次运行结束时写入。

### 2. 运行第四问的附加诊断

使用同一虚拟环境，给主程序加上 `--q4-diagnostics`。这会先运行常规实验，再对选定情景执行离散精确搜索、插入式调度对照、消融、耗时与最坏情景搜索。

```powershell
.\.venv\Scripts\python.exe -B main.py --output results_diagnostics --q4-diagnostics
```

```bash
.venv/bin/python -B main.py --output results_diagnostics --q4-diagnostics
```

| 可选参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--q2-relative-gap` | `0.005` | 第二问连续下界搜索的目标相对间隙，`0.005` 表示 0.5%。实际间隙见 `summary.json` 的 `q2_schedule.adaptive_bound`。 |
| `--q2-bound-budget` | `60000` | 第二问自适应连续下界允许的最大细分次数。 |
| `--q4-state-budget` | `20000` | 每次第四问离散精确搜索的状态数上限，仅在启用附加诊断时生效。 |

诊断输出包括 `q4_exact_search.csv`、`q4_ablation.csv`、`q4_search_efficiency.csv`、`q4_runtime.csv`、`q4_worst_search.csv`，以及有可行方案时的 `q4_exact_timelines.csv`。在 `q4_exact_search.csv` 中，`confirmed_on_pool=True` 表示相应搜索在指定候选站池和策略限制下给出了最优性证书；`termination=state_budget` 表示预算已用尽，此时找到的方案不能视为已证最优。

## 查看结果

先读 `summary.json`，再按问题查看详细表。下面链接到 `results/` 中的结果；使用 `--output` 指定新目录时，文件名相同。

| 文件 | 内容 |
| --- | --- |
| [summary.json](results/summary.json) | 四问数值、站位、离散与连续下界、成对细化记录、概率与扰动统计 |
| [q1_observable.png](results/q1_observable.png) | T1 的可观测分区、等质量最优弧和 V4 路径 |
| [q2_vehicle_timelines.csv](results/q2_vehicle_timelines.csv) | 优化方案、最近分配和最早就绪贪心的逐车时间轴 |
| [q2_intersection_errors.csv](results/q2_intersection_errors.csv) | 按交会角与站距对比线性误差、非线性误差分位数和拒绝率 |
| [q2_schedule_experiments.csv](results/q2_schedule_experiments.csv) | 同门槛调度和净空实验的站对、时间与精度 |
| [q2_experiment_timelines.csv](results/q2_experiment_timelines.csv) | 各调度对照的逐车时间轴 |
| [q2_schedule.png](results/q2_schedule.png) | 调度对比甘特图 |
| [q3_probability.png](results/q3_probability.png) | 方位误差达标概率随累计里程变化 |
| [q3_variance_shares.png](results/q3_variance_shares.png) | 三类方位误差的一阶方差贡献率随累计里程变化 |
| [q4_scenarios.csv](results/q4_scenarios.csv) | 各情景的开始时刻、完成率、最晚开始时间、原告警延迟和实际采用的策略 |
| [q4_vehicle_timelines.csv](results/q4_vehicle_timelines.csv) | 420 个情景、五种策略和五辆车的事件明细 |
| [q4_aggregate.csv](results/q4_aggregate.csv) | 按扰动类型统计完成率、最晚时刻和原告警延迟 |
| [q4_grouped.csv](results/q4_grouped.csv) | 按故障车或恢复时间进一步分组的结果 |
| [q4_pressure_scenarios.csv](results/q4_pressure_scenarios.csv) | 两种初始调度各自同口径的压力测试及原任务延迟 |
| [q4_pressure_aggregate.csv](results/q4_pressure_aggregate.csv) | 压力测试的完成率、最晚时间与代表性最差情景 |
| [q4_pressure_timelines.csv](results/q4_pressure_timelines.csv) | 压力测试逐车时间轴 |
| [q4_comparison.png](results/q4_comparison.png) | 完成率及带罚值的最晚开始时间对比 |

按 `scenario`、`policy`、`vehicle` 筛选时间轴。例如 `N000_50_V4` 表示东向 50 m 的新增告警与 V4 故障同时发生，`R30` 表示 30 s 恢复通信。`retain` 严格执行原指令；`retain_plus` 取消失效任务并追加新告警；`repair` 保留健康任务并改派故障待办；`replan_raw` 全撤回作对照；`replan` 比较上述策略与保留子集方案。逐情景表中 `selected_policy` 标出所选策略，`kept_jobs` 表示整体保留的原任务，`continued_legs` 表示保留原站及准备进度的单车动作。

`travel` 行的 `path_xy` 存储折线路径。`slew` 是转向，`wait_sync` 是等待另一辆车，`observe` 行的起止即观测开始和结束，`reset` 是观测后回正，`abort_reset` 是中止任务后回正，`failed` 是车辆退出，`at_station` 表示原地开始站内动作。时间列以秒计，坐标列以米计，角度列以度计。CSV 为带 BOM 的 UTF-8，便于 Excel 直接打开。

## 维护入口

| 模块 | 负责内容 |
| --- | --- |
| [geometry.py](geometry.py) | 物理数据、可观测性、交会几何、可视图路径 |
| [problem1.py](problem1.py) | 第一问的阴影、连续面积、最优弧和准备时间证明 |
| [problem2_accuracy.py](problem2_accuracy.py) | T2 圆弧正交站对和同步就绪的连续下界 |
| [experiments.py](experiments.py) | 同门槛调度、成对局部细化、视线净空敏感性和逐车次指向复核 |
| [scheduler.py](scheduler.py) | 候选站、最近法、束搜索、单车复用枚举与离散下界 |
| [errors.py](errors.py) | 交会精度验证、投影正态概率与误差模拟 |
| [disturbances.py](disturbances.py) | 扰动情景、故障和断联回放、按车续行与自适应重调度 |
| [analysis_metrics.py](analysis_metrics.py) | 连续下界和图表 |
| [main.py](main.py) | 执行顺序、几何与时序复核、结果导出 |

修改站点、障碍、事件逻辑或情景后，重新运行 `main.py`，用新的汇总和时间轴核对可行性。数学推导和算法步骤可用于移植到其他语言。
