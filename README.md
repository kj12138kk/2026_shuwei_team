# 2026 数维杯秋季赛 B 题

本项目解答车载双站观测的四个建模问题。先读 [数学建模解答](SOLUTION.md)，其中包含公式推导、可行性与最优性证明、算法步骤、误差检验、调度方案和扰动结果。

| 问题 | 主要方法 | 结果 |
| --- | --- | --- |
| 1. 可观测区与单站选择 | 圆盘几何、闭区间光线裁剪、仰角距离下界 | T1 最优站 (92.2231, 70.0000) m，距目标 7.5821 m |
| 2. 双站定位和车辆调度 | 方位交会雅可比、可视图、复用链枚举与方格下界 | T2 最小一阶 RMSE 为 0.1436 m。最晚观测开始为 10.3688 s，连续下界为 10.2477 s |
| 3. 指向误差 | 投影正态角度密度、概率积分、蒙特卡洛 | 最大累计里程 66.8373 m，80 m 时要求 $a+80b\le1.63675$ m |
| 4. 扰动重调度 | 事件回放、实时状态更新、字典序方案比较 | 82 个情景（含无扰动对照），提供各策略的指标和逐车时间轴 |

## 复现

使用 Python 3.9 或以上版本，在仓库根目录执行以下命令。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B main.py
```

macOS 或 Linux 可将解释器路径换成 `.venv/bin/python`。默认写入 `results/`。可用 `python main.py --output results_local` 生成一份独立结果。题面数据录在 `geometry.py` 中，运行只需要本目录的源码和依赖。仿真固定了随机种子。

## 查看结果

| 文件 | 内容 |
| --- | --- |
| [summary.json](results/summary.json) | 四问数值汇总、站位、精度、连续下界、概率和统计量 |
| [q1_observable.png](results/q1_observable.png) | T1 的可观测区域与解析最优站 |
| [q2_vehicle_timelines.csv](results/q2_vehicle_timelines.csv) | 优化方案与最近分配的逐车时间轴 |
| [q2_schedule.png](results/q2_schedule.png) | 调度对比甘特图 |
| [q3_probability.png](results/q3_probability.png) | 方位误差达标概率随累计里程变化 |
| [q4_scenarios.csv](results/q4_scenarios.csv) | 各情景的告警开始时刻、完成率、最晚开始时间和实际采用的策略 |
| [q4_vehicle_timelines.csv](results/q4_vehicle_timelines.csv) | 全部 82 个情景、四个策略和五辆车的事件明细 |
| [q4_aggregate.csv](results/q4_aggregate.csv) | 按扰动类型统计的方案均值 |
| [q4_comparison.png](results/q4_comparison.png) | 完成率及带罚值的最晚开始时间对比 |

按 `scenario`、`policy`、`vehicle` 筛选时间轴。例如 `N000_50_V4` 表示东向 $0^\circ$、距离 50 m 的新增告警与 V4 故障同时发生，`R30` 表示 30 s 恢复通信。`retain` 保持原任务，`retain_plus` 允许顺延新增任务，`replan_raw` 尝试撤回重排，`replan` 取两个可行策略中指标更优的一种。

`travel` 行的 `path_xy` 存储折线路径。`slew` 是转向，`wait_sync` 是等待另一辆车，`observe` 行的起止即观测开始和结束，`reset` 是观测后回正，`abort_reset` 是中止任务后回正，`failed` 是车辆退出，`at_station` 表示原地开始站内动作。时间列以秒计，坐标列以米计，角度列以度计。CSV 为带 BOM 的 UTF-8，便于 Excel 直接打开。

## 维护入口

| 模块 | 负责内容 |
| --- | --- |
| [geometry.py](geometry.py) | 物理数据、可观测性、交会几何、可视图路径 |
| [scheduler.py](scheduler.py) | 候选站、最近法、束搜索、单车复用枚举和连续细化 |
| [errors.py](errors.py) | 交会精度验证、投影正态概率与误差模拟 |
| [disturbances.py](disturbances.py) | 扰动参数、故障和断联回放、对照及重调度 |
| [analysis_metrics.py](analysis_metrics.py) | 连续下界和图表 |
| [main.py](main.py) | 执行顺序、几何与时序复核、结果导出 |

修改站点、障碍、事件逻辑或情景后，重新运行 `main.py`，用新的汇总和时间轴核对可行性。数学推导和算法步骤可用于移植到其他语言。
