# 数维杯 B 题论文插图

图按论文的叙述顺序编排。当前版本使用本项目 `results/` 的四问输出。图内采用英文标注，以下中文图注可随正文使用。

| 图 | 主题 | 文件 |
| --- | --- | --- |
| 场景图 | 初始三维视线与五车到站路径 | [PDF](fig00_scene_and_routes.pdf) · [SVG](fig00_scene_and_routes.svg) · [PNG](fig00_scene_and_routes.png) |
| 1 | T1 可观测区域、最快单站与 T2 正交双站 | [PDF](fig01_observability.pdf) · [SVG](fig01_observability.svg) · [PNG](fig01_observability.png) |
| 2 | 交会角、定位误差与近退化样本 | [PDF](fig02_intersection.pdf) · [SVG](fig02_intersection.svg) · [PNG](fig02_intersection.png) |
| 3 | 多车调度、精度约束与净空敏感性 | [PDF](fig03_schedule.pdf) · [SVG](fig03_schedule.svg) · [PNG](fig03_schedule.png) |
| 4 | 指向达标概率、误差贡献与实际站点复核 | [PDF](fig04_pointing.pdf) · [SVG](fig04_pointing.svg) · [PNG](fig04_pointing.png) |
| 5 | 扰动重调度、高压对照与案例回放 | [PDF](fig05_rescheduling.pdf) · [SVG](fig05_rescheduling.svg) · [PNG](fig05_rescheduling.png) |

PDF 与 SVG 保留矢量图形和可编辑文字，PNG 为 600 dpi。图幅均按 180 mm 双栏宽度绘制。在排版中以原宽插入可保留设计字号。图形选型与版式依据 `picture-draw` 中的 `chart-type-chooser` 和 `nature-figure-production`。

## 论文图注

### 场景图　初始观测几何与最优站点路径

**a**，100 m × 100 m 场地的等比例正交投影。方形标记位于 V1–V5 初始相机位置（离地 2 m），三角标记为 T1、T2、T3，离地高度分别为 45、35、55 m。实线连接满足车体、仰角和无遮挡条件的 10 个初始车–目标对。竖向虚线为目标的地面投影，地面可见处以 × 标记。正交视角中，被建筑体遮住的线段会随建筑遮挡裁切。**b**，俯视图中星标 $P^*$ 为第一问选定站点 $(83.3244, 62.5331)\,\mathrm{m}$。五车从各自起点独立求得的最短路径分别为 87.73、52.85、77.68、7.19、46.74 m，实际派车为 V4。浅色虚线轮廓为基于车体半径 1.5 m 的外切避障多边形，细灰线为所绘路径折点之间的可直达边，V5 的虚线路径与 V3 的部分路段重合。 [场地、视线判定与路由](../geometry.py) · [第一问站点](../results/summary.json)

### 图 1　单站可观测性与最优站位

**a**，100 m × 100 m 场地内 T1 的四类互斥分区。B1–B5 为建筑，V1–V5 为车辆起点。浅灰为车体中心禁入区，浅蓝为仰角超过 80° 的区域，浅橙为视线遮挡区，浅绿为可观测区。可观测面积由连续积分得到，为 5,608.77 m²。**b**，T1 最优半径为 7.58 m。绿色圆弧为等质量最优站位，V4 沿蓝线到达星标站点并在 5.06 s 完成准备。**c**，V5 与 V3 在距离 T2 各 5.82 m 处形成 90° 交会，虚线基线长 8.23 m，最小一阶距离均方根误差（DRMS）为 0.144 m。站点与目标使用东向、北向直角坐标。 [汇总数据](../results/summary.json) · [几何数据](../geometry.py) · [可观测性模型](../problem1.py)

### 图 2　交会角对双站定位误差的影响

**a**，单站方位误差标准差为 1° 时，两站等距 10、20 或 40 m 的一阶 DRMS 随标称交会角的变化。菱形为图 1c 的 T2 站对。**b**，两站均距目标 20 m 时，14 个标称交会角下线性误差与非线性交会误差的第 95 百分位数。每个角度使用 300,000 组方位误差样本，曲线显示筛选前统计量。**c**，在同一批样本上，实测交角距 0° 或 180° 小于 10° 时判为近退化并剔除，图中为相应剔除比例。**a** 使用连续公式，**b–c** 的连线辅助识别离散采样趋势。 [交会实验表](../results/q2_intersection_errors.csv) · [误差模型](../errors.py)

### 图 3　第二问的调度与净空对照

**a–b**，分别为题面名义约束与附加 DRMS ≤ 0.50 m、交会角至少 30° 的方案。两图共用时间轴，粗体车号标记任务复用车辆。蓝色观测条的左端为告警启动时刻。**c**，各方案对三项告警的最晚启动时刻。名义优化方案为 10.3521 s，最近分配为 13.1392 s，最早就绪贪心为 10.9690 s。精度对照下优化方案与最近分配分别为 25.0019 s 和 30.4530 s。**d**，以名义可行方案为基准，视线（LOS）净空设为 0.05 m 和 0.10 m 时最晚启动分别推迟 39.2 ms 和 77.2 ms。绿点为零净空条件下 10.3006 s 的连续下界，对应认证相对间隙约 0.50%。 [逐车事件](../results/q2_experiment_timelines.csv) · [调度对照表](../results/q2_schedule_experiments.csv) · [连续下界](../results/summary.json)

### 图 4　累计行驶与目标站距对指向可靠性的影响

**a**，目标水平距离固定为 40 m 时，绝对方位误差不超过 5° 的概率及蒙特卡洛估计。每个模拟点使用 600,000 次试验，误差线为 95% Wilson 区间。95% 概率上限对应累计里程 66.84 m，80 m 处的积分概率为 91.51%。**b**，40 m 工况下三个误差源的一阶方差份额，位置误差在 66.84 m 时占 84.6%。**c**，按各观测访问的实际站距重新计算概率。位置标准差与站距之比不超过 4.09% 时达到 95% 要求。名义方案 6/6 次访问达标，0.50 m DRMS 方案 4/6 次达标，独立 T2 最小 DRMS 站对 0/2 次达标。这里的分母按车次统计。 [概率与车次复核](../results/summary.json) · [方差贡献表](../results/q3_variance_shares.csv) · [概率模型](../errors.py)

### 图 5　扰动下的告警启动表现

**a**，419 个扰动案例按事件组合分组，括号为各组样本数。若存在未启动告警，汇总中的最晚启动时刻按 60 s 计。**b**，相同的 40 个高压案例分别用名义和 0.50 m DRMS 约束运行。局部修复和择优策略在两个口径下分别有 40/40 和 27/40 个案例启动全部告警。**c**，`N270_50_V5_R30` 的精度约束案例，5 s 发生 V5 故障并加入南向 50 m 处的新告警，通信在 30 s 恢复。时间轴显示局部修复后各车的动作及观测任务。**d**，原令保留、保留并追加、局部修复三种策略的累计告警启动数，最终分别为 1/4、3/4、4/4。择优策略在该场景采用局部修复，四项告警的最晚启动时刻为 56.89 s。 [常规扰动汇总](../results/q4_aggregate.csv) · [高压汇总](../results/q4_pressure_aggregate.csv) · [高压场景](../results/q4_pressure_scenarios.csv) · [逐车回放](../results/q4_pressure_timelines.csv)

## 复现

在项目根目录使用 [requirements.txt](../requirements.txt) 中的现有依赖运行。

```powershell
python -B -X utf8 pictures/generate_figures.py
```

仅重画某个主题时，例如第三问。

```powershell
python -B -X utf8 pictures/generate_figures.py --only pointing
```

`--only` 还接受 `scene`、`observability`、`intersection`、`schedule`、`rescheduling`，可一次列出多个主题。用 `--output <目录>` 可切换输出位置。图由同目录下的 `plot_*.py` 模块分别生成。模型或参数更新后，先按[项目说明](../README.md)刷新 `results/`，再重新导图并复核图注中的数值。
