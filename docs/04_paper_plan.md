# 论文写作路线

## 当前第一篇论文定位（A3.5 final后）

当前推荐题目方向收敛为：

> **Feasibility-Aware Autoregressive Decoding for Learned
> Continuous-Process Multi-Robot Allocation**

主贡献是相同hetero-GNN条件下Pair-Pointer相对静态decoder的确认性增益，
不是学习方法优于MILP/LNS。Untouched final为65.05%对40.28%，配对提升
24.77个百分点且统计显著；但hybrid load-balanced以更高coverage和更低
时间支配Pair-Pointer，MILP/LNS coverage也更高。完整证据与写作结构见
`docs/25_a3_5_first_paper_evidence_skeleton.md`。A4 warm-start属于未来独立
协议，不回填第一篇确认性结果。

## 1. 第一篇小论文：范围冻结

建议题目：

> **Feasibility-Aware Graph Learning for Continuous-Process Multi-Robot Task Allocation and Scheduling**

问题：在连续工艺曲线被切分为具有方向、交接、前驱、时间窗和资源语义的任务段后，如何在异构机器人能力、可达性、负载均衡和共享区约束下，快速产生可验证的分配、排序和调度？

输入：机器人/工具/初始状态；连续段曲线和方向；估计进入、执行、退出时间；可达性、路径长度和运动学风险；前驱、交接、优先级、时间窗和共享区。

输出：`assignment[segment, robot]`、每个机器人有序任务段、候选开始时间和修复/失败诊断。输出不是关节控制量，也不预测物理质量。

方法：解析曲线特征或可选PointNet几何encoder；机器人—任务—资源异构图；硬可行性mask；GNN/graph Transformer；sequential assignment/order decoder；确定性调度、CP-SAT repair或局部修复。

基线：贪心、负载均衡贪心、Hungarian+顺序启发式、CP-SAT/MILP、LNS/ALNS；所有方法共享同一约束、同一时间预算和同一冻结实例。

指标：Cmax、负载方差、总travel/setup时间、priority-weighted tardiness、可行率、违例/repair率、共享区冲突数（修复前后）、运行时间和小规模optimality gap；还报告任务数、机器人数和布局变化下的泛化。

第一篇不把路径规划、RL、物理模型同时宣称为创新。路径模块只提供边可达性、路径长度、预计时长和基础冲突检查。

## 2. 第一篇章节与图表

```text
1 Introduction: 连续工艺段与离散点MRTA的区别
2 Related Work: MRTA/VRP scheduling, constrained graph learning, curve encoding
3 Problem: segment semantics, heterogeneous constraint graph, objectives
4 Method: mask, decoder, verifier/repair
5 Protocol: synthetic/geometric benchmark, splits, baselines, budgets
6 Results: quality/runtime, constraints, scale, ablation, failures
7 Limitations: no factory deployment, no physical quality model
```

必需图表：系统接口图、连续段/异构图示例、严格split图、质量—时间Pareto、失败/repair案例、规模泛化表。真实几何案例仅在A6数据核验完成后独立标注。

## 3. 硕士论文叙事

毕业论文以模块闭环而非单一网络为主线：

```text
continuous process input
 -> allocation/scheduling (WP-A)
 -> geometric-kinematic route planning (WP-B)
 -> robust execution/compensation (WP-C)
 <- external process constraints/costs
```

WP-B扩展姿态、方向、时间参数化、共享空间重规划及其对分配代价的反馈。WP-C扩展定位误差、延迟/扰动、监督prior、安全投影和可选学习策略。物理团队模型只作为有版本和证据来源的外部约束/代价，不成为本论文的物理建模贡献。

若WP-C的RL仍不能超过强监督/优化基线，论文将其作为限制与执行层比较，而不强行维持RL创新叙事。

## A4a warm-start证据状态

`a4_warm_start_pilot_v1`不能补写为第一篇论文的确认性结果。唯一一次validation
运行发现MILP initializer标签污染与fixed-time incumbent截断语义错误，违反预注册
完整性门槛，正式分类为`A4A_PRIMARY_EVALUATION_INVALID_STOP`。固定50次迭代的
结果只可用于解释repair行为和设计未来协议，不可据此声称Pair-Pointer warm-start
优于或劣于static、load-balanced或MILP。第一篇论文的学习结论仍以已封闭的A3.5
matched-decoder实验为限，强优化/启发式方法继续作为能力与工程参照。

## A4b ordinary-search证据状态

新协议`a4b_neural_lns_dev_v1`仅建立评测、普通LNS/ALNS与未来destroy-ranking
数据接口，不改变A3.5论文主结论。修正后的1秒group-level coverage中random LNS与
ALNS均为62.5%；3秒仅round-robin达到66.7%，ALNS仍为62.5%。因此这些
development结果不能写成“neural guidance优于ordinary LNS”，也不能作为第一篇
论文的确认性学习结论。当前动作是`HOLD_A4B_LEARNED_DESTROY_TRAINING`。

事后语义审计确认v1的fixed-iteration与train-selected operator行来自固定时间截断
trace，不能作为迭代预算证据；fixed-time行仍仅作描述。独立恢复协议
`a4b_ordinary_lns_dev_v2`已完成96 train/48 development新数据、完整train/label门禁、
768条development trace、2,304行metric与12,484-step replay。四种方法在
0.5/1.0/3.0秒均为72.92%且成功/失败identity完全相同；exact-30下ALNS为76.04%、
random为72.92%，但多出的三个恢复首次发生在21.99--155.53秒。fixed-time trace
中位仅完成0--1个邻域，说明当前主比较受共享repair耗时支配，仍不能支持学习邻域
选择或ALNS superiority。若继续A4b，必须先用新协议、新ID和未见development建立
非退化ordinary-search recovery；否则停在当前工程基础。上述结果不改变A3.5论文
主张。

后续学习方向限定为Pair-Pointer-derived guided LNS，而非通用GNN destroy selector。
主因果实验固定`hybrid_load_balanced` initializer和全部repair/verifier/acceptance/
预算，仅改变atomic-unit destroy-set selection；matched static与frozen Pair-Pointer
initializer只能进入后续完整析因。当前ordinary-search semantic-parity recovery尚未
获授权，模型训练继续HOLD；未来结果也不得回填A3.5 sealed结论。
