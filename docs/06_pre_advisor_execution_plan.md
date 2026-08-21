# 分阶段实施与对接前执行计划

当前先完成第一篇小论文的A0–A4，不进行新的RL调参。详细18个月工作包见
`docs/09_masters_thesis_work_packages.md`；按周执行、测试、实验矩阵和冻结门见
`docs/09_phase1_continuous_allocation_execution_plan.md`。

| Gate | 必做内容 | 验收门槛 |
|---|---|---|
| A0 | 连续段/机器人/工具/资源schema；切分、前驱、时间窗、交接与共享区约束字典 | 10–20个手工可审计fixture、schema和约束测试 |
| A1 | 可达性/代价oracle；冲突验证；贪心、负载均衡贪心、Hungarian+排序、CP-SAT/MILP | 每个实例有可解释solver状态、可行/失败诊断和指标 |
| A2 | 参数曲线/工位/机器人配置基准；工作件/布局/实例级严格split | manifest hash、泄漏测试、数据卡和自动报告 |
| A3 | mask异构GNN/图Transformer、assignment/order decoder、solver imitation/warm-start | validation-only选择、冻结测试、全部强基线比较 |
| A4 | CP-SAT repair、LNS/ALNS、规模与动态到达/故障/扰动重规划 | 等预算quality/runtime Pareto、消融和失败案例 |
| A5 | 路径规划和执行补偿接口 | 边代价接口与模块集成测试；不要求物理建模 |
| A6 | 确认CAD/曲线/布局/日志后的held-out真实case | `REAL-GEOMETRY`或更高证据等级的独立报告 |

## A0–A4 的防泄漏与公平比较

- 以工件、布局、任务实例分组split；同一父曲线的相邻段和同一布局变体不能跨训练与冻结测试。
- 所有特征归一化、学习型cost和solver标签仅由训练组拟合/生成。
- CP-SAT/MILP报告time limit、optimality gap和不可行状态；GNN、LNS和启发式在相同约束与预算下评估。
- repair失败必须记为失败，不能删除；冲突数需同时报告repair前后。
- A3先做监督模仿或warm-start。RL只在A4的动态问题有明确MDP和优势假设时考虑。

## 对接触发条件

完成A0–A2后即可向老师展示可审计架构并请求真实几何资料；完成A3–A4后再请求真实案例验证。届时索取CAD/坐标系、连续工艺曲线及方向/速度、机器人基座/TCP/关节限位、工具和夹具、禁入区、交接/优先级/时间窗、历史程序/分配和执行日志。没有这些资料时，所有结论维持`SYNTHETIC`/`SIM-GEOMETRIC`。
