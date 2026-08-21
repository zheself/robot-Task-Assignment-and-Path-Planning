# 研究范围与主问题

本项目按“先小论文、后模块化扩展为硕士论文”执行。完整时间表见
`docs/09_masters_thesis_work_packages.md`；第一篇的可执行协议见
`docs/08_continuous_process_multi_robot_plan.md`。

## 1. 两个相接工作包

### WP-A：连续工艺多机械臂任务分配与调度（当前优先、第一篇小论文）

对象是滚边、涂胶、焊缝等连续曲线按明确工艺规则得到的**连续任务段**，而不是离散点。
研究任务切分、机器人归属、机器人内排序、负载均衡、优先级/时间窗、共享区冲突和调度。
另一成员负责离散点分配；本项目不得把点级任务或其结果混入主问题。

问题输入：连续段几何与方向、机器人/工具能力、可达性和代价、前驱/交接规则、时间窗、共享区/夹具资源。输出：段到机器人的分配、机器人内有序段列表、开始时间窗和待修复约束。

第一篇题目建议为：

> **Feasibility-Aware Graph Learning for Continuous-Process Multi-Robot Task Allocation and Scheduling**

核心方法是异构约束图、不可行边掩码、assignment/order decoder、确定性调度与repair；CP-SAT/MILP、LNS/ALNS和启发式是必需基线。PointNet最多编码稠密曲线/CAD点集，绝不是分配器或GNN本身。

### WP-B/C：路径规划与执行层（毕业论文后续扩展）

WP-B接收已分配的有序任务段，为每个机器人生成/优化初始路径，检查运动学可达性、姿态/方向约束、时间参数化和共享空间冲突，并把路径长度、时长和奇异性风险反馈给WP-A。

WP-C在已规划路径上接入执行补偿。已有UR5 FK/Jacobian、安全投影、定位误差prior和序列环境可复用；监督学习、深度学习或RL只有在动态重规划、延迟或扰动下能相对强基线给出明确增益时才成为正式方法。当前SAC/TD3未超过projected supervised prior，必须保留为失败结论。

## 2. 模块化接口与物理边界

```text
process curves + external constraints
       -> WP-A allocate/schedule
       -> WP-B plan/check/replan
       -> WP-C execute/compensate
       -> diagnostics/costs feed upward
```

物理团队可以提供速度/姿态上限、工艺风险/质量代价、连续执行或交接规则、优先级、夹具/禁入区等**外部接口**。本项目不建立或声称建立应力、接触力、板件塑性和真实滚边质量模型。

## 3. 证据边界

- 当前连续曲线和工位实例只能称 `SYNTHETIC` 或 `SIM_GEOMETRIC`。
- 静态UR5 CSV不能直接训练多机器人分配模型，也不是offline RL数据。
- 保守共享区或轨迹包络检查只是一种代理冲突检查，不等于真实碰撞安全保证。
- 未有验证CAD/工艺规则/执行日志前，不得声称真实产线部署、真实滚边验证或sim-to-real成功。

## 4. 成功判据

第一篇的成功不是“使用了GNN”，而是在同等硬约束与计算预算下，GNN候选加修复在冻结测试上相对CP-SAT/MILP、LNS/ALNS、Hungarian+排序和贪心，给出可报告的质量—时间折中；若没有优势，保留失败和速度/规模诊断。
