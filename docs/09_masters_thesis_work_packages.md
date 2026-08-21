# 硕士论文整体路线与工作包规划

规划跨度：约18个月；按验收门而非日历承诺推进。每个工作包只有在前一门通过后才扩大范围。

## 1. 最终论文目标

形成可审计的模块化闭环：

```text
continuous process segments
 -> feasibility-aware allocation/scheduling
 -> multi-arm geometric-kinematic planning
 -> robust execution compensation
 <- external physical/process constraints and costs
```

毕业论文的贡献是模块之间的约束一致性、接口与证据链；不是宣称由本人建立了全部物理模型。WP-A是第一篇独立小论文，WP-B/C为后续扩展与整合。

## 2. 工作包、时间窗与交付物

| 相对时间 | 工作包 | 目标与核心交付 | 决策门 |
|---|---|---|---|
| M0–M2 | WP0 / A0–A1 | schema、约束字典、fixtures、oracle、verifier、CP-SAT与启发式 | 无学习baseline可行且可解释 |
| M2–M4 | WP-A1 / A2 | 连续曲线/布局基准、严格manifest、自动报告 | 无泄漏且小实例可验证gap |
| M4–M7 | WP-A2 / A3–A4 | masked GNN、solver imitation/warm-start、repair、LNS/ALNS、尺度/动态评测 | 第一篇是否有质量—速度贡献 |
| M7–M9 | Paper-1 | 冻结协议、统计/失败分析、撰稿与投稿 | 不以弱基线或单一实例支撑结论 |
| M8–M12 | WP-B | 路径生成、姿态/可达性、时间参数化、共享区代理检查、代价回馈 | 路径接口可驱动分配重评估 |
| M11–M15 | WP-C | 定位prior、安全投影、执行扰动、监督/优化与可选学习 | RL只有超过强基线才保留为贡献 |
| M13–M17 | WP-D / A6 | 接入已核验CAD/规则/日志和物理团队接口，held-out case | 按REAL-GEOMETRY/HISTORICAL标签报告 |
| M16–M18 | Thesis | 统一实验、消融、限制、复现包和论文撰写 | 模块边界和证据级别一致 |

时间窗可重叠，但不能跳过A0–A2直接训练GNN，不能跳过路径验证直接声称多臂执行。

## 3. 第一篇小论文的范围控制

第一篇只回答：“连续任务段的约束图学习能否在强优化/启发式基线下，快速给出可验证的多机械臂分配、排序和调度？”

包含：连续段表示、异构图、mask、decoder、CP-SAT/MILP/LNS/启发式、repair、严格合成/几何评测。  
不包含：高保真路径执行、端到端RL、物理滚边模型、真实产线部署或质量提升。

## 4. 模块所有权与接口

| 模块 | 本项目责任 | 输入 | 输出 | 非本项目责任 |
|---|---|---|---|---|
| WP-A | 分解、分配、排序、调度、repair | 曲线、能力、约束、边代价 | assignment/order/schedule | 离散点分配 |
| WP-B | 路径/姿态/时间和代理冲突检查 | order/schedule、机器人几何 | route/cost/risk/repair constraints | 未验证的碰撞保证 |
| WP-C | 执行误差/安全与可选学习 | planned route、扰动/观测 | compensation/diagnostics | 强行证明RL有效 |
| physical interface | 接收并版本化外部限制/代价 | 团队提供的数据 | typed constraint/cost | 应力、接触、塑性、质量建模 |

## 5. 风险与转向规则

- 若GNN未优于/接近同预算LNS/CP-SAT，保留其扩展性和失败结果，转向学习型warm-start或停止将GNN作为贡献。
- 若真实几何迟迟无法核验，第一篇只作`SIM-GEOMETRIC`并完成方法论；不虚构产线结论。
- 若路径规划显示上层解大量不可达，回馈oracle/repair并重做A1假设，不以事后删除失败任务处理。
- 若RL不优于监督/优化基线，WP-C以安全执行接口和负结果为主，不占用第一篇研究范围。
- 若物理团队接口尚未稳定，使用显式占位约束并标`unverified`，不自建未经验证的物理代理。

## 6. 建议目录合同

未来实现保持独立：`src/.../allocation/`、`planning/`、`execution/`、`process_interface/`；对应`configs/`、`data/cards/`、`data/manifests/`、`tests/`和`reports/`子目录。公共对象只通过版本化schema和上述接口交换，避免把第一篇的分配逻辑耦合进旧UR5/RL环境。
