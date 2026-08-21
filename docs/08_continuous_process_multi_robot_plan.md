# 连续工艺多机械臂：分配、路径与执行接口规划

状态：A0–A2已通过；A3 W9 foundation已通过，W10尚未开始；A4及WP-B/C保持阻塞。  
范围：第一篇以WP-A为中心；WP-B/WP-C定义接口和后续扩展，非第一篇附加创新点。

## 1. 总体架构

```text
continuous curves + workcell + external process limits
          |
    segmenter / constraint builder
          |
  WP-A: allocation + order + schedule + repair
          |  {robot, ordered segments, start windows, constraints}
          v
  WP-B: route generation + kinematics + timing + conflict check
          |  {route, duration, risk, repair constraints}
          v
  WP-C: safety projection + error compensation + optional learning
          |  {execution diagnostics}
          +-------------------- feedback costs --------------------+
```

物理团队只从左侧提供有来源的约束和代价，如速度/姿态上限、连续/交接规则、工艺优先级、夹具禁入区、风险或质量代价。本项目不实现应力、接触、塑性或质量预测。

## 2. WP-A 第一篇小论文

### 研究对象和语义

任务不是CSV行或离散点。一条曲线按最小长度、工艺方向、允许交接位置、不可拆分段、前驱关系和资源规则切成segment；同一父曲线的顺序和连续性必须作为约束保存。

每个segment至少包括：`task_id, parent_curve_id, segment_index, sampled_curve,
start/end pose, tangent/direction, length, nominal speed, process duration,
tool requirement, priority, release/due window, predecessor, handoff policy,
shared zone`。机器人包括基座、工具/能力、初始状态、可用时间和运动学模型标识。

### 异构约束图及输出

节点为机器人R、任务段T、共享区/夹具Z；边为R–T可达性/代价/工具匹配，T–T前驱/连续/交接，T–Z占用，以及表示潜在时间冲突的关系。可选PointNet编码`sampled_curve`，但需保留弧长、起终点和方向；分配关系由heterogeneous GNN/graph Transformer处理。

网络输出masked `assignment[T,R]`、机器人内next/insertion score和可选时间优先分数。不可达、工具不匹配、违背固定规则的边在decoder前掩码。确定性scheduler/repair再保证每段唯一、前驱、资源和共享区约束；失败即记录失败。

### 强基线与实验

必须比较：贪心、负载均衡贪心、Hungarian+顺序启发式、CP-SAT/MILP、LNS/ALNS；CP-SAT/MILP给小规模oracle和gap。训练先使用solver imitation或warm-start，非端到端RL。比较时约束、实例和预算相同。

指标：Cmax、负载方差、travel/setup、priority tardiness、可行率、各类违例、repair率、共享区代理冲突（前/后）、runtime、small-instance gap、规模/布局泛化。split按workpiece/layout/instance，禁止父曲线/布局泄漏。

## 3. WP-B 路径规划接口

WP-B不改变WP-A的任务语义，只验证或精化已选边和顺序：

```text
estimate_edge(robot, segment, context) ->
  feasible, travel_time, process_time, path_length,
  kinematic_risk, conflict_proxy, confidence, diagnostics

plan_route(robot, ordered_segments, schedule) ->
  route, pose/direction compliance, timing, diagnostics

check_routes(all_routes, schedule) ->
  proxy_conflicts, resource_overlaps, repair_constraints
```

发展内容：初始路径、姿态/方向、IK/奇异性、时间参数化、共享区时空检查和重规划。只有在完整且验证过的机器人/夹具几何可用后，collision checker的证据范围才能提升；在此之前它只是proxy。

## 4. WP-C 执行与学习接口

已有UR5模块可提供局部安全投影、定位误差prior、风险和执行诊断。将来针对已分配/规划路径，可比较无补偿、supervised prior、优化/反馈和可选学习策略。RL需要明确的动态任务、扰动、延迟或在线重规划MDP；静态UR5点数据不能充当transition。当前RL失败不应通过继续调参掩盖。

## 5. 数据与证据升级

当前：程序化曲线、工作单元和约束为`SYNTHETIC`/`SIM-GEOMETRIC`。  
之后：核验CAD/工艺曲线/布局为`REAL-GEOMETRY`；只有真实执行记录才升级到`REAL-HISTORICAL`。  
需求：工件/工位坐标系、连续路径与方向/速度、机器人基座/TCP/关节限制、工具和夹具几何、禁入区、交接/优先级/时间窗及历史程序/日志。

## 6. 执行顺序

严格执行A0→A1→A2→A3→A4→A5→A6。A0–A4构成第一篇；A5是毕业论文路径/执行接口；A6是可信度增强而非提前宣称真实部署。详见`docs/06_pre_advisor_execution_plan.md`和`docs/09_masters_thesis_work_packages.md`。
