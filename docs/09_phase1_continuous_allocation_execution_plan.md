# Phase 1：连续工艺多机械臂任务分配与调度执行计划

状态：A0 `PASSED_AND_FROZEN_V1`；A1 `PASSED_FOUNDATION_V1`；A2 `PASSED_AND_FROZEN_V4`；A3 `FINAL_FAILED_BASELINE_FLOOR_V4_CLOSED`；A3.5 `DECODER_HYPOTHESIS_SUPPORTED_FINAL_CLOSED`；A4a `PRIMARY_INVALID_STOPPED`；A4b普通搜索基础 `COMPLETE_NEURAL_TRAINING_HELD`；A4动态扩展 `NOT_STARTED`  
适用范围：第一篇小论文 **Feasibility-Aware Graph Learning for Continuous-Process Multi-Robot Task Allocation and Scheduling**。  
计划周期：12个有效工作周；按验收门推进，不以日历日期强制绑定。  
上位规划：`docs/06_pre_advisor_execution_plan.md`、`docs/08_continuous_process_multi_robot_plan.md`、`docs/09_masters_thesis_work_packages.md`。

## 1. Phase 1 边界和完成定义

Phase 1只解决连续工艺段的任务切分、分配、机器人内排序、调度和确定性约束修复。一个任务段是具有曲线几何、方向、时长、前驱/交接和资源语义的工艺对象，绝不是独立采样点。

本阶段明确不做：应力/接触/板件塑性/质量模型；完整机器人—机器人—夹具几何碰撞规划；高保真执行控制；视觉；端到端RL或MARL。共享区和轨迹包络只作为可解释的**保守冲突代理**，不作碰撞安全保证。

Phase 1完成的严格定义：

1. 从版本化配置和固定seed出发，能生成连续曲线—工位—机器人实例，构造不可泄漏的train/validation/frozen-test manifest；
2. 能运行并报告贪心、负载均衡贪心、Hungarian+排序、CP-SAT/MILP及适用的LNS/ALNS；
3. 能训练可行性掩码的异构图模型作为solver imitation或warm-start，并只以validation选择模型；
4. 所有输出经过确定性验证/repair，失败和修复前后约束状态均被保存；
5. 对相同冻结实例、相同约束和相同预算，自动报告质量、可行性、repair与时间指标；
6. 所有结果标为`SYNTHETIC`或`SIM-GEOMETRIC`，不包含真实产线或物理质量结论。

## 2. 目标目录与接口合同

以下目录在相应周创建；本文件不要求现在创建它们。

```text
configs/allocation/
  schema_v1.json, constraints_v1.json, benchmark_v1.json, experiments/
data/cards/allocation/                 # 数据卡、生成假设与证据标签
data/manifests/allocation/             # immutable split + SHA-256
data/fixtures/allocation/              # 10–20个小型可读JSON实例
src/safe_residual_rl/allocation/
  schema.py, constraints.py, generation.py, oracle.py, verifier.py
  solvers/, graphs/, models/, repair/, evaluation/
tests/allocation/
reports/phase1_allocation/             # 紧凑JSON/CSV/Markdown/figures
outputs/phase1_allocation/             # ignored: 模型、solver traces、原始结果
```

### 固定对象

```text
ProcessSegment:
  id, parent_curve_id, segment_index, sampled_curve, start/end pose,
  direction/tangent, length, process_duration, tool_requirement,
  priority, release/due window, predecessors, handoff_policy, shared_zones

RobotSpec:
  id, base_pose, capabilities, availability_window, kinematic_model_id,
  initial_state, nominal_speed limits

ResourceSpec:
  id, type(shared_zone/fixture/no_go), capacity, availability_window

AllocationPlan:
  assignment, robot_orders, planned_start/end, solver_status,
  violated_constraints, repair_actions, diagnostics
```

### 跨模块稳定接口

```text
estimate_edge(robot, segment, context) ->
  feasible, travel_time, process_time, path_length,
  kinematic_risk, conflict_proxy, confidence, diagnostics

verify_plan(instance, allocation_plan) ->
  feasible, violations, proxy_conflicts, objective_terms

repair_plan(instance, candidate_plan, budget) ->
  repaired_plan, repair_status, repair_actions, remaining_violations
```

Phase 1的`estimate_edge`可以使用解析距离、速度上限和简化可达性；不得假装使用未验证的完整IK、URDF碰撞或物理模型。未来WP-B/C可替换这个实现，但不得改变字段语义。

## 3. 按周任务与验收

| 周 | Gate | 主要任务 | 当周冻结/验收 |
|---|---|---|---|
| W1 | A0 | 归纳连续段、机器人、资源和计划schema；写10个手工JSON fixture | **完成**：12个fixture；schema字段、单位、ID唯一性、父曲线连续性验证通过 |
| W2 | A0 | 约束字典：工具、可达性、唯一分配、前驱、交接、时间窗、共享区；定义目标函数 | **完成**：`schema_v1`、`constraints_v1`和SHA-256 manifest冻结 |
| W3 | A1 | 实现解析曲线特征、简化edge cost/reachability oracle和mask | **完成**：稳定原因码、代理证据卡和10项W3测试；完整项目57项测试通过 |
| W4 | A1 | 实现verifier及贪心/负载均衡贪心/Hungarian+顺序；建立CP-SAT/MILP小实例oracle | **完成**：确定性验证器/代理调度器和三种启发式统一输出；可行、掩码不可行和人工反例与预期一致 |
| W5 | A1 | 加入时间窗与共享区代理调度；CP-SAT时间限制、gap和状态记录 | **完成**：assignment MILP记录status/bound/gap/time limit；18项W4–W5测试及12-run JSON/CSV/Markdown smoke报告通过；完整项目75项测试通过 |
| W6 | A2 | 实现参数连续曲线、工位、机器人和约束的程序化生成器 | **完成**：line/arc/B-spline/closed-loop、2–8机器人及约束难度由固定seed复现，全部标为`SIM_GEOMETRIC` |
| W7 | A2 | 定义规模/难度族，构建按工件、布局、任务实例分组的split manifest | **v4完成并冻结**：408实例/216独立组/18难度单元；402个普通实例带独立哈希、可复核的构造式A1代理witness，6个负控制；零schema/hash/witness/split泄漏 |
| W8 | A2 | 批量运行非学习基线，生成质量—时间基线表、失败库和candidate solver labels | **v4验收通过**：8方法×408实例=3,264 runs；所有预注册完整性、覆盖率、状态和负控制检查通过；失败方法运行完整保留 |
| W9 | A3 | 图构建、特征标准化、异构GNN/图Transformer最小模型和masked decoder | **foundation v1通过**：192 train拟合词表/归一化，48 validation只评估；三类纯PyTorch模型、hard mask、atomic-unit decoder、11项A3测试和双重复smoke通过；frozen/stress未访问 |
| W10 | A3 | 全量solver imitation和validation checkpoint/family selection | **开发门通过**：3模型族×3 seeds、192 train/48 validation完整运行；`edge_mlp`以95.8% coverage入选，但低于强基线97.9–100%；frozen/stress未访问，最终评测须另行预注册 |
| W11 | A4 | CP-SAT repair、局部插入/reorder、LNS/ALNS；规模泛化与OOD布局测试 | 报告repair前后违例、成功率和时间；失败实例可复现 |
| W12 | A4 | 动态到达/机器人不可用/时间扰动的**重规划**评测；写Phase 1报告和论文提纲 | scope、配置、数据manifest和方法选择冻结；决定是否具备投稿条件 |

W12的动态重规划仍使用重新调用优化/启发式/GNN warm-start；不引入RL。若没有可靠动态需求或GNN收益，保留静态评测和失败结果，不扩展范围。

## 4. A0–A4 的测试清单

| 类别 | 必测内容 |
|---|---|
| Schema | JSON round trip；单位/有限值；ID、父曲线和segment索引唯一；不可拆分/交接规则；版本兼容性 |
| Constraints/masks | 工具不匹配、不可达、前驱环、时间窗、资源容量、共享区重叠；每个mask理由可追溯 |
| Oracle/verifier | 固定输入确定性；人工fixture中的可行/不可行期望；目标项可加和；计划唯一分配 |
| Solver | CP-SAT/MILP状态、超时和gap记录；贪心退化案例；Hungarian后排序可行性；LNS/repair不丢任务 |
| Generator/split | 固定seed可复现；不同seed有受控变化；工作件/布局/父曲线泄漏测试；训练专属归一化 |
| GNN | 节点/边排列不变性测试；invalid-edge logit掩码；decoder不产生重复/缺失任务；checkpoint不读frozen test |
| Repair/evaluation | repair不引入新硬违例；修复前后指标一致；相同实例/预算的配对比较；失败输出保留 |
| Reporting | 每次运行保存config、seed、manifest hash、版本、时间预算、指标、solver状态和失败案例 |

在任何A3训练前，A0/A1/A2测试必须全绿；Phase 1实现完成后再运行完整测试集。测试通过不等于真实工位可执行或碰撞安全。

## 5. 实验矩阵

### 实例因素

| 因素 | 训练/validation | Frozen test / stress |
|---|---|---|
| 机器人数 | 2–6 | 2–8，含未见机器人数量 |
| 任务段数 | 8–50 | 8–100；小规模子集用于optimality gap |
| 几何 | 直线、圆弧、B-spline、闭合边界 | 未见曲线组合、长度分布、方向模式 |
| 工位 | 规则基座、有限共享区 | 未见布局、紧窄共享区、资源瓶颈 |
| 工艺约束 | 前驱、工具、交接、时间窗 | 更密前驱、紧时间窗、不可拆分比例变化 |
| 动态扰动 | 可选轻度变化 | 新任务、机器人不可用、时间估计扰动 |

### 方法与评测矩阵

| 方法 | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|
| Greedy | ✓ | ✓ | 比较 | 比较 |
| Load-balanced greedy | ✓ | ✓ | 比较 | 比较 |
| Hungarian + sequencing | ✓ | ✓ | 比较 | 比较 |
| CP-SAT/MILP | ✓ | ✓ | teacher/oracle | repair/oracle |
| LNS/ALNS |  | baseline | baseline | 强基线 |
| Masked GNN/graph Transformer |  |  | 主方法 | warm-start/replanning |

所有单元报告以下指标：可行率、Cmax、负载方差、总travel/setup、priority-weighted tardiness、修复率、各类违例、共享区代理冲突前后、运行时间；小规模加optimality gap。每个frozen配置至少使用预注册实例和多个生成seed，报告聚合与逐实例失败。

## 6. 冻结标准

| 时点 | 必须冻结的对象 | 不得再用的信息 |
|---|---|---|
| A0结束 | schema/约束版本、fixture语义、目标项定义 | 不能因某方法失败改写约束 |
| A2结束 | 生成器版本、训练/validation/frozen manifest、instance seeds、预处理 | frozen布局、父曲线和实例不能用于调特征/权重 |
| A3开始 | 模型族、损失、训练预算、validation选择指标 | 不能按frozen test挑模型或seed |
| A4开始 | repair和LNS预算、动态事件协议、主指标权重 | 不能只对GNN放宽repair或时间预算 |
| W12 | 第一篇方法、场景、统计和失败列表 | 后续WP-B/C结果不能回填为第一篇主结果 |

任何冻结后修改均需新版本号、新manifest hash、修改理由和完整重跑；不能覆盖原结果。

## 7. Gate验收与停止规则

- **A0通过**：schema和约束字典覆盖全部第一篇硬约束；每项有正/反fixture和单元测试。
- **A1通过（2026-08-09，foundation v1）**：oracle、verifier和四类非学习基线在小实例上可解释；MILP的optimal/infeasible/limit状态及bound/gap/预算字段可完整记录。当前MILP仅优化assignment proxy load并接确定性调度，不宣称联合调度或运动规划最优；正式质量—时间基准仍属于A2。
- **A2 pilot通过（2026-08-09，pilot frozen v1）**：生成器、数据卡、grouped split、实例/manifest hash和13项A2测试完成；95-run基线表、23个保留的`schedule_infeasible`失败、candidate label及访问用途已冻结。该失败状态只表示当前assignment-first候选无法通过统一代理调度，不证明完整问题不可行。**A2正式门尚未通过**：仅8个train实例不足以训练/评价论文GNN；必须先冻结paper-scale v2的样本量、难度单元和可行性政策并重跑，之后才允许A3。
- **A2 paper v2失败（2026-08-10，frozen）**：v2样本量、难度、可行性政策和统计均在运行前冻结；完整2,040-run实验只失败`minimum_frozen_cell_candidate_coverage`。不得事后降低阈值或复用已观察的v2 frozen作新方法最终测试。先以train/validation改进deadline/resource-aware非学习调度，再以全新v3 frozen重新验收；A3继续阻塞。
- **A2 paper v3失败（2026-08-09，frozen）**：先仅用v2 train/validation开发minimum-slack调度、hybrid order selection和不依赖可行初始解的order-aware LNS，再以新seed、新前缀任务组冻结v3。完整3,264-run评测的完整性、train/validation、负控制和五个frozen单元均通过，但OOD-scale为11/24（45.8%），低于预注册50%。不得把“只差一个实例”改写为通过，也不得用v3 frozen调预算。下一轮必须加强scale-aware assignment/sequencing与小规模联合参考，并使用全新v4 frozen组；A3继续阻塞。
- **A2 joint/beam开发未通过（2026-08-09）**：小规模联合assignment/sequencing参考已能区分完整最优、限时incumbent、无incumbent限时、完整不可行与超规模；五个有效fixture及8个v3-train小例协议通过。beam-ALNS显式分支机器人/共享资源顺序，并在v3 validation达到47/48，但与order-aware LNS逐单元覆盖率完全持平，未满足“至少一个validation单元严格提升”的预声明门槛。因此没有生成v4，不得以train改善或conditional score替代失败门槛；A3继续阻塞。
- **A2 assignment-beam开发未通过（2026-08-10）**：系统分支atomic-unit机器人分配并接sequence beam的方案只访问v3 train/validation；validation precedence为10/12，低于order-aware LNS的11/12，违反“不允许任何cell退化”的预声明门槛。该方法被排除，不能以v4结果补选。
- **A2 paper v4通过并冻结（2026-08-10）**：v3 train/validation上的240实例readiness先证明构造式witness协议为100%确定、可验证并保持非时间窗语义；随后预注册全新seed/组的v4。402个普通实例均有隐藏于候选方法的A1代理witness，6个负控制保持设计不可行。完整3,264-run评测通过全部门槛：train 99.5%、validation 100%，六个frozen cell依次为100%、100%、83.3%、91.7%、50%、50%，负控制100%；零schema、泄漏、实例/witness哈希、witness验证或异常状态失败。manifest为`0c98f30e92697ce8b5eca724df0f7d1b7053293df1e792707487ecb6c71b5398`。A2到此关闭，A3解除阻塞；但v4 frozen/stress必须保持evaluation-only，并且通过不代表运动碰撞、真实执行或物理质量证据。
- **A3 W9 foundation通过（2026-08-10）**：在任何训练前冻结`a3_development_v1`；开发工作区只含192 train、48 validation及对应witness，loader硬拒绝frozen/stress。完成规范化异构图、train-only词表/normalizer、edge-MLP/hetero-GNN/graph-Transformer、模型内hard mask、atomic-unit assignment与precedence-aware order decoder。11项A3测试通过；16/8实例四epoch smoke同seed双跑checkpoint hash一致，validation代理可行coverage为7/8。该数值仅为W9工程检查，不能作为论文性能或frozen结论。W10仍须运行全部train/validation、三模型族与三seed，之后另行预注册一次性frozen评测。
- **A3 W10开发门通过（2026-08-10）**：严格按冻结配置完成3模型族×3 seeds；每个seed以validation coverage→conditional proxy score→atomic-unit accuracy选checkpoint，运行和聚合均只访问192 train/48 validation。`edge_mlp`三个seed均为46/48（95.8%），异构GNN均值89.6%，graph Transformer均值88.9%；因此按规则选择`edge_mlp`。它超过弱基线83.3–85.4%，但低于`hybrid_assignment_milp`的97.9%以及`hybrid_load_balanced`/`order_aware_lns`的100%。这通过的是开发完整性与选择协议，不是A3最终门，也不支持GNN优于强基线。frozen/stress仍未访问；下一步必须先冻结一次性最终评测协议。
- **A3一次性最终评测已预注册（2026-08-10，尚未运行）**：固定`edge_mlp`及seed 17/29/43三个checkpoint哈希、五个context/weak基线、三个strong基线、144个frozen-test实例的grouped统计、六个逐难度绝对/相对门槛、stress/负控制政策和五级失败/成功表述。配置SHA-256为`ce574b6b62c2218f8a2f7b3130646444cc00b60ccc69c6822bdde5d3f48ab756`。实现评测器时只能使用fixture和train/validation测试；记录代码哈希后才允许一次sealed访问。当前仍没有A3 frozen结果，A4继续阻塞。
- **A3一次性最终评测失败并关闭v4学习分支（2026-08-10）**：seal后唯一作业完整运行144 frozen-test与24 stress实例、3 checkpoints与8 baselines，共1,848行；所有完整性、哈希、mask/atomic-unit、witness与负控制检查通过。`edge_mlp`三seed总体coverage为55.6%/58.3%/56.3%，均值56.7%；虽略高于最佳context/weak的55.6%，但绝对门槛在dense precedence 40.3%、tight windows 37.5%、scale 15.3%失败，dense precedence相对弱基线门也失败。三个strong基线总体为68.8%、73.6%、79.2%，配对覆盖差异95% CI均完全低于0。因此分类为`A3_FINAL_FAILED_BASELINE_FLOOR`；不得在v4上重跑、调参或加repair挽救结果，A4等待solver-only或全新未见benchmark的方向决策。
- **A3.5 Feasible-Pair Pointer开发pilot通过继续门（2026-08-11，非frozen）**：在新seed、新ID前缀、与v2/v3/v4零group/instance重叠的`a3_5_pointer_pilot_v1`上，仅生成96 train与48 validation实例；协议和配置在生成前冻结。动作是不可拆分atomic unit与robot的自回归pair，hard mask覆盖可达性、重复选择和前驱，raw候选直接进入不变的A1 scheduler/verifier，不使用repair或RL。五模型族×三seed完整运行；`hetero_gnn_pair_pointer` coverage为79.2%/79.2%/77.1%，均值78.5%，相同编码器静态decoder均值50.7%，六个预注册继续检查全部通过，且零mask、atomicity和dead-end失败。但pointer仍低于MILP 79.2%与order-aware LNS 81.3%，推理中位时间由0.0175 s升至0.400 s。结论只允许写为“值得预注册全新未见最终协议”；本轮未生成frozen/stress，未解锁A4，也不改变A3 v4失败。
- **A3.5 sealed final已预注册但未运行（2026-08-11）**：主假设收敛为相同hetero-GNN下Pair-Pointer相对静态decoder的最终verifier coverage改善，而非超过MILP/LNS。固定已有六个checkpoint及state/file hash，计划生成全新seed/ID的六单元×12 group×2 variant，共72独立group/144实例。主统计为group内平均两个variant和三个匹配seed后的配对coverage差、单侧sign-flip检验及group-cluster bootstrap区间；强优化方法只作第二判据和质量—时间报告。配置SHA-256为`8c4d3cb7cc6e61ee589e98786ab78e77958d09694afb102d7f806eda9c208368`。当前尚未实现/封存最终评测器、未生成frozen、未运行最终评测；A4继续关闭。
- **A3.5 sealed final通过主假设并关闭（2026-08-11）**：严格完成13项targeted、146项非v4回归、validation-only Slurm preflight `941015`、source seal、一次生成`941022`和一次评测`941024`。72 group/144实例的untouched结果为Pair-Pointer 65.05%、matched static 40.28%，group-paired差+24.77个百分点，95% CI [+18.52,+31.48]，单侧p=0.0000099999；三个seed均改善、六个cell无回退、所有完整性检查通过，故分类`A3_5_DECODER_HYPOTHESIS_SUPPORTED`。但MILP/LNS/hybrid load-balanced分别为72.22%/79.86%/68.75%，Pair-Pointer未满足任何强基线非劣条件。论文只能主张动态decoder显著改善同编码器静态decoder，并完整报告其强优化差距和质量—时间权衡；不得重跑或用A4 repair改写结论。
- **第一篇论文证据骨架已建立（2026-08-11，描述性）**：仅消费immutable pilot/final输出，生成主结果、六单元、三seed、coverage–runtime Pareto、失败计数、development-vs-final、claim边界和方法动态状态图。hybrid load-balanced对Pair-Pointer的总体支配被明确展示；523个失败不做超出现有`schedule_infeasible`日志的事后细分。当前进入写作，不运行新模型。A4仅保留为未来“相同repair预算下warm-start增量价值”的新协议问题。
- **A4a warm-start开发pilot关闭（2026-08-11）**：新`a4wsp1`仅含96 train/48 validation，唯一Slurm array产生完整5,616行。聚合动作是`STOP_A4_LEARNING_WARM_START_BRANCH`。事后语义审计发现MILP失败时adapter替换为load-balanced state，以及fixed-time cutoff丢弃预算内已可行incumbent，故主结果分类`A4A_PRIMARY_EVALUATION_INVALID_STOP`，不得补丁重跑或生成frozen。未受影响的fixed-50描述性coverage为Pair 77.43%、static 67.71%、load-balanced 77.08%；Pair init约0.413 s而load约0.024 s。该结果不足以支持学习warm-start工程必要性。
- **A4b evaluator与ordinary LNS/ALNS基础完成（2026-08-13）**：新`a4bnlsd1`只含48 train/24 development实例，无validation/frozen/stress。修复真实initializer assignment provenance与monotonic cutoff，建立8种atomic-unit destroy、共享repair/verifier、ALNS权重、trace及96个等repair-budget候选标签。修正作业`942372`、`942373_[0-5]`全部exit 0；169项非frozen回归通过。1秒独立group coverage中ALNS与random均62.5%，3秒round-robin为66.7%而ALNS仍62.5%。最低非劣门仅以相等通过，且normalized-anytime reference/penalty与细粒度计时尚未冻结，故动作是`HOLD_A4B_LEARNED_DESTROY_TRAINING`，不得自动训练或生成frozen。
- **A4b ordinary-search恢复v2完成（2026-08-20）**：审计确认v1 fixed-iteration和train-selected operator行来自3秒截断trace，不能作为相应证据；fixed-time行保留为描述。新`a4blnsd2`仅含96 train/48 development实例，72个group、六cell均衡、无validation/frozen/stress，144个witness全部通过A1 verifier。v2分离fixed-time/exact-iteration，加入travel-aware结构化repair、细粒度计时、train-frozen primal-integral reference及fail-closed train/label gate。`981071`因旧60秒watchdog fail-closed，串行`983899`在无完整输出时取消；六worker `984111`完成312条trace后仅因NumPy版本字段拼写在元数据阶段失败。另行冻结的exact-hash metadata recovery未执行搜索；`984885`恢复/merge/train/label、`984886` smoke、`984887`六CPU development和`984888` replay/aggregate均exit 0。最终768条trace、2,304行metric、384/384 exact-30与12,484-step replay通过。四方法在0.5/1.0/3.0秒均为72.92%且identity-wise相同；exact-30下ALNS 76.04%、random 72.92%，但三个额外恢复首次发生在21.99--155.53秒，fixed-time中位仅完成0--1邻域。分类为`A4B_V2_DEVELOPMENT_COMPLETE`，但repair-dominated fixed-time不能支持learned destroy训练，`HOLD_A4B_LEARNED_DESTROY_TRAINING`不变。继续时必须另立新ID ordinary-search recovery协议并禁止用已观察development调参。
- **A4b后续方向纠正（2026-08-20）**：未来问题限定为Pair-Pointer-derived atomic-unit destroy guidance。必须明确冻结A3.5表示复用、权重迁移、蒸馏或仅结构继承，普通GNN不能改名。主实验固定hybrid initializer及共享repair/verifier/acceptance/预算；initializer只在后续完整析因中变化。任何模型训练前先通过另行授权的新ID semantic-parity repair recovery和最低search-opportunity/operator-dependence门禁；recovery通过也不自动解除HOLD。
- **A3通过**：学习模型严格使用train/validation完成拟合与选择，mask与verifier有效；冻结测试通过预注册的六个绝对难度门槛、不差于预定义弱基线并完整对比强基线。当前入选模型是`edge_mlp`而非GNN；是否优于强基线不作为最低工程通过条件，但决定论文创新表述。
- **A4通过/投稿门**：等预算下完成repair、尺度和动态协议；若GNN无质量—时间优势，论文仅保留为可扩展性诊断或停止GNN主张，绝不挑选有利实例。

立即停止扩展并回到前一门的情况：发现split泄漏；约束语义无法判定；基础solver/ verifier不一致；GNN通过无效边或repair漏洞取得优势；或结果依赖未声明的物理/碰撞假设。

## 8. Phase 1 报告包

每周产生简短状态条目；每个gate产生JSON、CSV、Markdown和必要图表。W12形成：问题和边界、schema/约束卡、benchmark卡、manifest、基线/主方法表、质量—时间图、ablation、逐实例失败案例、复现命令和明确限制。报告不得把`SIM-GEOMETRIC`写为真实滚边、真实碰撞安全或物理质量改善。
