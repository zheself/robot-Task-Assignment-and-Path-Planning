# 第一篇论文证据骨架：A3.5 Pair-Pointer

状态：**开始写作；不运行新模型**  
证据标签：**SIM_GEOMETRIC**  
推荐定位：**Feasibility-aware autoregressive decoding for learned continuous-process allocation**

## 1. 固定研究问题与核心结论

确认性问题是：在相同异构图编码器、训练数据、监督信号和硬可行性
mask框架下，动态自回归atomic-unit–robot Pair-Pointer是否比静态并行
decoder提高最终未修复候选的verifier coverage？

可以正式写：

> 在相同异构图编码器、训练数据、监督信号和硬可行性mask框架下，
> 引入动态状态更新与自回归atomic-unit–robot成对构造，将连续工艺分配
> 候选的verifier coverage从40.28%提高到65.05%，平均配对提升24.77个
> 百分点；该提升在task-group层面具有统计显著性，并在全部三个seed和
> 六个预注册难度单元上保持非负方向。

该结果支持动态、自回归构造相对matched static decoding的增益，也支持在
解码过程中显式维护负载、完成时间、位置、前驱和共享资源状态。由于两类
decoder均使用hard mask，不能把全部差异归因于hard mask本身；也不能把
SIM_GEOMETRIC结果推广为任意工位或真实产线结论。

## 2. 论文必须主动回答的反例

`hybrid_load_balanced`以68.75% coverage和0.0201 s中位时间同时优于
Pair-Pointer的65.05%和0.4460 s，在总体coverage–runtime平面上严格支配
Pair-Pointer。MILP和LNS coverage也分别达到72.22%和79.86%。

因此第一篇论文不能以“学习方法优于传统方法”为叙事。引言、结果与讨论
应直接回答：当前学习组件的价值是验证动态learned decoding相对静态
learned decoding的结构性增益，而不是已经替代成熟求解器。强优化方法是
能力上界和工程参照。

## 3. 方法贡献的允许范围

1. 连续工艺段—机器人—共享资源异构约束图，而非离散点分配；
2. atomic-unit–robot Feasible-Pair Pointer；
3. 动态负载、完成时间、位置、前驱满足和资源状态更新；
4. softmax前hard pair mask、deterministic greedy rollout；
5. 不变的A1 scheduler和独立verifier，失败保留且无repair；
6. matched decoder、固定checkpoint和untouched task-group统计协议。

训练标签统一称为**heterogeneous solver-generated verified incumbents**，
不得称为LNS expert、全局最优解或真实专家动作。

## 4. 证据层级

### 4.1 确认性主证据

- Pair-Pointer 65.05%，matched static 40.28%；
- group-paired difference +24.77个百分点；
- 95% cluster-bootstrap CI [+18.52,+31.48]；
- one-sided sign-flip p=0.0000099999；
- 三seed差值+22.22、+29.17、+22.92个百分点；
- 六单元差值均非负；
- 零hash、mask、atomicity、witness和访问完整性失败。

### 4.2 工程参照与负面证据

- Pair-Pointer没有对任何强基线达到预注册非劣；
- hybrid load-balanced总体支配Pair-Pointer；
- Pair-Pointer比static慢约73倍，比load-balanced慢约22倍；
- `scale`上Pair-Pointer虽由6.94%提高到31.94%，绝对coverage仍低；
- 523个失败行全部被当前分类器归为`schedule_infeasible`，现有日志不足以
  进一步声称具体根因比例。

### 4.3 描述性支持证据

Development增益+27.78个百分点，untouched final为+24.77个百分点，方向
复现但效应略缩小。该比较只用于展示稳定性，不构成新的确认性检验。

## 5. 论文图表清单

| 编号 | 作用 | 文件 |
|---|---|---|
| Fig. 1 | 方法架构与动态状态更新 | `figures/a3_5_sealed_final_v1/method_architecture_dynamic_state.pdf` |
| Fig. 2 | 六难度单元matched coverage | `figures/a3_5_sealed_final_v1/difficulty_cell_coverage.pdf` |
| Fig. 3 | 三seed稳定性 | `figures/a3_5_sealed_final_v1/seed_stability.pdf` |
| Fig. 4 | coverage–runtime Pareto与启发式支配 | `figures/a3_5_sealed_final_v1/coverage_runtime_pareto.pdf` |
| Fig. 5 | 全部保留失败计数 | `figures/a3_5_sealed_final_v1/failure_counts.pdf` |
| Fig. 6 | development与untouched final | `figures/a3_5_sealed_final_v1/development_vs_frozen.pdf` |
| Table 1 | 主结果、强基线和时间 | `a3_5_paper_main_results.csv` |
| Table 2 | seed结果 | `a3_5_paper_seed_stability.csv` |
| Table 3 | 单元结果 | `a3_5_paper_difficulty_cells.csv` |
| Table 4 | claim–evidence–boundary | `a3_5_paper_claim_evidence_boundary.csv` |

图表均为sealed输出的描述性重绘，不改变指标、不重新选择方法、不形成新的
确认性结论。PNG用于快速审阅，PDF用于论文排版。

## 6. 建议论文结构

```text
1 Introduction
  连续工艺段分配与静态并行learned decoding的局限
  明确不声称强求解器优势
2 Related Work
  连续任务MRTA、异构图学习、自回归组合解码、solver warm-start
3 Problem Formulation
  atomic unit、机器人、资源、前驱、窗口、verified coverage
4 Method
  heterogeneous encoder、dynamic Pair-Pointer、hard mask、state update
  unchanged scheduler/verifier
5 Experimental Protocol
  heterogeneous verified incumbents、matched checkpoints、grouped split
  development与一次sealed final、强基线与运行预算
6 Results
  confirmatory matched-decoder result
  cell/seed stability
  coverage–runtime与hybrid dominance
  retained failures
7 Discussion and Limitations
  为什么动态decoder有效、为什么尚不能替代heuristic/LNS
  SIM_GEOMETRIC、proxy scheduler、粗粒度失败标签、无真实部署
8 Conclusion
  支持decoder hypothesis；strong-baseline superiority未支持
```

## 7. A4只保留为新研究问题

A4尚未启动。若未来单独预注册，推荐假设为：

> 在完全相同的repair/LNS预算下，Pair-Pointer warm-start是否比matched
> static、hybrid load-balanced、MILP incumbent和cold start更快达到首个
> 可行方案或目标质量？

公平矩阵必须给所有初始解相同repair算法、时间/迭代预算、停止条件和目标
函数，并报告time-to-first-feasible、固定预算coverage、time-to-target、
最终代价、assignment修改数和搜索节点/迭代数。A3/A3.5 frozen永久关闭，
不得用其选择A4方法、预算或阈值。

## 8. A4a实施后的状态

上述问题已作为独立development-only协议`a4_warm_start_pilot_v1`实施，但没有产生
可用于论文的方法比较结论。唯一一次validation暴露MILP fallback标签污染和
fixed-time cutoff未保留截止前可行incumbent两项完整性缺陷，故按预注册规则分类为
`A4A_PRIMARY_EVALUATION_INVALID_STOP`并关闭。固定50迭代结果仅是描述性诊断：
Pair-Pointer三seed平均coverage为77.43%，static为67.71%，load-balanced为77.08%；
这些数字不能替代失效的主时间预算证据，也不能用于宣称warm-start superiority。
第一篇论文证据骨架不因此改变，任何后续修正研究都必须新立协议、新数据和新门槛。

## 9. A4b新问题与当前边界

A4b不再比较不同initializer，而把差异限定为相同`hybrid_load_balanced`、repair、
acceptance、scheduler/verifier、seed和预算下的destroy-set selection。新的
development-only基础已经实现，但没有训练Neural LNS。1秒group coverage中random
与ALNS均为62.5%，3秒round-robin为66.7%而ALNS仍为62.5%；因此当前只能写“建立了
可审计普通搜索与数据接口”，不能写学习邻域选择有效。动作是
`HOLD_A4B_LEARNED_DESTROY_TRAINING`。这一结果不回填、修正或重新解释A3.5 sealed
final；第一篇确认性主张仍严格限定为Pair-Pointer相对matched static decoder的提升。

后续审计发现上述v1 fixed-iteration/单operator选择行使用了被3秒截断的trace，故只
保留fixed-time描述。恢复协议`a4b_ordinary_lns_dev_v2`随后完成独立96/48
train/development数据、fail-closed train/label门禁、六cell development和replay。
最终768条trace与2,304行metric完整；四种方法在0.5/1.0/3.0秒均为72.92%且
identity-wise完全相同。exact-30下ALNS为76.04%、random为72.92%，但三个额外恢复
只在21.99--155.53秒出现，fixed-time中位仅完成0--1个邻域。该结果可作为普通搜索
瓶颈和未来协议设计的development证据，不能成为learned destroy、ALNS superiority
或第一篇论文的新确认性学习主张，也不改变A3.5主张或强基线差距。完整闭环见
`docs/37_a4b_v2_development_closure.md`。

## 10. A4b后续方向与A3.5的关系（非第一篇确认性结果）

A4b后续问题已纠正为Pair-Pointer引导求解器搜索，而非无来源的通用learned
destroy网络。未来主实验从完全相同的`hybrid_load_balanced`方案开始，仅比较
ordinary random/handcrafted/ALNS与Pair-Pointer-derived destroy-set selection；
repair、scheduler/verifier、acceptance、seed、迭代和端到端预算全部相同。

“Pair-Pointer-derived”必须有可审计的A3.5来源：冻结heterogeneous encoder或
unit/robot compatibility表示复用、声明的权重迁移、冻结模型蒸馏，或仅
atomic-unit–robot pointer结构继承。最后一种只能主张结构沿袭，不能主张继承了A3.5
权重或能力。matched static与frozen Pair-Pointer initializer仅用于后续完整3×2析因，
不得与不同guidance交叉混杂后声称因果收益。

这段未来方向不扩展本文件的确认性证据。A3.5主张仍只有Pair-Pointer相对matched
static decoder的sealed提升，并必须保留hybrid/MILP/LNS更强的披露。V2 fixed-time
只有0--1个已完成邻域，当前仍为`HOLD_A4B_LEARNED_DESTROY_TRAINING`。下一步只能
单独授权新ID/seed的ordinary-search semantic-parity recovery；即使通过，也需再次
授权Pair-Pointer-derived destroy-only pilot。计划见
`docs/28_a4b_neural_lns_research_plan.md`与
`docs/38_a4b_pair_pointer_guided_search_protocol_draft.md`。

A3.5 sealed checkpoint不因该未来方向自动开放。若Protocol P未另获固定hash只读加载
授权，只能研究Pair-Pointer architecture-derived guidance；不得读取sealed数据或
声称复用learned representation。未来成功层级也不能跳跃：同框架ordinary方法、
最强order-aware LNS、MILP/hybrid coverage-runtime envelope依次报告，只有有效匹配
下真正超过强基线才能写“优于传统方法”。这些均不是本篇sealed结果的新主张。
