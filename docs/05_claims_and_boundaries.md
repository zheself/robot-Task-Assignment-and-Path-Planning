# 结论与表述边界

每张图表、表格和摘要结论必须携带相应证据标签：

```text
SYNTHETIC       无真实校准的程序化实例
SIM-GEOMETRIC   程序化曲线、工作单元、机器人/资源几何与时序约束
REAL-GEOMETRY   已核验CAD/工艺曲线/布局，未必有真实执行日志
REAL-STATIC     真实离散定位测量
SIM-CALIBRATED  由真实定位数据校准的执行仿真
SIM-STRESS      人为但预注册的扰动测试
REAL-HISTORICAL 已确认语义的真实连续/前后记录
```

## WP-A allocation/scheduling

A3.5可以写：在相同hetero-GNN、训练数据、监督信号和hard-mask框架下，
动态自回归Pair-Pointer将sealed SIM_GEOMETRIC verifier coverage从40.28%
提高到65.05%，group-paired提升24.77个百分点且三个seed方向一致。

A3.5不能写：hard mask单独导致该提升；Pair-Pointer优于或非劣于强基线；
学习分配器具有工程必要性已被证明。`hybrid_load_balanced`在总体coverage和
runtime上支配Pair-Pointer，MILP/LNS coverage也更高，必须在摘要、结果或
限制中清晰披露。

可以写：在同等约束和预算的几何实例上比较连续段分配/排序/调度；以异构图、mask和确定性repair生成并验证候选；报告时间、质量、违例和失败；在数据确认后做held-out真实几何案例。

不能写：PointNet解决分配；静态定位CSV训练出分配器；保守共享区/轨迹包络等于真实碰撞安全保证；合成实例等于真实汽车产线部署；GNN未经验证天然满足硬约束；或没有物理模型却提升应力、接触力、板材塑性、滚边质量。

## WP-B path planning

可以写：在所声明的机器人模型/几何近似下评估可达性、姿态/方向约束、时间参数化、路径代价和代理冲突检查，并将这些量反馈至分配代价。

不能写：基础IK、距离或包络检查已验证完整机器人—机器人—夹具碰撞；未核验URDF/CAD/TCP下的真实可执行性；或没有控制器/真实日志便宣称工位节拍和在线重规划性能。

## WP-C execution/learning

可以写：真实静态定位数据校准了prior；安全投影在模型层限制动作；RL/学习只在明确的仿真/执行条件下与强基线比较。

不能写：静态CSV是offline RL transition；真实机器部署RL；sim-to-real成功；reward penalty是安全保证；跨日期差必然是物理漂移。当前SAC/TD3未击败projected supervised prior，不能宣称RL优越。

## External physical interface

可以接入版本化的速度/姿态界限、连续/交接规则、优先级、夹具禁入区和物理团队提供的风险/质量代价。必须注明提供方、版本、单位、适用范围和是否验证；不得将其改写成本项目已建的力、应力、接触或塑性模型。

## A4a warm-start边界

可以写：A4a建立了统一initializer adapter、identical-repair trace和独立开发数据，
并在一次validation中发现了两个评测完整性缺陷，因此按预注册规则停止。

不能写：A4a形成有效的fixed-time方法排名；Pair-Pointer warm-start已被证明优于或
劣于任何initializer；有缺陷的MILP标签行代表真实MILP incumbent；或用fixed-50
描述性结果替代预注册主指标。若未来修正，必须使用新协议和新数据，不能重跑本协议。

## A4b Pair-Pointer-derived guided LNS边界

可以写：A4b修复了新评测器的initializer provenance与anytime cutoff语义，建立了
八种atomic-unit destroy operator、共享repair/verifier的ordinary LNS/ALNS、
可重放trace与`search-generated neighborhood improvement labels`接口；全部证据为
`SIM_GEOMETRIC development-only`。

不能写：ALNS或任何学习邻域方法已优于random/handcrafted LNS；当前标签是真实专家
动作；A4b证明Pair-Pointer initializer价值；或development结果支持真实机器人、
碰撞安全、动态插单、路径规划或物理质量。当前没有Neural LNS checkpoint，也没有
A4b frozen benchmark；`HOLD_A4B_LEARNED_DESTROY_TRAINING`不得改写为方法通过。

补充边界：v1 fixed-iteration和train-selected operator结果不得再引用为有效预算/选择
证据。v2已经完整结束，可以写“768条trace、2,304行metric、384/384 exact-30与
12,484-step replay均通过；四种方法在0.5/1.0/3.0秒均为72.92%且identity-wise完全
相同；exact-30下ALNS为76.04%、random为72.92%”。必须同时写明ALNS仅多恢复三个
identity且首次恢复在21.99--155.53秒，fixed-time中位只完成0--1个邻域，因而不支持
ALNS superiority或直接训练learned destroy。任何更强ordinary-search recovery必须
另立新协议、新seed/ID，只能用fixture和全新train选择参数，不得在已观察的v2
development上调参。

未来A4b学习问题只能写为：在相同`hybrid_load_balanced` initializer、repair、
candidate cap、scheduler/verifier、acceptance、seed、迭代和端到端时间预算下，
Pair-Pointer-derived destroy-set selection是否优于ordinary LNS/ALNS。所谓
Pair-Pointer-derived必须披露为冻结表示复用、权重迁移、蒸馏或仅结构继承之一；普通
GNN destroy selector不能仅改名。主实验不得同时改变initializer和guidance；matched
static和frozen Pair-Pointer initializer只能进入后续完整3×2析因实验。

任何模型训练前必须另立ordinary-search semantic-parity recovery：新ID/seed/
train-development namespace，候选集合及顺序、selected state、verifier与随机转移签名
完全一致，并通过最低completed-neighborhood、initializer-failure recovery和
operator-dependent fixed-time ALNS门禁。即使recovery通过，也只能另行申请
Pair-Pointer-derived destroy pilot，不会自动解除HOLD。详见
`docs/38_a4b_pair_pointer_guided_search_protocol_draft.md`。

A3.5 sealed closure不自动授权A4b加载checkpoint。未来只有在新Protocol P得到明确
授权后，才可按固定file/state-dict hash只读加载，并且只能作用于全新Protocol-P数据；
禁止读取A3.5 frozen instance/witness/final trace，禁止重评、挑选、ensemble、替换、
微调或test-time adaptation。未获得该权限时只能称
`Pair-Pointer architecture-derived`，不能称复用A3.5 learned representation。

未来primary方法预先固定为“冻结Pair-Pointer encoder/compatibility representation +
新search-state destroy head + 不变repair”，其继承参数不得fine-tune；迁移、蒸馏和
结构继承只能是预注册消融，不能看结果后替换primary。所有graph/encoder/state/
selector/mask/canonicalization/repair/verifier成本必须计入端到端时间，缓存只能在计时
后建立并限于同一trace，普通方法使用相同数据准备边界。

成功表述按层级推进：先在同一搜索框架内超过random/handcrafted/ALNS；再单列最强
`order_aware_lns`；再报告MILP与hybrid的coverage-runtime envelope。不同backend只能
是强参考而不是相同因果公平性声明；只有有效匹配协议真正超过强基线后，才能写
Pair-Pointer guidance优于传统方法。
