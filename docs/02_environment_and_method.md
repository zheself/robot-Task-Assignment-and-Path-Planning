# 环境选择与最小可行方法

## 1. 三种环境取舍

| 维度 | 自定义 Gymnasium | MuJoCo | ManiSkill |
|---|---|---|---|
| 与静态误差数据匹配 | 最强，可直接定义误差场 | 中，仍需自建误差层 | 弱，现成任务不匹配 |
| UR5/KUKA 运动学 | 直接复用 DH/适配器 | 需 MJCF/URDF 与标定 | 需机器人资产与 task |
| 延迟/漂移/噪声 | 最透明、易验证 | 可实现但层次更复杂 | 可实现但调试成本高 |
| 关节/奇异性约束 | Jacobian/IK 显式实现 | 内置动力学更完整 | 控制器接口更完整 |
| 碰撞/夹具 | 初期简化 | 强 | 强 |
| 薄板塑性滚边 | 不支持 | 不支持 | 不支持 |
| Phase 1 成本 | 最低 | 中 | 高且与旧路线混淆 |

结论：Phase 1 使用自定义 Gymnasium。只有出现下列需求才升级：

- MuJoCo：需要关节动力学、碰撞、夹具和控制器响应；已有可靠 URDF/MJCF。
- ManiSkill：需要 GPU 并行、大规模视觉/点云或已有 SAPIEN 资产。
- Abaqus/LS-DYNA surrogate：获得真实滚边材料/接触/质量数据后，离线建代理模型；不把
  FE 求解放入 RL step。

## 2. Gymnasium MVP

环境必须是独立于 RL 算法的可测试状态机。

```text
reset:
  sample reference path
  sample one training-domain error prior realization
  sample episode drift/noise/delay
  initialize q from verified IK/FK-consistent pose

step(a_t):
  scale local residual
  convert through local path frame
  project via damped Jacobian/IK and hard constraints
  apply action delay/controller filter
  compute virtual actual TCP through calibrated error model
  produce observation, reward, termination, detailed info
```

`info` 至少记录：raw/projected action、projection reason、q command、condition number、
prior error、drift、noise、tracking error 和 reward terms。测试策略不能访问 prior 分项。

## 3. 基础模型

先做下列静态 error prior：

```text
P0 global mean bias
P1 linear/ridge on q and x_nominal
P2 random forest or gradient boosting
P3 Gaussian process or compact MLP
P4 optional mechanism features: FK discrepancy, Jacobian, workspace features
```

选择 prior 的标准不是同域最低训练误差，而是 group validation、calibration residual 和
跨工作空间稳定性。RL simulator 应至少使用两个 plausible priors 做 model uncertainty
stress test，避免策略只利用单一网络伪影。

## 4. 控制方法

```text
B0 no compensation
B1 global/session mean bias
B2 supervised pointwise compensation
B3 Jacobian/IK projected supervised compensation
B4 ILC for repeated reference path
B5 standard residual SAC
B6 standard residual TD3
M1 mechanism-guided residual RL (structured state + base action)
M2 M1 + hard safe projection + uncertainty/action gating (proposed full)
```

“机制引导”的有效组成必须逐一消融，不能只写在名称中：

- local Frenet/path frame action；
- FK/Jacobian projection；
- supervised base compensation；
- joint/singularity safety projection；
- prior uncertainty or out-of-domain gate。

## 5. 两阶段训练

Stage A：先完成不含 RL 的闭环。

```text
data audit -> FK verification -> prior fit -> path generator
-> no-comp/supervised/ILC -> scenario evaluation
```

Stage B：加入 RL。

```text
Gymnasium checker -> SAC/TD3 smoke -> multi-seed
-> full safe residual -> ablation -> held-out scenario suite
```

如果 B2/B4 已达到误差下限，RL 没有显著收益，应转向“安全与跨漂移鲁棒性”贡献，或
停止把 RL 作为主方法。该决策应由结果驱动。

## 6. 软件依赖

Phase 1：NumPy、SciPy、pandas、scikit-learn、Gymnasium。只有训练 SAC/TD3 时加入
PyTorch 与现代 SB3。明确不安装 ManiSkill/SAPIEN，不复现旧项目模型。

低维 prior 和环境测试用 CPU。多 seed RL 仍可优先 CPU；确认并行环境/GPU 有收益后再
使用 Slurm GPU。不要因集群有卡就扩大模型。

