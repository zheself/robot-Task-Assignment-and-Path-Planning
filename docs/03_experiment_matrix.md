# 严格实验矩阵

## 1. 研究假设

- H1：逐点监督补偿改善同域误差，但在连续路径中可能产生不平滑或不可执行补偿。
- H2：局部路径 residual + Jacobian 投影降低整轨迹误差和动作变化率。
- H3：训练域校准的 drift/noise randomization 改善跨日期/偏置/延迟鲁棒性。
- H4：硬安全投影降低 joint-limit、singularity、IK failure，不以显著精度损失为代价。
- H5：完整方法优于普通 residual SAC/TD3；否则不能宣称 mechanism guidance 有效。

## 2. 主方法对比

| ID | 方法 | 学习 | 连续策略 | 安全投影 |
|---|---|---|---|---|
| B0 | No compensation | 否 | 否 | 是 |
| B1 | Mean bias | 训练域统计 | 否 | 是 |
| B2 | Supervised error prior | 监督 | 逐点 | 否/报告失败 |
| B3 | Projected supervised prior | 监督 | 逐点 | 是 |
| B4 | ILC | 重复 trial | 是 | 是 |
| B5 | Residual SAC | RL | 是 | 仅 action box |
| B6 | Residual TD3 | RL | 是 | 仅 action box |
| M1 | Mechanism-guided residual RL | RL | 是 | Jacobian structured |
| M2 | M1 + hard safety/OOD gate | RL | 是 | 完整，提出方法 |

## 3. 测试场景

每类至少固定一组轻/中/重强度，强度从训练数据或标定误差估计：

```text
S0 in-domain unseen paths
S1 unseen path geometry/curvature
S2 unseen workspace region
S3 cross-date/session error distribution
S4 TCP translation/rotation bias
S5 joint-zero and DH perturbation
S6 action delay and low-pass response
S7 observation/measurement noise
S8 combined shift (pre-registered, not tuned on test)
```

跨日期真实点只能直接评估 prior 的误差预测/点补偿。RL 的连续跨日期结果仍是把 held-out
误差分布注入仿真后的结果，应写作 “held-out-date-calibrated stress test”，除非有对应
真实连续轨迹。

## 4. 指标

Accuracy：

```text
path RMSE, MAE, p95, max [mm]
normal/tangential/binormal error [mm]
endpoint error [mm]
fraction within tolerance
```

Control quality：

```text
mean/max compensation magnitude [mm]
first difference / second difference / jerk
path length and completion
```

Safety：

```text
joint-limit margin and violation count
Jacobian condition number / singularity events
IK failure and projection rate
unreachable command rate
```

Robustness：相对 S0 的误差退化率、worst-case 与 CVaR。Efficiency：训练 transitions、
wall-clock、推理延迟。

## 5. 统计协议

- RL 至少 5 个训练 seed；最终每 seed 使用相同的固定 scenario seeds。
- 主表报告 mean、std、bootstrap 95% CI；配对场景用 paired bootstrap/Wilcoxon。
- 对比以 episode/path 为独立样本，不能把每个 timestep 当独立重复。
- 所有方法使用相同 prior、split、路径、扰动和安全执行接口。
- test suite 在主实验前冻结；调参只看 validation suite。

## 6. 消融

```text
A1 remove supervised base
A2 world-XYZ action instead of local path frame
A3 remove error/action history
A4 remove Jacobian mechanism features
A5 penalty-only safety instead of hard projection
A6 remove OOD/uncertainty gate
A7 no calibrated randomization
A8 one prior model vs prior ensemble
```

## 7. 最小论文闭环

论文可提交的最低证据：

1. 一台机器人、完整数据卡和无泄漏跨 session/date split；
2. verified FK/Jacobian 与数据一致性；
3. B0--B4 与至少一种 residual RL；
4. 5 seeds、S0--S7、full method ablation；
5. 清楚区分真实点预测、data-calibrated continuous simulation；
6. 最好增加一组未用于任何校准的历史补偿前后 case study。

