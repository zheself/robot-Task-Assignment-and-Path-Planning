# 数据清洗、数据卡与严格划分

## 1. 原始数据原则

- 旧仓库 `data/` 只读，不复制 657 MB 二进制依赖包到新项目。
- 每个源文件记录相对路径、大小、SHA256、header、行数、机器人、日期和语义状态。
- `data_all.csv` 与其组成文件二选一；默认保留子文件以保留 session/path 分组信息。
- 不确定字段保留原值并标为 `unverified`，不静默重排。
- 统一后的内部单位为 rad/m/s；原始列和转换元数据同时保留。

## 2. 第一轮审计

对每个 CSV 输出：

```text
encoding, delimiter, row_count, columns
q min/max and finite count
nominal/real workspace range
declared XYZ error median/p95/max
last-three-columns-as-XYZ error median/p95/max
duplicate rows and duplicate file hashes
date/session/path identifiers
merged-file warning and column-order warning
```

人工核查队列：

1. `UR5_DATA/data08.csv`：确认 header 错还是数据值顺序错。
2. 根 `train00/test00`：确认机器人、单位、frame 和异常误差原因。
3. KR210 `data_all` 与 `data01...data011` 的包含关系。
4. KR240 `建模数据1...5` 与汇总文件的包含关系。
5. “手摇10个/验证/理论/补偿前后”与 MPI/TXT/CAM 的一一对应。
6. 现有师兄论文实际使用过的文件和 split，避免创新与测试复用。

## 3. 标准化表结构

建议 processed Parquet 一行一个测量点：

```text
sample_id
robot_id
collection_date
session_id
path_id
source_file_sha256
source_row

q_rad[6]
x_nominal_m[3]
orientation_nominal        optional, representation recorded
x_measured_m[3]
orientation_measured       optional
error_m[3] = x_measured_m - x_nominal_m

is_sequential
measurement_device
frame_id
unit_status
semantic_status
split
```

CSV 的宽列在处理层可保持独立标量列，模型输入时再组装向量。

## 4. 划分策略

所有 split 都在窗口化和标准化前完成。

### UR5 候选

- `train_domain`：经确认的早期基础批次，排除 `data_all` 和问题文件。
- `validation_domain`：不同文件/工作空间块或 2025-08-06 的一部分 session。
- `test_cross_session`：2025-08-07 或另一完整未见 session。
- `test_semantic_holdout`：`data08` 仅在列语义确认后作为独立测试或剔除。

具体日期角色必须依据元数据确认，不能仅凭目录名自动决定。

### KR210 候选

- 日期 A（2025-05-22）只用于误差 prior 和 simulator calibration。
- 日期 B（2025-05-29）只用于跨日期测试。
- 同日期再按原始子文件/path group 做 validation。
- 不同时使用 `data_all` 与子文件。

### KR240 候选

- 2025-07-17 建模文件只用于训练/验证。
- 2025-07-18 验证文件保持为 test-only。
- 不根据 test 指标反复调 reward 或 simulator randomization。

## 5. 防止双重使用

需要三层隔离：

```text
误差模型训练：只见 train groups
RL 训练：只访问由 train groups 校准并随机化的 simulator
最终测试：未见真实 group 统计 + 未见模拟轨迹/scenario
```

真实 test 数据不能用于：

- normalization；
- GP kernel/MLP hyperparameter selection；
- drift/noise幅度选择；
- reward weight tuning；
- early stopping；
- scenario difficulty selection。

如果用 test 日期估计 drift 再测试同一日期，只能称 oracle/stress-test upper bound，不能
作为主结果。

## 6. 连续路径合成

静态点本身可能无可靠时序。连续 episode 的参考路径应分为：

1. `interpolation paths`：在训练工作空间内连接可达点；
2. `analytic paths`：line/arc/S-curve/spline，限制在训练域 convex hull 附近；
3. `held-out workspace paths`：进入未见工作空间但保持机器人可达；
4. `historical paths`：若 CAM/MPI 能解析，完全保持为外部 case study。

路径生成不应把随机排列的静态点伪装为真实采样轨迹。所有合成路径明确标注
`synthetic_reference=true`。

## 7. 数据增强参数估计

- `e_static`：仅 train groups 的 cross-fitted supervised model。
- `measurement_noise`：同一点重复测量差；没有重复时只能把模型残差作为上界近似。
- `b_episode`：同点/近邻点跨日期的误差差；没有配准点时用 distribution shift 而非
  measurement drift 的措辞。
- DH/TCP/zero offset：优先来自标定文件；没有时作为明确的 sensitivity range。

每个随机化分布都在 data card 记录估计方法和证据等级。

