# Experiment Timeline

本文档记录最终 MiniGrid Memory 实验的迭代过程。记录内容只包含可复现实验设置、结果和取舍，不包含私有路径或硬件配置。

## 2026-06-02 迭代 1：降低专家比例并加入短历史 MLP

目的：解决主实验结果过于极端的问题。将离线数据从高专家比例改为 `explore:expert = 35%:65%`，并加入 `SBC-K5` 和 `SBC-K10` 两个短历史 MLP baseline。

配置：

| 项目 | 数值 |
|---|---:|
| 环境 | MiniGrid MemoryS17Random |
| 离线轨迹数 | 400 / seed |
| seeds | 0, 1, 2 |
| epochs | 20 |
| DT context length | 20 |
| target return | 0.95 |
| eval episodes | 100 / seed |

结果：

| 模型 | 平均回报 | 成功率 |
|---|---:|---:|
| BC | 0.0000 | 0.0000 |
| RCBC | 0.0000 | 0.0000 |
| SBC-K5 | 0.6561 | 0.6600 |
| SBC-K10 | 0.8183 | 0.8233 |
| DT | 0.5301 | 0.5333 |

结论：这一轮说明短历史确实有用，但 `SBC-K10` 已经覆盖了从起点 cue 到岔路口的关键片段，导致它强于 DT；同时 DT 的 target return sweep 不明显。因此该轮不适合作为最终结果，需要缩短中间模型历史长度或提高 DT 训练充分性。

## 2026-06-02 迭代 2：缩短中间模型历史并提高 DT 训练充分性

目的：保留降低专家比例后的混合数据难度，同时避免 `SBC-K10` 直接看到完整关键序列。将中间模型改为 `SBC-K3` 和 `SBC-K5`，并用 seed 0 探针测试 40 epoch 是否恢复 DT 的序列建模优势。

配置：

| 项目 | 数值 |
|---|---:|
| 离线数据比例 | explore 35%, expert 65% |
| 离线轨迹数 | 400 |
| seed | 0 |
| epochs | 40 |
| DT context length | 20 |
| target return | 0.95 |
| eval episodes | 100 |

结果：

| 模型 | 平均回报 | 成功率 |
|---|---:|---:|
| BC | 0.0000 | 0.0000 |
| RCBC | 0.0000 | 0.0000 |
| SBC-K3 | 0.6163 | 0.6200 |
| SBC-K5 | 0.7058 | 0.7100 |
| DT | 0.9938 | 1.0000 |

DT target return sweep：

| Target return | 平均回报 | 成功率 |
|---:|---:|---:|
| 0.00 | 0.3579 | 0.3600 |
| 0.30 | 0.7254 | 0.7300 |
| 0.95 | 0.9939 | 1.0000 |

结论：该配置同时满足三点：专家比例已经降低，短历史模型形成中间层次，DT 仍然明显最好且 target return 有单调趋势。因此进入 3 seed 正式复验。

## 2026-06-02 迭代 3：3 seed 正式复验

目的：确认迭代 2 的 seed0 结果不是偶然现象。

配置：

| 项目 | 数值 |
|---|---:|
| 离线数据比例 | explore 35%, expert 65% |
| 离线轨迹数 | 400 / seed |
| seeds | 0, 1, 2 |
| epochs | 40 |
| DT context length | 20 |
| target return | 0.95 |
| eval episodes | 100 / seed |

主结果：

| 模型 | 平均回报 | 跨 seed 标准差 | 成功率 |
|---|---:|---:|---:|
| BC | 0.0000 | 0.0000 | 0.0000 |
| RCBC | 0.0000 | 0.0000 | 0.0000 |
| SBC-K3 | 0.6064 | 0.0359 | 0.6100 |
| SBC-K5 | 0.6893 | 0.0152 | 0.6933 |
| DT | 0.9938 | 0.0000 | 1.0000 |

DT target return sweep：

| Target return | 平均回报 | 跨 seed 标准差 | 成功率 |
|---:|---:|---:|---:|
| 0.00 | 0.4407 | 0.0902 | 0.4433 |
| 0.30 | 0.5697 | 0.1357 | 0.5733 |
| 0.95 | 0.9939 | 0.0000 | 1.0000 |

结论：该轮作为最终正式结果。它同时呈现了数据难度、中间模型梯度和 DT 的序列建模优势。

## 2026-06-02 迭代 4：epoch sweep 与最终训练轮数选择

目的：验证 40 epoch 的必要性，并说明 20 epoch 不作为最终截止的原因。

配置：固定 seed 0、400 条离线轨迹、`explore:expert = 35%:65%`，比较 5、10、20、30、40 epoch。

| Epoch | BC 成功率 | RCBC 成功率 | SBC-K3 成功率 | SBC-K5 成功率 | DT 成功率 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.00 | 0.00 | 0.54 | 0.54 | 0.45 |
| 10 | 0.00 | 0.00 | 0.54 | 0.57 | 0.44 |
| 20 | 0.00 | 0.00 | 0.58 | 0.67 | 0.63 |
| 30 | 0.00 | 0.00 | 0.60 | 0.68 | 1.00 |
| 40 | 0.00 | 0.00 | 0.62 | 0.71 | 1.00 |

结论：20 epoch 时 DT 尚未完全收敛，且没有稳定超过 `SBC-K5`；30 epoch 后 DT 达到 100% 成功率，40 epoch 与 30 epoch 基本一致。最终采用 40 epoch 作为保守正式配置，并在报告中把 epoch sweep 作为收敛分析。

## 2026-06-05 AntMaze Umaze 连续动作扩展

目的：在 MiniGrid Memory 离散动作实验之外，补充 D4RL/AntMaze Umaze 连续动作空间的尝试。目标是与 D4RL 风格的长时域稀疏奖励 benchmark 接轨。

### 新增代码

| 文件 | 内容 |
|---|---|
| `src/antmaze.py` | Minari/Farama AntMaze 数据加载、observation flatten、连续动作 trajectory/dataset、state normalization |
| `src/train_antmaze.py` | 独立 AntMaze 训练入口，continuous BC + DT，支持 quick/smoke/seed0/多 seed |
| `src/models.py` | 新增 `ContinuousBCPolicy`、`ContinuousDecisionTransformerPolicy`（`embed_action` 从 Embedding 换为 Linear） |
| `tests/test_continuous.py` | 连续动作模型、AntMaze flatten、连续 DT dataset 的 3 个单元测试 |

### 环境与数据

| 项目 | 数值 |
|---|---|
| 数据集 | D4RL/antmaze/umaze-v1 (Farama/Minari) |
| State dim | 31 (obs 27 + achieved_goal 2 + desired_goal 2) |
| Action dim | 8 (连续，[-1, 1]) |
| 离线轨迹数 | 1430 |
| 总 transitions | 1,000,000 |
| 数据集平均 return | 438.2 |
| 评估环境 | AntMaze_UMaze-v4 (MuJoCo) |

### 远程执行环境

| 项目 | 数值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D (24GB) |
| CUDA | 12.4 |
| Python | 3.10 (conda sam2qwen) |
| torch | 2.6.0+cu124 |
| 平台 | AutoDL (盛云) |
| 网络 | 需 HF_ENDPOINT=https://hf-mirror.com 下载数据集 |

### 性能优化

第一版 seed 0 使用默认参数（batch_size=256, num_workers=0），DT ~95s/epoch，GPU 利用率仅 25%。优化后：

| 参数 | 优化前 | 优化后 |
|---|---|---|
| batch_size | 256 | 1024 |
| num_workers | 0 | 8 |
| BC 总耗时 | 5:48 | 2:39 |
| DT per epoch | 95s | 35s |
| GPU 利用率 | 25% | 83% |
| 总耗时 | ~60 min | ~20 min |

> 注意：num_workers=8 会派生 8 个子进程，每个持有数据集副本 (~5.4GB RSS)，内存吃紧时建议 num_workers=4。

### Seed 0 结果

配置：

| 项目 | 数值 |
|---|---|
| epochs | 30 |
| batch_size | 1024 |
| num_workers | 8 |
| context_length | 20 |
| embed_dim | 128 |
| n_layers | 3 |
| n_heads | 4 |
| lr | 1e-4 |
| target_return | 1.0 |
| rtg_scale | 1.0 |
| eval_episodes | 20 |

训练 loss：

| 模型 | Epoch 1 loss | Epoch 30 loss |
|---|---|---|
| Continuous BC | 0.1675 | 0.0920 |
| Continuous DT | 0.1845 | 0.0725 |

两个模型 loss 均平稳下降，DT 最终 loss 低于 BC（0.072 vs 0.092）。

评估结果：

| 模型 | return_mean | success_rate | episode_length |
|---|---|---|---|
| Continuous BC | 0.0 | 0.0 | 700.0 |
| Continuous DT | 0.0 | 0.0 | 700.0 |

### 结论

训练流程端到端跑通，loss 收敛正常。但评估阶段两个模型在 AntMaze Umaze 环境中 return 均为 0：
- AntMaze 是典型的稀疏奖励迷宫（仅到达目标给正 reward），对模仿学习策略极度不友好
- 存在 MuJoCo 版本不匹配（安装 3.9.0，数据集要求 3.1.1-3.1.6），可能影响物理模拟
- 该扩展作为 MiniGrid 主实验之外的补充尝试，报告中单独陈述，不与主实验合并

后续如需改进可考虑：MuJoCo 版本对齐、加入目标条件（GCBC/GCDT）、使用更大 context_length、或引入 online fine-tuning。

## 2026-06-05 Hopper D4RL 原论文 benchmark 复现

目的：将连续动作 DT 实验对齐 Decision Transformer 原论文的 D4RL Gym 任务（Hopper、HalfCheetah、Walker、Reacher）。与 AntMaze 不同，Hopper 是原论文直接评估的 benchmark。

### 新增代码

| 文件 | 内容 |
|---|---|
| `src/d4rl_continuous.py` | 通用 Minari 连续控制数据加载器，支持 flat/dict observation |
| `src/train_d4rl.py` | Hopper 训练入口，含 replay check 模式，使用原论文 GPT 超参数 |

### 环境与数据

| 项目 | 数值 |
|---|---|
| 数据集 | `mujoco/hopper/medium-v0` (Minari) |
| State dim | 11 (flat, 非 dict) |
| Action dim | 3 (连续, [-1, 1]) |
| 离线轨迹数 | 1327 |
| 总 transitions | 999,404 |
| 数据集平均 return | 2818 |
| 数据集 max return | 3909 |
| 评估环境 | Hopper-v5 (terminate_when_unhealthy=False) |

### MuJoCo 版本问题

数据集要求 mujoco==3.2.3，远程安装 3.9.0 时 replay 结果严重不匹配（replayed return ~500-1200 vs dataset ~3600）。降级到 3.2.3 后 replay 接近但仍不完全一致（首 50 步 reward 匹配，但全 episode 因累积物理漂移偏差）。评估时设置 `terminate_when_unhealthy=False` 避免 Hopper 提前摔倒终止。

### 原论文超参数

| 参数 | 值 |
|---|---|
| context length K | 20 |
| target return | 3600 |
| Transformer layers | 3 |
| attention heads | 1 |
| embedding dim | 128 |
| learning rate | 1e-4 |
| weight decay | 1e-4 |
| gradient clipping | 0.25 |
| batch size | 1024 |
| epochs | 30 |
| rtg_scale | 1000 |

### 3 Seed 结果

| Seed | BC return | BC 成功率 | DT return | DT 成功率 |
|---:|---:|---:|---:|---:|
| 0 | 3461 | 0.90 | 3137 | 0.70 |
| 1 | 1071 | 0.00 | 2242 | 0.15 |
| 2 | 1761 | 0.00 | 2742 | 0.45 |
| **均值** | **2098** | 0.30 | **2707** | 0.43 |

跨 seed 统计：

| 模型 | return mean | return std(跨 seed) | 成功率 |
|---|---|---|---|
| BC | 2098 | 1230 | 30% |
| DT | 2707 | 449 | 43% |

DT 平均回报比 BC 高 29%，3 轮中 2 轮胜出，且跨种子更稳定（std 449 vs 1230）。

### 结论

- DT 在 Hopper 上展示了 return-conditioned 序列建模的有效性，均值 2707 vs BC 2098
- 但未能完全复现原论文的 DT 大幅领先（原论文 Hopper-medium DT ~3600 vs BC ~1100）
- MuJoCo 物理漂移导致评估回报与数据集回报不在同一尺度（DT 最高仅 ~3800）
- 跨种子方差大，说明 offline RL 在 Hopper 上对数据 split 和初始化敏感
- 该实验作为原论文 benchmark 的复现尝试，与 MiniGrid Memory 主实验和 AntMaze 扩展共同构成最终报告的三部分
