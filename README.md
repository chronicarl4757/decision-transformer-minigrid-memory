# Decision Transformer on MiniGrid Memory

本项目是动态规划课程期末作业的强化学习论文复现项目，复现论文 **Decision Transformer: Reinforcement Learning via Sequence Modeling** 的核心思想：将离线强化学习轨迹建模为条件序列，用目标回报、历史状态和历史动作预测下一步动作。

最终主实验选择 **MiniGrid MemoryS17Random** 部分可观测记忆任务。该任务要求智能体先观察起点物体，随后在岔路口选择与起点物体匹配的一侧。当前状态在离开起点后不再包含起点物体信息，因此该任务可以检验 Transformer 是否能利用历史上下文完成记忆回忆。

## 方法

比较五个模型：

| 模型 | 输入 | 作用 |
|---|---|---|
| BC | 当前状态 | 无条件行为克隆 baseline |
| RCBC | 当前状态 + return-to-go | 公平的条件化前馈 baseline |
| SBC-K3 | 最近 3 步状态 | 短历史 MLP baseline |
| SBC-K5 | 最近 5 步状态 | 更长短历史 MLP baseline |
| DT | return-to-go + state/action history | Decision Transformer |

实验使用离线轨迹训练模型。脚本策略只用于生成离线数据；评估时所有模型只能使用部分观测特征，不能读取环境内部答案。

## 主要结果

主实验采用 MiniGrid Memory，统计口径为 `3 个 seed`。补充实验另行在 Hopper-medium 上运行了 `7 个 seed`，两者不要混用。

MiniGrid Memory 主实验 `3 个随机种子` 的汇总结果如下：

| 模型 | 平均回报 | 跨 seed 标准差 | 成功率 |
|---|---:|---:|---:|
| BC | 0.0000 | 0.0000 | 0.00 |
| RCBC | 0.0000 | 0.0000 | 0.00 |
| SBC-K3 | 0.6064 | 0.0359 | 0.61 |
| SBC-K5 | 0.6893 | 0.0152 | 0.69 |
| DT | 0.9938 | 0.0000 | 1.00 |

Target return sweep:

| Target return | 实际平均回报 | 成功率 |
|---:|---:|---:|
| 0.00 | 0.4407 | 0.4433 |
| 0.30 | 0.5697 | 0.5733 |
| 0.95 | 0.9939 | 1.0000 |

结果表明：在该部分可观测记忆任务中，只看当前状态的 BC 和 RCBC 无法完成目标；短历史 MLP 能完成一部分匹配，但仍不稳定；Decision Transformer 可以利用更长历史上下文完成起点物体与岔路口物体的匹配，并能通过 target return 生成更高回报行为。

补充实验 `Hopper-medium (seed0-6)` 的结果如下：

| 模型 | 平均回报 | 跨 seed 标准差 | 成功率* |
|---|---:|---:|---:|
| BC | 1899.7 | 1141.3 | 0.27 |
| DT | 2465.3 | 696.0 | 0.34 |

`*` Hopper 的成功率定义为 `return >= 3000` 的比例，只用于辅助观察回报分布。这个结果更适合解读为“与原论文方向一致的补充 benchmark”，不是统计显著的强结论。

## 复现方式

### 环境准备

推荐使用 Python 3.11。项目依赖已写入 `pyproject.toml` 和 `uv.lock`。

如果使用 `uv`：

```bash
uv sync
```

如果已有课程虚拟环境，也可以直接在该环境中安装/运行依赖，只需保证包含 `torch`、`gymnasium`、`minigrid`、`numpy`、`pandas`、`matplotlib`、`pytest` 和 `ruff`。

### 一键运行

项目提供一键脚本：

```bash
./run_project.sh
```

默认模式会依次执行：

1. `pytest`
2. `ruff check .`
3. 一个 quick smoke 训练，用于确认代码、环境和训练流程能跑通

如果要重新运行完整 3 seed 的 MiniGrid Memory 正式实验：

```bash
./run_project.sh --full
```

完整实验会重新生成：

| 输出 | 含义 |
|---|---|
| `outputs/memory_final_multiseed/` | Memory 主实验 3 个 seed 的 metrics、losses 和图表 |
| `outputs/memory_final_summary/` | Memory 主实验的多 seed 汇总表和主图 |

Hopper 补充实验使用独立入口 `src/train_d4rl.py`，输出位于 `outputs/hopper_medium/`。

### 手动运行

安装依赖后也可以手动运行测试：

```bash
python -m pytest
python -m ruff check .
```

运行 3 seed 实验：

```bash
for seed in 0 1 2; do
  python -m src.train \
    --env memory \
    --episodes 400 \
    --epochs 40 \
    --batch-size 128 \
    --context-length 20 \
    --n-layers 1 \
    --embed-dim 96 \
    --n-heads 4 \
    --lr 1e-3 \
    --eval-episodes 100 \
    --seed "$seed" \
    --target-return 0.95 \
    --eval-target-returns 0 0.3 0.95 \
    --output-dir "outputs/memory_final_multiseed/seed${seed}"
done

python -m src.summarize_runs \
  --runs-dir outputs/memory_final_multiseed \
  --output-dir outputs/memory_final_summary
```

生成最终报告：

```bash
pandoc -f markdown+tex_math_single_backslash reports/final_submission_report.md \
  -o dist/final_submission_report.pdf \
  --pdf-engine=xelatex \
  --number-sections \
  --toc \
  --listings \
  -V documentclass=ctexart \
  -V papersize=a4 \
  -V geometry:margin=2.4cm \
  -H reports/pdf_style.tex \
  --resource-path=.:reports
```

## 主要文件

| 文件 | 作用 |
|---|---|
| `run_project.sh` | 一键测试、lint、quick smoke；`--full` 可重跑完整实验 |
| `src/envs/memory.py` | MiniGrid Memory 部分观测 wrapper 与脚本策略 |
| `src/data.py` | 离线轨迹采集、return-to-go 计算、dataset 构造 |
| `src/models.py` | BC、RCBC、StackedBC 与 Decision Transformer 模型 |
| `src/train.py` | MiniGrid Memory 主实验训练、评估与 target sweep |
| `src/train_d4rl.py` | Hopper D4RL 补充实验训练入口 |
| `src/summarize_runs.py` | 多 seed 结果聚合和图表生成 |
| `src/summarize_epoch_sweep.py` | 训练轮数 sweep 汇总和图表生成 |
| `docs/experiment_timeline.md` | 实验迭代时间线 |
| `reports/final_submission_report.md` | 最终公共报告 |
| `dist/final_submission_report.pdf` | 可提交最终报告 PDF |
| `dist/literature_review.pdf` | 公开版文献阅读报告 PDF |
| `docs/2106.01345v2.pdf` | Decision Transformer 原论文 |

## 分享包内容

`dist/final_project_public_share.zip` 是面向组员分享的最小公开包，包含：

- 源代码：`src/`
- 测试：`tests/`
- 一键运行脚本：`run_project.sh`
- 最终实验结果：`outputs/`
- 原论文：`docs/2106.01345v2.pdf`
- 实验时间线：`docs/experiment_timeline.md`
- 最终报告与文献回顾：`dist/` 和 `reports/`

分享包不包含虚拟环境、缓存、本机路径、local 备份或模型 checkpoint。

## 局限性

MiniGrid Memory 主实验使用手工压缩的部分观测特征，而不是直接从原始图像端到端学习；同时主实验规模为 3 个 seed，适合作为课程复现与机制验证，不等同于完整 benchmark 复现。Hopper 结果虽然补充了与原论文 D4RL 更接近的实验，但仍受环境版本差异和统计规模限制。
