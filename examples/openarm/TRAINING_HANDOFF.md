# OpenArm + π0.5 训练交接文档

**日期**：2026-10-25
**用途**：把数据转换、π0.5 LoRA 微调、留出集验证的完整流程交接给协作者。文档中的路径/分区/代理都用 `<...>` 占位，请按自己的环境替换。

> 本仓库**不包含**原始数据、LeRobot 数据集、模型权重和 W&B 日志。

---

## 1. 任务与语言指令

任务：右臂依次把桌面上的药盒和输液瓶放入篮子，然后左臂提起篮子。

训练时写入所有帧的英文指令：

```
Using the right arm, pick up the medicine box and the IV bottle from the table one at a time, place them into the basket, then use the left arm to lift the basket.
```

推理时必须使用完全相同的字符串。

---

## 2. 代码基线

- 上游：`Physical-Intelligence/openpi` commit `215abfb`
- 本仓库是上游的 fork，新增/修改：
  - `src/openpi/policies/openarm_policy.py`（新增）
  - `src/openpi/training/config.py`（新 config、`DataConfig.episodes`、120/25 episode 列表）
  - `src/openpi/training/data_loader.py`（`EpisodeSubsetDataset`、`video_backend="torchcodec"`）
  - `pyproject.toml`（`wandb==0.22.3`）
  - `uv.lock`
- 代码许可证：Apache-2.0（保留 `LICENSE`）
- `pi05_base` 权重包含 PaliGemma/Gemma，受 `LICENSE_GEMMA.txt`（Gemma Terms of Use）约束：**不要提交权重**；下载/分发见第 12 节

---

## 3. 数据 schema

源数据为 HDF5，每个文件一个 episode，`schema_version=1.6`（旧数据 1.2 也可用），pipeline 1.8，30 Hz。

每条 episode 使用：

| 字段 | 维度 | 含义 |
|---|---|---|
| `observations/qpos` | 16 | 关节位置 + 夹爪（rad / m） |
| `observations/qvel` | 16 | 关节速度 |
| `actions` | 16 | 控制器实际收到的关节位置命令 |
| `observations/eef` | 14 | FK(qpos) 的 hand_tcp 位姿（world, xyzw，每臂 3+4） |
| `actions_eef` | 14 | FK(actions) 的 hand_tcp 位姿 |
| `observations/images/{head,left_wrist,right_wrist}` | JPEG | head 1280×720，wrist 640×480 |
| `timestamp` | T | ROS 时钟（非 epoch） |

关节顺序（16 维）：

```
L joint1..7, L finger, R joint1..7, R finger
```

**新增的触觉字段（`observations/tactile/...`）全部忽略**，转换脚本不读取它们。

本批训练数据统计：145 episodes / 76,959 帧 / 42.8 分钟。

---

## 4. 转换为 LeRobot

```bash
python examples/openarm/scripts/convert_openarm_data_to_lerobot.py \
  --data-dir <PATH_TO_RAW_HDF5_DIR> \
  --root <PATH_TO_LEROBOT_HOME> \
  --repo-id openarm/cropped_20261025_joint_eef \
  --include-eef \
  --task "Using the right arm, pick up the medicine box and the IV bottle from the table one at a time, place them into the basket, then use the left arm to lift the basket."
```

输出：

- 145 episodes / 76,959 帧
- **145 个 parquet + 435 个 mp4**（每 episode 3 路视频）
- 约 1.1GB
- 关节和 EE 路线共用这一份数据集（EE 路线读取 `observation.eef` / `action.eef`）

校验：

```bash
python examples/openarm/scripts/verify_lerobot_dataset.py \
  --root <PATH_TO_LEROBOT_HOME> \
  --repo-id openarm/cropped_20261025_joint_eef \
  --horizon 30
```

---

## 5. 训练/验证划分

- **120 训练 / 25 验证**，随机种子 42
- 文件：`examples/openarm/split/{split.json,train_episodes.json,val_episodes.json}`
- 索引是 **LeRobot episode index**（0–144，顺序与转换时排序后的源文件一致）
- `split.json` 里同时记录了 `source_index` / `source_file`，方便回查
- `src/openpi/training/config.py` 里的 `_TRAIN_EPISODES_20261025` 必须与 `train_episodes.json` 一致
- 验证集不参与训练、不参与 norm stats

验证集对应的源 episode 编号可由：

```bash
python - <<'PY'
import json
d = json.load(open('examples/openarm/split/split.json'))
print(','.join(str(e['source_index']) for e in d['val']))
PY
```

---

## 6. Norm stats

不建议用标准 `compute_norm_stats.py`：它会为每一帧解码 3 路视频，在登录节点上很慢，并且容易触发线程/视频解码问题。

使用 `compute_norm_stats_direct.py`，直接从 parquet 的 state/action 计算，并精确复刻：

- 关节：state=qpos；action delta = command - state（arm dims），夹爪绝对
- EE raw：state=[L p3,q4,grip, R p3,q4,grip]；action=dp3+rotvec3+grip（每臂）
- action chunk 使用与 LeRobot 相同的 clamp-to-episode-end padding

```bash
HF_LEROBOT_HOME=<PATH_TO_LEROBOT_HOME> python examples/openarm/scripts/compute_norm_stats_direct.py \
  --config pi05_openarm_20261025_lora
HF_LEROBOT_HOME=<PATH_TO_LEROBOT_HOME> python examples/openarm/scripts/compute_norm_stats_direct.py \
  --config pi05_openarm_20261025_eef_raw_lora
```

输出到 `assets/<config_name>/openarm/cropped_20261025_joint_eef/norm_stats.json`。仓库里也附了这两份文件（`examples/openarm/norm_stats/`），可直接复制到 `assets/`：

```bash
mkdir -p assets
cp -r examples/openarm/norm_stats/* assets/
```

---

## 7. 训练配置与超参

Configs（`src/openpi/training/config.py`）：

- `pi05_openarm_20261025_lora`（关节）
- `pi05_openarm_20261025_eef_raw_lora`（EE raw）

| 项 | 值 |
|---|---|
| 基座 | `pi05_base` |
| 模型 | `Pi0Config(pi05=True, action_horizon=30)` |
| LoRA | `paligemma_variant=gemma_2b_lora`，`action_expert_variant=gemma_300m_lora` |
| batch | 8 |
| LR | 5e-5 cosine，warmup 1000，decay 30k |
| EMA | 关闭 |
| 训练步数 | 上限 100,000,000（由人手动停止） |
| checkpoint | 每 2500 步，`keep_period=2500` |
| 全量微调 | 单卡 80GB OOM（batch=1 也 OOM），因此使用 LoRA |

动作/状态布局：

- 关节：state 16；action 16（`DeltaActions` mask `(7,-1,7,-1)`）
- EE raw：state 16；action 14（每臂 `dp3 + rotvec3 + grip`）
- **EE raw 不使用 `_eef_obs_to_cmd`**，不做任何平均偏差补偿；旧 corrected config/checkpoint 仅作 legacy 保留，不能与新 raw 流程混用

---

## 8. 训练命令

先配置环境：

```bash
cp examples/openarm/scripts/openarm_env.example.sh \
   examples/openarm/scripts/openarm_env.sh
# 编辑 openarm_env.sh：HF_LEROBOT_HOME、代理、FFmpeg lib、wandb 等
```

单卡（在 GPU allocation 内）：

```bash
srun -c 8 -p <YOUR_PARTITION> --gres=gpu:1 -t 7-00:00:00 \
  bash examples/openarm/scripts/run_training_job.sh \
    pi05_openarm_20261025_lora openarm_20261025_joint_run1 <PROXY_PORT> fresh
```

EE raw：

```bash
srun -c 8 -p <YOUR_PARTITION> --gres=gpu:1 -t 7-00:00:00 \
  bash examples/openarm/scripts/run_training_job.sh \
    pi05_openarm_20261025_eef_raw_lora openarm_20261025_eef_raw_run1 <PROXY_PORT> fresh
```

两个同时跑可用 `examples/openarm/scripts/launch_20261025_training.sh`（需设置 `PARTITION`）。

checkpoint 路径：

```
checkpoints/pi05_openarm_20261025_lora/openarm_20261025_joint_run1/<step>/
checkpoints/pi05_openarm_20261025_eef_raw_lora/openarm_20261025_eef_raw_run1/<step>/
```

W&B 可选；在 `openarm_env.sh` 里设置 `WANDB_BASE_URL/WANDB_ENTITY/WANDB_PROJECT`。没有外网时可用 `WANDB_MODE=offline`，训练后 sync。

---

## 9. 验证

离线验证脚本直接读 HDF5（不需要转 LeRobot）：

```bash
python examples/openarm/scripts/eval_checkpoints.py \
  --config pi05_openarm_20261025_lora \
  --checkpoint checkpoints/pi05_openarm_20261025_lora/openarm_20261025_joint_run1/<step> \
  --data-dir <PATH_TO_RAW_HDF5_DIR> \
  --episodes <comma-separated val source indices> \
  --out logs/eval_joint/<step>.json
```

EE raw 换成 `pi05_openarm_20261025_eef_raw_lora`，结果放 `logs/eval_eef/`。

成功率定义（离线代理，不是闭环任务成功率）：

- 关节 medium：第一步手臂 MAE ≤ 0.05 rad 且夹爪 ≤ 5mm
- EE medium：第一步位置 ≤ 1cm、旋转 ≤ 5°、夹爪 ≤ 5mm
- strict / loose 阈值见脚本

汇总：

```bash
python examples/openarm/scripts/summarize_eval.py --eval-root <path-containing-eval_joint-and-eval_eef>
```

最终选择 checkpoint 仍建议做真机闭环 rollout。

---

## 10. 环境

- Python 3.11（uv-managed）
- JAX 0.5.3，torch 2.7.1，lerobot 固定 commit
- `rerun-sdk` 在旧 glibc 上装不上：

```bash
uv sync --managed-python --python 3.11 --no-install-package rerun-sdk
uv pip install -e .
```

- 训练视频解码需要 FFmpeg 7 + `LD_LIBRARY_PATH`（torchcodec）
- `uv.lock` 中的源是某个 PyPI 镜像；如果对方网络不同，建议按自己的索引重新 `uv lock`
- 共享节点建议设置 `OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS`（见 `openarm_env.example.sh`）

---

## 11. 已知问题 / 决策

1. **pinned LeRobot `episodes=` bug**：过滤后 `episode_data_index` 只含被选 episode，但 `__getitem__` 仍用原始 episode index，导致 `IndexError`。修复：全量加载 + `EpisodeSubsetDataset` 将过滤索引映射到原始帧索引。
2. **EE raw**：不使用 `_eef_obs_to_cmd`；旧 corrected configs/checkpoints 仅作历史保留，不能与新流程混用。
3. **全量微调 OOM**：单卡 80GB 即使 batch=1 也 OOM；使用 LoRA。
4. **EE checkpoint 保存竞态**：Orbax 异步保存在某些运行中出现 `NOT_FOUND`（临时 params 分片）。建议两个作业的 checkpoint 步数错开、放不同节点，或改用同步保存。
5. **norm stats**：标准数据加载器需要解码视频，在共享登录节点上慢且容易触发线程问题；使用 `compute_norm_stats_direct.py`。
6. **过拟合风险**：旧数据 118 eps / 21 分钟曾训练 ~35 epochs；新数据 145 eps / 42.8 分钟更大，但仍建议用 25 条留出集选 checkpoint，不要只看训练 loss。
7. **触觉未使用**：转换脚本忽略 `observations/tactile/*`。

---

## 12. 许可证与合规

- openpi 代码：Apache-2.0，保留 `LICENSE`。
- `pi05_base` 包含 PaliGemma/Gemma 权重，受 `LICENSE_GEMMA.txt`（Gemma Terms of Use）约束：
  - **不要提交权重到本仓库**
  - 官方公开路径：`gs://openpi-assets/checkpoints/pi05_base`
  - 如需分发副本，附上 Gemma Terms、Notice 文本，并确保接收方接受 Section 3.2 使用限制：`Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms`
  - 本仓库训练出的微调 checkpoint 同样属于 Model Derivatives，按 Gemma Terms 处理
- 原始数据与转换后的数据集不包含在本仓库；分享前请确认数据授权。

---

## 13. 最小冒烟测试

1. 转换 1 条数据：
   ```bash
   python examples/openarm/scripts/convert_openarm_data_to_lerobot.py \
     --data-dir <RAW_DIR> --root <LEROBOT_HOME> \
     --repo-id openarm/smoke_test --include-eef --limit 1 --overwrite
   ```
2. 用 `verify_lerobot_dataset.py` 检查 loader 和视频。
3. 环境/训练循环可用 FakeData debug config 测试：
   ```bash
   python scripts/train.py debug_pi05 --exp-name=smoke --num-train-steps=2 --batch-size=1 --overwrite
   ```
4. 完整训练需要 145 条数据 + 120/25 划分 + norm stats。

---

## 14. 文件索引

| 文件 | 用途 |
|---|---|
| `src/openpi/policies/openarm_policy.py` | OpenArm joint/EE 输入输出、raw/legacy 开关 |
| `src/openpi/training/config.py` | 两个新 config、`DataConfig.episodes`、120/25 列表 |
| `src/openpi/training/data_loader.py` | `EpisodeSubsetDataset`、torchcodec |
| `scripts/convert_openarm_data_to_lerobot.py` | HDF5 → LeRobot |
| `scripts/verify_lerobot_dataset.py` | 转换校验 |
| `scripts/compute_norm_stats_direct.py` | parquet 直接算 norm stats |
| `scripts/eval_checkpoints.py` | 留出集评估 |
| `scripts/summarize_eval.py` | checkpoint 对比 |
| `scripts/checkpoint_sanity.py` | P0 真实帧推理自检 |
| `split/` | 120/25 划分 |
| `norm_stats/` | 两份 norm stats |
