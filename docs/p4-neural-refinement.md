# P4 神经边界精修

截至 2026-10-10，数据准备、MTL/MTL_BDR 粗预测缓存、真实 CUDA 神经训练和冻结评估已完成；当前暂停训练并交接。最佳权重和完整评估随云端仓库保存，实际绘图及P4最终验收仍未完成。此模块是独立研究入口，尚未接入默认 CLI。监督和指标使用数据集来源标注，没有独立人工真值，也没有 5ms 精度成功证据。最新恢复说明见 [训练交接](TRAINING_HANDOFF.md)，暂停期间不执行下列训练/评估示例。

## 数据与隔离

固定来源为 `AaronZ345/GTSinger` revision `d78cd68d00135b1995da03f25fccb3a221efd8c1`。180 段计划原件中，17 段因来源区间越过原 WAV 末端而排除；原件和标注不裁切。实际保留 163 段、18 首歌、2,436 个歌词来源边界，原始 48kHz 网格不变。

| 划分 | 段数 | 歌曲数 | 中 / 英歌曲 | 来源边界 |
| --- | ---: | ---: | ---: | ---: |
| train | 86 | 10 | 6 / 4 | 1,371 |
| val | 38 | 4 | 2 / 2 | 554 |
| test | 39 | 4 | 2 / 2 | 511 |

同歌曲的 Control/演唱技术变体保留在同一划分，旧歌曲分配保持不变；`EN-Tenor-1` 全部留在 test。歌曲身份和边界标签均为 dataset/unreviewed，`review_status=pending`；来源边界为 TextGrid 的首发声音素标注，不写入人类真值。原整歌 offset 保持 unknown/null。

可追溯输入和排除理由见 [准备结果](../reports/p4/training-source-20261010/quality-summary.json)、[划分](../reports/p4/training-source-20261010/splits.json)及该目录的 `source-manifest.json`。模型、配置、输入和运行代码摘要随报告及检查点保存，不能将路径相同视为摘要相同。

## 共享特征与模型

版本配置为 [configs/boundary-tcn.json](../configs/boundary-tcn.json)：训练配置 `boundary-training-config-v1`，模型配置 `boundary-tcn-v1`。

- 输入不重采样，first channel 与既有运行时一致。hop 为 `round(sr * .0025)`，48kHz 时 120 samples/2.5ms；帧中心从样本 0 开始。其他采样率记录实际整数 hop，而非强称恰好 2.5ms。
- 57 维声学输入为 48 个 centered 20ms periodic Hann log-Mel 加既有 9 个 DSP 特征。固定 HTK Mel 滤组覆盖 40–12,000Hz，采样率低于 24kHz 拒绝；窗口越界 zero pad，不猜测延迟校正。
- 每词以 coarse `start_sample` 建立 281 帧、物理半径 ±350ms 的窗口，追加 `[-1,1]` relative-position 通道，共 58 通道。mask 排除音频范围、物理半径及 coarse word end 外的位置；窗口中心绝不取自监督标签。
- 当前词首、前词末、后词首 ARPAbet 构成三个 16 维 embedding；stress stripped，未知音素用 UNK。
- 默认条件 TCN 为 128 channels、六个双 Conv1d 残差块，kernel=3，dilation=1/2/4/8/16/32，GroupNorm/dropout，总计 **609,026 参数**。
- 输出 heatmap logits、概率加权相对坐标的可微 `delta_seconds`、未校准 confidence logit。实际精修取有效热图 argmax，保留 coarse/raw frames 和 top candidates；跨词顺序冲突回退 coarse。热图质量和 confidence 不能解释为起点正确概率。

特征步长和浮点时间格式不是物理起点精度保证。局部精修也不保证全局歌曲结构，当前缺失粗候选显式保留 fallback。

## 训练协议

训练仅使用 `mtl:legacy` 的冻结粗候选。train 计算梯度，val 以实际 argmax/fallback 的相对来源 MAE 选择最佳检查点和 early stopping；test 不参与目标、loss、调参或选择。训练入口会校验全量输入摘要/划分，之后只解码 train/val WAV。

固定 seed=7，最多 60 epochs，patience=10，batch=32、gradient accumulation=2，AdamW lr=0.0003、weight decay=0.01。真实 CUDA float16 AMP/GradScaler、有限梯度检查和 clipping 必须启用，失败不暗中改为 CPU。热图/softmax/三路输出保持 float32。损失包括高斯 soft-label BCE、delta SmoothL1、窗口可达性 confidence 和同 minibatch 邻词一致性项；超出窗口的来源标签不计热图/delta loss，仍保留在 val 指标分母。

仅 train 使用原件/一套 seeded 波形增强特征缓存、±75ms 的复制粗中心 jitter 和轻度 feature gain/noise。波形增强包括 20–35dB Gaussian noise、因果迟到反射、0.8–1.2 volume、20–30dB 原创正弦 bass/triad 合成伴奏；同样本数、同网格，混响 direct path `impulse[0]=1`，无循环回绕或 padding 修饰。它不是实际配器或起点校准。val/test 保持原件、原粗中心，train jitter 不修改冻结预测或标签。

## 实际训练结果

[训练报告](../reports/p4/training-cuda-20261010/result.json) 为 completed：RTX 5060、PyTorch 2.7.1+cu128/CUDA 12.8、真实 float16 AMP，完成 36 epochs、792 optimizer steps，全部轮次记录有限梯度，AMP scale 均为 1024。第 26 轮验证最优，随后 10 轮无改进触发 `validation_early_stopping`；未以 test 选择检查点。

验证分母为 554 个来源边界，均有数值预测，无 missing/text mismatch；其中 20 个来源标签位于原粗窗口外，仍计入指标。最优验证 MAE **58.4439266681ms**，median 17.5ms、P90 127.3643ms、P95 250.875ms；与来源标注 ±5/10/20/50ms 一致的覆盖分别为 21.1191%/34.8375%/52.3466%/73.6462%，大于 500ms 的严重来源误差 8 个。这是用于选 epoch 的 val 结果，不是测试表现或独立人工准确率。

总墙钟时间 1,347.8245s（约 22.46min）；其中训练 epochs 与逐轮验证 1,291.3629s、进程 CPU time 4,668.9115s，特征缓存另耗墙钟 50.7171s、CPU time 555.6115s。36 轮的最大 PyTorch CUDA allocated/reserved 分别为 252,772,864/278,921,216 bytes（241.0625/266MiB），不代表整机显存或仅 GPU kernel 耗时。

最佳本地检查点 [boundary.pt](../reports/p4/training-cuda-20261010/boundary.pt) 为 2,456,370 bytes，文件摘要与报告一致：

```text
SHA256 967da4debb2a457cf45a2989335183781e41890f0d4c43af3e1b72292a5bb4da
```

## 恢复命令

CPU 可运行共享特征、窗口、新模型推理及轻量定点检查；训练 CLI 严格要求 CUDA。下列定点命令仅用于相关源码变化后的恢复验证，已有成功结果不重复执行：

```bash
cd /Users/xcy/Program/auto-lrc
.venv-baseline/bin/python -m pytest tests/test_boundary_neural.py tests/test_boundary_augmentation.py -q
```

CPU API 可显式加载上述本地权重，使用 `load_boundary_checkpoint(path, device="cpu")` 和 `refine_boundaries(audio, sr, candidates, loaded=runtime, device="cpu")`。`audio` 为原始 mono/channel-first ndarray，`candidates` 必须携带同一原始网格；无默认权重、下载或设备失败降级。

WSL 项目根为 `/home/xcy/auto-lrc`，沿用 `.venv-cuda`，原件在 `data/local/gtsinger-training-p4`。以下保留实际使用的冻结训练/评估命令；训练已有 completed 证据，禁止重跑同名目录，评估成功仍须其独立报告。新执行的输出目录须不存在：

```bash
cd /home/xcy/auto-lrc
.venv-cuda/bin/python scripts/train_boundary.py \
  --segments reports/p4/training-source-20261010/segments.jsonl \
  --splits reports/p4/training-source-20261010/splits.json \
  --audio-root data/local/gtsinger-training-p4 \
  --coarse-report reports/p4/coarse-cache-cuda-20261010/result.json \
  --config configs/boundary-tcn.json \
  --out-dir reports/p4/training-cuda-20261010

.venv-cuda/bin/python scripts/evaluate_boundary.py \
  --segments reports/p4/training-source-20261010/segments.jsonl \
  --splits reports/p4/training-source-20261010/splits.json \
  --audio-root data/local/gtsinger-training-p4 \
  --coarse-report reports/p4/coarse-cache-cuda-20261010/result.json \
  --checkpoint reports/p4/training-cuda-20261010/boundary.pt \
  --device cuda --batch-size 32 \
  --out-dir reports/p4/evaluation-cuda-20261010
```

粗缓存已有成功结果，复用经过摘要验证的 36 次旧预测和 290 次新增 CUDA 推理，勿重复相同推理。评估固定比较 coarse、DSP、neural、DSP+neural，报告 MAE/分位数、±5/10/20/50ms 来源覆盖、缺失/严重错误、分类组、少量测试歌曲 bootstrap、CPU/GPU 耗时和显存；failed 保留在覆盖分母。耗时不包含已缓存的粗模型推理。

| 目录 | 当前状态 |
| --- | --- |
| `reports/p4/training-source-20261010` | 数据准备 completed |
| `reports/p4/coarse-cache-cuda-20261010` | 冻结粗候选 completed |
| `reports/p4/training-cuda-20261010` | completed；36 轮、第 26 轮最佳、val 来源 MAE 58.4439ms |
| `reports/p4/evaluation-cuda-20261010` | completed；0 failures；完整JSON已取回并核对输入/源码/权重摘要 |

## 后台执行与结果同步

WSL 根目录可使用固定路径入口 [run_boundary_stage.sh](../scripts/run_boundary_stage.sh)，三个阶段默认通过 `nohup` 独立会话后台执行，可脱离 SSH。它不下载或同步数据，也不改变训练/评估参数：

```bash
cd /home/xcy/auto-lrc
bash scripts/run_boundary_stage.sh train
bash scripts/run_boundary_stage.sh status
# status 中 train.success=true 后：
bash scripts/run_boundary_stage.sh evaluate
# status 中 evaluate.success=true 后：
bash scripts/run_boundary_stage.sh plots
```

`--foreground` 可等待任一阶段并返回明确成功/失败。训练和评估输出沿用上表，绘图固定为 `reports/p4/plots-cuda-20261010`，包括冻结结果的曲线和六个来源最大误差案例；不进行新推理。归档为 `reports/p4/stage-runs-20261010/{train,evaluate,plots}/`，保留 `launch.json`、带进程出生标记的 `spawn.json`/`pid.json`、`stage.log`、`exit.json` 和 `status.json`。

重复命令仅显示已有活进程或复用已完成结果，不重新启动；归档和输出均拒绝覆盖。成功要求 runner 状态 completed、明确 exit code=0、阶段 `result.json`（绘图为 `index.json`）completed 三者同时成立；启动成功只是 started，并非实验成功。无退出码、失联 PID、失败或不完整记录会拒绝下游。未经此入口创建的已有输出显示 external_unverified，保持原件，不能补造退出码或从归档缺失推断成功；这种历史结果继续按原 CLI 和真实证据处理。

本次已完成训练使用原 CLI，因此 runner 的 external_unverified 表示缺少它自己的退出归档，不否定已同步训练报告和权重的真实 completed 证据；评估由维护者沿用上方原 CLI 执行，不为迁移归档重新训练。

SSH 恢复后先检查 `status`，再将上述三个实验目录与 `stage-runs-20261010` 同步到本地同名目录即可收回结果、图和运行日志；原始音频仍只在 `data/local`，无需重新同步。新执行前遇到失败/残留目录应保留归档、由维护者选择新的实验版本路径，不删除旧证据后自动重试。

## 许可与发布

代码许可分别见仓库根 [Apache-2.0](../LICENSE) 和上游声学实现 [t2l/mtl/LICENSE](../t2l/mtl/LICENSE)（MIT）。数据原文见本地 `data/local/gtsinger-representative/dataset_license.md` 或固定 revision 的 [dataset_license.md](https://huggingface.co/datasets/AaronZ345/GTSinger/blob/d78cd68d00135b1995da03f25fccb3a221efd8c1/dataset_license.md)：CC BY-NC-SA 4.0，要求署名、非商业、相同方式共享，另含责任/赔偿、代表雇主授权及引用论文的附加协议。原文是使用和传播的依据，代码许可不能替代数据条款。

权重随研究交接保存到用户云端仓库，未另行授予 MIT 或商业许可。上述检查点已记录训练来源和摘要，数据许可全文随 [交接归档](../reports/handoff/GTSINGER_LICENSE.md) 保留；传播和使用仍须处理完整数据条款和适用发布条件，见 [MODEL_CARD](../checkpoints/boundary/MODEL_CARD.md)。当前只用于研究验证，不宣称独立人工精度或 5ms 性能。
