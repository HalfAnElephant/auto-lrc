# auto-lrc 高精度对齐工程实施规格

> 执行变更（2026-10-10）：用户后续明确要求“先跳过人工阶段，完全采用训练数据”。本轮研发采用 GTSinger 发布标注作为训练和评测参考，人工复标、分歧裁决及歌曲身份人工确认不再阻塞 P1。仍按来源歌曲隔离 train/val/test，保留标注来源和未核实别名的限制。后续指标仅表示相对数据集标注的误差，不视作独立人工验证的 5ms 精度。下文保留原始规格；与本变更冲突处以最新用户要求为准。

> 流程变更（2026-10-10）：用户要求“本次研究跳过所有只读审查”。不再安排独立审查代理或 DeepSeek 只读审查；保留实现所需的代码阅读、功能测试和真实数据验证。既有审查记录仅为历史证据。

## 1. 任务与原则

仓库：https://github.com/HalfAnElephant/auto-lrc

目标：在原有自动歌词对齐能力上，增加高分辨率字词起点定位，最终争取在独立人工标注测试集上实现首字起点 MAE ≤5ms。

核心原则：

1. 优先采用粗对齐 + 局部精修，不重新训练整个大型模型。
2. 保留原有 API、模型检查点和 LRC 输出的向后兼容性。
3. 先验证数据、时间轴和声学特征，再尝试更复杂的模型。
4. 所有改进必须与原版进行定量对比，不凭主观试听判断精度。
5. 可以在 Mac 开发，GPU 训练与完整测试在 Arch Linux WSL2 中完成。
6. 每个阶段独立提交、测试、记录结果；某项方案无收益时可以撤回。
7. 不得通过人为设置输出小数位数、直接复制标注或使用测试集调参来宣称达到 5ms。

## 2. 现有代码需要重点检查的位置

| 文件                 | 工作内容                     |
| -------------------- | ---------------------------- |
| `main.py`            | CLI 接入、输出格式           |
| `t2l/t2l.py`         | 流水线编排、采样率、时间换算 |
| `t2l/phonetic.py`    | 多语言文本转音素             |
| `t2l/mtl/model.py`   | CNN、BiLSTM、MTL、BDR        |
| `t2l/mtl/wrapper.py` | 模型推理、对齐调度           |
| `t2l/mtl/utils.py`   | CTC 风格动态规划、路径回溯   |
| `t2l/mtl/data.py`    | 旧版训练数据逻辑             |
| `t2l/mtl/train.py`   | 原训练入口                   |

已识别的优先风险：

- 原始时间步长为 `256 / 22050 * 3 ≈ 34.83ms`，不适合直接输出 5ms 精度。
- AcousticModel 与 BoundaryDetection 的第二、第三层 BiLSTM 设置 `batch_first=False`，但上一层传入的张量为 batch-first，存在时序维度解释错误的风险。
- 非英语歌词通过转写后，再经 `g2p_en` 映射到英文音素集合，多语言发音表示可能不准确。
- BDR 当前主要作用于歌词行起点，而不是每个字的精细起点。
- `requirements.txt` 锁定 PyTorch 2.1.2，不能直接作为 RTX 5060 的目标 GPU 环境。
- 原训练脚本依赖缺失模块和外部 DALI 数据，不能假设开箱可运行。
- 原 CLI 默认不会使用 BDR，需要增加清晰的模型选择参数。

先审计并验证这些问题，不要在没有基准测试的情况下同时修改全部模型结构。

## 3. 阶段 P0：复现与基准测试

目标：让原项目成为可比较的 baseline。

执行：

1. 克隆代码、初始化 Git LFS、下载检查点。
2. 建立独立旧版环境与新版 CUDA 环境，记录依赖版本。
3. 跑通现有 `pytest` 和 `demofile`。
4. 分别测试 `MTL` 与 `MTL_BDR`，必要时增加纯净人声、混合音频两种输入。
5. 保存每个字的原始帧索引、预测时间、音素序列和对齐分数。
6. 核查实际声学模型输出帧率、STFT `hop_length`、padding、卷积输出形状与时间原点。
7. 复核 BiLSTM 维度问题，并分别对“保持旧行为”和“修正维度”进行消融实验。
8. 创建结构化 JSON 对齐结果，内部统一使用浮点秒或整数样本位置，不使用格式化后的 LRC 时间戳参与计算。

验收：原版结果可复现，新旧实现之间任何变化均有测试解释。

## 4. 阶段 P1：GTSinger 数据集

数据地址：

- https://huggingface.co/datasets/GTSinger/GTSinger
- https://github.com/AaronZ345/GTSinger

优先使用中文和英文，其余语言后续扩展。

数据处理：

1. 解析 WAV、TextGrid、JSON、MusicXML。
2. 提取每个词/字及其首个发声音素的起止时间。
3. 中国歌曲优先按汉字/音节映射；英文优先按词和可对应的音节处理，不强行把英文字符当作独立发音单位。
4. 保留原始 48kHz 音频及其真实时间轴。
5. 建立统一 `AudioSegment / LyricUnit / Phoneme / Boundary` 数据结构。
6. 将时间标注转换为原始音频采样点索引，同时保存 TextGrid 原始时间。
7. 按歌曲分组划分 train/val/test，严格避免同一录音、切片、伴奏版本跨集合；另设歌手独立测试集（可行时）。
8. 分类标记连读、气声、滑音、长音、休止、重唱及首辅音类型。
9. 先抽取少量数据做人工复标，评估 TextGrid 本身的边界误差。

重点：不能假定 GTSinger 标注具有 5ms 的绝对准确性。需明确标注的音素起点是否符合本项目的“实际演唱起点”定义。

对于无法稳定确定的边界，标记为 ambiguous，不强迫模型学习任意单点标签。

数据集主要用于干净人声评估与模型训练。额外构造带伴奏音频（需有合法来源），测试 Demucs 对结果的影响。

## 5. 阶段 P2：粗对齐增强

先保留原版声学模型，将它作为候选边界生成器。

实现统一接口：

```
coarse_align(audio, lyrics) -> list[BoundaryCandidate]
```

候选来源：

- 原版 MTL + CTC 风格 Viterbi 对齐。
- MTL + BDR。
- 可选 Whisper / faster-whisper ASR 辅助。
- 有条件时增加现成多语言音素强制对齐模型作为对照。

Whisper 的职责主要是校验歌词文本、发现漏唱、错词、重复行和大范围错位。

已有正确歌词时，应优先进行条件化歌词对齐，而不是重新识别并替换歌词。

新增：

- 显式语言识别与分句、分词处理。
- 歌词重复片段匹配。
- 行级锚点、静音区间约束。
- 长音、停顿和相同音素连续出现的路径约束。
- 局部窗口内的带状动态规划，减少整首歌曲的计算量。
- 每个候选保存来源、分数、置信度和搜索窗口。

不要简单平均不同模型的时间戳。应先验证评分是否可校准，再学习模型融合权重。

## 6. 阶段 P3：无需训练的高分辨率精修

这是最高优先级的精度实验。

对于每个粗对齐边界 `t0`，截取例如 ±150ms 的局部音频，并依据粗对齐置信度动态调整窗口。

用原始音频和人声分离结果提取：

- 短时能量与能量变化率。
- Spectral Flux / 谱变化。
- 高频噪声、爆破辅音与摩擦辅音线索。
- 基频及周期性变化（作为辅助，不直接等同于字词起点）。
- 局部语音活动概率。
- 可选的不同窗口长度 STFT 特征。

在 48kHz 音频上采用 120 samples（2.5ms）或 240 samples（5ms）的局部特征步长。

注意：特征步长并不等于模型误差。STFT 窗口长度、中心填充和边界延迟均需校准。

通过可调权重计算候选边界得分：

```
score(t) = w1 * onset(t) + w2 * phoneme_transition(t) + w3 * boundary_probability(t) + w4 * acoustic_consistency(t)
```

按音素类别调整特征权重；辅音起点和元音起点不能一律用能量阈值判断。

如分数不可靠，保留粗对齐时间并输出低置信度，不进行无根据的“精修”。

验收：在 GTSinger test 集上与 P0 对比，至少报告 MAE、P90、±5/10/20/50ms 覆盖率及失败样本。

## 7. 阶段 P4：轻量神经网络边界检测器

当 P3 无法满足目标时，训练独立的小模型，避免一开始重训大型 CNN–BiLSTM。

建议使用约 0.5M–2M 参数的 CNN/TCN Boundary Refiner。

输入：

- 以粗对齐时间为中心的局部音频窗口。
- 高分辨率 log-Mel / 频谱变化特征。
- 可选原始波形特征分支。
- 当前音素及前后音素的 embedding。
- 粗对齐概率、局部声学状态及歌词边界类型。

输出：

- 局部候选位置的起点概率热图。
- 相对粗对齐位置的时间偏移 `delta_t`。
- 置信度或不确定性估计。

损失函数：

```
L = λ1 * L_boundary + λ2 * L_regression + λ3 * L_consistency
```

其中：

- `L_boundary`：软标签 BCE / Focal Loss。
- `L_regression`：Smooth L1 / Huber。
- `L_consistency`：相邻字词边界顺序及声学一致性约束。

训练时对粗时间戳注入随机偏差，模拟真实推理误差；不要直接把真实边界作为输入中心而不给推理阶段同等条件。

增加轻微噪声、混响、音量变化和伴奏混合增强，但不能引入未补偿的时间平移。

训练采用 AMP、梯度累积和早停。先训练轻量模型，再根据消融效果考虑对原版 MTL 或 BDR 微调。

## 8. 阶段 P5：联合优化与多级精度

整体流程：

`Audio + Lyrics`
→ `Normalize & Phonemize`
→ `Vocal Separation (optional)`
→ `Coarse Alignment`
→ `Boundary Candidates`
→ `DSP Refinement`
→ `Neural Boundary Refinement`
→ `Monotonic Global Optimization`
→ `LRC Export`

根据置信度分配计算资源：

- 高置信度边界：快速局部修正。
- 中置信度边界：完整神经网络精修。
- 低置信度边界：扩大搜索窗口并重新进行局部对齐。
- 仍无法确认的边界：保持低置信度并提供人工复核接口。

全局优化应保证词序、行序和合法时间区间，同时允许相邻发音连续、重叠或弱分界的情况；不能机械强制固定最小字间隔。

对精修结果做统一时间轴映射，校准分离模型延迟、重采样延迟、STFT 中心偏移、音频解码起点以及音频切片偏移。

对于 MP3，特别检查编码器延迟和解码后的真实播放时间。

## 9. 阶段 P6：评估和消融

至少比较：

| 编号 | 系统                       |
| ---- | -------------------------- |
| A    | 原版 MTL                   |
| B    | 原版 MTL + BDR             |
| C    | MTL + DSP 精修             |
| D    | MTL + Neural Refiner       |
| E    | MTL + DSP + Neural Refiner |
| F    | E + 全局约束 / 可选 ASR    |

每个系统报告：

- MAE、Median AE、P90、P95。
- 误差 ≤5ms、10ms、20ms、50ms 的比例。
- 漏对齐率、错误歌词匹配率。
- 不同语言、辅音类型、唱法下的误差。
- 干净人声与伴奏混合音频的差异。
- 推理耗时、峰值显存、训练成本。
- 基于歌曲分组的置信区间。

5ms 必须使用独立复核过的人工标注集评价；如标注误差显著高于该目标，则先改善标注。

各阶段目标是逐步改善，例如先稳定进入 50ms、20ms、10ms 范围，最后探索 5ms。阶段阈值是研发里程碑，不是未经测量的精度保证。

需要建立失败样本可视化：波形、频谱、真实边界、候选边界、预测边界、误差和置信度。

## 10. 开发与运行环境

Mac：代码编辑、单元测试、少量数据调试，不承担训练任务。

Windows + Arch Linux WSL2：RTX 5060、32GB RAM，承担模型推理、数据处理和训练。

Codex 需要执行：

1. 检查 WSL2 GPU 是否可用、`nvidia-smi`、显存容量及驱动。
2. 检查 PyTorch CUDA 版本、GPU compute capability 与算子支持情况。
3. 为新版系统建立单独 Python 虚拟环境；不要直接混装原版 PyTorch 2.1.2 和新版 CUDA 包。
4. 优先采用支持 RTX 50 系显卡的正式 PyTorch CUDA 12.8 或兼容版本。
5. 旧环境用于 baseline 复现，新环境用于训练和最终部署。
6. 使用相同随机种子、数据切分和统一评测程序。
7. Windows 主机安装合适的 NVIDIA 驱动，WSL 内不要另外安装 Linux NVIDIA 显示驱动。
8. 数据优先存放在 WSL Linux 文件系统中，数据集体积较大时先下载中文/英文子集。
9. 如 GPU 显存不足，降低 batch size、缩短训练窗口、启用 AMP 和梯度累积，不降低最终时间轴精度。
10. 所有 GPU 依赖版本和环境检测结果保存到文档。

## 11. 建议新增的工程模块

建议在不破坏现有 `t2l` 结构的前提下增加：

```
t2l/
  alignment/
    coarse.py
    candidates.py
    constrained_dp.py
    refiner.py
    fusion.py
  boundary/
    features.py
    dsp.py
    model.py
    inference.py
  datasets/
    gtsinger.py
    annotations.py
  evaluation/
    metrics.py
    benchmark.py
    visualization.py
  audio/
    timeline.py
    preprocessing.py

scripts/
  prepare_gtsinger.py
  train_boundary.py
  evaluate_alignment.py

configs/
  baseline.yaml
  high_precision.yaml

tests/
  test_timeline.py
  test_boundary.py
  test_high_precision.py
```

路径为建议新增结构，不代表仓库中已经存在。

CLI 增加：

- `--precision fast|balanced|high|ultra`
- `--aligner mtl|mtl_bdr|hybrid`
- `--refiner none|dsp|neural|hybrid`
- `--device auto|cpu|cuda|mps`
- `--json-output`
- `--enhanced-lrc-output`
- `--confidence-output`

保持原 CLI 的默认兼容行为。

内部精度至少保留微秒级浮点或采样点时间，最终输出 LRC 时再舍入到 1ms；不能在中途反复舍入或截断。

保留标准 LRC，新增增强型逐字 LRC 与含置信度的 JSON 输出。

## 12. 最终交付与验收

要求 Codex 提供：

1. 能运行的增强版代码和兼容原项目的 CLI。
2. GTSinger 数据准备、训练、测试和评估脚本。
3. 轻量边界模型及可复现的训练配置。
4. 每个阶段的基准测试结果与消融表。
5. 误差分布图、失败案例及人工复核方式。
6. Mac 开发、WSL2 GPU 运行说明。
7. 一组不依赖 GPU 的基础 CI 测试，以及可选 GPU 集成测试。
8. 模型许可证、数据许可证与来源说明。
9. 最终技术报告：已实现的精度、仍存在的问题及下一阶段建议。

执行顺序严格遵循 P0 → P1 → P2 → P3 → P4 → P5 → P6。

每个阶段通过验收后再进入下一阶段。优先用最小改动验证精度收益，不为技术复杂度而引入模型。

**最终成功标准：在隔离的、人工复核的真实测试集上，MAE 达到或接近 5ms，并同时控制 P90 和严重错位率。未达到时如实报告，不得将 5ms 的输出步长当成实际定位误差。**
