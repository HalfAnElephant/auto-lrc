# P0 复现、布局消融与 GPU 验收结果

实验日期：2026-10-09。原版 MTL / MTL_BDR 与当前 legacy 输出在同一 CPU 环境、本 demo 上完全一致；各配置两次重复均一致。布局修正会显著改变预测，尚无独立真实标注，不能认定准确度改善，也不能宣称达到 5ms。P0 运行与复现验收已完成，GPU 结果另列如下；跨平台输出不保证相等。

## 实验边界与来源

- 同一完整 `demofile/original_track.mp3` 和 UTF-8 歌词；不做人声分离，`vocalize=False`。
- Mac arm64、Python 3.10.16、torch / torchaudio 2.1.2；完整依赖见 [冻结文件](../reports/p0/requirements-baseline.freeze.txt) 及各 manifest。环境和 Windows / WSL 状态见 [environments.md](environments.md)。
- seed=7，实际 CPU intra-op=2、inter-op=1，确定性算法开启；每次重复从种子重置和重新加载模型开始。这里只验证无 Demucs 的 seed 消费路径。
- 原版 commit 为 `2c475cc62dda50572d7a292fd88cc39e8aa2ae2c`。快照位于 `/tmp/auto-lrc-p0-original-2c475cc`。39 个普通 tracked 文件与 Git 内容逐字节一致；4 个 LFS 实体与该 commit 的 OID 一致，见 [source-verification.json](../reports/p0/original-threads2/source-verification.json)。原版源代码没有改动，外部 harness 只记录原对齐函数的输入/返回值。
- 原版与新版的歌词、音频、MTL / BDR 检查点及语言模型 SHA-256 均一致。当前运行代码见 [摘要](../reports/p0/demo-threads2/run-source-sha256.json)、[tracked diff](../reports/p0/demo-threads2/implementation.diff) 与 [runner 源码快照](../reports/p0/demo-threads2/benchmark.py)；后者用于审计，实际入口仍是仓库 `scripts/benchmark_baseline.py`。

第一次实验发现，Mac native backend 在导入 `t2l` 创建 MelSpectrogram 后，`set_num_threads(2)` 已无法生效，实际保持 8。该实验在 BDR 首轮中断；已完成的两次 MTL、错误配置和日志保留在 [original](../reports/p0/original/summary.json)，不计入本报告验收。之后将线程设置提前到 `t2l` 导入前，使用全新目录重跑。两个有效 manifest 均记录了实际 2 线程。

## 复现结果

每次输出 482 个字词单位、1712 个音素、9356 个模型时间帧。输出单位数不等于真实边界覆盖率。

| 实现 | 模型 | 布局 | 未校准全路径 score | 含加载的观察耗时 r1 / r2（秒） | 两次重复 |
| --- | --- | --- | ---: | ---: | --- |
| untouched 2c475cc | MTL | legacy | -11098.0625 | 9.535 / 6.867 | 一致 |
| 当前诊断接口 | MTL | legacy | -11098.0625 | 10.841 / 7.061 | 一致 |
| 当前诊断接口 | MTL | batch-first | -11174.775390625 | 9.010 / 7.559 | 一致 |
| untouched 2c475cc | MTL_BDR | legacy | -11196.181801110975 | 71.513 / 74.830 | 一致 |
| 当前诊断接口 | MTL_BDR | legacy | -11196.181801110975 | 75.137 / 76.358 | 一致 |
| 当前诊断接口 | MTL_BDR | batch-first | -11290.255047954935 | 86.264 / 80.909 | 一致 |

当前耗时为 `model_load_seconds + runtime_seconds`；原版为整次 `process()` 耗时。当前包含轻量形状 hooks，两者均受首次初始化、缓存和系统负载影响；这不是严格的性能对照。MTL 与 MTL_BDR 的 score 定义不同，且都未经标注校准，不能跨模型当成准确率。

原版与当前 legacy 的 **4 组逐次比较全部完全相等**，检查项为原始字词起止帧、全路径 score、音素序列、字词到音素的索引、输出帧数、增强 LRC 字符串。没有使用容差或 LRC 舍入值来比较原始时间。原版指纹和当前指纹覆盖字段不同，因此只用于各自重复性；跨版本用字段逐值比较。

证据：[原版 summary](../reports/p0/original-threads2/summary.json)、[新版 summary](../reports/p0/demo-threads2/summary.json)、[完整比较](../reports/p0/demo-threads2/comparison.json)。每轮 JSON、增强 LRC 和日志均保留在相应目录，新版还保存了标准 LRC。

## 真实形状与时间轴

下表来自完整 demo 推理时的 hooks，没有另跑合成输入。所有布局的张量输入/输出形状保持一致。

| 位置 | 实测形状 |
| --- | --- |
| 22,050 Hz 输入的首通道 | `[1, 7185392]` |
| MelSpectrogram | `[1, 128, 28068]` |
| 声学 CNN / 残差 CNN | `[1, 32, 128, 28068]` |
| MaxPool2d（kernel / stride = 2 × 3） | `[1, 32, 64, 9356]` |
| 声学第一层 LSTM | `[1, 9356, 256] → [1, 9356, 512]` |
| 声学第二、三层 LSTM | `[1, 9356, 512] → [1, 9356, 512]` |
| MTL 输出 | `[1, 9356, 41, 47]` |
| BDR 第一层 LSTM | `[1, 9356, 32] → [1, 9356, 64]` |
| BDR 第二、三层 LSTM | `[1, 9356, 64] → [1, 9356, 64]` |
| BDR 输出 | `[1, 9356, 1]` |

STFT 实测 `n_fft=win_length=512`、`hop_length=256`、`pad=0`、`center=True`、`pad_mode=reflect`。输出帧数正好为 `28068 // 3 = 9356`。保留旧映射 `frame × 768 / 22050`，名义时间步长约 **34.829932 ms**，帧率约 28.710938 Hz。

中心化 STFT 的第 0 帧以输入 sample 0 为中心；池化帧 k 覆盖 Mel 帧 3k、3k+1、3k+2，几何中点为 3k+1。该几何信息不等于实际发声起点校准；没有据此批量修正时间戳。MP3 解码、重采样、网络上下文以及可选人声分离的实际时间偏移仍需标注/信号实验。

## BiLSTM 布局消融

真实 hooks 确认两个模型的第二、三层均收到 `[batch=1, time=9356, features]`，旧 `batch_first=False` 却把长度为 1 的 axis 0 解释为时间、把 9356 解释为 batch。修正实验仅把本次新加载实例的所有 LSTM 设为 `batch_first=True`，保留参数、模型定义和生产默认行为。

下表的“位移”是同一单位在 batch-first 与 legacy 之间的 **预测起点差的绝对值**，不是相对真实标注的误差。90 分位数采用位置 `(n−1)×0.9` 的线性插值。

| 模型 | 起点改变数 / 总数 | 平均绝对位移（ms） | 中位绝对位移（ms） | 90 分位绝对位移（ms） | 最大绝对位移（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTL | 235 / 482 | 1485.908 | 0.000 | 5315.048 | 16091.429 |
| MTL_BDR | 274 / 482 | 1757.394 | 417.959 | 5475.265 | 16091.429 |

MTL 的终点改变 235 个，MTL_BDR 的终点改变 275 个。两种修正布局均改变增强 LRC，但各自两次重复完全一致。参数形状兼容和相同输出维度不足以证明旧检查点适用于新时序语义；当前证据支持保留 legacy 为默认，把修正布局继续作为待评估实验。

## 命令与复核

原版执行命令（输出目录运行前不存在）：

```sh
cd /tmp/auto-lrc-p0-original-2c475cc
/Users/xcy/Program/auto-lrc/.venv-baseline/bin/python -u run_original_p0_threads2.py
```

实际脚本冻结于 [reproduce.py](../reports/p0/original-threads2/reproduce.py)。重做实验时必须先准备同 commit 快照与 LFS 实体，并将脚本的 `OUT` 改为一个新目录；脚本主动拒绝已有目录。

当前执行命令（仓库根目录）：

```sh
.venv-baseline/bin/python scripts/benchmark_baseline.py \
  demofile/original_txt.txt demofile/original_track.mp3 \
  -o reports/p0/demo-threads2 --variants mtl,mtl_bdr \
  --lstm-layouts legacy,batch-first --repeats 2 --seed 7 --threads 2
```

纯读复核已保存结果，不重新推理、不写文件：

```sh
.venv-baseline/bin/python reports/p0/demo-threads2/compare.py
```

[比较脚本](../reports/p0/demo-threads2/compare.py) 也接受 `--original DIR --current DIR`。保存的 `comparison.json` 是该脚本的 stdout。

## RTX 5060 / WSL2 真实验收

隔离环境为 Python 3.10.16、torch / torchaudio 2.7.1+cu128，RTX 5060 compute capability 12.0；CUDA 卷积、LSTM、STFT 和 86 包依赖检查通过。没有修改 Linux 显示驱动或原 Mac 环境依赖版本。实际 freeze、安装兼容处理与恢复命令见 [environments.md](environments.md)。

| 阶段 | 真实范围 | 结果 | 观察耗时（秒） | PyTorch 峰值分配显存（MiB） |
| --- | --- | --- | ---: | ---: |
| MTL | 完整 demo，不分离人声 | CUDA forward、482 个完整单位、有限数值通过 | 20.169 | 2238.62 |
| MTL_BDR | 完整 demo，不分离人声 | 声学/BDR CUDA forward、482 个完整单位、有限数值通过 | 85.179 | 2243.59 |
| mdx_extra | 30–36 秒片段、默认 4 成员集成 | 每个成员 CUDA forward，输出 `[2,132300]`、22050Hz、有限人声 | 分离 7.950 | 1605.90 |
| 默认 CLI | 完整 demo，默认 MTL + mdx_extra 人声分离 | CUDA 对齐、482 单位、JSON/标准/增强 LRC 一致通过 | 30.387 | 未测量 |

Demucs 首次加载含四个官方检查点下载，耗时 384.831 秒；不把下载混入分离推理时间。默认 CLI 使用已缓存检查点，且线程、执行入口与单阶段脚本不同，表格不构成性能对照。所有 GPU 阶段只运行一次，不声明 GPU 重复性或确定性。

可复核产物：`reports/p0/gpu/{mtl,mtl_bdr,demucs,default-cli}/result.json`，各自保留 JSON/LRC，Demucs 另存输入片段和分离人声；默认 CLI 保留完整日志。单阶段使用 [verify_gpu_runtime.py](../scripts/verify_gpu_runtime.py)，拒绝 CPU 回退和复用输出目录。远端运行源码摘要与当前本机来源一致。

默认 CLI 实际命令：

```sh
.venv-cuda/bin/python main.py demofile/original_txt.txt demofile/original_track.mp3 \
  -o reports/p0/gpu/default-cli \
  --json-output reports/p0/gpu/default-cli/alignment.json \
  --enhanced-lrc-output reports/p0/gpu/default-cli/enhanced.lrc
```

GPU 与 Mac CPU 输出并非逐位一致：MTL 有 1/482 个起点改变 1 帧；MTL_BDR 有 79/482 个起点不同，最大差 649 帧（约 22.60 秒）。[跨平台比较](../reports/p0/gpu/comparison.json)保存原始计数与 score。这里同时改变了操作系统、torch 版本、计算设备和数值后端，尚未隔离具体原因；不能将差异归因于某一个因素，更不能当作精度改善。后续评估应固定运行环境，BDR 的路径敏感性须纳入失败分析。

## 未覆盖事项

没有独立复核的人工标注，因此准确度 MAE / P90 / ±5、10、20、50ms 覆盖率均为 **unavailable**。以上两次重复只证明本机、固定输入与固定 seed 下的输出重复性，不证明跨平台逐位一致、标注精度、GPU 性能或 Demucs 后的时间轴。

CPU 实验没有运行人声分离。GPU 验收包括真实分离，但未测量人声分离带来的边界延迟或准确度，也不覆盖训练。独立审查产物见 [reports/p0/review](../reports/p0/review)，整个阶段的状态由 [WORK_PROGRESS.md](WORK_PROGRESS.md) 统一维护。
