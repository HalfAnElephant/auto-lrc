# P0 baseline runner

从仓库根目录使用独立的旧版 Python 环境执行。脚本仅用于 CPU 声学模型复现；默认关闭人声分离，不修改生产模型结构或检查点。

```bash
.venv-baseline/bin/python scripts/benchmark_baseline.py \
  demofile/original_txt.txt demofile/original_track.mp3 \
  --variants mtl,mtl_bdr --lstm-layouts legacy,batch-first \
  --repeats 2 --seed 7 --threads 2 -o reports/p0/baseline
```

输出目录必须不存在，避免覆盖先前实验。只运行原版时省略 `--lstm-layouts`；默认 `legacy`。`--vocalize 1` 使用现有 Demucs 默认配置，可能下载模型；这条分支遵循现有自动设备选择，声学模型仍使用 CPU。

每个重复实验先重置 Python、NumPy 和 PyTorch 的随机种子，再重新加载检查点。模型构造到对齐之间不再次重置随机种子。基准预加载模型早于歌词和音频处理，而原 `process()` 在处理之后加载；与原版是否完全一致须比较实际结果，不能仅凭相同 seed 推定，尤其不能把关闭分离时的结论推广到 Demucs 分支。`batch-first` 仅在加载后的实例上把所有 LSTM 设为 `batch_first=True`，记录每层原值与实验值。参数名称与形状兼容并不代表旧检查点适合修正后的模型，也不代表精度改善。

线程配置在导入音频模块之前设置，避免 macOS 原生后端在 MelSpectrogram 初始化后固定线程数；manifest 记录实际生效的线程数。早期未生效的实验须单独保留并排除出规定配置的比较。

产物：

- `manifest.json`：命令、环境与依赖版本、Git 状态、输入及所用模型的 SHA-256、配置和精度限制。
- 每次运行的 `.json`：原始字词结果、音素、未舍入的帧与秒、全局分数、LSTM 配置、真实推理期间的模块输入输出形状。
- 每次运行的 `.lrc` / `.standard.lrc`：增强与标准格式；内部比较不读取这些格式化时间戳。
- `summary.json`：耗时、结果数及重复性。指纹比较原始字词结果、LRC 与全局分数，排除耗时；只有一次运行时重复性结果为 `null`。

`runtime_seconds` 包含歌词处理、音频解码、可选分离、特征、声学推理与对齐，以及轻量审计 hooks；检查点加载时间单独记录。重复实验会受到文件缓存和首次库初始化影响，不能直接视为 GPU 部署性能。

时间轴审计读取实际 MelSpectrogram 的 FFT、hop、padding 与 center 设置，记录卷积、池化、LSTM 与模型输出的形状。当前池化输出 k 组合 Mel 帧 3k、3k+1、3k+2，其几何中点是 3k+1，而原输出仍按 `k * 768 / 22050` 秒计时。CNN 上下文、双向网络和池化进一步影响声学响应，不能仅据该中点给所有时间戳加固定偏移。解码、重采样、人声分离延迟与真实发声起点仍需另行校准。

没有独立复核的真实边界时，MAE、P90、覆盖率保持不可用。原始全局对齐分数未经校准，不能解释为字词置信度或准确率。P0 复现通过也不代表达到 5ms 精度。

诊断模式若收到非有限路径分数，会抛出 `AlignmentValueError`，不导出含 NaN/Infinity 的 JSON。默认非诊断接口保留原有行为。多个 CLI 输出文件按顺序写入，文件系统写入失败时不保证整组文件原子提交。
