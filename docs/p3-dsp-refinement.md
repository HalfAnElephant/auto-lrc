# P3：无训练 DSP 边界精修

2026-10-10：P3 的特征、精修、固定配置选择及真实 WSL 评估已完成。默认系统保留 `coarse`；验证集最佳试验为 `low_prior`，但未通过启用门槛。下一阶段为 P4 轻量边界模型训练。本阶段按用户要求未安排外部或独立只读审查。

## 实现与时间几何

`t2l/boundary/features.py` 的 `extract_features(audio, sample_rate, hop_samples=120, window_ms=20.0)` 接受 mono 或 channel-first NumPy 音频，沿用第一声道，返回 `sample_positions / features / metadata`。位置为原音频整数采样点，不重采样、不平移。48kHz 下 hop=120 为2.5ms；原生44.1kHz分离音频仍使用其自己的采样网格。

主谱/周期性窗为20ms，能量/高频辅助窗最多5ms。窗以 `sample_positions` 为中心，超出音频部分补零；频谱使用 periodic Hann。特征包括能量、能量正变化、谱通量、谱变化、高频比例及其正变化、周期性及其正变化、活动代理。周期性仅为辅助相关线索，活动代理不是模型 VAD 概率。中心窗会看到未来采样点，可能在真实起点之前响应；报告明确 `latency_calibrated=false`、校正0样本，未猜测或扣除半窗延迟。2.5ms 特征步长不构成2.5ms或5ms误差保证。

`t2l/boundary/dsp.py` 提供冻结的 `DSPConfig` 和 `refine_boundaries(audio, sample_rate, candidates, config, features=None)`。默认在粗候选附近 ±150ms 内评分，依据粗模型首音素类别加权多种线索，并加粗起点先验。只建议起点，结束时间仍保留粗候选含义；弱证据、收益不足、缺失粗对齐或顺序冲突时显式 fallback。保留原候选字段和细节，`refinement` 记录状态、原因、原起点及 `delta_samples`；置信度仍为 null/uncalibrated。精修函数不接收来源标注，也不读取人工标签。

## 固定选择规则

输入为 P1 的18段来源模式划分及 P2 已完成的 `mtl:legacy / mtl_bdr:legacy` 粗报告。固定 train/val/test 为7/3/8段、96/37/98个来源边界。所有误差均相对公开数据集的 `textgrid:first_phone` 标注，`independent_human_accuracy=false`；没有写入或伪造人工真值。当前 val 仅中文，test 仅英文；歌曲别名及来源起点未获人工确认。

预声明网格如下，另以不精修的 coarse 为基线。三种配置均使用 `min_gain=0.05 / hop_samples=120 / radius_seconds=0.15`。

| 配置 | prior_weight | min_evidence |
| --- | ---: | ---: |
| conservative | 0.25 | 0.25 |
| low_prior | 0.10 | 0.25 |
| high_prior | 0.50 | 0.50 |

只读取 val 标签选择，两模型合并 MAE 最低的配置作为 `best_trial_config`，并列按声明顺序。启用该最佳试验还要求：MAE 严格改善、evaluated 数不下降、全部边界分母上的 ±5/10/20/50ms 覆盖均不下降、`>500ms` 严重错误数不增加。未通过时 `selected=none / selected_config=null`，默认保留 coarse。没有从 test 选择替代配置。

验证结果合并两模型，相同37个边界各计一次，总分母74：

| 配置 | MAE ms | ±5ms | ±10ms | ±20ms | ±50ms | >500ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 151.75 | 2.70% | 2.70% | 8.11% | 18.92% | 2 |
| conservative | 135.27 | 2.70% | 5.41% | 5.41% | 24.32% | 2 |
| low_prior | 132.63 | 0.00% | 2.70% | 5.41% | 29.73% | 2 |
| high_prior | 144.07 | 2.70% | 5.41% | 8.11% | 21.62% | 2 |

`low_prior` 的 MAE 最低，但 ±5ms 覆盖从2.70%降至0%，±20ms从8.11%降至5.41%，故启用失败。选择文件保存整个网格、规则、val结果、输入摘要及 DSP/features 源码摘要；evaluate 核验这些摘要并冻结配置。coarse 报告必须 completed、输入一致且行覆盖/歌曲/划分匹配，原 WAV 也经摘要校验。缺失或文本错配保留在指标分母，不以丢弃失败提高覆盖。

## 真实 WSL 固定评估

evaluate 始终报告已冻结的 `best_trial_config=low_prior` DSP 试验，即使默认选择仍是 coarse；只运行这个 val 最佳试验，没有在 test 扫描网格。train 指标仅作诊断，不参与选择。两模型在本子集结果相同，下表每模型分母独立计数，未将重复预测当额外独立样本。

| 集合 | 边界数 | coarse MAE ms | DSP试验 MAE ms |
| --- | ---: | ---: | ---: |
| train | 96 | 105.26 | 95.86 |
| val | 37 | 151.75 | 132.63 |
| test（仅英文） | 98 | 82.58 | 79.96 |

| test 指标 | coarse / selected | DSP试验 |
| --- | ---: | ---: |
| MAE ms | 82.58 | 79.96 |
| P90 ms | 166.64 | 173.25 |
| P95 ms | 217.66 | 243.53 |
| ±5ms | 5.10% | 3.06% |
| ±10ms | 11.22% | 10.20% |
| ±20ms | 19.39% | 23.47% |
| ±50ms | 45.92% | 53.06% |
| >500ms | 1 | 2 |

平均误差下降，但 P90/P95、±5/10ms覆盖和严重错误恶化，不能视为达到5ms目标或稳定收益。`selected` 指标与 coarse 相同。报告18行 completed、结构性 `failure_count=0`；这不表示没有大误差。WSL Python3.10.16 / NumPy1.26.3，DSP 在 CPU 执行，GPU为 null；总墙钟2.839秒、CPU2.829秒，不包含已复用的 P2 模型推理。

## 受控混音诊断

复用 P1 中英文各一首训练歌曲的两段、25个来源边界及已完成的粗预测/Demucs结果，在同一固定 `low_prior` 下比较清唱、原创合成伴奏混音、原生44.1kHz分离音频；没有混音专用调参或重跑模型。两模型结果相同，下表每模型独立分母25。

| 版本 | coarse MAE ms | DSP MAE ms | coarse P90 ms | DSP P90 ms | coarse ±5ms | DSP ±5ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 90.39 | 69.82 | 174.60 | 128.20 | 4.00% | 0.00% |
| mixture | 121.32 | 126.38 | 222.59 | 254.10 | 0.00% | 4.00% |
| separated | 118.25 | 102.31 | 174.60 | 183.80 | 4.00% | 0.00% |

混音 MAE/P90恶化；清唱和分离音频的 MAE 改善也未带来5ms覆盖改善。此处仅为两段训练诊断，不是独立 test、不覆盖自然配器，也不能证明逐音素分离延迟已校准。保留原/分离网格，不施加平移。WSL CPU 实测墙钟1.403秒。

## 证据、验证与恢复

完整机器结果索引见 [P3 报告](../reports/p3/README.md)。分布图和6个最大误差窗口见 [图索引](../reports/p3/dsp-figures-20261010/index.json)；每窗呈现波形、频谱及局部候选评分、粗/DSP候选和来源标注位置。图中的来源位置不代表人工声学真值。

已完成的定点检查为 feature39 + DSP20 + benchmark40 + mixture1 = **100项通过**；benchmark40沿用已有成功结果。文档收口未重跑测试、模型或云端 CI。无本阶段外部审查。

以下为归档调用，须在仓库根执行；归档输出目录已存在，脚本会拒绝覆盖。相关输入未变化时直接读取现有结果。若新实验确有必要，改用全新的 `--out-dir`，不要改写旧证据。

```bash
# Mac：仅 val 配置选择
.venv-baseline/bin/python scripts/benchmark_dsp_gtsinger.py \
  --segments reports/p1/representative-v2/segments.jsonl \
  --splits reports/p1/source-mode-20261010/splits.json \
  --audio-root data/local/gtsinger-representative \
  --coarse-report reports/p2/coarse-cuda-20261010/result.json \
  --out-dir reports/p3/dsp-select-20261010

# WSL：固定选择后的全划分评估，不调参
.venv-cuda/bin/python scripts/benchmark_dsp_gtsinger.py \
  --segments reports/p1/representative-v2/segments.jsonl \
  --splits reports/p1/source-mode-20261010/splits.json \
  --audio-root data/local/gtsinger-representative \
  --coarse-report reports/p2/coarse-cuda-20261010/result.json \
  --selection-file reports/p3/dsp-select-20261010/selection.json \
  --out-dir reports/p3/dsp-evaluate-wsl-20261010

# WSL：复用已有两段训练混音/分离推理
.venv-cuda/bin/python scripts/benchmark_dsp_mixtures.py \
  --segments reports/p1/representative-v2/segments.jsonl \
  --splits reports/p1/source-mode-20261010/splits.json \
  --audio-root data/local/gtsinger-representative \
  --coarse-report reports/p2/coarse-cuda-20261010/result.json \
  --mixture-report reports/p1/mixtures-cuda-20261010/result.json \
  --selection-file reports/p3/dsp-select-20261010/selection.json \
  --out-dir reports/p3/dsp-mixtures-wsl-20261010
```

可视化额外依赖为 `requirements-research.txt` 中的 `matplotlib==3.8.4`；NumPy/SciPy沿用既有版本，原推理 requirements 未改。WSL 混音/分离 WAV 的恢复路径见 [P1 数据文档](p1-data-preparation.md)。

## 下一阶段

P3 交付完成，但没有5ms成功证据。按规格进入 P4，训练独立的小型边界模型；训练/调参仅用 train/val，已有 test 歌曲继续隔离。扩大训练数据时仍须保留父歌曲、同音频及歌手策略，不能将当前 test 调回训练。用户跳过人工阶段的决定保留，后续仍须将指标称为来源标注误差；不将当前小规模英文 test 宣称为最终多语言精度验收。
