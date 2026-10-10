# Boundary TCN Model Card

状态：**真实 CUDA 训练和冻结评估 completed，当前暂停交接；最佳权重保存到用户云端仓库，P4最终验收及实际绘图未完成**（2026-10-10）。本卡记录 P4 架构、冻结训练协议和来源验证结果。默认歌词 CLI 仍未启用本模型。检查点没有 optimizer/scaler/RNG 状态，不能从原轮次无缝续训。

## 用途与接口

研究用途：在已有 MTL 粗词起点附近，基于原始音频特征预测局部边界候选。它不识别歌词，不替换原声学模型，不提供独立人工起点真值或全局对齐保证。2.5ms 是 48kHz 下的特征 hop，不是准确率；禁止据此宣称 5ms 精度已实现。

模型输出热图、可微相对偏移和 **uncalibrated** confidence。推理取有效热图 argmax，保留 coarse/raw frames、top candidates 和不确定性诊断；缺失窗口和顺序冲突显式回退粗候选，不自动修正音频或真值。

新检查点格式为 `boundary-checkpoint-v1`，字段含严格 `config`、`model_state_dict`、训练来源 metadata。加载必须指定新权重路径，使用 `load_boundary_checkpoint(path, device="cpu"|"cuda"|"mps"|"auto")`；auto 为 CUDA 可用时 CUDA，否则 CPU，显式设备失败不降级。无内置权重或自动下载，不能载入旧 MTL/BDR 检查点。

## 架构

配置：[configs/boundary-tcn.json](../../configs/boundary-tcn.json)，模型版本 `boundary-tcn-v1`，默认 **609,026 参数**。

| 项目 | 定义 |
| --- | --- |
| 原始网格 | 不重采样；hop=`round(sr*.0025)`，48kHz 时 120 samples/2.5ms，frame center 从 0 开始 |
| 声学输入 | 48 维 20ms centered periodic Hann log-Mel（固定 40–12,000Hz）+9 维 DSP，共 57；sr<24kHz 拒绝 |
| 候选窗口 | 281 帧、coarse 起点 ±350ms；追加 [-1,1] relative-position，共 58 通道，越界 zero pad+mask |
| 音素上下文 | 当前首、前词末、后词首 ARPAbet，各 16 维 embedding；stress stripped，未知为 UNK |
| TCN | 128 channels，六个双 kernel-3 Conv1d 残差块，dilation 1/2/4/8/16/32，GroupNorm/dropout |
| 输出 | heatmap logits、weighted relative delta_seconds、未校准 confidence_logit；AMP 下 heads 为 float32 |

窗口中心仅来自 coarse `start_sample`，不读取来源/人工标签；有效位置受原 WAV、物理半径和 coarse word end 限制。centered 窗口没有猜测的 latency 校正。未知原整歌 offset 保持 unknown/null。

## 数据与训练

来源：[AaronZ345/GTSinger](https://huggingface.co/datasets/AaronZ345/GTSinger/tree/d78cd68d00135b1995da03f25fccb3a221efd8c1)，固定 revision `d78cd68d00135b1995da03f25fccb3a221efd8c1`。180 段计划中排除 17 段来源区间越界，保留原件，不裁切成有效标注。实际 163 段、18 首歌、2,436 个来源边界：

| 划分 | 段数 / 歌曲 | 来源边界 | 语言 |
| --- | --- | ---: | --- |
| train | 86 / 10 | 1,371 | 中英（6/4 首） |
| val | 38 / 4 | 554 | 中英（2/2 首） |
| test | 39 / 4 | 511 | 中英（2/2 首） |

所有歌曲变体同 split；`EN-Tenor-1` 全部 test，既有歌曲划分保留。歌曲身份和 TextGrid 首发声音素边界均 **dataset/unreviewed**，人工字段为空，review pending。监督和评估仅是与发布数据标注的差异，`independent_human_accuracy=false`。

固定 seed=7，train-only 梯度、合成波形/特征增强和复制粗中心 jitter；val/test 原音频不增强。混响是 unit direct path+正延迟反射，无循环回绕；合成伴奏是原创振荡器，不是实际配器，不校准起点。真实 CUDA float16 AMP、gradient accumulation=2（batch32）、AdamW、最多60 epochs/patience10；只按 val 相对来源 MAE 选择 epoch 和 early stopping，test 不用于选择。置信头目标为窗口可达性，仍不应作为起点正确概率。

## 验证与限制

实现定点检查已有 neural 53 项、augmentation 25 项通过，含 CPU bfloat16 前向/反传、坐标/边界/fallback/检查点契约；这些结果不是真实训练或 GPU 性能验收。

[真实训练报告](../../reports/p4/training-cuda-20261010/result.json) 记录 RTX 5060、PyTorch 2.7.1+cu128/CUDA 12.8、float16 AMP，共 36 epochs/792 optimizer steps，全部轮次梯度 finite、AMP scale 1024。第 26 轮最优，之后 10 轮无改善按 val early stopping；test 未用于选择。

最优 val 的 554 个来源边界全部纳入（missing/text mismatch=0，20 个来源标签在原粗窗口外仍计入）。来源 MAE **58.4439266681ms**、median 17.5ms、P90 127.3643ms、P95 250.875ms；±5/10/20/50ms 来源覆盖 21.1191%/34.8375%/52.3466%/73.6462%，大于 500ms 的来源误差 8 个。此处仅是选 epoch 的 val 结果，不能作为 test 或独立人工起点准确率。

总墙钟 1,347.8245s（约 22.46min）；训练及逐轮验证 1,291.3629s/CPU time 4,668.9115s，特征缓存另为 50.7171s/CPU time 555.6115s。逐轮 PyTorch CUDA 峰值 allocated/reserved 为 252,772,864/278,921,216 bytes（241.0625/266MiB），不包括整机 GPU 占用口径。

本地 [最佳检查点](../../reports/p4/training-cuda-20261010/boundary.pt) 2,456,370 bytes，已核对文件 SHA256 与报告一致：

```text
967da4debb2a457cf45a2989335183781e41890f0d4c43af3e1b72292a5bb4da
```

| 实验 | 证据位置 | 状态 |
| --- | --- | --- |
| 数据准备 | `reports/p4/training-source-20261010` | completed |
| MTL/BDR 粗候选缓存 | `reports/p4/coarse-cache-cuda-20261010` | completed |
| 神经训练/最佳检查点 | `reports/p4/training-cuda-20261010` | completed；best epoch 26 / 36 |
| 冻结评估 | `reports/p4/evaluation-cuda-20261010` | completed；0 failures；完整JSON已取回并核对摘要 |

冻结 test 共511个来源边界，八组方法均无 missing/text mismatch，0 failures。MTL coarse/DSP/neural/DSP+neural 来源MAE为99.172/91.343/51.516/50.537ms；MTL_BDR为98.014/90.291/50.704/49.735ms。纯neural >500ms 来源误差为7个，其他方法为5个；完整分类、分位数、覆盖、bootstrap及耗时/显存在 [评估报告](../../reports/p4/evaluation-cuda-20261010/result.json)，使用LFS保存。仅18首歌曲和有限歌手，不能推断其他语言、真实配器/分离误差或独立人工起点准确率；仅四首测试歌曲的bootstrap区间也可能不稳定。训练窗口外的来源标签仍计入验证/测试分母，不能以丢失难例抬高覆盖。

## 许可与来源

代码许可见根 [LICENSE](../../LICENSE)（Apache-2.0）及 [t2l/mtl/LICENSE](../../t2l/mtl/LICENSE)（上游 MIT）。**这不等于权重已获 MIT 或商业许可。** 最佳权重随研究交接保存到用户云端仓库，未另行授予权重许可；使用和传播须处理训练数据条款。原数据许可全文随 [交接归档](../../reports/handoff/GTSINGER_LICENSE.md) 保留。

数据许可原文见 `data/local/gtsinger-representative/dataset_license.md` 和固定 revision 的 [dataset_license.md](https://huggingface.co/datasets/AaronZ345/GTSinger/blob/d78cd68d00135b1995da03f25fccb3a221efd8c1/dataset_license.md)：CC BY-NC-SA 4.0，包含署名、非商业、相同方式共享等要求，并附责任/赔偿、代表雇主授权及引用论文条款。使用和后续传播须保留并处理完整条款；本卡不另行替数据或权重授许可。

完整维护命令及待回填字段见 [P4 文档](../../docs/p4-neural-refinement.md)。
