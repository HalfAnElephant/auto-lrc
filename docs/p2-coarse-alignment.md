# P2 粗对齐候选与局部约束

2026-10-10：P2 已完成接口、受约束解码和真实 CUDA 对照。按用户要求使用数据集来源标注，不进行人工复标或独立只读审查。没有宣称精度改善。

```python
from t2l.alignment import coarse_align

candidates = coarse_align("song.wav", ["第一行歌词", "第二行歌词"],
                          method="mtl_bdr", language="zh", vocalize=False)
rows = [candidate.to_dict() for candidate in candidates]
```

`method` 支持 `mtl`、`mtl_bdr`，可传 `load_mtl_model()` 返回的相容预加载 tuple。输入为音频路径及原始字符串/行列表；使用实际推理字词单位，不重写歌词。显式语言为 zh/en/ja/ko/ru，auto 使用既有语言判定；候选文本元数据另外保留脚本和 unknown，不能将所有拉丁字母当作英文。英文显式模式只将发音输入转小写，原文字保留。

候选包含行/词/重复出现次序、音素、原始帧及时间、原始解码音频采样率和长度、原始网格上的名义坐标、未校准全路径分数、局部搜索窗口和失败说明。`confidence=null`，没有把分数伪装成概率。缺失或越界起点保留为 unaligned，不能以裁剪制造有效边界；半开末端超过原音频时裁候选末端并留痕，原始模型帧和秒始终保留。采样点换算不构成声学原点校准。

## 可选受约束解码

默认仍走原版路径，原 `process()`、CLI 与检查点不变。传入 `constraints` 时启用新 NumPy CTC/Viterbi：

```python
candidates = coarse_align("song.wav", ["hello world"], language="en",
    constraints={"mode": "constrained",
                 "word_windows": [[0, 40], [20, 90]],
                 "onset_windows": [[0, 10], [20, 50]],
                 "silence_intervals": [[0.0, 0.1]],
                 "fallback_to_legacy": True})
```

示例窗仅说明格式，须按实际音频/词数提供。`word_windows` 为每词全部音素可占用的半开帧范围，`onset_windows` 仅限制首音素首次进入；歌词行首也可用同一机制约束。`silence_intervals` 为原始解码时间轴秒，按既有名义帧网格投影；该网格物理原点尚未校准。窗口来自粗预测或独立输入，禁止从测试真值生成。

相邻同音素必须经过 blank，长音可持续占用，同一词/行重复出现仍是独立状态；静音允许 blank 和显式词间 pause 类别，禁止其他发声音素。BDR 只在指定行首音素首次进入时计分。新返回区间为音素占用的半开区间；旧版 end 保持原有“下词或终止帧”约定，两者用 `frame_interval_convention` 区分。

仅保留两行活动浮点分数与活动带的 int8 回溯，窄带减少状态计算；宽带回溯仍可能为 O(T×P)。非法参数直接失败；合法约束不可达时抛 `NoAlignmentPathError`，默认包装器返回原版路径并明确 `satisfied=false` 和回退原因，严格模式可禁用回退。不做候选均值融合。Whisper/外部强制对齐是原规格可选项，本阶段未引入。

## 验证与结果

相关本地检查189项通过：候选/文本41、受约束路径67、运行时及原wrapper/process34、原CLI16、基准输入/指标31。路径测试含小状态穷举最优分数、重复音素、长音、静音pause、首次BDR、不可达和窄/全带同解。语法与diff检查通过；未跑等价云端CI。

RTX5060 / torch2.7.1+cu128 上18段×2模型×2解码=72次推理全部完成。真实forward hooks确认CUDA输入/输出及有限值，源码摘要与Mac一致；实际输出924候选，无缺失/错配/回退。局部带固定为原版预测两侧0.35秒，未读取标注生成窗、未用test选择参数。完整报告在 [coarse-cuda-20261010/result.json](../reports/p2/coarse-cuda-20261010/result.json)。

四种配置在这批数据的起点完全相同，以下是相对发布TextGrid的误差：

| 划分 | 边界数 | MAE | P90 | ±5ms | ±20ms | ±50ms | >500ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 96 | 105.26ms | 168.93ms | 1.04% | 6.25% | 28.13% | 2 |
| val | 37 | 151.75ms | 215.36ms | 2.70% | 8.11% | 18.92% | 1 |
| test | 98 | 82.58ms | 166.64ms | 5.10% | 19.39% | 45.92% | 1 |

72次调用累计12.232秒，峰值已分配108.32MiB/保留278MiB，不含预加载模型时间；未建立重复推理统计，不解释为稳定性能收益。此子集val只有中文、test只有英文，最终多语言评测仍需扩充。P2保留默认旧解码，后续P3验证高分辨率声学精修，不能把本阶段接口完成解释为5ms准确率达标。
