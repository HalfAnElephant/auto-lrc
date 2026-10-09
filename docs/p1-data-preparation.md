# P1 数据准备与人工复核

2026-10-10：代表性子集、来源分类、歌曲/歌手隔离划分工具和本地盲标工具已实现；**P1 整体未验收**。歌曲身份及独立人工边界仍待复核，没有精度结论，不进入 P2–P6。

## 数据来源与实际范围

[GTSinger 官方仓库](https://github.com/AaronZ345/GTSinger)目前指向 [AaronZ345/GTSinger](https://huggingface.co/datasets/AaronZ345/GTSinger)。本次固定 revision `d78cd68d00135b1995da03f25fccb3a221efd8c1`，未混用旧命名空间 `GTSinger/GTSinger`。

| 样本 | 音频 | 歌词单位 / 停顿 / 呼吸 |
| --- | --- | --- |
| Chinese / ZH-Alto-1 / Breathy / 不再见 / Control_Group / 0000 | 48kHz，384000 samples，8秒 | 11 / 0 / 1 |
| English / EN-Alto-1 / Glissando / my love / Control_Group / 0000 | 48kHz，445008 samples，9.271秒 | 14 / 5 / 0 |

以上两段是 smoke 子集，连同源许可共9个文件、2,577,629字节。代表性子集另固定20段、81文件、25,807,238字节，包含6个歌名候选、5位歌手及 Control/Breathy/Glissando 组；仅读取对应目录的官方元数据，没有下载整库。

两种子集均保留 mono、24-bit PCM、48kHz 原始 WAV，未重采样或裁切。每段获取 WAV、JSON、MusicXML 和 TextGrid，部分英文文件名是 `0000_TextGrid`。代表性子集有两段源 JSON 的末尾越过 WAV 一个采样点，保持原件并显式排除：`ZH-Tenor-1/Breathy/慢慢喜欢你/Control_Group/0000`、`ZH-Tenor-1/Glissando/给电影人的情书/Control_Group/0000`。可用18段、155.963秒、中文6段/英文12段、231歌词单位，另有33呼吸/11停顿。

原件在被 Git 忽略的 `data/local/gtsinger-p1/` 和 `data/local/gtsinger-representative/`；显式清单为 [gtsinger-samples.json](../data/gtsinger-samples.json) 和 [gtsinger-representative.json](../data/gtsinger-representative.json)。WAV 校验官方 LFS SHA-256，其余文件校验 Git blob OID；输出另保存内容摘要和源 URL，原件不改写。

数据使用 [CC BY-NC-SA 4.0 与附加条款](https://huggingface.co/datasets/AaronZ345/GTSinger/blob/d78cd68d00135b1995da03f25fccb3a221efd8c1/dataset_license.md)。完整许可与样本一起保留，附加条款存在 `M4Singer` 字样，应以源文件原文为准；本次未发布/再分发音频，也没有推断商用或伴奏授权。

## 安装与可恢复命令

数据准备使用 Python 3.9/3.10、`praatio==6.2.2`，盲标波形/频谱显示另用 `numpy==1.26.3`；不导入 torch、模型或 fastText。安装 [requirements-data.txt](../requirements-data.txt)，或复用已有 `.venv-baseline`（numpy 已存在，仅新增 praatio）：

```sh
uv pip install --python .venv-baseline/bin/python -r requirements-data.txt
.venv-baseline/bin/python scripts/fetch_gtsinger_samples.py \
  --input-root data/local/gtsinger-p1
.venv-baseline/bin/python scripts/prepare_gtsinger.py \
  --input-root data/local/gtsinger-p1 \
  --manifest data/gtsinger-samples.json --out-dir reports/p1/samples-v2
.venv-baseline/bin/python scripts/fetch_gtsinger_samples.py \
  --manifest data/gtsinger-representative.json \
  --input-root data/local/gtsinger-representative
.venv-baseline/bin/python scripts/prepare_gtsinger.py \
  --input-root data/local/gtsinger-representative \
  --manifest data/gtsinger-representative.json \
  --out-dir reports/p1/representative-v2 --on-invalid exclude
```

下载仅接受锁定的 smoke/representative-v1 清单。已有文件仅在大小与摘要匹配时复用；不匹配时失败，保留现场。网络失败清理当前临时文件、保留已成功文件。单文件限4MiB，总量分别限8MiB/100MiB，验证完成才排他发布文件。`build_gtsinger_representative_manifest.py --output NEW_PATH` 可重新核查固定20段的官方元数据，拒绝覆盖清单。

prepare 先验证所有来源；成功时排他发布完整目录，失败不留半份结果；已有目录拒绝覆盖。重做时选择新目录。默认 `--on-invalid fail`；显式 `exclude` 仅隔离读取器拒绝的段，并完整记录原因，不忽略摘要/下载错误。旧报告保留，当前带分类字段的结果以 `samples-v2`、`representative-v2` 为准。读取接口：

```python
from pathlib import Path
from t2l.datasets import load_segment

segment = load_segment(Path("data/local/gtsinger-p1"),
    "Chinese/ZH-Alto-1/Breathy/不再见/Control_Group/0000",
    dataset_revision="d78cd68d00135b1995da03f25fccb3a221efd8c1")
record = segment.to_dict()
```

## 时间与失败契约

`AudioSegment / LyricUnit / Phoneme / Boundary` 显式区分来源标注、候选边界和独立人工真值。中文保留字/音节单位，英文保留词，`<SP>`/空白为停顿、`<AP>`为呼吸，二者无歌词起点。

候选取 TextGrid 中该词首个非停顿/呼吸音素。数值秒保留，不经 LRC 格式化；整数样本采用 `Decimal(str(seconds)) × 实际采样率` 最近整数、半值向上取整。praatio 返回数值时间，归一化字段不声称保留原始数字的词法写法；原 TextGrid 文件和摘要仍保留，JSON 原字段及完整 TextGrid tiers 另存。

时间只对应下载片段自己的 WAV。官方切分会将片段起点归零，未提供足以恢复完整录音的 offset，`source_recording_offset_samples` 保持 `null`；不能从 `0000/0001` 序号推算整歌时间。

非法/非有限/越界/倒序时间、音素跨词、数组长度错误、损坏或截断音频、缺失/歧义文件均抛带来源路径的 `ValueError`。合法但不一致的 JSON/TextGrid 内容并列保存为 `validation_issues`，不自动修补。

MusicXML 保留乐谱音高、时值、歌词、连音；仅完整、顺序一致且单声部的歌词链建立精确对应，音高冲突或不支持结构保持 unknown，不把时值当作实录秒。英文保存词及音素核、乐谱音节；无法证实的音素到乐谱音节映射为空。首个发声音素含完整原始起止时间。

分类字段带 source/rule/evidence：休止、呼吸、气声和滑音来自注释；长音为乐谱持续至少2个四分音符的筛查规则，首辅音类别按 [CMUdict 音素表](https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones) 或中文拼音映射。中文 `r` 等尚未纳入可靠规则的音素保留 unknown，不借用英文类别强行赋值。仅明确的 tie/slur 支持乐谱连音标记；连读、重唱及不确定对应保持 unknown。以上不是经声学复核的真值。

## 真实集成结果与盲标

[代表性质量摘要](../reports/p1/representative-v2/quality-summary.json)记录20请求/18可用/2排除、231待复标起点；其余所查字段冲突0。所有独立人工标签与置信度为空，精度 `unavailable`、划分 `unassigned`。没有冲突只说明来源一致，不表示标注准确。smoke 当前结果有25待复标起点，适合首轮人工试标。

定点检查：读取/分类53项、获取/整理39项、分组25项、人工标签/API21项通过，共138项。真实数据的摘要校验与整理另行执行；18段均有完整乐谱对应，23单位符合长音筛查规则。桌面1440px/手机390px Playwright检查画布非空、无横向溢出、无页面异常，播放控件、逐采样点微调及非法保存拒绝通过；截图在 `reports/p1/review/ui-*.png`，不将浏览器播放状态视为人工听辨。独立 [审查及处理记录](../reports/p1/review/review-dispositions.md)区分实质修复、保守失败契约与未复现的判断。复用P0成功结果，不重复模型实验。输出含本机绝对路径，换机器须重新prepare。

WSL集成的20个原件均48kHz，得到相同18可用/2排除及231空标签；排他发布/拒绝覆盖和原结果摘要不变检查通过。只新增 praatio6.2.2，87个包依赖检查通过。原执行摘要在 `reports/p1/wsl/wsl-validation-20261010/integration-checks-final.json`；审查后的小范围源码更新另记，不把旧集成结果称为新代码的重跑结果。

- `manual-review.csv` / `manual-review.json`：盲标表，仅列音频、文字、单位索引和空人工字段，隐藏候选时间。
- `REVIEW.md`：先独立听辨/观察波形，再看候选的操作说明。
- `annotation-candidates.json`：复标完成后的来源对照表。
- `segments.jsonl`：完整归一化记录、原注释、各级数值时间及来源。
- `source-manifest.json` / `quality-summary.json`：版本、许可、校验与质量摘要。

本地盲标界面（macOS / Linux / WSL；保存锁使用 `fcntl`，不支持原生 Windows）：

```sh
.venv-baseline/bin/python scripts/review_gtsinger.py \
  --form reports/p1/samples-v2/manual-review.json \
  --session data/local/reviews/xcy-smoke-v2.json --port 8765
```

打开 `http://127.0.0.1:8765`，首次界面不展示任何候选。标注坐标始终是原始 WAV 整数采样点；播放速度仅影响试听。明确保存才写独立 session，原件与盲标表不改写。会话记录复核者、UTC时间、置信度、历史修订及原件指纹；跨标签页/进程的旧保存拒绝覆盖。HTTP保存验证Host、显式同源Origin与本次进程token；这是可信本机上的复标工具，不提供用户认证。复用相同命令恢复，另一复核者使用另一个 session。图标为本地打包的 Lucide 1.54.0，许可原文在 `scripts/review_ui/LICENSE`，浏览器不请求外部 CDN。

人工复标后再执行来源误差审计：

```sh
.venv-baseline/bin/python scripts/audit_gtsinger_annotations.py \
  --form reports/p1/samples-v2/manual-review.json \
  --candidates reports/p1/samples-v2/annotation-candidates.json \
  --review data/local/reviews/xcy-smoke-v2.json \
  --second-review data/local/reviews/second-smoke-v2.json \
  --output reports/p1/NEW_AUDIT.json
```

报告 MAE/Median/P90/P95、±5/10/20/50ms覆盖率与标注者分歧。`reviewed_count`统计已完成的第一份标签；未标、ambiguous、第二复标未完成、缺少可比较来源候选分别列出，均不进入来源误差指标。两人差异超过阈值时等待裁决，不平均两个时间戳生成真值。`inter_reviewer_mae_ms` 是全部有效复标对之间的差异诊断，包含待裁决分歧，范围显式写入报告；另报仅在阈值内的 `inter_reviewer_agreed_mae_ms`，二者都不是来源/模型精度。该报告测量来源注释误差，不是对齐模型精度；姓名和时间戳本身也不能证明独立性或裁决已经完成。

本阶段操作定义为“首个可辨的演唱音素起点”，包括清辅音噪声、送气或爆破等实际声学线索，不自动等同于元音周期起点。复标者应先记录定义、独立标签与身份；不确定边界填写 `ambiguous` 和原因，不强迫单点真值。至少需要独立复标与分歧裁决，不能将发布标注直接复制为人工标签。

## 按歌曲分组与后续边界

默认 `song_id` 是语言/歌曲标题的规范化键，无法自动识别别名、译名或同曲不同语言。划分前须提供人工核实的 `song_review.json`：

```json
{"reviewed_by": "实际核实者", "song_ids": {
  "来源song_id": "规范歌曲身份",
  "同曲别名song_id": "规范歌曲身份"
}}
```

随后用 `scripts/split_gtsinger.py --segments ... --song-review ... --output ...` 生成划分。固定 seed 与歌曲身份哈希，输入顺序不影响结果；整首歌及已标明派生录音、同内容音频不能跨集合。每段必须有读取器保存的有效 WAV SHA-256，不退回路径作为内容身份。缺失身份映射/摘要、冲突父子关系/同音频身份、重复段 ID、不足3首歌时拒绝默认三分划分。比例按歌曲数近似，绝不拆歌凑样本量。

可重复指定 `--held-out-singer EN-Tenor-1`：该歌手演唱的所有歌曲整体留在 test，即使同曲也有其他歌手版本。输出 `singer_independent_test_segment_ids` 指明严格歌手独立的测试子集；其他 test 歌手可能出现在 train/val，不将全 test 冒称歌手独立。若保留歌曲后无法形成非空 train/val，明确失败。目前未提供人工歌曲身份映射，故未生成实际划分。

P1 验收仍需要：人工确认歌名别名与共同录音/技巧组身份，生成固定划分，完成首轮独立盲标及第二复核/分歧裁决，再评估 TextGrid 的实际误差。合法伴奏来源与混音延迟实验也未完成。当前18段仅为代表性检查子集，不能支持训练收益或5ms精度结论。
