# 工作进度与恢复入口

## 最新暂停交接（2026-10-10）

用户要求暂停继续训练并将当前成果推送云端。以 [TRAINING_HANDOFF.md](TRAINING_HANDOFF.md) 为最新恢复入口，下文第23轮记录是历史状态。真实 CUDA 训练已 completed：36轮、最佳第26轮、val 来源 MAE 58.4439266681ms；最佳权重及完整训练报告已保存在 `reports/p4/training-cuda-20261010/`。检查点只有模型参数和训练元数据，不含 optimizer/scaler 状态，不能从原轮次无缝续训，当前 CLI 也无 resume 功能。

完整冻结评估 JSON 已取回：completed、0 failures，八组 test 方法均纳入511个来源边界，missing/text mismatch=0；MTL/MTL_BDR 的 test neural 来源 MAE 分别为51.516/50.704ms。报告与 WSL 原件摘要一致，输入/源码/检查点与训练报告相符。2026-10-10交接时通过远程终端检查，未发现 train_boundary/evaluate_boundary/run_boundary_stage 活进程，本地也没有训练进程；不自动启动后续实验。数据与混音归档见 `reports/handoff/`。实际 P4 绘图和最终验收仍未完成，P5/P6未实现。相关测试沿用已有成功结果，交接时只验证归档完整性，不重跑训练和评估。

恢复执行：2026-10-09；更新至2026-10-10。**P0完成，P1来源标注模式、P2粗对齐及P3 DSP精修试验已验收，P4正在执行。用户要求跳过人工阶段、完全采用训练数据，并跳过所有只读审查；后续仅做必要功能/数据/运行验证。P3最佳DSP试验未通过val覆盖门槛，默认保留coarse，没有5ms成功证据。扩展数据已取得并整理为163段、18首歌，MTL/BDR粗预测已缓存；609,026参数神经模型已在WSL启动训练，Windows本机日志确认到第23轮，临时最佳val MAE59.6519ms（第19轮），最终结果尚未取回。指标仅为相对数据集标注误差。**

## 用户目标与约束

完整实施规格及最新变更保存在 [progress/high-precision-spec.md](progress/high-precision-spec.md)。按 P0 → P1 → P2 → P3 → P4 → P5 → P6 顺序执行，每阶段按更新后的要求验收；保留原 API、检查点、默认 CLI/LRC 行为。采用数据集标签不等于将训练数据用于测试或伪造人工真值；仍固定独立 train/val/test，调参仅使用 train/val。Mac 用于开发和基础验证；RTX 5060 / Arch Linux WSL2 用于后续 GPU 实验。用户补充账号 `Haer` 后，现有 SSH 密钥已登录 Windows。WSL2 发行版 `archlinux`、Linux用户 `xcy`；实机 RTX 5060 / 8151MiB显存 / compute capability12.0 / 驱动617.42，WSL可用内存约15GiB。

当前验证渠道选定为本地，不重复安排等价云端 CI。新增基础 CPU CI 和手动 CUDA 集成工作流，运行范围及 runner 配置见 [ci.md](ci.md)；本次未推送或触发云端任务，不把工作流存在视为 CI 通过。

用户另要求“本次研究跳过所有只读审查”。后续不安排 DeepSeek 或其他独立只读审查；保留实现、测试及真实数据验证。来源模式这次仅做过 worker 配置干跑，未启动外部审查，不能记为审查通过。

## 已落盘的代码

- P4进行中：`expand_gtsinger_training.py` 按固定revision及歌曲清单扩展到180段原件，排除17段区间越界后保留163段；原件及标注不裁切。train/val/test为86/38/39段、10/4/4首歌、1371/554/511个来源边界，各划分均含中英文，旧歌曲划分及EN-Tenor-1歌手留出保持不变。准备结果在 `reports/p4/training-source-20261010/`。`cache_training_coarse.py` 已完成36次经摘要验证的复用及290次新增CUDA推理，结果在 `reports/p4/coarse-cache-cuda-20261010/`，不得重复相同推理。609,026参数TCN、原始采样网格特征、波形增强、训练/冻结评估、图表脚本及后台恢复入口已落盘；真实CUDA训练已启动，P4最终验收未完成。

- P3：`t2l/boundary/features.py` 在原采样网格提取2.5ms步长的多种高分辨率线索，`dsp.py` 提供带粗起点先验、音素类别权重和显式fallback的局部精修；置信度未校准。`benchmark_dsp_gtsinger.py` 仅val选择、固定配置评估，`benchmark_dsp_mixtures.py` 复用两段训练混音/分离预测，`plot_dsp_failures.py` 生成分布及6个失败窗口。默认推理路径未改；方法和恢复命令见 [p3-dsp-refinement.md](p3-dsp-refinement.md)。

- P2：`t2l/alignment/` 提供 `coarse_align()`、`BoundaryCandidate`、语言/重复歌词元数据和可选带状CTC/Viterbi。原有wrapper/process只增加opt-in参数；默认路径不改。候选保留原始帧、采样网格、未校准分数、缺失边界和明确回退；无人工置信度。`benchmark_coarse_gtsinger.py` 已在WSL完成72次真实CUDA对照；接口、约束、结果及范围见 [p2-coarse-alignment.md](p2-coarse-alignment.md)。

- `main.py`：新增 `--aligner mtl|mtl_bdr`、`--json-output PATH`、`--enhanced-lrc-output PATH`；输出预检防止覆盖输入或路径别名冲突。
- `t2l/mtl/wrapper.py`：`align(return_details=True)` 可返回原始帧、音素/索引映射、模型与帧元数据、未校准全路径分数；默认返回值保持原契约。
- `t2l/t2l.py`：`process(return_details=True)` 返回 `lrc / units / alignment`；浮点秒直接从原始帧换算，缺失项为 null。未改变模型结构、默认时间轴或对齐算法。
- `scripts/benchmark_baseline.py`：MTL/MTL_BDR、legacy/batch-first 布局消融、随机种子、重复运行、真实推理形状 hooks、输入/模型来源和环境记录。布局修正只作用于新加载实例，不改模型定义或检查点。
- 修改三个原测试模块，并新增 `tests/test_baseline_benchmark.py`、`tests/test_timeline.py`。
- [p0-methodology.md](p0-methodology.md) 说明基准方法；[environments.md](environments.md) 说明 Mac/WSL 环境与恢复命令；[p0-results.md](p0-results.md) 汇总真实结果。
- P1：`t2l/datasets/` 提供四联文件读取、显式数据结构、带来源/规则/证据的分类和歌曲/歌手隔离划分；受限下载/整理/划分 CLI 及 `review_gtsinger.py`、`audit_gtsinger_annotations.py` 提供盲标与来源误差审计。人工标签保持原始 WAV 整数采样点，保存含指纹、历史修订和跨进程保护，候选不出现在盲标界面。`requirements-data.txt` 为可选数据依赖，未改变推理依赖版本。
- `adjudicate_gtsinger.py` 与 `t2l/datasets/adjudication.py` 补齐显式人工裁决和参考导出，来源审计可接收 `--adjudication`。裁决绑定两份完整盲标快照及原始音频身份；超限/歧义未裁决时保持 pending，阈值内保留第一人的原始采样点及第二人的确认，不平均。原 session 不改写，导出不构成独立隔离测试集验收；操作见 [数据文档](p1-data-preparation.md#分歧裁决与参考导出)。
- 固定 revision `d78cd68d00135b1995da03f25fccb3a221efd8c1` 的中英文原件已取得：smoke 原件在 `data/local/gtsinger-p1/`，代表性原件在 `data/local/gtsinger-representative/`，均 Git 忽略。代表性清单含20段、81文件、25,807,238字节、6个歌名候选和5位歌手。当前分类结果在 `reports/p1/samples-v2/`（25个待复标边界）和 `reports/p1/representative-v2/`（18可用/2排除、231个待复标边界）。两段源 JSON 末端越过 WAV 一个采样点，原件保留，不自动裁切；其他所查字段冲突0。
- 18段可用样本均完成保守的 MusicXML 歌词对应，231单位中23个符合乐谱长音筛查规则；乐谱时值不转换成实录秒。连读、重唱和不支持的音素/乐谱结构保持 unknown。人工真值为空，精度 unavailable，划分 unassigned。
- `prepare_gtsinger_mixtures.py` 生成原创合成伴奏及6dB受控混音，逐采样点保留48k原始网格、原件摘要和父歌曲身份；独立伴奏CC0，人声/混音仍遵循数据集条款。`benchmark_gtsinger_mixtures.py` 对清唱/混音/原生44.1k分离结果运行MTL/MTL_BDR，保留原始预测、受限波形互相关及实际CUDA前向证据。两脚本不修改产品推理接口。
- `.github/workflows/tests.yml` 在 push/PR 上执行单个 Linux/Python3.10 CPU job，使用独立测试依赖子集，不获取模型/数据；`gpu-integration.yml` 仅手动触发，复用现有真实 CUDA 验证入口，需外部预配置 runner。另建 `/home/xcy/auto-lrc-ci-20261010/.venv-ci` 完成 CPU 依赖安装及251项测试，最终保留原始 LFS 指针验证模型文件不参与测试。范围与证据见 [CI 本地验收](../reports/ci-20261010/README.md)。

P0 已通过 60 项轻量测试、`compileall`、`git diff --check` 和两轮 DeepSeek 只读审查。恢复中修复了 benchmark 线程初始化顺序、非有限诊断分数和失败报告序列化；新版8次推理及定点复审均已完成：4个配置各重复两次结果一致，同环境原版/legacy逐值一致。GPU 环境、MTL/BDR 完整 demo、Demucs 四成员片段及完整默认 CLI 均通过真实验收。

P1 定点测试共138项通过（读取/分类53、获取/整理39、歌曲/歌手分组25、人工标签/API21）。真实下载/校验/prepare、原始48kHz与盲标导出检查通过；桌面1440px/手机390px Playwright检查画布非空、无横向溢出、无页面异常，截图在 `reports/p1/review/ui-*.png`。独立 DeepSeek 只读审查分别覆盖数据和复标/划分，提出的问题及处理依据见 `reports/p1/review/review-dispositions.md`；首个数据审查触达上限后，沿用同会话限定收尾成功。旧 `deepseek.md` 仅覆盖早期版本。P0成功结果复用，未重复模型或等价CI。

受控混音新增53项定点测试及语法检查通过，两脚本独立DeepSeek审查完成。WSL实际生成18段/155.963秒混音，选其中中英文各一段完成12次对齐及2次四成员Demucs分离，报告在 `reports/p1/mixtures-cuda-20261010/`。源码/检查点与Mac摘要一致；本次18段混音、伴奏及分离WAV保存在WSL的 `data/local/mixtures/`，提交仅含文本证据。原有成功检查未重跑。

显式裁决流程的本地全量回归342项通过、0失败/0跳过；DeepSeek初审后规范了 API/CLI 的同值阈值表示，并补齐缺失裁决文件及同一 session/硬链接别名拒绝。修复后仅重跑受影响的两个模块，100项通过，语法检查通过；其余251项输入未变，复用全量成功结果。定点复审触达轮数上限后，同会话限定收尾成功，未发现新的实质缺陷，失败回合另保留。原件/复标记录未变，真实人工标签仍为0/25。证据及审查处理见 [裁决流程验收记录](../reports/p1/adjudication-20261010/README.md)，未运行云端CI或重复模型推理。

## 已核查事实

- 原始 HEAD：`2c475cc`，分支 `main`，origin 为 HalfAnElephant/auto-lrc。开始时仅有用户自有未跟踪的 `AGENTS.md`；不纳入本次实现提交。
- 本地阶段提交：P0 为 `96ad584`；P1 数据准备和复标工具为 `4f1b131`，受控混音对照为 `879c166`，基础CI为 `a9118b8`，整体验收仍待人工复核。未推送。P0 两个 Demucs WAV 仅作本机证据，不纳入提交。
- 根目录不存在 `.codegraph/`，本次按普通源码入口检查。
- Git LFS 本地初始化及 pull 已成功，三个检查点和语言模型是真实文件。
- 原版 STFT hop=256，采样率 22050Hz，时间池化 stride=3，输出名义步长约 34.83ms。实际音频边界原点仍未校准。
- 两个模型的第二、第三层 LSTM 设置 batch_first=False，而输入为 batch-first；维度问题已由源码确认，精度影响尚未实测。
- Demo MP3：44.1kHz、双声道、容器时长 325.868118 秒，ffprobe 报告 start_time=0.025057 秒。不能直接用该值修正歌词时间。
- 旧训练模块依赖缺失的 `t2l.mtl.test`、外部 DALI 等，不能当成可运行训练入口。

## 当前验证状态

- P3：100项定点检查通过（feature39、DSP20、benchmark40、mixture1），benchmark40成功结果复用；未安排外部或独立只读审查。Mac仅val选择，最佳试验low_prior的MAE151.75→132.63ms，但±5/20ms覆盖下降，不通过gate，selected=none。WSL固定试验18段completed、0结构失败，test相对来源MAE82.58→79.96ms，但P90166.64→173.25ms、P95217.66→243.53ms、>500ms错误1→2、±5ms5.10%→3.06%；默认保留coarse，未达5ms。test仅英文；混音仅中英文两段训练诊断，未额外调参。DSP全划分CPU墙钟2.839秒，混音诊断1.403秒，复用P2/P1模型推理。结果及失败图在 [reports/p3/](../reports/p3/README.md)，保持固定训练隔离，不重跑成功实验。

- P2：189项相关检查通过（候选41、DP67、运行时/原wrapper/process34、原CLI16、基准31），无模型的其余未变检查复用；72次CUDA对照完成，源码摘要一致，无缺失/错配/回退。四种配置起点一致，test相对来源MAE82.58ms/P90166.64ms，±5ms5.10%；该test仅英文98边界，非独立人工精度。累计推理12.232秒，峰值分配/保留108.32/278MiB，不含模型预加载。报告 `reports/p2/coarse-cuda-20261010/result.json`，不要重跑。

- P1 来源模式：41项划分定点测试通过，18段真实划分及来源歌曲/WAV摘要隔离、歌手保留和输入顺序不变检查通过；train/val/test为7/3/8段、96/37/98个来源边界。EN-Tenor-1四段为严格歌手独立测试子集，其涉及的两首歌整体留test。报告及固定划分在 `reports/p1/source-mode-20261010/`。当前val仅中文、test仅英文，不能用这个小子集代表最终多语言评测；扩充时仍须将已有歌曲固定留在原集合。没有改写原始标注或人工字段。

- `git diff --check`：通过（保存进度时）。
- Python 3.10.16 虚拟环境 `.venv-baseline`：已创建。
- 直接安装旧 requirements 失败于 fasttext 隔离构建缺少 pybind11；已改为先装构建依赖，再 `--no-build-isolation` 安装原 requirements。
- 旧环境运行时导入、NLTK资源和真实模型加载已通过；torch 2.1.2、torchaudio 2.1.2、pytest 8.3.5、fasttext 0.9.2、numpy 1.26.3。完整冻结版本复用 `reports/p0/requirements-baseline.freeze.txt`。
- 原版 `2c475cc` 的 MTL / MTL_BDR 各重复两次已通过，帧、音素、LRC、score 完全一致，结果在 `reports/p0/original-threads2/`；新版完整消融8次已完成且重复性通过，结果在 `reports/p0/demo-threads2/`。早期8线程中断实验保留在 `reports/p0/original/`，不纳入规定配置验收。
- DeepSeek worker 首轮真实只读审查成功，原文在 `reports/p0/review/deepseek-initial.md`；已修复其诊断序列化问题，定点复审成功，原文在 `reports/p0/review/deepseek-followup.md`，未发现新的正确性缺陷。
- WSL `.venv-cuda` 已完成：Python3.10.16、torch/torchaudio2.7.1+cu128，86包兼容、语言入口/NLTK/真实 CUDA 算子通过。下载超时与 fasttext/GCC15 构建问题已解决并记录，未修改系统配置。
- GPU MTL/BDR 都得到482个完整单位；Demucs 六秒片段四成员全部执行；默认CLI完整demo在30.387秒完成分离、CUDA对齐和三种输出验证。结果在 `reports/p0/gpu/`。单阶段耗时/显存及范围见报告，不以一次运行声明重复性。
- 跨平台 MTL 有1个起点差1帧；BDR有79个起点不同，最大649帧。设备、torch与系统同时变化，尚未隔离原因；后续固定环境评估并保留此风险。独立人工标注和精度指标仍不可用。
- WSL P1 集成通过，报告在 `reports/p1/wsl/wsl-validation-20261010/`：20个原始 WAV 均48kHz，18可用/2排除，231标签 pending；Linux排他目录发布和重复运行拒绝覆盖通过，原报告摘要不变。仅添加 praatio6.2.2，numpy及torch版本不变，87包依赖检查通过。集成报告保留执行时的源码摘要；审查后3个更新文件已同步，Mac/WSL摘要一致，WSL语法检查3/3通过，记录在 `post-review-code-sync.json`。读取器/分类/清单等7个原集成输入未变，未重跑prepare或模型，不混写旧证据。
- 受控混音实机实验用时12.900秒，峰值已分配/保留显存约1.75/2.22GiB。相对清唱预测，中文混音/分离后起点平均绝对变化50.66/9.50ms，英文37.32/42.29ms；不是人工精度。八个波形窗口在±50ms内的相关峰值均位于0样本，不能据此断言发声边界已校准；未施加任何平移。英文44.1k分离输出按既有floor长度比原片段短约2.268微秒，未补齐或往返48k。未重复推理，不声称重复性。
- 基础 CI 本地验证：独立 Arch Linux WSL / Python3.10.16 / torch与torchaudio2.1.2+cpu 环境，`pip check`、251项完整测试及 `compileall` 通过；最终无模型实体，仅4个原始 LFS 指针，0失败/0跳过。49个源码/依赖/CPU工作流摘要与Mac一致。两个工作流的 actionlint 和所有 Bash run 块语法检查通过；Linux 原始 GPU run 块的路径/失败停止夹具通过，未运行模型。DeepSeek 只读审查指出解析可执行文件 symlink 会绕过虚拟环境，已改为检查目录并保留原调用路径，夹具验证修复。原审查触达轮数上限后限定收尾成功，原文和处理依据均在CI报告；不把初次失败记为审查成功。未注册 runner、推送或运行 GitHub Actions。

## 恢复顺序

1. 保留现有 dirty 工作和用户 `AGENTS.md`，不要重建环境或覆盖任何实验目录。
2. Mac已完成原版4次、新版8次实验及两轮DeepSeek审查。检查 `docs/p0-results.md` 和 `reports/p0/demo-threads2/comparison.json`；勿重跑成功实验。
3. WSL `/home/xcy/auto-lrc/.venv-cuda` 已安装且通过实际推理；冻结文件在 `reports/p0/environment/requirements-cuda.freeze.txt`。保留环境、uv/NLTK/Demucs缓存，无需重新安装。
4. 源码、3检查点、lid和demo已同步至 `/home/xcy/auto-lrc`；Windows中转文件为 `C:/Users/Haer/auto-lrc-p0-runtime-20261009.tar.gz`（本任务创建，可在验收后清理）。
5. `scripts/verify_gpu_runtime.py` 的三个阶段均已运行成功；完整默认CLI另外验收。所有结果已从WSL同步回 `reports/p0/gpu/`，包括原始诊断/分离片段/日志；不重跑相同输入。
6. SSH：`ssh -p 12345 -o BatchMode=yes -o StrictHostKeyChecking=yes Haer@127.0.0.1 'wsl.exe -d archlinux -- sh -s'`，stdin提供Linux脚本，必须先 `cd /home/xcy/auto-lrc`。使用现有密钥，无需再次询问密码。
7. 先读 [p1-data-preparation.md](p1-data-preparation.md)。用户已跳过人工阶段，旧盲标/裁决工具仅保留作将来可选复核，不再等待人工记录，也不把来源标签写入 `human_truth_*`。旧报告的 pending/unassigned 是执行时的历史状态，不改写旧证据。
8. 使用显式来源模式固定 train/val/test 及歌手独立子集，标签直接引用 `segments.jsonl` 的 `units[].boundary.candidate_sample/candidate_seconds`（`textgrid:first_phone`）；人工置信度和复核状态仍为空/pending。完成来源模式验收后进入P2，保持独立划分及来源指标解释。不重跑相关输入未变的成功检查。
9. 受控混音与两段GPU对照已完成，无需重跑；恢复路径和完整命令在数据文档。WSL原始混音位于 `data/local/mixtures/representative-v1-20261010/`，分离音频位于 `data/local/mixtures/native-vocals-20261010/`。现有报告与音频目录均不可覆盖。合成夹具不代表真实配器覆盖或人工起点真值。
10. 基础 CI 的安装/测试成功结果见 `reports/ci-20261010/lfs-pointer/`，初次展开模型的结果保留在上层目录；相关输入未变时复用，不重建环境或重跑。可选 GPU runner 的外部配置见 [ci.md](ci.md)，本次没有注册 runner 或触发远程任务。
11. P3已完成：读取 [p3-dsp-refinement.md](p3-dsp-refinement.md) 和 `reports/p3/README.md`，保留selected=none及固定low_prior试验结果；归档目录不可覆盖。进入P4独立轻量边界模型训练，仅使用train/val，已有test歌曲继续隔离。可视化额外依赖为 `requirements-research.txt` 的matplotlib3.8.4，原推理requirements未改。

## 尚未完成

人工歌曲身份确认、独立复标/裁决及 TextGrid 相对人工边界的误差依用户要求暂缓，不再作为本轮研发门槛；来源模式开发子集、P2接口/解码和P3 DSP试验已验收，默认保留coarse。合法原创合成伴奏的受控实验已完成，真实匹配伴奏覆盖及逐音素分离延迟校准仍待验证。原整歌偏移未知，来源分类不代表声学真值。P4模型、训练、冻结评估和绘图代码已完成定点验证；真实训练最终报告、冻结权重、隔离测试指标和真实绘图尚待取回/执行，P4未验收，不进入P5/P6，也无5ms成功证据。

### P4当前恢复点

- 原CLI训练输出为WSL `/home/xcy/auto-lrc/reports/p4/training-cuda-20261010/`，日志为Windows `C:\Users\Haer\auto-lrc-p4-training-log.txt`。2026-10-10约11:32远程桌面读取到epoch23，best epoch19、val MAE59.6519ms；扩展val粗基线100.46768ms。此日志不是最终completed结果或test指标。
- 原SSH会话已以255关闭；随后新连接无握手，最新探测为本机12345无监听/Connection refused。Windows本机此前确认sshd Running，用户确认主机和映射在线；不能据连接异常推断训练失败或完成。Mac锁定令电脑操作工具无法继续读取桌面，已请用户解锁。禁止重启训练或覆盖原输出。
- 连接恢复后，先读取训练 `result.json`，只有status=completed且输入/运行代码/检查点摘要吻合，才执行既有冻结评估命令。若failed或running，保留结果并处理实际状态；不要复制临时最佳权重来声称最终训练成功。
- 新增 `scripts/run_boundary_stage.sh` 可对新阶段保存后台运行和退出证据。当前旧CLI训练不经过runner，其status应为external_unverified；不能补造退出码。恢复时对该旧实验沿用原CLI及实际结果证据。
- P4实现定点验证共215项通过：数据扩展44、粗缓存8、分类指标4、神经接口53、增强25、训练/评估50、绘图31。最后局部修改已仅重跑对应受影响项，其余成功结果复用；未执行等价云端CI。后台runner另完成11组执行契约夹具，真实GPU评估和图片渲染仍未执行。
