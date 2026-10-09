# 工作进度与恢复入口

恢复执行：2026-10-09；更新至2026-10-10。**P0 运行与复现验收完成；P1 代表性数据、来源分类、隔离划分与盲标工具已实现，人工复核和整体验收未完成。没有独立人工真值，不得宣称达到 5ms，也不得跳过 P1 验收进入 P2–P6。**

## 用户目标与约束

完整实施规格保存在 [progress/high-precision-spec.md](progress/high-precision-spec.md)。按 P0 → P1 → P2 → P3 → P4 → P5 → P6 顺序执行，每阶段验收后才推进；保留原 API、检查点、默认 CLI/LRC 行为。Mac 用于开发和基础验证；RTX 5060 / Arch Linux WSL2 用于后续 GPU 实验。用户补充账号 `Haer` 后，现有 SSH 密钥已登录 Windows。WSL2 发行版 `archlinux`、Linux用户 `xcy`；实机 RTX 5060 / 8151MiB显存 / compute capability12.0 / 驱动617.42，WSL可用内存约15GiB。

当前验证渠道选定为本地，不重复安排等价云端 CI。仓库原先没有 CI；用户要求的基础 CI 属于后续交付，目前没有新增工作流。

## 已落盘的代码

- `main.py`：新增 `--aligner mtl|mtl_bdr`、`--json-output PATH`、`--enhanced-lrc-output PATH`；输出预检防止覆盖输入或路径别名冲突。
- `t2l/mtl/wrapper.py`：`align(return_details=True)` 可返回原始帧、音素/索引映射、模型与帧元数据、未校准全路径分数；默认返回值保持原契约。
- `t2l/t2l.py`：`process(return_details=True)` 返回 `lrc / units / alignment`；浮点秒直接从原始帧换算，缺失项为 null。未改变模型结构、默认时间轴或对齐算法。
- `scripts/benchmark_baseline.py`：MTL/MTL_BDR、legacy/batch-first 布局消融、随机种子、重复运行、真实推理形状 hooks、输入/模型来源和环境记录。布局修正只作用于新加载实例，不改模型定义或检查点。
- 修改三个原测试模块，并新增 `tests/test_baseline_benchmark.py`、`tests/test_timeline.py`。
- [p0-methodology.md](p0-methodology.md) 说明基准方法；[environments.md](environments.md) 说明 Mac/WSL 环境与恢复命令；[p0-results.md](p0-results.md) 汇总真实结果。
- P1：`t2l/datasets/` 提供四联文件读取、显式数据结构、带来源/规则/证据的分类和歌曲/歌手隔离划分；受限下载/整理/划分 CLI 及 `review_gtsinger.py`、`audit_gtsinger_annotations.py` 提供盲标与来源误差审计。人工标签保持原始 WAV 整数采样点，保存含指纹、历史修订和跨进程保护，候选不出现在盲标界面。`requirements-data.txt` 为可选数据依赖，未改变推理依赖版本。
- 固定 revision `d78cd68d00135b1995da03f25fccb3a221efd8c1` 的中英文原件已取得：smoke 原件在 `data/local/gtsinger-p1/`，代表性原件在 `data/local/gtsinger-representative/`，均 Git 忽略。代表性清单含20段、81文件、25,807,238字节、6个歌名候选和5位歌手。当前分类结果在 `reports/p1/samples-v2/`（25个待复标边界）和 `reports/p1/representative-v2/`（18可用/2排除、231个待复标边界）。两段源 JSON 末端越过 WAV 一个采样点，原件保留，不自动裁切；其他所查字段冲突0。
- 18段可用样本均完成保守的 MusicXML 歌词对应，231单位中23个符合乐谱长音筛查规则；乐谱时值不转换成实录秒。连读、重唱和不支持的音素/乐谱结构保持 unknown。人工真值为空，精度 unavailable，划分 unassigned。

P0 已通过 60 项轻量测试、`compileall`、`git diff --check` 和两轮 DeepSeek 只读审查。恢复中修复了 benchmark 线程初始化顺序、非有限诊断分数和失败报告序列化；新版8次推理及定点复审均已完成：4个配置各重复两次结果一致，同环境原版/legacy逐值一致。GPU 环境、MTL/BDR 完整 demo、Demucs 四成员片段及完整默认 CLI 均通过真实验收。

P1 定点测试共138项通过（读取/分类53、获取/整理39、歌曲/歌手分组25、人工标签/API21）。真实下载/校验/prepare、原始48kHz与盲标导出检查通过；桌面1440px/手机390px Playwright检查画布非空、无横向溢出、无页面异常，截图在 `reports/p1/review/ui-*.png`。独立 DeepSeek 只读审查分别覆盖数据和复标/划分，提出的问题及处理依据见 `reports/p1/review/review-dispositions.md`；首个数据审查触达上限后，沿用同会话限定收尾成功。旧 `deepseek.md` 仅覆盖早期版本。P0成功结果复用，未重复模型或等价CI。

## 已核查事实

- 原始 HEAD：`2c475cc`，分支 `main`，origin 为 HalfAnElephant/auto-lrc。开始时仅有用户自有未跟踪的 `AGENTS.md`；不纳入本次实现提交。
- 本地阶段提交：P0 为 `96ad584`；P1 数据准备和复标工具单独提交，整体验收仍待人工复核。未推送。P0 两个 Demucs WAV 仅作本机证据，不纳入提交。
- 根目录不存在 `.codegraph/`，本次按普通源码入口检查。
- Git LFS 本地初始化及 pull 已成功，三个检查点和语言模型是真实文件。
- 原版 STFT hop=256，采样率 22050Hz，时间池化 stride=3，输出名义步长约 34.83ms。实际音频边界原点仍未校准。
- 两个模型的第二、第三层 LSTM 设置 batch_first=False，而输入为 batch-first；维度问题已由源码确认，精度影响尚未实测。
- Demo MP3：44.1kHz、双声道、容器时长 325.868118 秒，ffprobe 报告 start_time=0.025057 秒。不能直接用该值修正歌词时间。
- 旧训练模块依赖缺失的 `t2l.mtl.test`、外部 DALI 等，不能当成可运行训练入口。

## 当前验证状态

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

## 恢复顺序

1. 保留现有 dirty 工作和用户 `AGENTS.md`，不要重建环境或覆盖任何实验目录。
2. Mac已完成原版4次、新版8次实验及两轮DeepSeek审查。检查 `docs/p0-results.md` 和 `reports/p0/demo-threads2/comparison.json`；勿重跑成功实验。
3. WSL `/home/xcy/auto-lrc/.venv-cuda` 已安装且通过实际推理；冻结文件在 `reports/p0/environment/requirements-cuda.freeze.txt`。保留环境、uv/NLTK/Demucs缓存，无需重新安装。
4. 源码、3检查点、lid和demo已同步至 `/home/xcy/auto-lrc`；Windows中转文件为 `C:/Users/Haer/auto-lrc-p0-runtime-20261009.tar.gz`（本任务创建，可在验收后清理）。
5. `scripts/verify_gpu_runtime.py` 的三个阶段均已运行成功；完整默认CLI另外验收。所有结果已从WSL同步回 `reports/p0/gpu/`，包括原始诊断/分离片段/日志；不重跑相同输入。
6. SSH：`ssh -p 12345 -o BatchMode=yes -o StrictHostKeyChecking=yes Haer@127.0.0.1 'wsl.exe -d archlinux -- sh -s'`，stdin提供Linux脚本，必须先 `cd /home/xcy/auto-lrc`。使用现有密钥，无需再次询问密码。
7. 先读 [p1-data-preparation.md](p1-data-preparation.md) 和 `reports/p1/samples-v2/REVIEW.md`。当前本地盲标服务地址 `http://127.0.0.1:8765`；恢复命令见文档，session 为 `data/local/reviews/xcy-smoke-v2.json`。尚无人保存真实标签。第二复核者须使用独立 session；候选时间另存，应复标完成后再查看。不能把原标注复制为人工真值。
8. 核实6个歌名候选、同曲别名/技巧组/配对语音/派生混音身份，提供有实际复核者的歌曲身份映射，再生成固定 train/val/test 及可行的歌手独立子集。先完成25个 smoke 边界的独立复标、第二复核及分歧裁决，执行来源误差审计；扩充正式隔离测试集。禁止跳过P1验收进入P2–P6，不重跑相关输入未变的成功检查。

## 尚未实现

P1歌曲身份人工复核、实际固定划分、独立边界复标/分歧裁决、来源标注误差和整体验收仍未完成；合法伴奏来源及混音延迟实验也未完成。原整歌偏移未知，来源分类不代表声学真值。P2 粗对齐增强、P3 DSP 精修、P4 轻量网络训练、P5 全局优化、P6 精度评测/失败可视化，以及 precision/refiner/device/confidence 等后续 CLI 均未实现。没有轻量边界模型、没有训练结果或5ms成功证据。
