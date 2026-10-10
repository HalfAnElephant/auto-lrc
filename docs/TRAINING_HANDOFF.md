# 训练交接与暂停恢复

更新时间：2026-10-10。**用户已要求暂停继续训练，保存全部当前成果供他人接手。现在不要启动训练或评估，不重复成功实验，不覆盖旧目录。** 下列命令是恢复说明，只有后续明确恢复工作时才执行对应实验。本文件优先说明最新暂停点；旧进度记录中“第 23 轮、最终报告未取回”属于较早状态。

交接时通过 Windows 远程终端执行 WSL `ps`，未发现 `train_boundary`、`evaluate_boundary` 或 `run_boundary_stage` 活进程，Mac 亦无训练进程。训练已自然早停，无需中断写盘。最佳检查点只含模型参数、配置和训练元数据，**没有 optimizer/scaler/RNG 断点状态，训练 CLI 没有 resume 功能**；他人可加载最佳模型做研究，若需新训练必须明确新实验目录和协议，不能声称接续原轮次。

## 当前成果与证据边界

P0 环境/基线、P1 来源模式数据、P2 粗对齐、P3 DSP 试验已完成原定验收。P4 模型及训练、冻结评估、绘图工具已实现，真实训练完成，但 **P4 尚未最终验收，P5/P6 尚未实现**。默认 CLI 尚未启用神经精修；P3 默认仍保留 coarse。

| 项目 | 当前事实 |
| --- | --- |
| 数据 | 固定 GTSinger revision，180 段原件中排除 17 段标注区间越界，保留 163 段、18 首歌 |
| 划分 | train 86 段/10 歌/1,371 边界；val 38/4/554；test 39/4/511；各划分均含中英文 |
| 隔离 | 同歌及变体同 split，已有歌曲分配保留，`EN-Tenor-1` 全部 test |
| 粗缓存 | completed；36 次经摘要验证的复用、290 次新增 CUDA 推理，无需再跑 |
| 神经训练 | completed；36 epochs，best epoch 26，val 来源 MAE 58.4439266681ms，10 轮无改进后 early stop |
| 冻结评估 | completed、0 failures；完整 JSON 已同步 Mac，与远端 SHA256 相同，输入/运行源码/权重摘要已核对 |
| 图片 | 绘图代码和定点检查已完成，实际 P4 图片尚未执行 |

完整 [冻结评估报告](../reports/p4/evaluation-cuda-20261010/result.json) 的 test 来源 MAE 如下，单位 ms。全部八种方法均纳入511个来源边界，missing/text mismatch=0；分类、bootstrap、逐边界预测、耗时和显存保留在完整 JSON 中：

| 粗模型 | coarse | DSP | neural | DSP+neural |
| --- | ---: | ---: | ---: | ---: |
| MTL | 99.172 | 91.343 | 51.516 | 50.537 |
| MTL_BDR | 98.014 | 90.291 | 50.704 | 49.735 |

完整 JSON 为320,551,753 bytes，Mac 与 WSL 的 SHA256 均为 `2a85d1708e6cd4e0ed9efe52ec2ba972005f81b123ce08624c25a6e2db9ef852`。纯 neural 的 >500ms 来源误差为7个，coarse/DSP/DSP+neural为5个；平均误差改善不代表所有困难案例改善。

全部监督/指标依据数据集 TextGrid 首发声音素来源标签，歌曲身份及边界仍 dataset/unreviewed、人工字段为空、`review_status=pending`。它们不是独立人工真值或人工精度；2.5ms 特征 hop 不等于达到 5ms 准确度。test 未用于选择最佳 epoch。用户跳过人工阶段和只读审查，不等于允许混用划分或伪造人工标签。

## 必须一同保留的文件

| 路径 | 用途及状态 |
| --- | --- |
| [configs/boundary-tcn.json](../configs/boundary-tcn.json) | 冻结模型/训练配置 |
| [reports/p4/data-plan-20261010/](../reports/p4/data-plan-20261010/) | `manifest.json`、固定 split overrides、246,784,466 bytes/180 段的受限获取计划 |
| [reports/p4/training-source-20261010/](../reports/p4/training-source-20261010/) | 已冻结的 segments/splits/source manifest/质量与排除报告 |
| [reports/p4/coarse-cache-cuda-20261010/result.json](../reports/p4/coarse-cache-cuda-20261010/result.json) | MTL/BDR 原始候选及来源/运行摘要 |
| [reports/p4/training-cuda-20261010/](../reports/p4/training-cuda-20261010/) | completed `result.json`、最佳 `boundary.pt`、`training.log`；已同步 Mac |
| [reports/p4/evaluation-cuda-20261010/](../reports/p4/evaluation-cuda-20261010/) | completed 完整 JSON，使用 Git LFS 保存 |
| [reports/handoff/](../reports/handoff/) | Mac早期原件/混音和WSL训练原件/混音归档、许可、文件摘要及恢复说明 |
| `reports/p4/plots-cuda-20261010/` | 尚未执行，未来不得覆盖同名既有结果 |

训练在 RTX 5060、Python 3.10.16、NumPy 1.26.3、PyTorch 2.7.1+cu128/CUDA 12.8 运行。模型 609,026 参数，57 维声学输入加 relative position，281 帧、±350ms；真实 float16 AMP、792 optimizer steps、全部轮次有限梯度。总墙钟 1,347.8245s；逐轮训练/验证 1,291.3629s，最大 PyTorch allocated/reserved 252,772,864/278,921,216 bytes。详细协议与口径见 [P4 文档](p4-neural-refinement.md) 和 [模型卡](../checkpoints/boundary/MODEL_CARD.md)。

最佳权重 2,456,370 bytes；本地文件与训练报告的 SHA256 已核对一致：

```text
967da4debb2a457cf45a2989335183781e41890f0d4c43af3e1b72292a5bb4da
```

取得云端仓库副本后先执行 `git lfs pull` 并核对以上文件是否实际存在，特别是权重内容而不是空占位。`data/local/` 被 Git 忽略，但当前原件和混音已打包到 `reports/handoff/` 的 LFS 归档，恢复说明见该目录README；也可按固定manifest重新获取。不要改写归档中的绝对来源路径或格式来适配新机器：运行入口按 `--audio-root` 重建实际音频路径并核验 WAV SHA256/网格，原 JSON 摘要必须保持不变。

## 环境与数据恢复

现有 WSL 项目为 `/home/xcy/auto-lrc`，发行版 `archlinux`，Linux 用户 `xcy`。已有 `.venv-cuda` 可直接复用，不重复安装。Windows SSH 用户为 `Haer`，HAER 映射进入 Windows SSH 后再调用 `wsl.exe`，不是直接的 Linux SSH：

```bash
ssh -p 12345 -o BatchMode=yes -o StrictHostKeyChecking=yes \
  Haer@127.0.0.1 'wsl.exe -d archlinux -- sh -s' <<'SH'
cd /home/xcy/auto-lrc
.venv-cuda/bin/python -c 'import torch, numpy; print(torch.__version__, torch.version.cuda, numpy.__version__)'
SH
```

仅在新 Linux/WSL 副本且 `.venv-cuda` 不存在时恢复以下环境。先安装兼容宿主 NVIDIA 驱动，WSL 内不安装 Linux 显示驱动；完整安装失败记录与官方 wheel 备用来源见 [环境文档](environments.md)。不要把旧 `requirements.txt` 混装进 CUDA 环境。

```bash
set -eu
test ! -e .venv-cuda
uv venv --python 3.10.16 .venv-cuda
uv pip install --python .venv-cuda/bin/python torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-cuda/bin/python \
  pip setuptools==69.0.3 wheel pybind11==2.11.1 Cython==3.0.8 numpy==1.26.3
CXXFLAGS='-include cstdint' CFLAGS='-include cstdint' \
  uv pip install --python .venv-cuda/bin/python --no-build-isolation fasttext==0.9.2
uv pip install --python .venv-cuda/bin/python --no-build-isolation -r requirements-cuda.txt
uv pip install --python .venv-cuda/bin/python -r requirements-data.txt -r requirements-research.txt
.venv-cuda/bin/python -m nltk.downloader averaged_perceptron_tagger cmudict
uv pip check --python .venv-cuda/bin/python
```

`cstdint` 只注入 fasttext 单次构建以适配 GCC 15；不要永久设置或传播到其他包。原 CUDA 依赖证据在 [freeze](../reports/p0/environment/requirements-cuda.freeze.txt)；后续可选数据/绘图依赖分别为 praatio 6.2.2、matplotlib 3.8.4。复用粗缓存的 P4 训练/评估不重新执行旧声学模型；若未来确需生成新粗预测，旧检查点和 fastText 文件需通过 Git LFS 恢复，但当前不要重复缓存推理。

现有原件在 WSL `data/local/gtsinger-training-p4`，Mac 早期子集和 WSL 原件均已作为归档保存。新机器优先从 `reports/handoff/` 恢复；若无法取得归档，使用已保存计划获取四联文件和许可，不重新 plan 或下载整集：

```bash
.venv-cuda/bin/python scripts/expand_gtsinger_training.py fetch \
  --manifest reports/p4/data-plan-20261010/manifest.json \
  --input-root data/local/gtsinger-training-p4
```

fetch 验证固定 revision、大小/摘要，复用相同文件并拒绝异内容覆盖。来源固定为 `AaronZ345/GTSinger` revision `d78cd68d00135b1995da03f25fccb3a221efd8c1`。已有冻结 `segments.jsonl`/`splits.json` 必须优先复用；下列 prepare 只用于明确需要重建数据证据的新目录，不替换训练输入或伪装为原报告（绝对路径可能改变输出摘要）：

```bash
.venv-cuda/bin/python scripts/expand_gtsinger_training.py prepare \
  --manifest reports/p4/data-plan-20261010/manifest.json \
  --input-root data/local/gtsinger-training-p4 \
  --on-invalid exclude \
  --out-dir reports/p4/training-source-restored
```

## 加载现有模型与原实验命令

模型加载 API 必须显式给新检查点，不自动下载或使用旧 MTL/BDR 权重。以下只展示加载；暂停期间没有执行：

```python
from t2l.boundary.inference import load_boundary_checkpoint, refine_boundaries

runtime = load_boundary_checkpoint(
    "reports/p4/training-cuda-20261010/boundary.pt", device="cpu")
# audio: 原始 mono [N] 或 channel-first [C,N] ndarray，first channel；不要重采样。
# candidates: 同一音频的 BoundaryCandidate/dict，中心来自 coarse，不能来自来源标签。
# refined = refine_boundaries(audio, sample_rate, candidates, loaded=runtime, device="cpu")
```

`device="cuda"` 显式要求 CUDA；失败不会暗中转 CPU。训练入口严格要求真实 CUDA AMP，不提供 CPU 训练或断点续训。以下为已执行训练的原命令存档，**不要再次执行；它会拒绝已有目录，不能删除目录绕过**：

```bash
.venv-cuda/bin/python scripts/train_boundary.py \
  --segments reports/p4/training-source-20261010/segments.jsonl \
  --splits reports/p4/training-source-20261010/splits.json \
  --audio-root data/local/gtsinger-training-p4 \
  --coarse-report reports/p4/coarse-cache-cuda-20261010/result.json \
  --config configs/boundary-tcn.json \
  --out-dir reports/p4/training-cuda-20261010
```

下列冻结评估命令已在 WSL 完成，完整结果已取回，**不要重跑**。它会核验 completed 训练报告、检查点、配置、输入及训练/推理代码摘要；不修改源码以绕过校验，也不换检查点挑选 test 表现：

```bash
.venv-cuda/bin/python scripts/evaluate_boundary.py \
  --segments reports/p4/training-source-20261010/segments.jsonl \
  --splits reports/p4/training-source-20261010/splits.json \
  --audio-root data/local/gtsinger-training-p4 \
  --coarse-report reports/p4/coarse-cache-cuda-20261010/result.json \
  --checkpoint reports/p4/training-cuda-20261010/boundary.pt \
  --device cuda --batch-size 32 \
  --out-dir reports/p4/evaluation-cuda-20261010
```

后续允许恢复、完整评估已取回并通过来源一致性检查后，绘图不运行模型，可在 WSL 或恢复相同依赖的 CPU 环境执行。输出必须不存在：

```bash
.venv-cuda/bin/python scripts/plot_boundary_results.py \
  --training-report reports/p4/training-cuda-20261010/result.json \
  --evaluation-report reports/p4/evaluation-cuda-20261010/result.json \
  --segments reports/p4/training-source-20261010/segments.jsonl \
  --splits reports/p4/training-source-20261010/splits.json \
  --audio-root data/local/gtsinger-training-p4 \
  --failure-cases \
  --out-dir reports/p4/plots-cuda-20261010
```

产物包括训练曲线、test 误差 CDF、六个最大来源误差案例及 `index.json`；CDF 保留 missing/text mismatch 分母。仅 `index.json.status=completed` 不代替实际图片内容检查。

已有后台入口 `bash scripts/run_boundary_stage.sh status` 可只读显示状态。此次训练/评估沿用旧 CLI，runner 缺少自己的退出归档会报 external_unverified；不要制造 exit code、重训或删除原件。新 runner 的 completed 需 exit=0、runner status 和结果 status 三者一致，不能把“启动成功”当实验成功。

## 接手顺序

1. 保持用户暂停，不自动训练、评估或调参；先确认云端副本包含本文件列出的源码、配置、文本证据和最佳权重，保留旧报告不可覆盖。
2. 复用已取回的完整冻结评估 JSON；completed/0 failures、八组511个test来源边界分母、与训练相同输入/源码/权重均已核对。先阅读各方法/分类/覆盖/时延及严重误差，再决定后续实验；不重复相同评估。
3. 后续获准恢复时执行冻结绘图并检查实际图片，回填最终评估证据、模型卡与阶段进度；完成 P4 最终验收后才讨论 P5/P6。不要把当前 neural 或 DSP+neural 自动接入默认产品路径。
4. 任何新实验使用明确的新目录/版本，保留 train/val/test、歌手留出和来源语义；只能用 train/val 选择，不能拿已见 test 调参后继续称同一隔离评估。

数据许可为源 `dataset_license.md` 的 CC BY-NC-SA 4.0 及附加责任/授权/引用条款，原文保存在 [数据许可](../reports/handoff/GTSINGER_LICENSE.md) 和原件归档中。代码根 [LICENSE](../LICENSE)（Apache-2.0）与 [MTL 上游许可](../t2l/mtl/LICENSE)（MIT）不能替代数据或权重条款；此次将最佳权重和数据归档保存到用户云端仓库，未另授 MIT 或商业许可。云端交接不构成许可变更或 5ms 性能声明。
