# 基础 CI 与可选 CUDA 集成

基础工作流 `.github/workflows/tests.yml` 在 push、pull request 和手动触发时运行。单个 Ubuntu 22.04 / Python 3.10 job 安装 PyTorch/torchaudio 2.1.2 官方 CPU wheel 与 `requirements-ci.txt`，执行依赖检查、完整现有 pytest 和语法检查。测试使用合成/临时夹具，不获取 LFS 模型、Demucs 权重、GTSinger 或 NLTK 语料，不要求 GPU。JUnit 结果保留7天。

新 Linux CPU 环境的安装步骤与工作流一致：

```sh
python3.10 -m venv .venv-ci
.venv-ci/bin/python -m pip install pip==24.0 setuptools==69.0.3 wheel==0.42.0
.venv-ci/bin/python -m pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
.venv-ci/bin/python -m pip install -r requirements-ci.txt
.venv-ci/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-ci/bin/python -m pytest -q --junitxml=reports/ci-cpu/junit.xml
.venv-ci/bin/python -m compileall -q main.py t2l scripts tests
```

CPU 依赖子集不包含完整多语言推理依赖，不能代替真实模型环境。已有 baseline/CUDA 环境无需重新安装。工作流配置和本地检查也不代表尚未执行的 GitHub Actions 已通过。

本地独立 Linux CPU 环境已完成依赖检查、251项完整测试和语法检查；最终测试仅保留仓库的4个 LFS 指针。环境、源码摘要、JUnit及审查处理依据见 [本地验收证据](../reports/ci-20261010/README.md)。该结果不证明 Ubuntu runner 或远程工作流已运行。

## CUDA runner 配置

`.github/workflows/gpu-integration.yml` 只响应手动 `workflow_dispatch`，不由 PR 自动触发。运行它需要已经注册、带 `auto-lrc-gpu` 标签的 Linux x64 自托管 runner；本次没有注册 runner，也没有推送或触发远程任务。`actions/checkout` v5 需要 runner 版本至少2.327.1。

runner 应有 Git LFS、实际可用 CUDA、完整项目推理依赖、NLTK 语料和可解码 demo MP3 的音频库。依赖和 RTX 5060 配置见 [环境说明](environments.md)。在仓库 Actions 变量中设置 `AUTO_LRC_GPU_PYTHON`，指向 checkout 目录外的现有 CUDA Python 绝对路径，例如目标 WSL 上的 `/home/xcy/auto-lrc/.venv-cuda/bin/python`。工作流不安装系统驱动，也不改动该虚拟环境。Demucs 缓存缺失时，其加载器会获取模型。

手动选择待验证的代码 ref 和 `all`、`mtl`、`mtl_bdr` 或 `demucs` 阶段。工作流直接复用 `scripts/verify_gpu_runtime.py`：

- MTL 与 MTL_BDR：完整 supplied demo，关闭人声分离；检查真实 CUDA 前向、有限输出和完整单位区间。
- Demucs：demo 第30秒起的6秒片段，运行默认 `mdx_extra` ensemble 的全部成员；不是整首分离或完整 CLI 验收。
- CUDA 不可用、检查点/输入缺失、输出不合法时失败，不允许以 CPU 回退作为通过。

输出目录按 run id / attempt / stage 区分，已存在目录拒绝覆盖。首个失败阶段会停止后续阶段；已执行阶段的 JSON、LRC 和 traceback 上传并保留7天，不上传分离 WAV。报告包含实际依赖、源码/输入/检查点摘要、设备、耗时和显存，不含人工精度结论。

已完成的 P0 CUDA 和 P1 混音验收见 [工作进度](WORK_PROGRESS.md)，相关输入未变化时复用成功证据。未来仅在相关源码、环境、输入变化或有具体新风险时执行相应阶段。
