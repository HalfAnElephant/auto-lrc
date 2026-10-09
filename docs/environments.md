# P0 环境与恢复

P0 的 Mac 复现环境与 GPU 环境保持独立。2026-10-09 已通过 HAER 的本地 `127.0.0.1:12345` 映射，以 Windows 用户 `Haer` 和已有 SSH 密钥登录；Arch Linux WSL2、RTX 5060、PyTorch CUDA 算子与项目模型的真实推理均已验证。

## 已建立的 Mac baseline

在仓库根目录执行，Python 3.10.16、PyTorch / torchaudio 2.1.2；保持 `requirements.txt` 的原始版本：

```sh
git lfs install --local
git lfs pull
uv venv --python 3.10.16 .venv-baseline
uv pip install --python .venv-baseline/bin/python pip setuptools==69.0.3 wheel pybind11==2.11.1 Cython==3.0.8 numpy==1.26.3
uv pip install --python .venv-baseline/bin/python --no-build-isolation -r requirements.txt
.venv-baseline/bin/python -m nltk.downloader averaged_perceptron_tagger cmudict
.venv-baseline/bin/python -m pytest -q
```

`fasttext==0.9.2` 的构建脚本未正确声明 `pybind11`；直接隔离构建会失败。先安装构建依赖，再关闭构建隔离，仅影响此虚拟环境。最终解析的完整依赖版本保存在 `reports/p0/requirements-baseline.freeze.txt`；其中 `av` 和部分传递依赖原本没有固定版本。

运行时工作目录必须为仓库根目录，检查点和 `lid.176.ftz` 按工作目录解析。测试无需加载检查点、下载 Demucs 或使用 GPU。完整 demo 及布局消融另用 `scripts/benchmark_baseline.py`，不属于轻量回归测试。

## Windows SSH 接入状态

只读探针收到 `SSH-2.0-OpenSSH_for_Windows_9.5`，说明该映射先进入 Windows SSH 服务，不能直接把它当作 WSL shell。完整探针见 [连接报告](../reports/p0/environment/windows-ssh-probe-2026-10-09.md)。默认用户 `xcy` 初次认证失败；用户补充 Windows 账户 `Haer` 后，已有密钥认证成功，`whoami` 返回 `haer\haer`。不要在聊天或仓库中保存密码、私钥。

`wsl.exe --list --verbose` 确认默认发行版实际名为 `archlinux`、VERSION 为 2。通过 `wsl.exe -d archlinux -- ...` 启动并运行探针，Linux 用户为 `xcy`，HOME 为 `/home/xcy`。Windows SSH 启动的默认工作目录是 `/mnt/c/Users/Haer`，执行项目命令时必须切换到 Linux 项目目录。

```sh
ssh -p 12345 -o BatchMode=yes -o StrictHostKeyChecking=yes \
  Haer@127.0.0.1 'wsl.exe -d archlinux -- uname -a'
```

实测环境：Arch Linux，内核 `6.6.87.2-microsoft-standard-WSL2`；GPU `NVIDIA GeForce RTX 5060`，驱动 `617.42`，显存 `8151 MiB`，compute capability `12.0`。WSL 可见内存约 `15 GiB`、swap `4 GiB`，Linux 文件系统可用约 `950 GiB`；用户说明的宿主 `32 GB` 不能当作 WSL 实际内存。系统 Python `3.14.4`、uv `0.11.6`；已有 gcc/g++，未找到 ffmpeg/ffprobe。完整机器可读结果见 [WSL 探针](../reports/p0/environment/wsl2-probe-2026-10-09.json)。

启动时出现 systemd 用户会话失败和 fish 配置引用缺失的 `/home/xcy/.cargo/env.fish` 提示，命令仍成功完成。没有改动这些用户设置。GPU 可被 `nvidia-smi` 识别不等于 PyTorch 或项目模型已通过。

## Arch Linux WSL2 / RTX 5060 的隔离 CUDA 环境

数据和环境放在 Linux 文件系统，例如 `~/auto-lrc`，避免放在 `/mnt/c`。Windows 宿主安装兼容 NVIDIA 驱动；**WSL 内不安装 Linux NVIDIA 显示驱动**，依据 [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

连接后先保存系统证据：

```sh
uname -a
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# 若 PATH 未包含它，使用 /usr/lib/wsl/lib/nvidia-smi
```

保留旧 baseline 环境，另建环境。PyTorch 2.7.1 / torchaudio 2.7.1 CUDA 12.8 组合来自 [官方版本安装页](https://pytorch.org/get-started/previous-versions/)；[2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)记录了 Blackwell / CUDA 12.8 支持。该组合现已在目标 RTX 5060 上通过实测。

上述三个官方来源于 2026-10-09 重新核对。安装时先清点现有环境与 Linux 文件系统空间，在不覆盖已有工作的项目目录创建独立 `.venv-cuda`。以下命令用于新环境恢复；现有环境无需重复安装。运行时依赖单独解析并保存实际 freeze，没有复制旧 freeze 后只替换两个 torch 版本号。

```sh
uv venv --python 3.10 .venv-cuda
uv pip install --python .venv-cuda/bin/python torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv-cuda/bin/python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory)
print('compiled architectures:', torch.cuda.get_arch_list())
x = torch.randn(2, 16, 64, device='cuda')
with torch.inference_mode():
    conv = torch.nn.Conv1d(16, 32, 3).cuda()(x)
    seq, _ = torch.nn.LSTM(16, 32, batch_first=True).cuda()(x.transpose(1, 2))
    spectrum = torch.stft(x[0, 0], n_fft=32, hop_length=8,
                          window=torch.hann_window(32, device='cuda'), return_complex=True)
    assert all(torch.isfinite(y).all() for y in (conv, seq, spectrum))
torch.cuda.synchronize()
print('CUDA convolution/LSTM/STFT smoke passed')
PY
```

不要直接把旧 `requirements.txt` 安装进此环境，它会覆盖 torch 版本且可能冲突。[requirements-cuda.txt](../requirements-cuda.txt) 仅列当前推理入口所需依赖，并让解析器选择与新 torch 相容的传递依赖；例如旧 `sympy==1.12` 不可照搬到 torch 2.7.1。必须先执行官方 CUDA wheel 安装，再安装该文件，保留已装的 `+cu128` wheel：

```sh
uv pip install --python .venv-cuda/bin/python pip setuptools==69.0.3 wheel pybind11==2.11.1 Cython==3.0.8 numpy==1.26.3
CXXFLAGS='-include cstdint' CFLAGS='-include cstdint' uv pip install --python .venv-cuda/bin/python --no-build-isolation fasttext==0.9.2
uv pip install --python .venv-cuda/bin/python --no-build-isolation -r requirements-cuda.txt
.venv-cuda/bin/python -m nltk.downloader averaged_perceptron_tagger cmudict
uv pip check --python .venv-cuda/bin/python
uv pip freeze --python .venv-cuda/bin/python
```

GPU 环境不包含与当前推理入口无关的 onnxruntime/faster-whisper 等旧依赖。旧检查点加载、音频解码、MTL/BDR/Demucs 的实际算子与完整默认 CLI 已在独立阶段验证，详见 [P0 结果](p0-results.md)。

Arch 的 GCC `15.2.1` 编译 fasttext `0.9.2` 时，`src/args.cc` 使用 `uint64_t` 却未包含 `<cstdint>`，首次构建失败。上面的命令仅为 fasttext 本次构建强制包含该标准头，保留依赖版本与上游源码，不更改系统编译器或永久环境变量；其他包不继承该标志。

首次官方 cu128 索引安装在 `pypi.nvidia.com` 的 `nvidia-cufile-cu12==1.13.0.11` 下载超时，重试 3 次后失败；没有把失败当作安装通过。恢复时保留 torch/torchaudio 官方 CUDA wheel，并让同版本 NVIDIA 传递依赖由官方 PyPI 解析，复用 uv 缓存；未修改系统代理或证书。以下 wheel URL 和 SHA-256 来自官方 cu128 索引（2026-10-09 核对）：

```sh
uv pip install --python .venv-cuda/bin/python \
  'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3' \
  'torchaudio @ https://download-r2.pytorch.org/whl/cu128/torchaudio-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=2ba0816eee659e343851a9c5dc60c8e1eb819a3969b29268fab27d3143273d78' \
  --index-url https://pypi.org/simple
```

此恢复安装已成功，torch `2.7.1+cu128`、torchaudio `2.7.1+cu128`；torch 识别 RTX 5060，CUDA `12.8`，编译架构含 `sm_120` 和 `compute_120`。最小推理依赖安装成功，NLTK `averaged_perceptron_tagger` / `cmudict` 下载完成；t2l 入口、语言识别与 g2p 导入及样例转换通过。`uv pip check` 检查 86 包全部兼容，实际冻结结果保存在 [CUDA freeze](../reports/p0/environment/requirements-cuda.freeze.txt)，[导入探针](../reports/p0/environment/cuda-runtime-imports-2026-10-09.json)保存结果。GPU 算子与完整模型验收结果由主代理统一整合，不能用安装或导入结果替代模型通过。

## 阶段边界

目前不训练、不下载整套 GTSinger、不声明精度收益。P0 的环境与 baseline 运行验收已完成，进入 P1 的少量数据准备、按歌曲划分及人工标注复核。任何 5ms 结论必须来自隔离且经独立人工复核的测试集。
