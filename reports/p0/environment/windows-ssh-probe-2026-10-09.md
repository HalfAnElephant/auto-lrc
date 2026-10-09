# Windows / WSL2 接入探针

日期：2026-10-09（Asia/Shanghai）。范围：HAER 映射、现有 SSH 认证及 Windows/WSL2/GPU 的只读检查。用户补充账户名后已登录 Windows 并启动 WSL2，尚未安装或修改远端环境。

## 已证实

| 项目 | 结果 |
| --- | --- |
| 用户提供的映射 | 本地 `12345` → 目标 `127.0.0.1:22` |
| 本机连接端点 | `127.0.0.1:12345` |
| SSH 服务 banner | `SSH-2.0-OpenSSH_for_Windows_9.5` |
| 现有配置的默认用户 | `xcy`；没有此端点的专用 Host 配置 |
| 首次主机密钥 | ED25519，`SHA256:+Ggd1bJ71eUEtTCZ4QRRPW9IiH8jCiS8yLzNtTdbUXo` |
| 批处理密钥认证 | 失败，SSH 退出码 255 |
| 远端返回 | `xcy@127.0.0.1: Permission denied (publickey,keyboard-interactive).` |
| 用户补充的 Windows 账户 | `Haer` |
| 正确账户认证 | 成功，`whoami` 返回 `haer\haer` |
| WSL 发行版 | 实际名 `archlinux`，VERSION `2` |
| WSL Linux 账户 | `xcy`，HOME `/home/xcy` |

首次使用严格主机检查时报告未登记密钥，随后使用 `StrictHostKeyChecking=accept-new` 登记新主机；没有遇到或绕过主机密钥冲突。该首次信任记录位于当前 Mac 的 `~/.ssh/known_hosts`，并非用户独立确认的服务器指纹。认证探针使用 `BatchMode=yes` 和 8 秒连接超时，不交互输入密码，不探测其他用户名。

```sh
ssh -p 12345 -o BatchMode=yes -o ConnectTimeout=8 \
  -o StrictHostKeyChecking=yes Haer@127.0.0.1 whoami
```

后续登录与探针分别证明 Windows 登录、Arch WSL2 和 GPU 识别成功，详细结果见 [机器可读 WSL 探针](wsl2-probe-2026-10-09.json)。服务 banner 本身不能作为这些结果的证据。

## 尚缺信息与下一步

认证问题已由使用正确账户 `Haer` 解决，无需改变目标授权密钥。不要提交密钥、密码或凭证到仓库。

认证成功后按顺序检查：

1. Windows `whoami`、`wsl.exe --list --verbose`；使用实际发行版名，确认 VERSION 为 2。
2. WSL 内 `uname -a`、`/etc/os-release`、`id`、`pwd`、Python/uv/ffmpeg 可用性，以及项目目录和 Linux 文件系统空间。
3. `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`；PATH 未包含时尝试 `/usr/lib/wsl/lib/nvidia-smi`。
4. 在 Linux 文件系统的独立目录建立 `.venv-cuda`，候选 Python 3.10、torch/torchaudio 2.7.1、CUDA 12.8 wheel；保留已有 baseline 环境。
5. 保存实际 torch/CUDA/cuDNN、compute capability、编译架构、显存和 convolution/LSTM/STFT 实测；再验证旧检查点、音频解码、MTL、BDR 与 Demucs。仅安装成功或算子探针通过均不足以宣称完整项目通过。

已完成第 1–3 项：Arch Linux，WSL2 内核 `6.6.87.2-microsoft-standard-WSL2`；RTX 5060，驱动 `617.42`，显存 `8151 MiB`，compute capability `12.0`；Linux 磁盘可用约 `950 GiB`；Python `3.14.4` 与 uv `0.11.6`。尚未建立项目 CUDA 环境，PyTorch 算子与项目 GPU 推理仍为 **unavailable**，不得将 P0 目标 GPU 环境标为完整通过。

## 官方依据（2026-10-09 核对）

- [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)列出 Linux/Windows 的 torch 2.7.1、torchaudio 2.7.1 与 `cu128` 安装源。
- [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)说明 Blackwell 支持及 CUDA 12.8 预编译 wheel；目标 RTX 5060 的实际算子支持仍需实测。
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)要求依赖 Windows NVIDIA 驱动，WSL 内不安装 Linux NVIDIA 显示驱动。

详细候选命令在 [环境文档](../../../docs/environments.md)。这些是官方安装兼容依据，不是目标机器的运行证据。
