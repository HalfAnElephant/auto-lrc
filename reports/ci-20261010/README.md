# CI 配置的本地验收

2026-10-10，本轮使用本地验证渠道。没有推送、注册 GPU runner 或触发 GitHub Actions；配置存在不代表云端 CI 已通过。运行范围及恢复命令见 [docs/ci.md](../../docs/ci.md)。

## CPU 依赖与回归

另建 Arch Linux WSL 隔离目录 `/home/xcy/auto-lrc-ci-20261010`，Python3.10.16，torch/torchaudio2.1.2+cpu；CUDA 编译支持为空且不可用。按照 CPU 工作流的步骤安装测试依赖，`pip check` 和 `compileall` 均 exit0。251项测试通过，0失败、0错误、0跳过；仅有一项既有 librosa `pkg_resources` 弃用警告。没有声学模型推理或数据集下载。

| 资源条件 | 结果 | 证据 |
| --- | --- | --- |
| 初次源码归档展开了本地 LFS 模型 | 251通过，4.567秒 | [environment.json](environment.json)、[junit.xml](junit.xml) |
| 换为 Git 中的4个原始 LFS 指针 | 251通过，3.871秒 | [lfs-pointer/environment.json](lfs-pointer/environment.json)、[JUnit](lfs-pointer/junit.xml) |

第二次仅改变模型资源，核实 `checkout(lfs: false)` 的资源条件；原模型保留在隔离目录的 `data/local/ci-available-model-backup/`。源码和依赖未变，复用安装、依赖检查及语法检查结果。最终报告中49个源码、依赖及 CPU 工作流 SHA256 均与本机一致；冻结依赖随对应环境报告保存。Arch Linux WSL 的结果不等同于 Ubuntu 22.04 runner 的执行结果。

## 工作流与 GPU 调度

两个工作流使用官方 action 的固定 SHA。`actionlint` v1.7.12（声明自定义 GPU label 和仓库变量）无错误，四个原始 Bash run 块均通过 `bash -n`。未安装 `shellcheck` 或 `pyflakes`，没有执行这两项扩展检查。

最终 GPU YAML 的原始 run 块在 WSL GNU `realpath`9.10 / Bash5.3.9 上通过以下行为夹具，证据及 YAML 摘要见 [gpu-guard.json](gpu-guard.json)：

- 外部虚拟环境的 Python symlink 保留原调用路径，按 MTL、MTL_BDR、Demucs 顺序执行。
- checkout 内虚拟环境及指向它的外部目录 symlink 均拒绝，未调用任何阶段。
- 第二阶段返回37时，工作流原样失败并停止，不调用 Demucs。

夹具使用 shell stub，未运行 CUDA、模型或 pytest；不是 GPU 推理验收。既有 P0/P1 CUDA 成功结果未重跑。

## 独立审查处理

DeepSeek worker 以 `deepseek-flash[1m]`、`Read,Glob,Grep` 只读审查。首次触达8轮上限，[原始结果](deepseek-review.json) 为失败；沿用会话限定收尾，[收尾结果](deepseek-review-close.json) 成功。审查覆盖工作流、测试依赖、真实 GPU 验证入口、测试与相关导入；未审查 README 新段落或工作进度文档。

发现的阻断问题：对 `GPU_PYTHON` 整体 `realpath` 后会运行虚拟环境的基础解释器，并可能放过 checkout 内虚拟环境。最终修复仅规范化 Python 所在目录以做边界检查，执行仍使用原虚拟环境路径；上面的真实 Linux 夹具验证该修复。审查时 CPU 安装/测试尚未结束，依赖闭合性当时仅为源码推断；最终 CPU 报告补上实际验证。未把原审查中的失败或未执行检查视为通过。

人工边界复核及 P1 整体验收仍未完成；本轮 CI 配置不提供任何定位精度结论。
