# 暂停训练交接归档

2026-10-10：保留当前研究成果，供他人恢复环境并继续研究。当前没有活跃训练进程，不自动开始下一阶段；最新状态见 [训练交接](../../docs/TRAINING_HANDOFF.md)。虚拟环境、包缓存、临时锁、重复源码传输包不是训练成果，不归档；环境依赖版本及恢复命令已保留。既有代码和报告按原路径提交，不改写训练输入或报告摘要。

## 文件

| 文件 | 内容 |
| --- | --- |
| `mac-data-20261010.tar.gz` | Mac的P1 smoke、代表性原件及smoke混音；含源许可、四联文件、原创伴奏许可 |
| `auto-lrc-data-handoff-20261010.tar.gz` | WSL的P4训练原件（180段计划原件，163段实际有效）和已有混音/分离音频；按 `data/local/` 原路径归档，含源许可 |
| [GTSINGER_LICENSE.md](GTSINGER_LICENSE.md) | 固定数据revision的完整许可与附加条款可读副本；原始字节文件另保存在压缩包内 |

最佳权重在 `reports/p4/training-cuda-20261010/boundary.pt`，完整评估在 `reports/p4/evaluation-cuda-20261010/result.json`。大文件和两个数据归档使用 Git LFS；普通源码clone中的指针不是实际成果内容。

交接验证：WSL数据归档和完整评估JSON的SHA256与远端原件一致；P4归档的721个计划文件逐项核对固定manifest的大小及git blob/SHA256摘要通过，未发现绝对路径、父目录穿越或链接成员。训练及评估的输入、全部记录的运行源码和最佳权重摘要与当前文件一致；八组test分母均为511、missing/text mismatch=0。已有实现定点检查215项通过及runner的11组契约夹具结果复用，未重跑训练/评估或等价本地测试；云端CI状态须以此次push实际触发结果为准。

## 新机器恢复

从仓库根目录执行，先取得LFS对象，再检查压缩包内容：

```bash
git lfs install
git lfs pull
shasum -a 256 -c reports/handoff/SHA256SUMS
tar -tzf reports/handoff/mac-data-20261010.tar.gz
tar -tzf reports/handoff/auto-lrc-data-handoff-20261010.tar.gz
```

仅在不存在对应数据目录的新副本解包，禁止覆盖已有原件：

```bash
test ! -e data/local/gtsinger-p1
test ! -e data/local/gtsinger-representative
test ! -e data/local/gtsinger-training-p4
test ! -e data/local/mixtures
tar -xzf reports/handoff/mac-data-20261010.tar.gz
tar -xzf reports/handoff/auto-lrc-data-handoff-20261010.tar.gz
```

两个归档的混音子目录不同；原始音频不重采样，绝对来源路径与JSON摘要不改写。按训练交接说明的 `--audio-root data/local/gtsinger-training-p4` 运行，保留固定segments/splits/coarse/config。无法取得归档时可按固定manifest/revision重新获取，不能重新划分数据或拿test调参。

检查点只保存最佳模型和训练元数据，没有优化器、AMP scaler或RNG状态；当前训练CLI不支持resume。后续新训练必须采用新的实验目录并明确协议，不能将加载最佳权重称为从第36轮无缝恢复。实际P4图片尚未执行，P4最终验收、P5/P6仍待接手。

数据及人声/混音适用GTSinger条款，原创独立伴奏许可随各混音目录保存。此次云端归档不另授MIT或商业权重许可，也不构成独立人工精度或5ms性能声明。
