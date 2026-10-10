# P3 实测证据

2026-10-10：无训练 DSP 精修的实现、val选择、固定 WSL 评估及两段训练混音诊断已完成。默认保留 coarse。完整方法、命令及限制见 [DSP 文档](../../docs/p3-dsp-refinement.md)。本阶段无外部或独立只读审查。

| 证据 | 范围与状态 |
| --- | --- |
| [select/result.json](dsp-select-20261010/result.json) | Mac CPU，仅3段val；completed，0结构失败；墙钟0.330秒 |
| [selection.json](dsp-select-20261010/selection.json) | 预声明3配置及合并val指标；best_trial_config=low_prior，selected=none |
| [evaluate/result.json](dsp-evaluate-wsl-20261010/result.json) | WSL CPU，18段全划分固定评估；completed，0结构失败；墙钟2.839秒、CPU2.829秒；GPU null |
| [mixtures/result.json](dsp-mixtures-wsl-20261010/result.json) | WSL CPU，中英文各一段训练诊断；completed，墙钟1.403秒；复用既有推理 |
| [figures/index.json](dsp-figures-20261010/index.json) | 误差分布图及6个失败窗口的逐项来源索引 |

选择只读取 val 标签，两模型共74次边界评估；low_prior MAE151.75→132.63ms，但 ±5ms覆盖2.70%→0%、±20ms覆盖8.11%→5.41%，未通过不降低覆盖的门槛。没有从 test 选择其它配置。

固定 test 每模型98边界，MTL/MTL_BDR结果相同：coarse→DSP试验 MAE82.58→79.96ms，P90166.64→173.25ms，P95217.66→243.53ms，±5ms5.10%→3.06%，`>500ms`错误1→2。selected 仍等于 coarse。未达到5ms目标；0结构失败不等于无大误差。

两段混音均属于 train，共25边界/模型。clean、mixture、separated 的 coarse→DSP MAE分别为90.39→69.82、121.32→126.38、118.25→102.31ms。此处不构成独立测试或自然伴奏覆盖；没有额外调参、重采样、时间平移或模型推理。分离音频保留原生44.1kHz网格。

所有指标相对公开 `textgrid:first_phone` 标注，`independent_human_accuracy=false`，未伪造人工真值或已确认歌曲身份。原划分保持7/3/8段、96/37/98边界；val仅中文、test仅英文。训练隔离保留，不能将本子集结果泛化为最终多语言精度。

输入身份与代码身份由报告完整 SHA256 绑定：

| 输入 | SHA256 |
| --- | --- |
| P1 segments.jsonl | `4b2cfad5009a1aa8707d971221766b188af1b8c3ef1b86972eefde816c6cc4f5` |
| P1 splits.json | `f37c6f801e7a624b5f942bfce735674b705492d186f18b0106e0ec2bb182b589` |
| P2 coarse result.json | `7c8e309177f55d472936e90fe0d5e7513ccf3dc452dc9d9e125ddff37144f59e` |
| DSP 源码 | `449638924051253d2c1c35340f66b1e07f1ace61a3009bc12434a551095a45ee` |
| features 源码 | `16ba53c01301aaeef5b481c63ee26dd03a47fc6e5139dbd7e14794de91ddb6d8` |
| selection.json | `6be76adf996fc1bd401747945d692f6669f209fa12669b81cd298ed3feae98d7` |

原 WAV 也经摘要校验，候选细节与缺失/错配分母留在结果中。定点检查 feature39 + DSP20 + benchmark40 + mixture1 = **100项通过**；benchmark40成功结果复用，文档收口没有重跑检查。恢复时读取现有证据，禁止覆盖归档目录；下一阶段 P4。
