# Interference alignment for integrated sensing and communications

BIA（盲干扰对齐）与 TIM（拓扑干扰管理）的 ISAC 仿真。仓库保留原有 MATLAB 探索代码，并加入可复现的 **同原始符号、直接叠加、统一 QPSK 接收机** Python 扩展。

The Python extension compares BIA/TIM encoding against direct superposition with identical raw payloads, matched symbol/sensing energies, and the same marginal bit-MAP communication receiver. It includes the archived 12-figure result set, raw Monte Carlo samples, and a portable reproduction entry point.

**复现限制：** 完整重跑可以生成并内部验证 12 组图，但部分感知 SIC 曲线对数值环境敏感。本次 320 行通信均值全部与归档一致；640 行感知统计中有 130 行 SIC 相关均值不一致。`verify` 检查结果内部一致性，`compare` 单独检查是否与历史归档一致，不把二者混同。需要准确重画已归档曲线时使用 `plot`。详见 [验证记录](docs/VALIDATION.md)。

## 从这里开始

- 查看现有结果：[12 组图总览](extensions/shared_source/reference/overview.png)、[合并 PDF](extensions/shared_source/reference/all_figures.pdf)、[结果说明](extensions/shared_source/reference/结果说明.md)。下载仓库后还可在浏览器打开 `extensions/shared_source/reference/index.html`。
- 查看模型与比较约束：[同源符号比较定义](extensions/shared_source/SHARED_SOURCE_COMPARISON.md)。
- 复现实验：使用根目录的 `reproduce.py`，无需 MATLAB，也无需访问外部 API。

## 安装与运行

建议 Python 3.10 或更新版本。维护验证使用 Python 3.10.16、NumPy 2.0.1、Matplotlib 3.10.1。只需 NumPy 与 Matplotlib；如安装 PyMuPDF，验证器还会检查 PDF 页数和可提取文字。

在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe reproduce.py test
.venv\Scripts\python.exe reproduce.py verify
.venv\Scripts\python.exe reproduce.py quick
```

Linux/macOS 把 `.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。已有合适环境时可直接使用 `python`。需要使用本次验证的依赖版本时，安装 `requirements-tested.txt`（建议搭配 Python 3.10）。

| 命令 | 用途 | 默认输出 |
|---|---|---|
| `python reproduce.py test` | 运行 6 项既有模型测试及 4 项入口/结果比较检查 | 终端 |
| `python reproduce.py verify` | 从原始样本检查归档统计、功率、符号、速率上限及源码哈希 | 终端，不改写归档 |
| `python reproduce.py quick` | 通信 100 帧/点、感知 400 次/点的端到端试跑 | `results/shared_source_quick/` |
| `python reproduce.py full` | 通信 2,000 帧/点、感知 6,000 次/点的完整实验 | `results/shared_source_full/` |
| `python reproduce.py plot` | 使用归档数据重画 12 组图，不重跑 Monte Carlo | `results/shared_source_replot/` |
| `python reproduce.py verify --out results/shared_source_full` | 检查指定运行结果 | 终端 |
| `python reproduce.py compare --out results/shared_source_full` | 比较新旧统计均值；发现差异时退出码为 1 | 终端 |

`quick`、`full`、`plot` 均支持 `--out`；非空输出目录会被拒绝，已有结果不会自动覆盖。示例：

```powershell
python reproduce.py full --out results/reproduction_01
python reproduce.py full --config my_config.json --out results/seed_02
```

从 `extensions/shared_source/configs/shared_source.json` 复制配置后修改种子或样本数。12 图入口需要保留四类波形及 0、10、20、30、35 dB 的 SNR 点；任意新实验网格需要同步修改绘图模块。快速结果只用于检查流程，不能替代完整结果。完整重跑耗时依硬件而异，历史运行约 232 秒（不含绘图），不是性能保证。

## 本次纳入的扩展

| 项目 | BIA | TIM |
|---|---:|---:|
| 每帧独立通信符号数（两格式相同） | 6 | 20 |
| 编码方案实际时隙数 | 6 | 10 |
| 直接叠加实际时隙数 | 4 | 4 |
| 共同窗口 QPSK 信息率上限 | 2 bit/channel use | 4 bit/channel use |
| 直接叠加按实际时长的信息率上限 | 3 bit/channel use | 10 bit/channel use |

两格式共享原始通信及感知符号，匹配逐信息符号能量和每发射机感知总能量，并满足逐时隙**期望**功率上限。通信端使用同一有限 QPSK 边缘后验/逐比特 MAP 接收规则。感知端比较编码方案的投影 LS 与直接叠加下的 SIC、TIN。固定感知序列包括 DFT、LFM、Walsh 和 QPSK。

解释结果时保留以下条件：

- 共同窗口与实际发送时长是两种不同归一化；不能据共同窗口曲线推断所有时间口径下均有优势。
- 信息率是有限 QPSK 互信息的 Monte Carlo 估计，不是实现纠错码后的净吞吐量。
- 通信接收端假设已知通信信道和感知波形，可精确抵消感知分量；感知接收端不知道通信数据。
- 通信和感知使用不同 Monte Carlo 批次，tradeoff 比较均值，不能据此计算同帧联合中断概率。
- 零观测误码不意味着真实 BER 为零；图中相应位置显示上限标记。
- 感知 SIC 的初始硬判决可能依赖接近零的浮点符号，固定种子不能保证跨数值环境复现所有 SIC 曲线。当前版本保留原实现；确定性的平局判决规则需要另行验证后建立新基线。
- 本扩展是复基带模型，不声称实现了 LDPC、脉冲成形、真实射频链路或原论文所有场景。

## 目录

```text
reproduce.py                        # 统一复现入口
requirements*.txt                   # 依赖与已测试版本
docs/INTEGRATION.md                  # 来源、迁移范围和后续维护方式
extensions/shared_source/
  run_shared_source.py               # 原有同源符号 Monte Carlo 入口
  verify_shared_source.py            # 验证器（默认只读）
  configs/shared_source.json         # 完整实验配置
  isac/                             # 模型、接收机、序列、绘图
  tests/                            # 原有 6 项有效版本测试
  reference/                        # 12 图、CSV、原始 NPZ、配置和历史 manifest
  import_provenance.json             # 导入文件哈希及归档一致性记录
tests/                              # 复现入口的边界检查
results/                            # 新运行结果（Git 忽略）
*.m                                 # 原有 MATLAB 文件，保持原样
```

## 原 MATLAB 代码

根目录保留 `BIA_main.m`、`test.m`、`test0629.m` 及其辅助函数。它们是早期探索版本，运行条件和指标口径与本次 Python 扩展不同。部分脚本调用 MATLAB 通信函数；`BIA_main.m` 引用的 `sengene` 未包含在原仓库中。本次没有重写或验证这些脚本，也不把其中的 BER 输出当作新扩展的复现结果。

旧 Python 联合图采用过不同的基线符号数及通信检测器，因此不属于本次有效主结果。维护时请以 `shared_source` 目录的定义、测试与归档为基准。详细范围见 [整合说明](docs/INTEGRATION.md)。
