# 维护验证记录

验证日期：2026-10-29。原仓库基准提交：`eb24cfa2e6ca3784be467a0866594d389e2eacef`。

## 已执行

| 检查 | 结果 |
|---|---|
| 原成果包与源码项目结果逐文件比对 | 58 个文件 SHA256 一致 |
| 原有同源符号模型/接收机测试 | 6 项通过 |
| 新复现入口/比较检查 | 4 项通过，覆盖归档保护、已有结果保护、绘图网格约束、数值差异识别 |
| 归档验证 | 800 组原始样本统计、552 个绘图数据点、36 个图导出通过 |
| 完整配置重跑 | 通过；通信 2,000 帧/点，感知 6,000 次/点 |
| 重跑数值阶段耗时 | 358.15 秒（不含绘图） |
| 重跑通信统计均值与归档比较 | 320 / 320 行一致 |
| 重跑感知统计均值与归档比较 | 510 / 640 行一致；130 行 SIC 相关均值有差异 |
| 基于归档数据重新绘图 | 12 组图完成；主统计表 960 / 960 行一致 |
| 原 MATLAB 文件 | 8 个文件哈希保持一致 |
| 参考归档 | 62 个文件哈希保持一致 |

运行环境：Python 3.10.16、NumPy 2.0.1、Matplotlib 3.10.1，Windows。测试复用现有 E 盘 ISAC 环境；项目和本次新结果位于 D 盘。

复查命令：

```text
python reproduce.py test
python reproduce.py verify
python reproduce.py full --out results/another_full_run
python reproduce.py verify --out results/another_full_run
```

CSV 均值比较采用相对容差 `1e-9`、绝对容差 `1e-11`，并检查每行样本数；未通过放宽容差隐藏 SIC 差异。`verify` 检查新结果内部一致性，`compare` 判断是否与历史归档一致。完整重跑的 `compare` 返回退出码 1，属于已记录的差异，不应标记为全部数值复现成功。

## SIC 数值敏感性

本次差异限于感知 SIC：BIA 的 LFM、Walsh 和部分 QPSK 设置，以及 TIM 的 QPSK 设置。通信和投影 LS/TIN 的均值均在上述容差内一致。

对 BIA/LFM、rho=0.3、SNR=10 dB 的初始 LS 残差做诊断，共 `36000` 个软判决中，实部绝对值小于 `1e-12` 的有 `18000` 个，虚部同样有 `18000` 个。原 SIC 直接用实/虚部是否小于零进行硬判决；接近零时的符号可受浮点计算影响，再被后续非线性迭代放大。把参考功率改变一个浮点相邻间隔的诊断也保存在 `sic_numerics_audit.json`。这支持原 SIC 存在数值敏感性的判断，但不把它夸大为所有差异唯一成因的完整证明。

运行 `python -B tools/audit_sic_numerics.py` 可复查诊断；具体数值依 NumPy/BLAS 环境而变化。当前保留原科学源码和原归档。确定性的平局判决、排序策略或其他接收机修订应另行审查并建立新结果版本。

新生成 12 图总览及图 03 已目视检查，未发现明显截断或缺失。

本次完整运行的逐点报告在 `results/shared_source_full/verification.json`、`archive_comparison.json` 与 `run_environment.json`，结果目录被 Git 忽略。参考归档内的历史验证记录保持原样。

未安装可选 PyMuPDF，本次没有重新执行 PDF 页数/文字提取检查；PNG/PDF/SVG 文件存在性、SVG 文本节点、图表 CSV 与原始统计的对应关系已由验证器检查。原 MATLAB 脚本未运行。
