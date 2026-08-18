# 数学建模题目 C：天津市统计数据分析与预测

本仓库用于保存小组数学建模题目 C 的原始资料、数据处理代码、分析结果和论文源文件。

## 目录约定

```text
.
├── TJMML_C.pdf          # 题目原文
├── data/
│   ├── raw/              # 从官方入口下载的原始文件，只做归档，不直接覆盖
│   ├── processed/        # 清洗、整理后的分析数据
│   ├── unified/          # 固定分支快照、统一切分和数据资产审计
│   └── metadata/         # 来源、下载时间、字段说明和数据字典
├── code/
│   ├── src/              # 可复用的数据处理、建模代码
│   ├── scripts/          # 可直接运行的脚本
│   └── notebooks/        # 探索性分析与过程记录
├── tex/                  # LaTeX 论文源文件、参考文献和自定义样式
├── build/tex/            # LaTeX 编译产物（默认不纳入版本控制）
├── figures/              # 论文图表（默认不纳入版本控制）
├── docs/                 # 方案、会议记录和阶段性说明
└── tests/                # 数据与模型的回归检查
```

## 当前阶段

1. 已建立目录和版本控制忽略规则。
2. 题目给出的天津市统计局数据入口已登记在 [`data/metadata/sources.csv`](data/metadata/sources.csv)。
3. 原始下载文件统一放入 `data/raw/`，下载失败或网页仅提供在线表格时记录在来源清单中，不把网页截图当作正式数据。
4. 题目已使用 MinerU 重新解析，主 Markdown 为 `docs/TJMML_C.md`，旧的本地回退版保存在 `docs/TJMML_C_fallback.md`。
5. LaTeX 模板已并入 `tex/template/`，模板编译产物归档在 `build/tex/template/`。
6. 模板已改为引用仓库内字体；不再依赖 `STHeiti`、Times New Roman、Arial 或 `simkai.ttf` 等系统字体。
7. 五个业务分支的独有数据和模型交付已合入，并在 `data/unified/branch_data_inventory.csv` 中按固定提交记录哈希与使用决策。
8. `data/unified/primary_train.csv` 保留 2023 年及以前的 22 条 canonical 证据；用户指定的 2010—2023 模拟增强建模层另存为 `outputs/unified_model_benchmark/primary_train_augmented.csv`，2024 年两条实际观测只作执行流程 pseudo-holdout。
9. 传统与 ML 固定适配器已在“模拟增强”和“未模拟 canonical 共同行（official-source）”两轨重跑；原分支声明代表无法共同排名，不能据此判定原分支赢家。统一报告见 `docs/unified_branch_model_comparison.md`。

## 可复现约定

- 原始资料不在分析代码中硬编码；采集脚本读取 `data/raw/`，建模脚本读取经审计的 canonical/unified 数据层。
- 每个外部数据文件都在来源清单中记录 URL、获取时间、文件名和备注。
- `build/tex/`、`figures/` 和 `data/processed/` 的生成结果默认不提交，确认稳定后再按需解除忽略。

## 统一比较复现

```bash
.venv/bin/python scripts/build_unified_branch_data.py
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg \
  .venv/bin/python code/scripts/compare_branch_models.py
.venv/bin/python -m unittest discover -s tests -v
```

模型依赖版本记录在 `code/requirements-ml.txt`。统一数据生成器仅使用 Python 标准库；模型比较固定随机种子 `20260817`。
