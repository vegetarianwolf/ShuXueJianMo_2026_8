# 统一分支数据层

此目录由 `scripts/build_unified_branch_data.py` 可重复生成。五个审计 tip 均固定到下列不可变 40 位提交 SHA，脚本不会解析实时分支指针。蓟州游客量与旅游综合收入的**唯一目标真值**是集成前 `main` 固定提交中的 `data/jizhou_tourism_economy/official_annual_summary_2010_2025.csv`；统一副本的 SHA-256 为 `9660557e72f2b6ea052800f1d2af7974a84633f18141999537038081a40aa905`。

## 数据契约

- `canonical_official_annual_2010_2025.csv`：上述 main 文件的逐字节副本，不接受其他分支覆盖。
- `benchmark_observations.csv`：供统一评测使用的完整 canonical 长表；字段为 `split_id, split, metric, year, value, unit, status, source_ids, quality_note, is_observed, cutoff_year`，保留原定 2019 时间切分标记以便复核跨疫情表现。
- `primary_train.csv`：最终训练/交叉验证契约，包含 `year <= 2023` 的全部 22 条非空 canonical preferred 目标记录。推导记录不被伪装成实测，仍保留 `is_observed=false` 供严格证据敏感性分析。
- `primary_test.csv`：本次重跑中执行隔离的 pseudo-holdout，仅含 2024 年 2 条实际观测；2024 不进入滚动选模或压力测试。由于模型代码形成时 2024 数据已经存在，它不是真正前瞻的未知样本。
- `rolling_origin_folds.csv`：min-train=5、外层测试年不晚于 2023 的 102 条 fold 记录。训练 role 与 `primary_train.csv` 使用相同 canonical preferred 证据契约；每折训练年份严格早于测试年，测试 role 只允许实际观测。
- `stress_train.csv` / `stress_test.csv`：cutoff=2019 的跨疫情压力测试，共 19/3 条；stress test 仅覆盖 2020—2023，明确排除 2024 pseudo-holdout。
- `sensitivity_imputations.csv`：邱分支的 8 条补值记录；全部 `is_observed=false`、`excluded_from_benchmark=true`，不得用作测试标签。
- `tianjin_gdp_2010_2025.csv`：从 `origin/111` 的 `GDP数据` 工作表按表头提取的 16 行辅助宏观数据。
- `tianjin_tourism_benchmark.csv`：从 `origin/111` 的 `天津市旅游基准` 工作表按表头提取的 4 行外部比较数据。
- `branch_data_inventory.csv`：5 个审计 tip 的数据资产清单，共 328 行，记录 Git blob、SHA-256 与纳入/排除决策。扫描范围为各 tip 的 `data/**`、表格型 `outputs/**` 和 `111完整示例` 工作簿。

测试标签不会被任何补值、目标值、情景值或模型预测覆盖。天津市 GDP 和旅游基准只作辅助特征/外部背景，也不构成蓟州目标标签。

## 已审计 tip

- `main` → `579333d9746a9cd0e12877bbcfbac4721f3ed33b`
- `origin/111` → `71c7ddcd287e3713f05e50651a8d52508daf5b89`
- `origin/邱志烨-数据搜索` → `47cc10d6c1333bfee4a121573566250c59374415`
- `codex/jizhou-tourism-modeling` → `66e27eb5a29bbf3abd51dc2dc1af4b8e41fc349c`
- `codex/jizhou-tourism-ml` → `3709084fc84614223ee00979494aa82b458296fe`
