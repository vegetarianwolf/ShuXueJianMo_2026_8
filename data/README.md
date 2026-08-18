# 数据目录

- `raw/`：官方网页、Excel、CSV、PDF 等原始资料。保留原始文件名或使用稳定的英文文件名，并在 `metadata/sources.csv` 中记录对应关系。
- `processed/`：经过字段统一、缺失值处理和单位核验的数据集。
- `unified/`：五个固定分支快照的数据资产清单、canonical 副本、统一训练/测试切分、滚动折和敏感性 sidecar。
- `metadata/`：来源清单、数据字典、下载日志和版本信息。

数据处理时请保留原始值、单位、统计口径和年份，不直接在 `raw/` 中修改下载文件。

统一评测只把 `jizhou_tourism_economy/official_annual_summary_2010_2025.csv` 作为目标标签来源。`unified/sensitivity_imputations.csv` 中的补值全部禁止进入 benchmark 标签。

`unified/primary_train.csv` 是 physical 证据层；用户指定的模拟增强训练层位于 `outputs/unified_model_benchmark/primary_train_augmented.csv`，模拟行保留 `is_simulated`、生成方法和可见截止年，不能冒充观测。
