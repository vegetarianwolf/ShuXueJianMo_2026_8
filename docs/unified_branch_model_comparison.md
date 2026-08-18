# 统一分支模型比较报告

## 结论先行

**不能判定原分支赢家。** 原传统分支声明的主模型是 `post_2022_level_break`，而用户本轮统一协议明确不运行任何断点模型；原 ML 分支声明的是 `ridge_regime` 模型族和 `raw_target_ridge_alpha_0.1` 点路径。因此原声明代表没有一组可共同排名的滚动预测。

在用户指定的模拟增强适配器轨上，描述性第一是 `raw_target_ridge_alpha_0.1`（macro-sMAPE 14.745%）；在更适合作为稳健性依据的未模拟 canonical 共同行轨（official-source）上，描述性第一是 `raw_target_ridge_alpha_0.1`（14.661%）。两轨的适配器相对顺序一致，但这仍不能替代原分支声明模型的共同排名。 这些表只比较“共同训练行上的固定规格适配器”，不是原分支胜负。

模拟增强轨是假设性分析，不是真实观测证据：log 插值和 log 增长尾推在结构上更贴近 log-linear OLS，而且会把游客量 2020—2022 的未知疫情路径平滑成趋势点。报告因此优先用未模拟共同行轨判断稳健性，并把两轨并列展示。

滚动稳定性框架采用截至 2023 年的 expanding-origin 外层验证；2024 年仅作最终单年 pseudo-holdout（每个目标 n=1），不用于本次执行的排序。模型族代码是在 2024 数据已经存在后形成，统一数据又是 final-vintage 回溯版，因此 2024 不是研究设计层真正“未见”的前瞻测试。2019 截断结果仅作跨疫情/恢复阶段压力测试，不与 pseudo-holdout 混称。

## 分支数据合并结论

- 共审计并合入 5 个唯一业务 tip。`main`、传统模型分支和 ML 分支的核心 `data/` 树完全相同，因此没有重复拼接同一批年度观测。
- `origin/111` 的独有天津市 GDP（2010—2025）与天津市旅游基准已规范化为独立辅助表；天津市口径不会覆盖蓟州区 GDP，也不直接充当蓟州目标标签。
- `origin/邱志烨-数据搜索` 的 8 条补全值保存在 `sensitivity_imputations.csv`，全部标记为非观测并排除在统一训练、测试和排名之外；其全样本标准化、异常检测等派生列也没有直接进入回测。
- 两个分支中的旧版工作簿、预测和情景交付均保留在 Git 历史/原路径，但不会覆盖集成前 `main` 固定提交的 canonical 真值。逐文件来源、blob、SHA-256 和纳入决策见 `data/unified/branch_data_inventory.csv`。

## 可比协议

- 随机种子固定为 `20260817`；所有模型读取同一统一数据层。
- 滚动外层最小训练记录数为 5，游客量测试年为 `[2015, 2016, 2017, 2018, 2019, 2023]`，综合收入测试年为 `[2015, 2017, 2018, 2019, 2021, 2023]`；外层测试最晚到 2023 年。
- 每折模拟只读取该折测试年前已存在的 physical training rows：内部缺口用两侧训练边界在 log 尺度插值；训练尾部到 `test_year-1` 用最近至多 3 个训练区间的 annualized log-growth 中位数外推。逐点方法、源年、边界见 `simulated_training_points.csv`；绝不读取外层测试、未来官方值或邱分支 sidecar。
- 固定主适配器是传统 `no_break_log_linear_common_rows` 与 ML `raw_target_ridge_alpha_0.1`。两者逐折使用完全相同的 physical/simulated/effective 行；脚本运行时强制验证这一不变量。
- ML 全候选及新建 `ml_inner_selector` 只作探索；其预处理、调参和选择均限定在每个外层折训练内部，但不能据其事后名次宣布分支赢家。
- 2024 pseudo-holdout 训练集每目标记录数为 `{'tourism_comprehensive_income': 11, 'tourist_visits': 11}`，测试集严格为游客量和综合收入各一条 2024 实际值；隔离只成立于本次重跑的执行流程。
- `data/unified/primary_train.csv` 是 2024 前的 physical 证据层；`outputs/unified_model_benchmark/primary_train_augmented.csv` 是用户指定的 2010—2023 建模训练层，包含 physical 与 simulated 行并保留 `is_simulated/method/known_through_year`，可直接与 `data/unified/primary_test.csv` 配对。
- 主排序指标是先按目标计算 sMAPE，再对两个目标 50/50 等权；不跨单位汇总 RMSE。表中同时给出相对 `naive_last` 的 skill 与最坏点误差。
- 传统断点模型 `post_2022_level_break`、`strict_evidence_level_break` 在所有 scope 均为 `not_executed_user_protocol`，不生成预测或误差。原生 `pre_covid_exponential` 和会排除 2020—2022 的 `no_break_log_linear` 仅保留在未模拟 canonical 敏感性表。

## 用户指定的模拟增强固定适配器排序（假设性）

| descriptive_rank | branch | model | macro_smape_percent | macro_smape_skill_vs_naive | beats_naive_all_targets | worst_target | worst_target_smape_percent | worst_point_smape_percent | tie_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 14.745 | 0.321 | True | tourism_comprehensive_income | 16.489 | 60.689 | unique |
| 2 | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 15.634 | 0.280 | True | tourism_comprehensive_income | 17.240 | 74.375 | unique |

该表的 `descriptive_rank` 只描述用户指定规格在模拟伪标签上的相对误差；`robust_winner` 固定为 false。模拟点会改变目标路径，不能当成新增事实或原分支的历史声明流程。

## 未模拟 canonical 共同行排序（official-source，稳健性优先）

| descriptive_rank | branch | model | macro_smape_percent | macro_smape_skill_vs_naive | beats_naive_all_targets | worst_target | worst_target_smape_percent | worst_point_smape_percent | tie_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 14.661 | 0.276 | False | tourism_comprehensive_income | 16.404 | 51.002 | unique |
| 2 | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 18.001 | 0.111 | False | tourism_comprehensive_income | 20.402 | 76.106 | unique |

这里不生成任何 benchmark 伪标签。传统 common-row OLS 与 raw Ridge 都吃每折全部 physical canonical training rows，测试仍只用官方实际；canonical 可包含官方来源的同比反推、回列或 supporting 值，并以 `status/is_observed` 明示，例如 2010 年综合收入是 `inferred_from_yoy`，不是 strict observed。该轨优先于模拟轨用于判断结论是否由伪标签驱动；即使两轨顺序一致，也只能称适配器相对排序。

## 两轨逐目标核对

| branch | model | metric | n_test | smape_percent | naive_smape_percent | smape_skill_vs_naive | training_track |
| --- | --- | --- | --- | --- | --- | --- | --- |
| codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | tourism_comprehensive_income | 6 | 16.404 | 27.892 | 0.412 | official_only_physical_rows |
| codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | tourism_comprehensive_income | 6 | 20.402 | 27.892 | 0.269 | official_only_physical_rows |
| codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | tourist_visits | 6 | 12.919 | 12.627 | -0.023 | official_only_physical_rows |
| codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | tourist_visits | 6 | 15.600 | 12.627 | -0.235 | official_only_physical_rows |
| codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | tourism_comprehensive_income | 6 | 16.489 | 25.668 | 0.358 | user_simulated_augmentation |
| codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | tourism_comprehensive_income | 6 | 17.240 | 25.668 | 0.328 | user_simulated_augmentation |
| codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | tourist_visits | 6 | 13.000 | 17.785 | 0.269 | user_simulated_augmentation |
| codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | tourist_visits | 6 | 14.029 | 17.785 | 0.211 | user_simulated_augmentation |

## 原分支声明方法核对

| branch | model | original_declaration | rolling_execution_status | simulated_track_macro_smape_percent | official_track_macro_smape_percent | jointly_rankable_original_declared_representatives |
| --- | --- | --- | --- | --- | --- | --- |
| codex/jizhou-tourism-modeling | post_2022_level_break | primary_model | not_executed_user_protocol | — | — | False |
| codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | recommended_point_forecast | fixed_adapter_executed_but_historical_selection_saw_2024 | 14.745 | 14.661 | False |
| codex/jizhou-tourism-ml | ridge_regime | selected_ml_models for both targets | exploratory_candidate_retrained_with_inner_tuning | 18.083 | — | False |

传统原声明模型没有执行值，`jointly_rankable_original_declared_representatives=false`。`raw_target_ridge_alpha_0.1` 本次每折固定 alpha 且不读取外层测试，但原 ML 分支推荐它时已检查含 2024 的回测；`ridge_regime` 仅作模型族声明审计。因此没有合法的原分支级胜者。

## 全候选探索性排名

| exploratory_rank | branch | model | macro_smape_percent | macro_smape_skill_vs_naive | worst_point_smape_percent | stability_flag |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | codex/jizhou-tourism-ml | gaussian_process | 14.613 | 0.327 | 68.177 | unique |
| 2 | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 14.745 | 0.321 | 60.689 | unique |
| 3 | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 15.634 | 0.280 | 74.375 | unique |
| 4 | codex/jizhou-tourism-ml | ridge_regime | 18.083 | 0.168 | 73.697 | unique |
| 5 | codex/jizhou-tourism-ml | ml_inner_selector | 18.206 | 0.162 | 73.697 | unique |
| 6 | codex/jizhou-tourism-ml | svr_rbf | 20.105 | 0.075 | 47.850 | unique |
| 7 | codex/jizhou-tourism-ml | robust_ml_ensemble | 20.254 | 0.068 | 73.697 | unique |
| 8 | codex/jizhou-tourism-ml | bayesian_ridge | 20.312 | 0.065 | 73.612 | unique |
| 9 | codex/jizhou-tourism-ml | huber_regime | 21.188 | 0.025 | 75.391 | unique |
| 10 | codex/jizhou-tourism-ml | naive_last | 21.726 | 0.000 | 54.172 | unique |
| 11 | codex/jizhou-tourism-ml | random_forest | 24.536 | -0.129 | 46.087 | unique |
| 12 | codex/jizhou-tourism-ml | spline_ridge | 37.279 | -0.716 | 58.938 | unique |

全候选表用于诊断，不可从中事后挑一个最优候选再宣称无偏胜者。并列通过 1e-10 精度判定；`worst_point_smape_percent` 暴露平均值掩盖的失稳。两种断点模型无论代数上是否可识别，都因用户协议禁止而不执行、不展示误差。

## 未模拟 canonical 分支原生策略敏感性

| metric | branch | model | n_test | smape_percent | smape_skill_vs_naive |
| --- | --- | --- | --- | --- | --- |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 6 | 16.404 | 0.412 |
| tourism_comprehensive_income | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 6 | 20.402 | 0.269 |
| tourism_comprehensive_income | codex/jizhou-tourism-modeling | no_break_log_linear | 6 | 26.819 | 0.038 |
| tourism_comprehensive_income | codex/jizhou-tourism-modeling | pre_covid_exponential | 6 | 26.819 | 0.038 |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | naive_last | 6 | 27.892 | 0.000 |
| tourist_visits | codex/jizhou-tourism-ml | naive_last | 6 | 12.627 | 0.000 |
| tourist_visits | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 6 | 12.919 | -0.023 |
| tourist_visits | codex/jizhou-tourism-modeling | no_break_log_linear | 6 | 15.600 | -0.235 |
| tourist_visits | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 6 | 15.600 | -0.235 |
| tourist_visits | codex/jizhou-tourism-modeling | pre_covid_exponential | 6 | 15.600 | -0.235 |

`pre_covid_exponential` 与原生 `no_break_log_linear` 会删除部分统一训练行，故只能用来解释“分支原生过滤策略会怎样”，不能与共同行适配器混作纯算法比较。

## 2024 最终单年 pseudo-holdout

| metric | branch | model | n_test | mae | mape_percent | smape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | ml_inner_selector | 1 | 2.044 | 0.925 | 0.920 |
| tourism_comprehensive_income | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 1 | 3.636 | 1.645 | 1.632 |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 1 | 15.495 | 7.011 | 7.266 |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | naive_last | 1 | 29.500 | 13.348 | 14.303 |
| tourist_visits | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 1 | 59.302 | 2.244 | 2.269 |
| tourist_visits | codex/jizhou-tourism-ml | ml_inner_selector | 1 | 65.341 | 2.472 | 2.442 |
| tourist_visits | codex/jizhou-tourism-ml | naive_last | 1 | 280.000 | 10.594 | 11.187 |
| tourist_visits | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 1 | 878.320 | 33.232 | 28.497 |

2024 pseudo-holdout 每个目标只有一个点，`MAE = absolute error`，MAPE/sMAPE 都没有稳定性含义；它只检查本次重跑中冻结后的方法能否跨到下一年。两种断点模型仍不执行。任何 2024 误差都没有回流到本次重跑的超参数、候选选择或主排序；但由于模型代码和方法讨论形成时 2024 已存在，它不能支持真正的前瞻泛化声明。

## 2019 截断跨阶段压力测试（不含 2024）

| metric | branch | model | n_test | test_years | mape_percent | smape_percent | smape_skill_vs_naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 2 | 2021;2023 | 50.175 | 36.546 | 0.131 |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | naive_last | 2 | 2021;2023 | 54.736 | 42.044 | 0.000 |
| tourism_comprehensive_income | codex/jizhou-tourism-ml | ml_inner_selector | 2 | 2021;2023 | 75.598 | 53.441 | -0.271 |
| tourism_comprehensive_income | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 2 | 2021;2023 | 92.589 | 62.217 | -0.480 |
| tourist_visits | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 1 | 2023 | 56.310 | 43.939 | 0.082 |
| tourist_visits | codex/jizhou-tourism-ml | naive_last | 1 | 2023 | 62.950 | 47.880 | 0.000 |
| tourist_visits | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 1 | 2023 | 96.167 | 64.941 | -0.356 |
| tourist_visits | codex/jizhou-tourism-ml | ml_inner_selector | 1 | 2023 | 116.698 | 73.697 | -0.539 |

压力测试的 physical training 只到 2019，测试为 2020—2023 中现有官方实际值：综合收入 2021/2023、游客量 2023；每个测试点单独生成截至 `test_year-1` 的模拟尾部，2021 实际不会流入 2023 压力测试训练。ML 的 pandemic/post-2022 特征在 physical training 中未见，输出属于跨阶段外推，不等于识别疫情或恢复效应。

## 模型源码固定

| branch | path | pinned_commit | git_blob_oid | sha256 | validated |
| --- | --- | --- | --- | --- | --- |
| codex/jizhou-tourism-modeling | code/scripts/model_jizhou_tourism.py | 66e27eb5a29bbf3abd51dc2dc1af4b8e41fc349c | 86ed05d5ccb2086532293d81ba02640aa35dd3de | 8622be9dee45afdb10458eaa38501053fd3035913e77ab78bb67a2d51fe2cdc2 | True |
| codex/jizhou-tourism-ml | code/scripts/model_jizhou_tourism_ml.py | 3709084fc84614223ee00979494aa82b458296fe | 81ba267b066c0d60a446e862ae100be0767ce6e9 | 688195945f5e35ec585962484fb30d43884f2a32d008cf4c08b6e354e98c7355 | True |

runner 在建模前校验两个可执行模型文件的 Git blob 与 SHA-256；任一字节漂移都会拒绝运行。原声明摘要也做字段校验。

## 分支覆盖与未纳入原因

- `codex/jizhou-tourism-modeling`：存在可执行 Python 实现；用户轨只跑无断点共同行 OLS，原生 pre-COVID/no-break 仅作敏感性，所有断点均不执行。
- `codex/jizhou-tourism-ml`：存在可执行 Python 实现，实跑全部 7 个候选族、稳健集成、raw-target Ridge、naive 和训练内选择器。
- `origin/111`：只有示例论文和工作簿，**缺少可执行模型源代码**；其中数据只能作统一数据层的旁证，不能把论文数值冒充为同一 split 下的重跑结果。
- `origin/邱志烨-数据搜索`：有数据预处理、补值和模型建议，**没有模型实现**；补值 sidecar 明确排除在标签、外层训练和测试之外。
- `main`：提供 canonical 目标真值并接收最终报告；没有另一套可独立执行的分支模型。

## 复现与限制

输入模式：`observations=data/unified/benchmark_observations.csv;primary=physical_unified_files;rolling=physical_unified_file;stress=physical_unified_files`。loader 会逐行验证 physical split/fold 与 `benchmark_observations.csv` 的目标值、状态、来源、观测标志、年份和 role 完全一致。`train_only_hyperparameters.csv` 可核对调参上限；`model_applicability.csv` 保留禁用断点与 ML 支持状态；逐点预测可复算所有指标。

```bash
.venv/bin/python scripts/build_unified_branch_data.py
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg .venv/bin/python code/scripts/compare_branch_models.py
.venv/bin/python -m unittest discover -s tests -v
```

样本总量很小、测试年份不规则，2020—2022 又存在目标缺口；canonical 还是包含同比反推、回列修订的 final-vintage 数据。模拟伪标签会低估结构冲击，不增加信息量；sMAPE 和时间外推也无法替代结构解释。应联合查看未模拟 canonical 共同行轨、模拟轨、逐目标 naive skill、最坏误差、2019 stress 和 2024 单点，不能引用“稳健冠军”或“原分支赢家”结论。
