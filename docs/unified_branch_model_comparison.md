# C题：蓟州区旅游经济趋势预测与对策分析——统一分支模型比较报告

## 题目要求覆盖矩阵

| 题目 | 题面任务 | 直接产物 | 解释边界 |
| --- | --- | --- | --- |
| 问题1 | 整理四项指标并建立简单增长模型 | problem1_indicator_summary.csv；problem1_simple_growth_parameters.csv；problem1_simple_growth_diagnostics.csv；problem1_simple_growth_forecasts_2026_2030.csv；q1_required_indicators.png | pre_covid_exponential 只描述2010—2019疫情前可用值 |
| 问题2 | 比较模型、预测2026—2030并评价合理性 | problem2_forecasts_2026_2030.csv；problem2_final_model_diagnostics.csv；两张Q2图 | 2025目标proxy不进训练；所有区间均为模型条件区间 |
| 问题3 | 三情景预测、因素敏感性和政策建议 | problem3_scenario_forecasts_2026_2030.csv；problem3_policy_sensitivity.csv；q3_scenarios_sensitivity.png | 透明会计情景，不是历史因果识别 |
| 资料说明 | 论文注明所有数据的实际获取日期 | problem_source_access_dates.csv 核验 canonical 引用的 23 个唯一来源 | 这23项均记录 accessed=2026-08-17；不推广到 sources.csv 全部条目 |

本报告把题面要求与计算产物逐项对齐。sMAPE、naive skill、滚动验证和 pseudo-holdout 是为回答“模型是否适用、合理、哪个更好”而自行选择的审计工具，**不是题面直接指定的评价指标**。

canonical annual summary 实际引用的 23 个唯一 `source_id` 均能在 `data/metadata/sources.csv` 中定位，且 `notes` 记录的实际获取日期均为 2026-08-17；逐项核对见 `problem_source_access_dates.csv`。这个结论只覆盖 canonical 真正引用的来源，不声称 `sources.csv` 全部条目都有日期。

## 问题1：指标整理与简单增长模型

四项题目指标统一到 2010—2025 年历；“覆盖数”只统计 canonical 非空值，不把 benchmark 模拟点冒充事实。游客量在 2020—2022 和 2025 缺失，综合收入在 2016、2020、2022、2025 缺失；宏观指标虽覆盖 16/16，但 2019 存在资料口径边界，不能把整段机械解释为同口径因果趋势。GDP 和第三产业增加值只用于历史背景与预测合理性核对，不参与跨单位模型评分；旅游综合收入是收入总量口径，不是增加值，不能把“综合收入/GDP”解释为旅游增加值贡献率。

清洗规则是：每个指标每年只保留 canonical preferred 值，强制转为数值并保留原始 `status/source_ids`；游客量统一为 `10k_persons`（表中解释为万人次），收入、GDP、第三产业增加值统一为 `100m_cny`（亿元）。缺失保持缺失，不用 0 代替。双侧 log 插值只在 Q2 的模型训练契约中生成，均标记 `is_simulated=true`，不是 Q1 指标事实或官方观测。

| indicator_label_cn | unit | nonmissing_count | missing_years | first_nonmissing_year | first_nonmissing_value | last_nonmissing_year | last_nonmissing_value | cagr_2010_2019_percent | cagr_2010_2018_percent | cagr_2019_2025_percent | growth_2023_2024_percent | status_boundary_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 旅游接待人次 | 10k_persons | 12 | 2020;2021;2022;2025 | 2010 | 848.000 | 2024 | 2643.000 | 14.193 | 15.361 | — | 11.849 | 2020-2022 and 2025 are missing |
| 旅游综合收入 | 100m_cny | 12 | 2016;2020;2022;2025 | 2010 | 35.521 | 2024 | 221.000 | 18.607 | 18.912 | — | 15.405 | 2010 is inferred_from_yoy; 2016,2020,2022,2025 are missing |
| 地区生产总值 | 100m_cny | 16 |  | 2010 | 215.470 | 2025 | 311.940 | — | 7.408 | 6.023 | 3.662 | 2019 macro series has a documented scope break; 2025 is official_initial |
| 第三产业增加值 | 100m_cny | 16 |  | 2010 | 136.230 | 2025 | 195.170 | — | 7.014 | 6.087 | 4.424 | 2019 macro series has a documented scope break; 2025 is official_initial |

宏观指标的 `cagr_2010_2019_percent` 因跨越口径断点而置空：GDP 的 2010—2018 / 2019—2025 分段 CAGR 为 7.408% / 6.023%，第三产业为 7.014% / 6.087%。2018→2019 的 GDP 与第三产业表观变化分别为 -42.457% 和 -41.571%，应读作资料口径断裂，不是经济活动骤降。

旅游目标的恢复路径也不是简单回到疫情前趋势：游客量 2023 较 2019 仍为 -15.607%，2024 较 2019 为 -5.607%，但 2023→2024 同比增长 11.849%；综合收入 2021 较 2019 为 -33.333%，2023 已较 2019 高 16.061%，2023→2024 再增长 15.405%。这些断点是后续模型不能只延长疫情前指数曲线的直接证据。

题目1的简单增长模型固定为 `pre_covid_exponential`：对 2010—2019 年可用 canonical 值拟合 `log(value)=β0+β1(year-2010)`。参数表给出模型条件标准误、t 检验和 95% t 区间：

| metric | parameter | estimate | standard_error | t_value | p_value | ci95_lower | ci95_upper | df_resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tourist_visits | intercept | 6.801 | 0.029 | 237.455 | 0.000 | 6.735 | 6.867 | 8 |
| tourist_visits | year_index | 0.134 | 0.005 | 25.057 | 0.000 | 0.122 | 0.147 | 8 |
| tourism_comprehensive_income | intercept | 3.636 | 0.030 | 120.226 | 0.000 | 3.564 | 3.707 | 7 |
| tourism_comprehensive_income | year_index | 0.170 | 0.006 | 29.505 | 0.000 | 0.156 | 0.183 | 7 |

| metric | n | annual_growth_rate_percent | r_squared_log | adjusted_r_squared_log | rmse_original_units | mape_percent | aicc_log | loocv_log_rmse | durbin_watson | jarque_bera_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tourist_visits | 10 | 14.389 | 0.987 | 0.986 | 85.890 | 3.829 | -56.946 | 0.060 | 1.783 | 0.713 |
| tourism_comprehensive_income | 9 | 18.487 | 0.992 | 0.991 | 4.814 | 4.078 | -49.679 | 0.062 | 1.127 | 0.651 |

诊断量只描述疫情前小样本拟合；R² 是 log 尺度，RMSE/MAPE 是原尺度，AICc 与 LOOCV 仅供同一目标规格核对，不能证明疫情后的结构稳定性。

若把 Q1 简单模型不加结构修正地机械外推到 2026—2030，结果如下；点值是 log 条件均值指数化后在原尺度上的条件中位数，不是带 lognormal 偏差修正的算术均值，也不作为 Q2 主预测：

| metric | model | year | forecast | mean_ci95_lower | mean_ci95_upper | prediction_interval95_lower | prediction_interval95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tourism_comprehensive_income | pre_covid_exponential | 2026 | 572.504 | 486.050 | 674.337 | 466.889 | 702.012 |
| tourism_comprehensive_income | pre_covid_exponential | 2027 | 678.345 | 568.353 | 809.624 | 547.292 | 840.780 |
| tourism_comprehensive_income | pre_covid_exponential | 2028 | 803.754 | 664.557 | 972.107 | 641.345 | 1007.290 |
| tourism_comprehensive_income | pre_covid_exponential | 2029 | 952.347 | 777.010 | 1167.249 | 751.361 | 1207.095 |
| tourism_comprehensive_income | pre_covid_exponential | 2030 | 1128.411 | 908.459 | 1401.616 | 880.050 | 1446.863 |
| tourist_visits | pre_covid_exponential | 2026 | 7723.692 | 6670.167 | 8943.618 | 6420.785 | 9290.986 |
| tourist_visits | pre_covid_exponential | 2027 | 8835.043 | 7538.678 | 10354.334 | 7273.852 | 10731.313 |
| tourist_visits | pre_covid_exponential | 2028 | 10106.305 | 8519.862 | 11988.152 | 8237.870 | 12398.520 |
| tourist_visits | pre_covid_exponential | 2029 | 11560.487 | 9628.374 | 13880.313 | 9327.324 | 14328.316 |
| tourist_visits | pre_covid_exponential | 2030 | 13223.908 | 10880.769 | 16071.635 | 10558.583 | 16562.047 |

![问题1四项指标](../outputs/unified_model_benchmark/q1_required_indicators.png)

## 问题2：模型评判与 2026—2030 预测

模型形式先由截至 2023 年的统一滚动验证冻结，再把 2024 canonical 目标加入最终系数重拟合。最终训练契约为每目标 2010—2024 共 15 个年度位置：12 个 physical canonical 值和 3 个训练期内双侧 log 插值；**不生成、不读取、不使用 2025 目标值**。其中 2010 年综合收入虽是 canonical physical 行，但状态为 `inferred_from_yoy`，不是 strict observed。

模型评判首先落到两个题面目标的未模拟 canonical expanding-origin 回测（每目标 6 个外层实际测试点）：

| metric | model | n_test | smape_percent | naive_smape_percent | smape_skill_vs_naive | worst_point_smape_percent |
| --- | --- | --- | --- | --- | --- | --- |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 6 | 16.404 | 27.892 | 0.412 | 51.002 |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 6 | 20.402 | 27.892 | 0.269 | 76.106 |
| tourism_comprehensive_income | pre_covid_exponential | 6 | 26.819 | 27.892 | 0.038 | 76.106 |
| tourism_comprehensive_income | naive_last | 6 | 27.892 | 27.892 | 0.000 | 54.063 |
| tourist_visits | naive_last | 6 | 12.627 | 12.627 | 0.000 | 16.928 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 6 | 12.919 | 12.627 | -0.023 | 43.450 |
| tourist_visits | no_break_log_linear_common_rows | 6 | 15.600 | 12.627 | -0.235 | 74.363 |
| tourist_visits | pre_covid_exponential | 6 | 15.600 | 12.627 | -0.235 | 74.363 |

Q1 的 `pre_covid_exponential` 在游客量/综合收入上的未模拟滚动 sMAPE 分别为 15.600% / 26.819%，fixed raw Ridge 则为 12.919% / 16.404%；这直接说明 Q1 疫情前简单模型在疫情后不再适合作为主预测。它机械外推到 2026 年已达游客量 7723.7 万人次、收入 572.5 亿元，到 2030 年达 13223.9 万人次和 1128.4 亿元，远离恢复期实际与政策情景尺度。

在未模拟轨上，固定 raw Ridge 的两目标等权 macro-sMAPE 为 14.661%，无断点 common-row OLS 为 18.001%；在模拟增强轨上二者分别为 14.745% 和 15.634%。但 Ridge 在游客量上的未模拟 sMAPE 12.919% 仍略高于 naive 的 12.627%，因此不能称为全目标稳健赢家。这里并列给出两个冻结模型的未来路径，供趋势型与恢复特征型假设交叉核对。所有选型只比较同一目标原尺度上的外层滚动 sMAPE；下方不同 target/LOOCV 尺度的拟合诊断不能横向替代该选型依据。

两种固定规格分别是无断点 common-row log OLS 和 `raw_target_ridge_alpha_0.1`。OLS 区间直接采用分支源码的 Student-t 公式，是平均 log 响应区间指数化；Ridge 使用固定种子 `20260817` 的 10,000 次固定设计残差 bootstrap。两者都把插值当作给定训练值，都是模型条件区间，不保证在重复抽样意义下达到 95% 覆盖率，也不是五年同时置信带。

| metric | model | year | forecast | mean_ci95_lower | mean_ci95_upper | prediction_interval95_lower | prediction_interval95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 2026 | 280.569 | 216.536 | 363.538 | 169.218 | 465.193 |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 2026 | 241.159 | 226.710 | 253.346 | 217.193 | 262.391 |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 2027 | 314.179 | 236.816 | 416.815 | 187.136 | 527.469 |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 2027 | 255.098 | 239.710 | 267.541 | 230.307 | 276.252 |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 2028 | 351.814 | 258.897 | 478.080 | 206.749 | 598.666 |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 2028 | 269.036 | 252.505 | 281.971 | 243.919 | 290.194 |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 2029 | 393.958 | 282.953 | 548.512 | 228.212 | 680.085 |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 2029 | 282.975 | 265.092 | 296.691 | 256.321 | 304.420 |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 2030 | 441.151 | 309.172 | 629.468 | 251.693 | 773.221 |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 2030 | 296.914 | 277.361 | 311.606 | 269.985 | 318.666 |
| tourist_visits | no_break_log_linear_common_rows | 2026 | 3840.327 | 3024.303 | 4876.532 | 2409.279 | 6121.381 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 2026 | 3039.552 | 2816.695 | 3241.784 | 2663.151 | 3400.422 |
| tourist_visits | no_break_log_linear_common_rows | 2027 | 4151.991 | 3199.308 | 5388.362 | 2574.961 | 6694.871 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 2027 | 3251.479 | 3009.761 | 3459.219 | 2862.306 | 3605.159 |
| tourist_visits | no_break_log_linear_common_rows | 2028 | 4488.949 | 3383.263 | 5955.984 | 2749.567 | 7328.668 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 2028 | 3463.405 | 3198.831 | 3678.913 | 3071.283 | 3815.780 |
| tourist_visits | no_break_log_linear_common_rows | 2029 | 4853.252 | 3576.821 | 6585.193 | 2933.572 | 8029.138 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 2029 | 3675.332 | 3387.801 | 3901.521 | 3255.351 | 4027.887 |
| tourist_visits | no_break_log_linear_common_rows | 2030 | 5247.121 | 3780.634 | 7282.449 | 3127.490 | 8803.314 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 2030 | 3887.259 | 3574.444 | 4128.023 | 3449.140 | 4251.025 |

| metric | model | training_n | simulated_training_n | target_scale | r_squared | rmse | mape_percent | aicc | loocv_rmse | loocv_scale | durbin_watson | jarque_bera_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tourist_visits | no_break_log_linear_common_rows | 15 | 3 | log | 0.792 | 387.132 | 15.268 | -47.715 | 0.204 | log | 0.269 | 0.587 |
| tourist_visits | raw_target_ridge_alpha_0.1 | 15 | 3 | raw | 0.953 | 141.164 | 5.357 | 160.277 | 204.475 | raw | 1.356 | 0.811 |
| tourism_comprehensive_income | no_break_log_linear_common_rows | 15 | 3 | log | 0.872 | 22.423 | 15.904 | -45.281 | 0.214 | log | 0.580 | 0.724 |
| tourism_comprehensive_income | raw_target_ridge_alpha_0.1 | 15 | 3 | raw | 0.971 | 8.905 | 7.403 | 77.379 | 13.580 | raw | 2.131 | 0.956 |

`target_scale/loocv_scale` 明示：OLS 的 R²、AICc 与 LOOCV 在 log 尺度，Ridge 在 raw 尺度，二者及两个不同单位目标之间不可直接比较。OLS 点预测是 log 条件均值指数化后的原尺度条件中位数；没有做 lognormal 均值修正。

Ridge 的标准化参数及 bootstrap 百分位区间如下；没有为 Ridge 系数伪造 t 检验或 p 值：

| metric | parameter | estimate | bootstrap_ci95_lower | bootstrap_ci95_upper | feature_training_mean | feature_training_scale |
| --- | --- | --- | --- | --- | --- | --- |
| tourist_visits | intercept | 2028.737 | 1955.836 | 2099.838 | — | — |
| tourist_visits | year_index | 915.628 | 763.234 | 1013.132 | 0.700 | 0.432 |
| tourist_visits | pandemic_2020_2022 | -227.530 | -310.164 | -105.998 | 0.200 | 0.400 |
| tourist_visits | post_2022 | -396.268 | -484.326 | -257.440 | 0.133 | 0.340 |
| tourism_comprehensive_income | intercept | 114.642 | 109.992 | 119.074 | — | — |
| tourism_comprehensive_income | year_index | 60.222 | 50.683 | 66.358 | 0.700 | 0.432 |
| tourism_comprehensive_income | pandemic_2020_2022 | -20.801 | -26.227 | -13.228 | 0.200 | 0.400 |
| tourism_comprehensive_income | post_2022 | -3.660 | -9.728 | 4.855 | 0.133 | 0.340 |

Ridge 特征固定为 `year_index=(year-2010)/10`、`pandemic_2020_2022=1[2020≤year≤2022]`、`post_2022=1[year≥2023]`，再按最终训练样本标准化。系数只用于预测，不是因果效应；最终 2020—2022 目标是训练期双侧 log 插值，因此疫情 dummy 尤其受模拟路径驱动，不能解释为已识别的疫情冲击。

![问题2模型评判](../outputs/unified_model_benchmark/q2_model_judgement.png)

![问题2预测](../outputs/unified_model_benchmark/q2_forecast_2026_2030.png)

**题目导向的选模结论：** 两条统一滚动轨上，fixed raw Ridge 的 macro-sMAPE 都低于无断点 common-row OLS，因此把 `raw_target_ridge_alpha_0.1` 作为 2026—2030 的主点预测：游客量从 3039.6 万人次增至 3887.3 万人次，综合收入从 241.16 亿元增至 296.91 亿元；OLS 族保留为 Q1 简单增长模型和 Q2 的解释性趋势对照。到 2030 年，OLS 给出 5247.1 万人次和 441.15 亿元，明显高于 Ridge，原因是 OLS 把全期平均 log 增长持续外推，而 Ridge 在 raw 目标上同时估计时间趋势、疫情和 2023 年后恢复水平，外推更平缓。两条路径都为正且随时间增长，但 OLS 区间更宽、对长期指数趋势更敏感；再考虑 Ridge 的游客量未模拟回测没有胜过 naive、样本很小，主预测只是相对更审慎的固定规格，不能称为稳健胜者。

合理性衔接上，Ridge 2030 年游客量 3887.3 万人次和收入 296.91 亿元均落在 Q3 三情景包络（游客量 2620.2—4055.8 万人次；收入 238.66—407.10 亿元）内，但呈现“客流高于政策基准、收入低于政策基准”的组合。其隐含名义人均次消费由 2026 年 793.4 元降至 2030 年 763.8 元，相比 2024 canonical 的 836.2 元下降 8.65%；因此把 Ridge 作为偏保守风险主线时，应同步监测客单价，而不能只看客流。

## 问题3：政策锚定三情景与敏感性

三情景沿用已合并分支的透明政策锚定口径：2025 年游客量 2800 万人次、综合收入 231 亿元均是政府目标 proxy，**不是实际观测**；对应名义人均次消费为 825 元/人次。基准情景取收入年增 8%、人均消费年增 3%；乐观情景取 12%/4%；悲观情景先在 2026 年施加收入水平冲击 −15%，随后收入年增 5%、人均消费年增 2%。所有游客量均由 `收入=游客量×人均次消费/10000` 恒等式反推。

| scenario_label_cn | scenario | year | tourist_visits | tourism_comprehensive_income | nominal_spend_per_visit |
| --- | --- | --- | --- | --- | --- |
| 乐观情景 | optimistic_assumption | 2026 | 3015.385 | 258.720 | 858.000 |
| 乐观情景 | optimistic_assumption | 2027 | 3247.337 | 289.766 | 892.320 |
| 乐观情景 | optimistic_assumption | 2028 | 3497.132 | 324.538 | 928.013 |
| 乐观情景 | optimistic_assumption | 2029 | 3766.143 | 363.483 | 965.133 |
| 乐观情景 | optimistic_assumption | 2030 | 4055.846 | 407.101 | 1003.739 |
| 基准情景 | baseline_policy_anchor | 2026 | 2935.922 | 249.480 | 849.750 |
| 基准情景 | baseline_policy_anchor | 2027 | 3078.443 | 269.438 | 875.242 |
| 基准情景 | baseline_policy_anchor | 2028 | 3227.882 | 290.993 | 901.500 |
| 基准情景 | baseline_policy_anchor | 2029 | 3384.575 | 314.273 | 928.545 |
| 基准情景 | baseline_policy_anchor | 2030 | 3548.875 | 339.415 | 956.401 |
| 悲观情景 | pessimistic_assumption | 2026 | 2333.333 | 196.350 | 841.500 |
| 悲观情景 | pessimistic_assumption | 2027 | 2401.961 | 206.167 | 858.330 |
| 悲观情景 | pessimistic_assumption | 2028 | 2472.607 | 216.476 | 875.497 |
| 悲观情景 | pessimistic_assumption | 2029 | 2545.330 | 227.300 | 893.007 |
| 悲观情景 | pessimistic_assumption | 2030 | 2620.193 | 238.665 | 910.867 |

OAT 敏感性只围绕基准情景，每次只改变一项假设：客源年增速 ±2 个百分点、人均消费年增速 ±1 个百分点、2026 政策协同水平乘数 ±5%，以及突发冲击从 0 到 −15%。水平乘数和冲击在该恒等式中都表现为持续水平位移，不能解释成两个独立可加机制。

各因素的扰动宽度和单位不同（±2 个百分点、±1 个百分点、±5%、0 至 −15%），所以图中的影响幅度不是标准化弹性，不能据此作严格“杠杆效率”排序；这里只展示在指定假设幅度下的非因果压力结果。

| factor | setting | metric | baseline_2030 | scenario_2030 | delta_2030 | delta_percent |
| --- | --- | --- | --- | --- | --- | --- |
| external_shock | high | tourism_comprehensive_income | 339.415 | 339.415 | 0.000 | 0.000 |
| external_shock | high | tourist_visits | 3548.875 | 3548.875 | 0.000 | 0.000 |
| external_shock | low | tourism_comprehensive_income | 339.415 | 288.503 | -50.912 | -15.000 |
| external_shock | low | tourist_visits | 3548.875 | 3016.544 | -532.331 | -15.000 |
| new_format_spend_growth | high | tourism_comprehensive_income | 339.415 | 356.214 | 16.799 | 4.950 |
| new_format_spend_growth | high | tourist_visits | 3548.875 | 3548.875 | 0.000 | 0.000 |
| new_format_spend_growth | low | tourism_comprehensive_income | 339.415 | 323.255 | -16.160 | -4.761 |
| new_format_spend_growth | low | tourist_visits | 3548.875 | 3548.875 | 0.000 | 0.000 |
| policy_coordination_multiplier | high | tourism_comprehensive_income | 339.415 | 356.386 | 16.971 | 5.000 |
| policy_coordination_multiplier | high | tourist_visits | 3548.875 | 3726.319 | 177.444 | 5.000 |
| policy_coordination_multiplier | low | tourism_comprehensive_income | 339.415 | 322.444 | -16.971 | -5.000 |
| policy_coordination_multiplier | low | tourist_visits | 3548.875 | 3371.431 | -177.444 | -5.000 |
| source_market_growth | high | tourism_comprehensive_income | 339.415 | 373.044 | 33.629 | 9.908 |
| source_market_growth | high | tourist_visits | 3548.875 | 3900.493 | 351.618 | 9.908 |
| source_market_growth | low | tourism_comprehensive_income | 339.415 | 308.256 | -31.159 | -9.180 |
| source_market_growth | low | tourist_visits | 3548.875 | 3223.085 | -325.790 | -9.180 |

量化建议（均为情景计算，不是历史因果效应）：

- **客源拓展 KPI**：把隐含客源年增速从 4.854% 提高到 6.854%；情景计算的 2030 游客量为 3900.5 万人次，较基准增加 351.6 万人次，收入增加 33.63 亿元。该增量是 OAT 假设结果，不证明投放会因果地产生同等增量。
- **新业态消费 KPI**：把名义人均次消费年增速从 3% 提高到 4%；在客源路径不变时，2030 收入为 356.21 亿元，较基准增加 16.80 亿元。该计算未识别业态投资的历史弹性或成本。
- **政策协同 KPI**：以 2026 路径水平乘数 1.05 作压力目标；2030 游客量和收入分别较基准增加 177.4 万人次、16.97 亿元。乘数是外生假设，不能当作政策因果系数。
- **风险预案 KPI**：监测预订量、客单价和交通可达性，使 2026 冲击幅度尽量高于 −15%；若形成持续 −15% 水平缺口，2030 游客量和收入将比基准少 532.3 万人次、50.91 亿元。损失仅为压力测试，不是风险概率预测。
- **年度校准 KPI**：以基准 2030 的 3548.9 万人次和 339.41 亿元作为可滚动修订的假设锚，而非硬承诺；每年用新实际更新偏差。锚值来自 2025 proxy 和设定增速，不具有因果或概率保证。

![问题3情景与敏感性](../outputs/unified_model_benchmark/q3_scenarios_sensitivity.png)


## 附录：统一分支模型回测与审计

### 结论先行

**不能判定原分支赢家。** 原传统分支声明的主模型是 `post_2022_level_break`，而用户本轮统一协议明确不运行任何断点模型；原 ML 分支声明的是 `ridge_regime` 模型族和 `raw_target_ridge_alpha_0.1` 点路径。因此原声明代表没有一组可共同排名的滚动预测。

在用户指定的模拟增强适配器轨上，描述性第一是 `raw_target_ridge_alpha_0.1`（macro-sMAPE 14.745%）；在更适合作为稳健性依据的未模拟 canonical 共同行轨（official-source）上，描述性第一是 `raw_target_ridge_alpha_0.1`（14.661%）。两轨的适配器相对顺序一致，但这仍不能替代原分支声明模型的共同排名。 这些表只比较“共同训练行上的固定规格适配器”，不是原分支胜负。

模拟增强轨是假设性分析，不是真实观测证据：log 插值和 log 增长尾推在结构上更贴近 log-linear OLS，而且会把游客量 2020—2022 的未知疫情路径平滑成趋势点。报告因此优先用未模拟共同行轨判断稳健性，并把两轨并列展示。

滚动稳定性框架采用截至 2023 年的 expanding-origin 外层验证；2024 年仅作最终单年 pseudo-holdout（每个目标 n=1），不用于本次执行的排序。模型族代码是在 2024 数据已经存在后形成，统一数据又是 final-vintage 回溯版，因此 2024 不是研究设计层真正“未见”的前瞻测试。2019 截断结果仅作跨疫情/恢复阶段压力测试，不与 pseudo-holdout 混称。

### 分支数据合并结论

- 共审计并合入 5 个唯一业务 tip。`main`、传统模型分支和 ML 分支的核心 `data/` 树完全相同，因此没有重复拼接同一批年度观测。
- `origin/111` 的独有天津市 GDP（2010—2025）与天津市旅游基准已规范化为独立辅助表；天津市口径不会覆盖蓟州区 GDP，也不直接充当蓟州目标标签。
- `origin/邱志烨-数据搜索` 的 8 条补全值保存在 `sensitivity_imputations.csv`，全部标记为非观测并排除在统一训练、测试和排名之外；其全样本标准化、异常检测等派生列也没有直接进入回测。
- 两个分支中的旧版工作簿、预测和情景交付均保留在 Git 历史/原路径，但不会覆盖集成前 `main` 固定提交的 canonical 真值。逐文件来源、blob、SHA-256 和纳入决策见 `data/unified/branch_data_inventory.csv`。

### 可比协议

- 随机种子固定为 `20260817`；所有模型读取同一统一数据层。
- 滚动外层最小训练记录数为 5，游客量测试年为 `[2015, 2016, 2017, 2018, 2019, 2023]`，综合收入测试年为 `[2015, 2017, 2018, 2019, 2021, 2023]`；外层测试最晚到 2023 年。
- 每折模拟只读取该折测试年前已存在的 physical training rows：内部缺口用两侧训练边界在 log 尺度插值；训练尾部到 `test_year-1` 用最近至多 3 个训练区间的 annualized log-growth 中位数外推。逐点方法、源年、边界见 `simulated_training_points.csv`；绝不读取外层测试、未来官方值或邱分支 sidecar。
- 固定主适配器是传统 `no_break_log_linear_common_rows` 与 ML `raw_target_ridge_alpha_0.1`。两者逐折使用完全相同的 physical/simulated/effective 行；脚本运行时强制验证这一不变量。
- ML 全候选及新建 `ml_inner_selector` 只作探索；其预处理、调参和选择均限定在每个外层折训练内部，但不能据其事后名次宣布分支赢家。
- 2024 pseudo-holdout 训练集每目标记录数为 `{'tourism_comprehensive_income': 11, 'tourist_visits': 11}`，测试集严格为游客量和综合收入各一条 2024 实际值；隔离只成立于本次重跑的执行流程。
- `data/unified/primary_train.csv` 是 2024 前的 physical 证据层；`outputs/unified_model_benchmark/primary_train_augmented.csv` 是用户指定的 2010—2023 建模训练层，包含 physical 与 simulated 行并保留 `is_simulated/method/known_through_year`，可直接与 `data/unified/primary_test.csv` 配对。
- 主排序指标是先按目标计算 sMAPE，再对两个目标 50/50 等权；不跨单位汇总 RMSE。表中同时给出相对 `naive_last` 的 skill 与最坏点误差。
- 传统断点模型 `post_2022_level_break`、`strict_evidence_level_break` 在所有 scope 均为 `not_executed_user_protocol`，不生成预测或误差。原生 `pre_covid_exponential` 和会排除 2020—2022 的 `no_break_log_linear` 仅保留在未模拟 canonical 敏感性表。

### 用户指定的模拟增强固定适配器排序（假设性）

| descriptive_rank | branch | model | macro_smape_percent | macro_smape_skill_vs_naive | beats_naive_all_targets | worst_target | worst_target_smape_percent | worst_point_smape_percent | tie_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 14.745 | 0.321 | True | tourism_comprehensive_income | 16.489 | 60.689 | unique |
| 2 | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 15.634 | 0.280 | True | tourism_comprehensive_income | 17.240 | 74.375 | unique |

该表的 `descriptive_rank` 只描述用户指定规格在模拟伪标签上的相对误差；`robust_winner` 固定为 false。模拟点会改变目标路径，不能当成新增事实或原分支的历史声明流程。

### 未模拟 canonical 共同行排序（official-source，稳健性优先）

| descriptive_rank | branch | model | macro_smape_percent | macro_smape_skill_vs_naive | beats_naive_all_targets | worst_target | worst_target_smape_percent | worst_point_smape_percent | tie_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | 14.661 | 0.276 | False | tourism_comprehensive_income | 16.404 | 51.002 | unique |
| 2 | codex/jizhou-tourism-modeling | no_break_log_linear_common_rows | 18.001 | 0.111 | False | tourism_comprehensive_income | 20.402 | 76.106 | unique |

这里不生成任何 benchmark 伪标签。传统 common-row OLS 与 raw Ridge 都吃每折全部 physical canonical training rows，测试仍只用官方实际；canonical 可包含官方来源的同比反推、回列或 supporting 值，并以 `status/is_observed` 明示，例如 2010 年综合收入是 `inferred_from_yoy`，不是 strict observed。该轨优先于模拟轨用于判断结论是否由伪标签驱动；即使两轨顺序一致，也只能称适配器相对排序。

### 两轨逐目标核对

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

### 原分支声明方法核对

| branch | model | original_declaration | rolling_execution_status | simulated_track_macro_smape_percent | official_track_macro_smape_percent | jointly_rankable_original_declared_representatives |
| --- | --- | --- | --- | --- | --- | --- |
| codex/jizhou-tourism-modeling | post_2022_level_break | primary_model | not_executed_user_protocol | — | — | False |
| codex/jizhou-tourism-ml | raw_target_ridge_alpha_0.1 | recommended_point_forecast | fixed_adapter_executed_but_historical_selection_saw_2024 | 14.745 | 14.661 | False |
| codex/jizhou-tourism-ml | ridge_regime | selected_ml_models for both targets | exploratory_candidate_retrained_with_inner_tuning | 18.083 | — | False |

传统原声明模型没有执行值，`jointly_rankable_original_declared_representatives=false`。`raw_target_ridge_alpha_0.1` 本次每折固定 alpha 且不读取外层测试，但原 ML 分支推荐它时已检查含 2024 的回测；`ridge_regime` 仅作模型族声明审计。因此没有合法的原分支级胜者。

### 全候选探索性排名

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

### 未模拟 canonical 分支原生策略敏感性

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

### 2024 最终单年 pseudo-holdout

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

### 2019 截断跨阶段压力测试（不含 2024）

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

### 模型源码固定

| branch | path | pinned_commit | git_blob_oid | sha256 | validated |
| --- | --- | --- | --- | --- | --- |
| codex/jizhou-tourism-modeling | code/scripts/model_jizhou_tourism.py | 66e27eb5a29bbf3abd51dc2dc1af4b8e41fc349c | 86ed05d5ccb2086532293d81ba02640aa35dd3de | 8622be9dee45afdb10458eaa38501053fd3035913e77ab78bb67a2d51fe2cdc2 | True |
| codex/jizhou-tourism-ml | code/scripts/model_jizhou_tourism_ml.py | 3709084fc84614223ee00979494aa82b458296fe | 81ba267b066c0d60a446e862ae100be0767ce6e9 | 688195945f5e35ec585962484fb30d43884f2a32d008cf4c08b6e354e98c7355 | True |

runner 在建模前校验两个可执行模型文件的 Git blob 与 SHA-256；任一字节漂移都会拒绝运行。原声明摘要也做字段校验。

### 分支覆盖与未纳入原因

- `codex/jizhou-tourism-modeling`：存在可执行 Python 实现；用户轨只跑无断点共同行 OLS，原生 pre-COVID/no-break 仅作敏感性，所有断点均不执行。
- `codex/jizhou-tourism-ml`：存在可执行 Python 实现，实跑全部 7 个候选族、稳健集成、raw-target Ridge、naive 和训练内选择器。
- `origin/111`：只有示例论文和工作簿，**缺少可执行模型源代码**；其中数据只能作统一数据层的旁证，不能把论文数值冒充为同一 split 下的重跑结果。
- `origin/邱志烨-数据搜索`：有数据预处理、补值和模型建议，**没有模型实现**；补值 sidecar 明确排除在标签、外层训练和测试之外。
- `main`：提供 canonical 目标真值并接收最终报告；没有另一套可独立执行的分支模型。

### 复现与限制

输入模式：`observations=data/unified/benchmark_observations.csv;primary=physical_unified_files;rolling=physical_unified_file;stress=physical_unified_files`。loader 会逐行验证 physical split/fold 与 `benchmark_observations.csv` 的目标值、状态、来源、观测标志、年份和 role 完全一致。`train_only_hyperparameters.csv` 可核对调参上限；`model_applicability.csv` 保留禁用断点与 ML 支持状态；逐点预测可复算所有指标。

获取日期核对严格限定在 canonical annual summary 实际引用的 23 个唯一来源：它们在 `data/metadata/sources.csv` 的 `notes` 中都记录 `accessed=2026-08-17`，详见 `problem_source_access_dates.csv`。未被 canonical 使用的来源元数据不在这项完整性声明内。

```bash
.venv/bin/python scripts/build_unified_branch_data.py
MPLCONFIGDIR=/tmp/jizhou-mpl XDG_CACHE_HOME=/tmp/jizhou-xdg .venv/bin/python code/scripts/compare_branch_models.py
.venv/bin/python -m unittest discover -s tests -v
```

样本总量很小、测试年份不规则，2020—2022 又存在目标缺口；canonical 还是包含同比反推、回列修订的 final-vintage 数据。模拟伪标签会低估结构冲击，不增加信息量；sMAPE 和时间外推也无法替代结构解释。应联合查看未模拟 canonical 共同行轨、模拟轨、逐目标 naive skill、最坏误差、2019 stress 和 2024 单点，不能引用“稳健冠军”或“原分支赢家”结论。

## 题目导向综合结论

- **问题1：增长背景。** 疫情前 `pre_covid_exponential` 估计游客量年增长 14.389%、综合收入年增长 18.487%；2023—2024 canonical 实际增长分别为 11.849% 和 15.405%。恢复仍为正但结构已变，Q1 曲线机械外推到 2030 年会达到 13223.9 万人次/1128.4 亿元，故只能作反例对照；GDP/第三产业按 2019 口径断点分段核对，综合收入不是增加值。
- **问题2：主预测与不确定性。** 两条统一滚动轨均由 fixed raw Ridge 取得较低 macro-sMAPE，故主点预测采用该规格：游客量 3039.6→3887.3 万人次，综合收入 241.16→296.91 亿元；无断点 OLS 作为更陡的趋势对照。Ridge 游客量未模拟回测仍略逊 naive，且其 2030 隐含人均次消费较 2024 低 8.65%，所以结论是“带客单价下行风险的相对审慎主线”，不是稳健冠军。
- **问题3：情景与行动。** 政策基准到 2030 年为 3548.9 万人次和 339.41 亿元；在指定扰动下，客源年增速提高 2 个百分点对应游客量 +351.6 万人次、收入 +33.63 亿元，同时应为 −15% 持续冲击下的 50.91 亿元收入缺口准备预案。各 OAT 扰动宽度不同，不能当作标准化杠杆排名；全部增量来自 2025 目标 proxy 与透明假设，不代表政策因果效应或实现概率。
