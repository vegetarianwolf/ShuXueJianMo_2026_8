from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


BASE = Path(r"D:\junk mass\小乱七八糟\数学建模\26国赛\shared files\data\jizhou_tourism_economy")
OUT = Path(r"D:\junk mass\小乱七八糟\数学建模\26国赛\shared files\outputs\01a00ea1-ee9b-7fe3-a9c3-5934549ee5ec")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE / name, encoding="utf-8-sig")


def pivot_metric(frame: pd.DataFrame, *, scope: str | None = None) -> pd.DataFrame:
    if scope is not None:
        frame = frame.loc[frame["metric_scope"] == scope].copy()
    return frame.pivot_table(index="year", columns="metric", values="value", aggfunc="first")


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    return (series - series.min()) / span if span else pd.Series(0.0, index=series.index)


def anomaly_rows(panel: pd.DataFrame, columns: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    thresholds: dict[str, dict] = {}
    for column, label in columns.items():
        growth = np.log(panel[column]).diff()
        valid = growth.dropna()
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        z = zscore(valid)
        thresholds[column] = {
            "label": label,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(lower),
            "upper": float(upper),
        }
        for year, value in valid.items():
            iqr_flag = bool(value < lower or value > upper)
            z_flag = bool(abs(z.loc[year]) > 3)
            rows.append(
                {
                    "year": int(year),
                    "metric": column,
                    "metric_cn": label,
                    "log_yoy": float(value),
                    "yoy_percent": float(math.expm1(value) * 100),
                    "z_score": float(z.loc[year]),
                    "iqr_lower": float(lower),
                    "iqr_upper": float(upper),
                    "iqr_flag": iqr_flag,
                    "z_flag": z_flag,
                    "final_flag": iqr_flag or z_flag,
                    "treatment": "保留并设置结构/口径哑变量" if (iqr_flag or z_flag) else "保留",
                }
            )
    return pd.DataFrame(rows), thresholds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    annual = read_csv("official_annual_summary_2010_2025.csv")
    gaps = read_csv("supplemental_gap_evidence_2016_2025.csv")
    supply_long = read_csv("official_tourism_supply_observations_2012_2024.csv")
    related_long = read_csv("official_related_observations_2014_2025.csv")
    supply = pivot_metric(
        supply_long,
        scope="above_designated_size_accommodation_and_catering",
    )
    related = pivot_metric(related_long)

    panel = annual[[
        "year",
        "preferred_visitor_10k_persons",
        "visitor_status",
        "preferred_comprehensive_income_100m_cny",
        "comprehensive_status",
        "preferred_gdp_100m_cny",
        "gdp_status",
        "preferred_tertiary_100m_cny",
        "tertiary_status",
        "source_ids",
        "quality_note",
    ]].copy()
    panel = panel.set_index("year").join(supply[["guest_rooms", "beds", "room_revenue"]]).join(related)

    panel["visitor_model_10k_persons"] = panel["preferred_visitor_10k_persons"]
    panel["income_model_100m_cny"] = panel["preferred_comprehensive_income_100m_cny"]
    panel["visitor_data_role"] = np.where(panel["visitor_model_10k_persons"].notna(), "official_preferred", "missing")
    panel["income_data_role"] = np.where(panel["income_model_100m_cny"].notna(), "official_preferred", "missing")
    panel["visitor_imputation_method"] = ""
    panel["income_imputation_method"] = ""
    panel["visitor_low"] = panel["visitor_model_10k_persons"]
    panel["visitor_high"] = panel["visitor_model_10k_persons"]
    panel["income_low"] = panel["income_model_100m_cny"]
    panel["income_high"] = panel["income_model_100m_cny"]

    # 2016: transparent algebraic back-calculation from 2017 value and official growth rate.
    panel.loc[2016, "income_model_100m_cny"] = 108.7866
    panel.loc[2016, "income_data_role"] = "algebraic_derived_diagnostic"
    panel.loc[2016, "income_imputation_method"] = "130/1.195（取整值反推，仅诊断）"
    panel.loc[2016, ["income_low", "income_high"]] = [104.0, 113.8]

    # 2025 visitor: secondary reported actual, sensitivity only.
    panel.loc[2025, "visitor_model_10k_persons"] = 2691.0
    panel.loc[2025, "visitor_data_role"] = "secondary_reported_sensitivity"
    panel.loc[2025, "visitor_imputation_method"] = "二手报道年度值，仅用于敏感性/锚点"
    panel.loc[2025, ["visitor_low", "visitor_high"]] = [2600.0, 2800.0]

    # 2021-2022 visitor total is constrained to 2803. Allocate with an equal-weight
    # composite of same-scope room-revenue share and comprehensive-income proxy share.
    room_share_2021 = supply.loc[2021, "room_revenue"] / supply.loc[[2021, 2022], "room_revenue"].sum()
    income_share_2021 = 110.0 / (110.0 + 55.7491)
    composite_share_2021 = (room_share_2021 + income_share_2021) / 2
    visitors_2021 = 2803.0 * composite_share_2021
    visitors_2022 = 2803.0 - visitors_2021
    panel.loc[2021, "visitor_model_10k_persons"] = visitors_2021
    panel.loc[2022, "visitor_model_10k_persons"] = visitors_2022
    panel.loc[[2021, 2022], "visitor_data_role"] = "aggregate_constrained_imputation"
    panel.loc[[2021, 2022], "visitor_imputation_method"] = "2021—2022合计2803；客房收入份额与综合收入代理份额等权分配"
    room_allocation_2021 = 2803.0 * room_share_2021
    income_allocation_2021 = 2803.0 * income_share_2021
    panel.loc[2021, ["visitor_low", "visitor_high"]] = [min(room_allocation_2021, income_allocation_2021), max(room_allocation_2021, income_allocation_2021)]
    panel.loc[2022, ["visitor_low", "visitor_high"]] = [2803.0 - max(room_allocation_2021, income_allocation_2021), 2803.0 - min(room_allocation_2021, income_allocation_2021)]

    # 2022 and 2025 income: constraint-consistent scenario, not observations.
    panel.loc[2022, "income_model_100m_cny"] = 55.7491
    panel.loc[2022, "income_data_role"] = "target_implied_diagnostic"
    panel.loc[2022, "income_imputation_method"] = "160/(1+187%)（2023目标与目标增速反推）"
    panel.loc[2022, ["income_low", "income_high"]] = [50.0, 65.0]
    panel.loc[2025, "income_model_100m_cny"] = 214.2509
    panel.loc[2025, "income_data_role"] = "aggregate_constraint_scenario"
    panel.loc[2025, "income_imputation_method"] = "五年累计余量270-2022情景值55.7491"
    panel.loc[2025, ["income_low", "income_high"]] = [200.0, 231.0]

    # 2020: non-random shock. A local two-proxy activity index uses the geometric mean
    # of the same-scope room-revenue ratio and local retail-sales ratio (1-19.1%).
    room_ratio_2020 = supply.loc[2020, "room_revenue"] / supply.loc[2019, "room_revenue"]
    retail_ratio_2020 = 1 - 0.191
    activity_ratio_2020 = math.sqrt(room_ratio_2020 * retail_ratio_2020)
    panel.loc[2020, "visitor_model_10k_persons"] = panel.loc[2019, "preferred_visitor_10k_persons"] * activity_ratio_2020
    panel.loc[2020, "income_model_100m_cny"] = panel.loc[2019, "preferred_comprehensive_income_100m_cny"] * activity_ratio_2020
    panel.loc[2020, "visitor_data_role"] = "shock_proxy_scenario"
    panel.loc[2020, "income_data_role"] = "shock_proxy_scenario"
    proxy_note = "2019值×sqrt(2020客房收入比×(1-社零降幅19.1%))"
    panel.loc[2020, "visitor_imputation_method"] = proxy_note
    panel.loc[2020, "income_imputation_method"] = proxy_note
    panel.loc[2020, ["visitor_low", "visitor_high"]] = [
        panel.loc[2019, "preferred_visitor_10k_persons"] * min(room_ratio_2020, retail_ratio_2020),
        panel.loc[2019, "preferred_visitor_10k_persons"] * max(room_ratio_2020, retail_ratio_2020),
    ]
    panel.loc[2020, ["income_low", "income_high"]] = [
        panel.loc[2019, "preferred_comprehensive_income_100m_cny"] * min(room_ratio_2020, retail_ratio_2020),
        panel.loc[2019, "preferred_comprehensive_income_100m_cny"] * max(room_ratio_2020, retail_ratio_2020),
    ]

    panel["per_visit_spend_yuan"] = panel["income_model_100m_cny"] * 10000 / panel["visitor_model_10k_persons"]
    panel["pandemic_dummy"] = panel.index.to_series().between(2020, 2022).astype(int)
    panel["recovery_dummy"] = (panel.index.to_series() >= 2023).astype(int)
    panel["macro_scope_break_dummy"] = (panel.index.to_series() >= 2019).astype(int)
    panel["supply_scope_break_dummy"] = panel.index.to_series().between(2016, 2017).astype(int)

    core_columns = {
        "visitor_model_10k_persons": "游客量",
        "income_model_100m_cny": "旅游综合收入",
        "preferred_gdp_100m_cny": "GDP",
        "preferred_tertiary_100m_cny": "第三产业增加值",
        "per_visit_spend_yuan": "人均次消费",
    }
    for column in core_columns:
        panel[f"ln_{column}"] = np.log(panel[column])
        panel[f"z_{column}"] = zscore(panel[column])
        panel[f"minmax_{column}"] = minmax(panel[column])

    anomaly, thresholds = anomaly_rows(panel, core_columns)
    flagged = anomaly.loc[anomaly["final_flag"]].copy()
    flag_map = flagged.groupby("year")["metric_cn"].agg("、".join)
    panel["anomaly_flag"] = panel.index.to_series().map(flag_map).fillna("")
    panel["anomaly_treatment"] = np.where(panel["anomaly_flag"].eq(""), "保留", "保留；在模型中使用疫情/恢复/口径断点哑变量并做稳健性检验")

    # Machine-readable export ordering.
    panel = panel.reset_index()
    output_cols = [
        "year",
        "preferred_visitor_10k_persons",
        "visitor_status",
        "visitor_model_10k_persons",
        "visitor_data_role",
        "visitor_imputation_method",
        "visitor_low",
        "visitor_high",
        "preferred_comprehensive_income_100m_cny",
        "comprehensive_status",
        "income_model_100m_cny",
        "income_data_role",
        "income_imputation_method",
        "income_low",
        "income_high",
        "preferred_gdp_100m_cny",
        "gdp_status",
        "preferred_tertiary_100m_cny",
        "tertiary_status",
        "per_visit_spend_yuan",
        "guest_rooms",
        "beds",
        "room_revenue",
        "all_resident_disposable_income",
        "social_retail_sales",
        "road_mileage",
        "general_public_budget_expenditure",
        "culture_tourism_sports_media_expenditure",
        "pandemic_dummy",
        "recovery_dummy",
        "macro_scope_break_dummy",
        "supply_scope_break_dummy",
        "anomaly_flag",
        "anomaly_treatment",
    ]
    for column in core_columns:
        output_cols += [f"ln_{column}", f"z_{column}", f"minmax_{column}"]
    processed = panel[output_cols]
    processed.to_csv(OUT / "蓟州区旅游经济_建模主表_2010_2025.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    anomaly.to_csv(OUT / "蓟州区旅游经济_异常检测结果.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    imputation_log = []
    for year in panel["year"]:
        row = panel.loc[panel["year"] == year].iloc[0]
        for target, official, model, role, method, low, high, unit in [
            ("游客量", "preferred_visitor_10k_persons", "visitor_model_10k_persons", "visitor_data_role", "visitor_imputation_method", "visitor_low", "visitor_high", "万人次"),
            ("旅游综合收入", "preferred_comprehensive_income_100m_cny", "income_model_100m_cny", "income_data_role", "income_imputation_method", "income_low", "income_high", "亿元"),
        ]:
            if pd.isna(row[official]) or row[role] != "official_preferred":
                imputation_log.append({
                    "year": int(year), "target": target, "official_value": row[official], "model_value": row[model],
                    "low": row[low], "high": row[high], "unit": unit, "data_role": row[role], "method": row[method],
                    "allow_as_official": "否", "recommended_use": "敏感性/约束/建模补全；论文表格必须标注",
                })
    imputation_frame = pd.DataFrame(imputation_log)
    imputation_frame.to_csv(OUT / "蓟州区旅游经济_缺失补全记录.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    needs = pd.DataFrame([
        ["必须补充", "2020—2022、2025游客量与综合收入精确年度实际", "决定疫情冲击、恢复速度和2026—2030预测锚点", "蓟州区统计局/文旅局、政府信息公开申请、天津统计年鉴原表", "当前补值只作情景，不可替代实测"],
        ["必须补充", "2016同口径旅游综合收入", "消除收入序列中间断点", "蓟州统计年鉴原表、区统计局档案", "现有108.7866为取整同比反推"],
        ["必须补充", "旅游指标定义、调查范围和修订历史", "防止把口径变化误判为增长或冲击", "区统计局指标解释/统计制度文件", "至少明确全域/景区、重复到访和名义/实际口径"],
        ["必须补充", "旅游收入平减指数或服务类CPI", "将名义收入转为不变价并识别真实增长", "国家统计局、天津市统计局价格指数", "可用天津服务类CPI近似并做敏感性"],
        ["可选补充", "连续月度游客量、收入、景区客流、酒店入住率", "增加样本量并支持SARIMAX/ETS/干预分析", "区文旅局、重点景区、住宿业统计、OTA合作", "目标不少于60个连续月份"],
        ["可选补充", "客源地、过夜率、停留时长、复游率、人均消费", "把收入变化拆分为数量、停留和消费机制", "运营商信令、问卷、OTA、文旅大数据平台", "必须统一年度/节假日口径"],
        ["可选补充", "交通流、停车、铁路/公交客运与天气事件", "解释可达性和外部冲击", "交通运输部门、高德/百度开放数据、中国气象数据网", "按日或月对齐，并记录闭园/暴雨事件"],
        ["可选补充", "旅游专项投入、项目投产时间、营销曝光", "估计政策弹性与滞后效应", "财政决算、发改项目库、采购公告、宣传平台后台", "文旅体传媒支出不能直接当旅游专项投入"],
        ["可选补充", "相似区县同口径面板", "支持DID、合成控制与相对趋势校验", "天津其他区、北京近郊、河北相似县区统计公报/EPS", "先做口径映射，再进行面板分析"],
    ], columns=["priority", "data_needed", "reason", "recommended_source", "assumption_or_limit"])
    needs.to_csv(OUT / "蓟州区旅游经济_数据补充清单.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": len(processed),
        "official_missing": {
            "visitor": int(annual["preferred_visitor_10k_persons"].isna().sum()),
            "income": int(annual["preferred_comprehensive_income_100m_cny"].isna().sum()),
            "gdp": int(annual["preferred_gdp_100m_cny"].isna().sum()),
            "tertiary": int(annual["preferred_tertiary_100m_cny"].isna().sum()),
        },
        "completed_missing": {
            "visitor": int(processed["visitor_model_10k_persons"].isna().sum()),
            "income": int(processed["income_model_100m_cny"].isna().sum()),
        },
        "imputation_count": len(imputation_frame),
        "anomaly_flag_count": int(anomaly["final_flag"].sum()),
        "anomaly_years": sorted(int(x) for x in anomaly.loc[anomaly["final_flag"], "year"].unique()),
        "thresholds": thresholds,
        "imputation_parameters": {
            "room_share_2021": room_share_2021,
            "income_share_2021": income_share_2021,
            "composite_share_2021": composite_share_2021,
            "activity_ratio_2020": activity_ratio_2020,
            "room_ratio_2020": room_ratio_2020,
            "retail_ratio_2020": retail_ratio_2020,
        },
    }
    (OUT / "preprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    workbook_payload = {
        "summary": summary,
        "annual": annual.replace({np.nan: None}).to_dict("records"),
        "gaps": gaps.replace({np.nan: None}).to_dict("records"),
        "supply": supply_long.replace({np.nan: None}).to_dict("records"),
        "related": related_long.replace({np.nan: None}).to_dict("records"),
        "processed": processed.replace({np.nan: None}).to_dict("records"),
        "imputations": imputation_frame.replace({np.nan: None}).to_dict("records"),
        "anomalies": anomaly.replace({np.nan: None}).to_dict("records"),
        "needs": needs.replace({np.nan: None}).to_dict("records"),
    }
    (OUT / "workbook_payload.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFLAGGED\n", flagged[["year", "metric_cn", "yoy_percent", "z_score", "iqr_flag", "z_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
