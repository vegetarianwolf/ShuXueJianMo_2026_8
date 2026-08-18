from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\junk mass\小乱七八糟\数学建模\26国赛\shared files")
DATA_DIR = ROOT / "data" / "jizhou_tourism_economy"
OUT_DIR = ROOT / "outputs" / "01a00ea1-ee9b-7fe3-a9c3-5934549ee5ec"


def records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.replace({np.nan: None})
    return frame.to_dict(orient="records")


annual = pd.read_csv(DATA_DIR / "official_annual_summary_2010_2025.csv")
tourism = pd.read_csv(DATA_DIR / "official_tourism_observations_2010_2025.csv")
macro = pd.read_csv(DATA_DIR / "official_macro_observations_2010_2025.csv")
related = pd.read_csv(DATA_DIR / "official_related_observations_2014_2025.csv")

# The following values are an auditable scenario-assisted completion, not new
# official observations.  They are written to the workbook as editable inputs
# and formulas, and every resulting row retains an imputation label.
five_year_visitors = float(
    tourism.loc[
        (tourism["period_type"] == "five_year_total")
        & (tourism["metric"] == "tourist_visits"),
        "value",
    ].iloc[0]
)
five_year_income = float(
    tourism.loc[
        (tourism["period_type"] == "five_year_total")
        & (tourism["metric"] == "tourism_comprehensive_income"),
        "value",
    ].iloc[0]
)
target_2025_visitors = float(
    tourism.loc[
        (tourism["period_start"] == 2025)
        & (tourism["metric"] == "tourist_visits")
        & (tourism["value_status"] == "target"),
        "value",
    ].iloc[0]
)
target_2025_income = float(
    tourism.loc[
        (tourism["period_start"] == 2025)
        & (tourism["metric"] == "tourism_comprehensive_income")
        & (tourism["value_status"] == "target"),
        "value",
    ].iloc[0]
)

visitor_remainder_2021_2022 = five_year_visitors - 2363.0 - 2643.0 - target_2025_visitors
income_2022 = five_year_income - 110.0 - 191.5 - 221.0 - target_2025_income
visitor_2021 = visitor_remainder_2021_2022 * 110.0 / (110.0 + income_2022)
visitor_2022 = visitor_remainder_2021_2022 * income_2022 / (110.0 + income_2022)
visitor_2020 = 2800.0 * 0.55
income_2020 = 165.0 / 2800.0 * visitor_2020 * 0.90
income_2016 = 21.9172 * 5.0
gdp_2017 = (430.0 + 380.0) / 2.0

clean = annual[[
    "year",
    "preferred_visitor_10k_persons",
    "preferred_comprehensive_income_100m_cny",
    "preferred_gdp_100m_cny",
    "preferred_tertiary_100m_cny",
]].copy()
clean.columns = ["year", "visitors_raw", "income_raw", "gdp_raw", "tertiary_raw"]
clean["visitors_clean"] = clean["visitors_raw"]
clean["income_clean"] = clean["income_raw"]
clean["gdp_clean"] = clean["gdp_raw"]
clean["tertiary_clean"] = clean["tertiary_raw"]

visitor_fill = {2020: visitor_2020, 2021: visitor_2021, 2022: visitor_2022, 2025: target_2025_visitors}
income_fill = {2016: income_2016, 2020: income_2020, 2022: income_2022, 2025: target_2025_income}
for year, value in visitor_fill.items():
    clean.loc[clean["year"] == year, "visitors_clean"] = value
for year, value in income_fill.items():
    clean.loc[clean["year"] == year, "income_clean"] = value
clean.loc[clean["year"] == 2017, "gdp_clean"] = gdp_2017

# Fill tertiary value added by interpolating the tertiary/GDP share between
# adjacent years where both values are available. This is robust to the level
# break because the ratio remains interpretable inside each published system.
known_share = clean.dropna(subset=["gdp_clean", "tertiary_raw"]).set_index("year")
share_by_year = known_share["tertiary_raw"] / known_share["gdp_clean"]
for year in clean.loc[clean["tertiary_clean"].isna(), "year"]:
    earlier = share_by_year.index[share_by_year.index < year].max()
    later = share_by_year.index[share_by_year.index > year].min()
    share = share_by_year.loc[earlier] + (year - earlier) / (later - earlier) * (
        share_by_year.loc[later] - share_by_year.loc[earlier]
    )
    gdp_value = float(clean.loc[clean["year"] == year, "gdp_clean"].iloc[0])
    clean.loc[clean["year"] == year, "tertiary_clean"] = gdp_value * share

for col in ["visitors_clean", "income_clean", "gdp_clean", "tertiary_clean"]:
    clean[f"{col}_z"] = (clean[col] - clean[col].mean()) / clean[col].std(ddof=1)
clean["visitor_yoy"] = clean["visitors_clean"].pct_change()
clean["income_yoy"] = clean["income_clean"].pct_change()

anomaly_stats = {}
for source, label in [("visitor_yoy", "游客量同比"), ("income_yoy", "综合收入同比")]:
    values = clean[source].dropna()
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flagged = clean.loc[(clean[source] < lower) | (clean[source] > upper), "year"].astype(int).tolist()
    anomaly_stats[label] = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "flagged_years": flagged,
        "decision": "保留；作为疫情冲击或恢复反弹的结构状态，不删除、不缩尾",
    }

coverage = (
    related.groupby(["metric", "unit"])
    .agg(n=("value", "size"), start_year=("year", "min"), end_year=("year", "max"))
    .reset_index()
)

model_schemes = [
    {
        "question": "问题1",
        "scheme": "方案1｜基础适配（主推荐）",
        "model": "分段Gompertz增长模型＋疫情干预项＋滚动窗口更新",
        "principle": "Gompertz曲线把旅游经济理解为由较快增长逐步转向容量约束的过程；在疫情年份加入干预项，并在疫情前、冲击期和恢复期允许增长速度发生变化。每获得一个新年度数据就向前滚动窗口重新估计参数，使基准曲线不会长期受早期样本支配。",
        "fit": "2010—2019年游客量和综合收入整体增长，而2020—2022年存在结构冲击，2023年后又快速恢复。模型参数少、经济含义清楚，适合16期年度短样本，也能满足问题1“简单增长模型＋适用性评价”的要求。",
        "innovation": "不再用一条全时期曲线硬拟合，而是引入变点、疫情干预和滚动更新；估计时按observed、revised、inferred、target等状态设置质量权重，降低低证据数据对曲线的影响。",
        "limitation": "容量上限对样本末端较敏感；2020—2022和2025含假设补全值，可能改变恢复斜率；模型仍主要描述趋势，不能充分解释政策、客源和新业态之间的非线性作用。",
    },
    {
        "question": "问题1",
        "scheme": "方案2｜创新融合（增强分析）",
        "model": "质量加权贝叶斯局部线性趋势＋隐马尔可夫状态切换",
        "principle": "把无法直接观察的真实旅游趋势看成随时间缓慢变化的状态，再用隐马尔可夫链在“常态增长、外部冲击、恢复反弹”三种状态间切换。不同来源的数据设置不同观测误差，最终得到趋势、状态概率及其可信区间。",
        "fit": "官方数据同时包含修订值、约数、反推值、累计值和目标值，且存在多处缺失。状态空间模型能够自然处理缺失，贝叶斯框架能把证据等级转化为不确定性，状态切换则对应疫情与恢复阶段。",
        "innovation": "将数据来源质量直接写入观测方差，而非把所有数值等权处理；同时输出每年处于不同发展状态的概率，避免人为先固定断点。",
        "limitation": "16个年度样本不足以稳定估计过多转移概率，结果对先验分布敏感；模型解释和实现难度高于分段曲线，适合作为稳健性或答辩亮点，不宜单独承担主结论。",
    },
    {
        "question": "问题2",
        "scheme": "方案1｜基础适配（主推荐）",
        "model": "干预ARIMAX＋阻尼ETS的滚动误差加权组合",
        "principle": "ARIMAX利用游客量或收入自身的滞后关系，并加入GDP、第三产业和疫情干预变量解释变化；阻尼ETS负责捕捉平滑但逐步放缓的趋势。通过滚动预测计算两类模型近期误差，误差小的模型获得更高组合权重，再用残差模拟形成95%预测区间。",
        "fit": "年度样本短且主要目标只有游客量和综合收入，低参数模型比深度学习稳定；宏观解释变量虽不完整但可形成少量核心因子，疫情虚拟变量可以阻止极端年份扭曲长期系数。",
        "innovation": "在经典ARIMAX基础上加入对数变换、口径断点和疫情干预，并让ARIMAX与ETS权重随滚动误差动态变化，而不是固定平均；区间同时传播残差与外生变量情景的不确定性。",
        "limitation": "ARIMAX需要给出2026—2030年外生变量路径；GDP与第三产业高度相关，可能产生共线性；验证期较短使组合权重波动，应限制滞后阶数并报告权重敏感性。",
    },
    {
        "question": "问题2",
        "scheme": "方案2｜创新融合（增强分析）",
        "model": "GM(1,N)小样本先验＋贝叶斯结构时间序列（BSTS）动态融合",
        "principle": "灰色GM(1,N)先从少量、不完全信息中给出稳定的长期趋势，再把这一趋势作为BSTS的先验或一个候选预测源；BSTS进一步分解局部趋势、外生回归和疫情干预，利用贝叶斯模型平均得到最终预测及后验区间。",
        "fit": "灰色模型对小样本和缺失较宽容，BSTS能够处理趋势随时间变化及参数不确定性，适合本题“样本少、来源质量不一、冲击明显、必须给区间”的组合特征。",
        "innovation": "将灰色系统的弱信息预测与贝叶斯状态空间融合，并按证据等级调整观测噪声；不是简单取均值，而是根据后验概率动态分配模型权重。",
        "limitation": "GM模型的近指数累积规律可能与疫情冲击冲突；BSTS结果受先验和变量选择影响，短样本下后验区间可能较宽；实现和解释成本高于ARIMAX＋ETS。",
    },
    {
        "question": "问题3",
        "scheme": "方案1｜基础适配（主推荐）",
        "model": "动态弹性情景模型＋拉丁超立方Monte Carlo＋龙卷风敏感性",
        "principle": "以问题2基准预测为自然延续路径，把政策投入、核心客源和新业态指数的变化乘以相应弹性，形成乐观、基准和悲观路径；再在每个参数区间内进行拉丁超立方抽样，观察游客量和收入分布，并按输出变化幅度排序关键因素。",
        "fit": "题目明确要求三情景、敏感性和定量建议，但历史政策变量很稀疏。弹性模型参数少、透明，允许把规划目标和专家区间直接转为可审计假设，适合校赛论文时间限制。",
        "innovation": "弹性不设为永久常数，而是加入政策生效时滞和逐年动态权重；联合抽样保留因素间相关性，相比单纯上下浮动10%的方法能给出概率分布和超标风险。",
        "limitation": "弹性多由短历史或专家区间估计，因果解释有限；若政策、客源和新业态存在强交互，单一弹性会低估非线性；建议需同时报告参数区间而非只给一个点值。",
    },
    {
        "question": "问题3",
        "scheme": "方案2｜创新融合（答辩亮点）",
        "model": "系统动力学（SD）＋贝叶斯网络风险传播＋多目标鲁棒优化",
        "principle": "系统动力学描述游客、接待能力、消费、财政再投入和新业态供给之间的反馈；贝叶斯网络计算宏观下行或公共事件向客流和收入传播的概率；多目标鲁棒优化在不同风险情景下选择兼顾收入增长、投入成本和波动风险的政策组合。",
        "fit": "问题3不仅要求预测，还要求把政策转成3—5条可操作建议。该组合可以表现“投入改善供给、供给吸引客流、收入反哺投入”的循环，也能把突发事件概率纳入政策方案比较。",
        "innovation": "把因果反馈、概率风险和决策优化连成一体，使模型输出从情景曲线升级为政策组合；可使用分布鲁棒约束避免建议只在单一预测路径上有效。",
        "limitation": "系统结构、条件概率和成本约束需要大量补充数据或专家赋值，主观性较强；若参数未经校准，复杂模型可能制造虚假的精确性，建议作为扩展模型并用基础方案交叉验证。",
    },
]

supplementary_needs = [
    {
        "category": "必须补充",
        "variable": "2020、2022、2025年度游客量与综合收入；2021游客量",
        "current_gap": "年度实际值缺失；仅有2021—2025累计值、2025目标值及部分年度收入",
        "why": "两项核心因变量必须形成连续序列，且疫情冲击幅度直接影响趋势与预测区间",
        "recommended_source": "蓟州区政府工作报告/统计公报原件、区文旅局年度总结、天津市文旅局；优先申请内部统计表",
        "generated_assumption": "用五年累计约束、2025目标替代值和2021/2022收入比例分配游客余量；所有结果标记为假设补全",
        "uncertainty": "对2025目标替代值做±5%至±10%敏感性分析，并重新分配2021—2022余量",
    },
    {
        "category": "必须补充",
        "variable": "2016年同口径旅游综合收入",
        "current_gap": "仅有旅游直接收入21.9172亿元",
        "why": "疫情前趋势模型需要连续收入序列",
        "recommended_source": "2017年政府工作报告附件、蓟州区文旅局、后续统计表回列值",
        "generated_assumption": "按2011、2013—2015、2017年综合收入/直接收入约5倍的稳定关系桥接为109.586亿元",
        "uncertainty": "桥接倍数在4.8—5.2之间扰动",
    },
    {
        "category": "必须补充",
        "variable": "旅游收入价格平减指数",
        "current_gap": "现有收入均为当年价，未形成实际收入",
        "why": "名义价格上涨会夸大旅游经济真实增长",
        "recommended_source": "国家统计局/天津统计局CPI，优先住宿、餐饮、交通、文娱分项构造旅游价格指数",
        "generated_assumption": "当前工作簿暂保留名义收入；正式建模前以2025年为100进行平减",
        "uncertainty": "比较综合CPI与旅游分项加权指数两种口径",
    },
    {
        "category": "必须补充",
        "variable": "2026—2030政策投入、客源市场和新业态情景参数",
        "current_gap": "现有财政功能分类支出不等于旅游专项投入，未来值未给出",
        "why": "问题3三情景和量化建议必须有可调参数",
        "recommended_source": "蓟州区“十五五”规划、财政预算、文旅项目清单、京津冀协同文件；必要时专家打分",
        "generated_assumption": "标准化为基准1.00的指数，乐观/悲观区间由规划目标与历史波动共同设定",
        "uncertainty": "使用三角分布或PERT分布，报告5%、50%、95%分位数",
    },
    {
        "category": "可选补充",
        "variable": "京津客源占比、移动/高速交通流量、搜索热度",
        "current_gap": "现有表中没有年度客源结构与高频需求先行指标",
        "why": "可提升客流预测的及时性并解释区域协同效应",
        "recommended_source": "景区票务与运营商脱敏数据、交通运输部门、高速出入口流量、百度指数/微信指数",
        "generated_assumption": "未生成具体数值，避免用无依据的网络热度替代官方客流",
        "uncertainty": "先做相关性与滞后检验，再决定是否纳入",
    },
    {
        "category": "可选补充",
        "variable": "住宿供给、入住率、重点景区接待量、节假日与天气",
        "current_gap": "仅有天津市级星级饭店与A级景区结构校验数据",
        "why": "可解释人均消费、新业态供给和季节/节假日波动",
        "recommended_source": "天津市文旅局、蓟州区文旅局、气象局、酒店协会及重点景区运营数据",
        "generated_assumption": "若无法获得，仅作为情景解释变量，不强行生成年度观测",
        "uncertainty": "采用区间或等级变量，避免伪造精确值",
    },
]

payload = {
    "annual": records(annual),
    "tourism_recent": records(tourism[tourism["period_start"] >= 2020]),
    "related_coverage": records(coverage),
    "model_schemes": model_schemes,
    "supplementary_needs": supplementary_needs,
    "computed_check": records(clean.round(10)),
    "anomaly_stats": anomaly_stats,
    "constants": {
        "five_year_visitors": five_year_visitors,
        "five_year_income": five_year_income,
        "target_2025_visitors": target_2025_visitors,
        "target_2025_income": target_2025_income,
        "visitor_remainder_2021_2022": visitor_remainder_2021_2022,
        "income_2022": income_2022,
        "visitor_2021": visitor_2021,
        "visitor_2022": visitor_2022,
        "visitor_2020": visitor_2020,
        "income_2020": income_2020,
        "income_2016": income_2016,
        "gdp_2017": gdp_2017,
    },
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "analysis_data.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({"output": str(OUT_DIR / "analysis_data.json"), "rows": len(annual)}, ensure_ascii=False))
