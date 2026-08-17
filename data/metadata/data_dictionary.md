# 初始数据字典

`data/raw/tj_tourism_economy_seed.csv` 是第一轮资料整理结果，不是最终建模数据集。

| 字段 | 含义 | 单位 | 说明 |
| --- | --- | --- | --- |
| `year` | 年份 | 年 | 对 `annual` 行表示全年 |
| `period_type` | 统计周期 | - | 当前种子数据以年度为主 |
| `tourist_visits_10k_persons` | 旅游接待人次 | 万人次 | 不能与半年度或季度值混用 |
| `tourism_revenue_10k_cny` | 旅游收入 | 万元 | 需进一步核对“直接收入/综合收入”口径 |
| `gdp_100m_cny` | 地区生产总值 | 亿元 | 统计公报口径，增长速度按可比价计算 |
| `tertiary_value_100m_cny` | 第三产业增加值 | 亿元 | 统计公报口径 |
| `value_status` | 值状态 | - | `observed` 为页面直接给出，`inferred` 为由同比增速反推 |

## 当前缺口

- 旧种子表的缺口说明已被本轮政府数据扩充更新。请以 `data/jizhou_tourism_economy/README.md` 和四张 `official_*.csv` 为准。
- 当前仍缺 2020 年旅游实际、2021 年精确游客量、2022 年旅游实际、2025 年旅游实际以及 2016 年综合收入。
- 题目要求的是“旅游综合收入”，站内历史页面的“旅游收入”多为直接收入，二者必须分开。

## 官方扩充数据字段

### `official_tourism_observations_2010_2025.csv`

| 字段 | 含义 |
| --- | --- |
| `period_start` / `period_end` | 观测覆盖的起止年份；年度值两者相同，五年累计值为 2021—2025 |
| `period_type` | `annual` 或 `five_year_total` |
| `metric` | `tourist_visits`、`tourism_direct_income` 或 `tourism_comprehensive_income` |
| `value` / `unit` | 数值及单位；游客为万人次，收入为亿元 |
| `value_status` | 实际、修订、推导、目标、累计等状态 |
| `metric_scope` | 明确区分游客范围、直接收入和综合收入 |
| `source_id` | 与 `data/metadata/sources.csv` 对应的来源标识 |
| `evidence_tier` | 1 为可直接复核政府原件，2 为官方索引/缓存复核，3 为政府站辅助附件 |
| `publication_date` / `accessed_date` | 发布日期与本次获取日期 |

### `official_macro_observations_2010_2025.csv`

| 字段 | 含义 |
| --- | --- |
| `metric` | `gdp` 或 `tertiary_value_added` |
| `value_status` | `official_initial`、`official_revised`、`official_final`、`official_yearbook`、`provisional_estimate` 或推导值 |
| `vintage` | 公报/年鉴版本，便于选择最新版并复核修订 |

### `official_related_observations_2014_2025.csv`

相关变量采用长表，`metric` 包括社零、固投、居民收入、常住人口、公路里程、一般公共预算支出和文化旅游体育与传媒支出。单位随 `unit` 字段变化，不能跨指标直接拼接。

### `official_annual_summary_2010_2025.csv`

这是按规则选择的宽表。每个数值旁均有状态列，且 `source_ids` 指向长表和来源清单。空白表示未找到可核验值，不表示 0。

## 旧种子表注意事项

`data/raw/tj_tourism_economy_seed.csv` 保留为第一轮历史整理，不建议直接读取：其数据行比表头多一个字段，且把直接收入和综合收入放在同一列。为了保护原始整理痕迹，本轮没有覆盖该文件。
