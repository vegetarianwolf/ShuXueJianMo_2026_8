# Ridge macro-scope harmonization comparison

This directory compares the unchanged Ridge algorithm on the documented mixed-scope
GDP/tertiary series and an approximate bridge to the post-2019 level. The bridge
multiplies 2010-2018 values by `2019 revised current-price value / (1 + 2019
bulletin comparable-price growth) / 2018 old-scope value`.

This is an **approximate bridge for sensitivity analysis, not an official current-price backcast**.
Comparable-price growth and current-price levels are not
the same accounting object. The 2.1%/4.4% rates come from the 2019 bulletin
initial release, but the 219.62/136.91 anchors are later-yearbook revisions; no
matching finalized rates were found, so the bridge also has a **vintage mismatch**.

- GDP bridge factor: `0.5635980725`.
- Tertiary-value-added bridge factor: `0.5596613466`.
- Fixed-lambda macro sMAPE, original -> harmonized: `22.032030% -> 8.280545%`.
- Nested-tuned macro sMAPE, original -> harmonized: `18.297223% -> 8.832112%`.
- All four 2019 target/model break-fold comparisons improve; improvement range:
  `54.735398` to `55.993625` percentage points.

Protocol: expanding-origin tests for 2015-2023; lambda selection uses data through
2023; the final fit ends in 2024; 2025 is an official-initial holdout and is never
used for fitting; forecasts cover 2026-2030. `tourism_ridge_invariance.csv` records
why the existing tourism Ridge remains numerically unchanged: GDP and tertiary
value added are absent from its target and feature dependency graph.

Official sources: [Jizhou 2019 bulletin](https://www.tjjz.gov.cn/zwgk/zfxxgkqjjg/tjj1/fdzdgknr34/tjxx34/202107/t20210705_5495855.html);
[NBS unified-accounting and historical-revision Q&A](https://www.stats.gov.cn/sj/sjjd/202302/t20230202_1896273.html);
[Jizhou 2020 yearbook XLS](https://www.tjjz.gov.cn/zwgk/zfxxgkqjjg/tjj1/fdzdgknr34/tjxx34/202111/W020211117618518239847.xls).
