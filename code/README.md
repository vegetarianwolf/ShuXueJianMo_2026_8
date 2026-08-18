# 代码目录

- `src/`：可被脚本或 notebook 复用的模块。
- `scripts/`：下载、清洗、建模和生成图表的命令行脚本。
- `notebooks/`：探索性分析，最终结论应回写到可复现脚本或论文。

建议将数据入口、字段映射和模型参数集中管理，避免在 notebook 单元中散落路径和参数。

本题当前可复现入口：

- `scripts/model_jizhou_tourism.py`：传统对数线性模型。
- `scripts/model_jizhou_tourism_ml.py`：机器学习候选与原生滚动回测。
- `scripts/compare_branch_models.py`：读取 `data/unified/`，执行统一回测并生成题目 1—3 的增长模型、2026—2030 预测区间、三情景、敏感性、报告和审计产物；所有断点模型按用户协议不执行。
- `scripts/optimize_ridge_model.py`：保留固定 `alpha=0.1` 基线，用严格嵌套扩展窗口验证调节 Ridge 的单一惩罚强度 `lambda`（即 scikit-learn 参数 `alpha`），并生成优化前后报告、预测区间、边界诊断与稳健性产物。
- `scripts/render_tjmml_c_visuals.py`：生成报告中的四项核心指标、模型评判、未来预测与情景敏感性图。
