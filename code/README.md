# 代码目录

- `src/`：可被脚本或 notebook 复用的模块。
- `scripts/`：下载、清洗、建模和生成图表的命令行脚本。
- `notebooks/`：探索性分析，最终结论应回写到可复现脚本或论文。

建议将数据入口、字段映射和模型参数集中管理，避免在 notebook 单元中散落路径和参数。

本题当前可复现入口：

- `scripts/model_jizhou_tourism.py`：传统对数线性模型。
- `scripts/model_jizhou_tourism_ml.py`：机器学习候选与原生滚动回测。
- `scripts/compare_branch_models.py`：读取 `data/unified/`，执行逐折模拟增强轨与未模拟 canonical 共同行稳健性轨、2024 pseudo-holdout 和跨阶段压力测试；所有断点模型按用户协议不执行。
