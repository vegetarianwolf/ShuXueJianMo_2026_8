# CSV 分表导出

本目录将“蓟州区旅游经济政府数据扩充版”工作簿的六张工作表逐张导出，便于 Python、R、MATLAB、数据库和其他建模工具直接读取。

- 编码：UTF-8（无 BOM）
- 分隔符：英文逗号
- 表头：首行
- 缺失值：空字段，含义为“未找到可核验值”，不等于 0
- 比率：`01_annual_summary_2010_2025.csv` 末两列使用小数，例如 `0.25` 表示 25%

文件映射和行列数见 `manifest.csv`。编号 01—06 与工作簿工作表顺序一致；这些 CSV 是从工作簿可见值生成的，不应与原始政府文件混淆。

Python 读取示例：

```python
import pandas as pd

annual = pd.read_csv(
    "data/jizhou_tourism_economy/csv_exports/01_annual_summary_2010_2025.csv"
)
```
