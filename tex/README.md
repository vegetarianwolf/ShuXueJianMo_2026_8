# LaTeX 目录

论文源文件、参考文献、宏包配置和自定义样式放在此处；编译产物统一输出到 `build/tex/`。

本次采用的模板位于 `tex/template/`，入口文件为 `数模通用模板.tex`，模板说明见 [`tex/template/README.md`](template/README.md)。

建议正式论文另建 `tex/main.tex`，使用相对路径引用 `../figures/` 中的图表和 `../data/processed/` 中的稳定结果；不要直接修改模板示例文件。
