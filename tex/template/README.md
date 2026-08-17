# 数模通用 LaTeX 模板

此目录来自本项目指定的模板文件夹，已将源文件、字体、参考文献样式、示例代码和示例图并入仓库。

## 入口文件

- `数模通用模板.tex`：XeLaTeX 入口文件。
- `cumcmthesis.cls`：数学建模论文文档类。
- `gbt7714-numerical.bst`、`ref.bib`：参考文献样式和示例文献库。
- `fonts/`：中文字体资源。

## 编译约定

请从仓库根目录执行，并将编译产物写入 `build/tex/template/`：

```bash
cd tex/template
xelatex -interaction=nonstopmode -output-directory=../../build/tex/template 数模通用模板.tex
```

仓库中的 `build/tex/template/` 已保留原模板提供的编译结果作为参考，但该目录默认被 Git 忽略。

## 当前环境提示

本机 XeLaTeX 检查在字体加载阶段停止，错误为找不到模板依赖的系统字体 `STHeiti`。这属于字体环境依赖，不是模板正文错误；模板源文件保持原样。若本机仍无法识别该字体，可在 TeX 环境中安装/注册对应字体，或后续为本项目增加本地字体映射。
