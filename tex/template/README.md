# 数模通用 LaTeX 模板

此目录来自本项目指定的模板文件夹，已将源文件、字体、参考文献样式、示例代码和示例图并入仓库。

## 入口文件

- `数模通用模板.tex`：XeLaTeX 入口文件。
- `cumcmthesis.cls`：数学建模论文文档类。
- `gbt7714-numerical.bst`、`ref.bib`：参考文献样式和示例文献库。
- `fonts/`：中文、英文字体资源，论文编译不依赖系统字体注册。

## 编译约定

请从仓库根目录执行，并将编译产物写入 `build/tex/template/`：

```bash
cd tex/template
xelatex -interaction=nonstopmode -output-directory=../../build/tex/template 数模通用模板.tex
```

仓库中的 `build/tex/template/` 已保留原模板提供的编译结果作为参考，但该目录默认被 Git 忽略。

## 字体说明

- 中文正文使用仓库内的 Source Han Serif CN；黑体、楷体、仿宋使用仓库内的 Fandol 字体。
- 英文正文使用仓库内的 TeX Gyre Termes/Heros，分别作为 Times/Arial 的跨平台替代。
- 代码字体使用模板原有的 YaHei/Consolas 和 Fira Code 文件。
- Fandol 字体许可证保存在 `fonts/Fandol-COPYING`；来源为 [CTAN fandol](https://ctan.org/pkg/fandol)。

模板类文件已改为显式加载这些相对路径字体，不再依赖 `STHeiti`、Times New Roman、Arial 或 `simkai.ttf`。
