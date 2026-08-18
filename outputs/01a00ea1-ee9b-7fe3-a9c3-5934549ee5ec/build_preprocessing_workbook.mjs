import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/junk mass/小乱七八糟/数学建模/26国赛/shared files/outputs/01a00ea1-ee9b-7fe3-a9c3-5934549ee5ec";
const payload = JSON.parse(await fs.readFile(path.join(outputDir, "workbook_payload.json"), "utf8"));
const outputPath = path.join(outputDir, "蓟州区旅游经济_预处理汇总数据集.xlsx");

const wb = Workbook.create();
const sheetNames = [
  "使用说明",
  "严格年度数据",
  "缺失补全",
  "建模主表",
  "异常检测",
  "异常图表",
  "转换说明",
  "供给辅助",
  "数据补充",
  "模型建议",
];
for (const name of sheetNames) wb.worksheets.add(name);

const C = {
  navy: "#17324D", teal: "#0C6B70", blue: "#2A6F97", green: "#2A9D8F",
  orange: "#E78B48", red: "#D95F59", yellow: "#E9C46A", white: "#FFFFFF",
  paleBlue: "#EAF3F8", paleGreen: "#E8F5F1", paleOrange: "#FFF1E6",
  paleRed: "#FCEBE8", paleYellow: "#FFF8DD", gray: "#F3F5F7",
  midGray: "#D7DEE5", darkGray: "#53606B",
};

function title(sheet, text, endCol) {
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange("A1").values = [[text]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: C.navy,
    font: { bold: true, color: C.white, size: 18 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 36;
}

function subtitle(sheet, range, text, fill = C.paleBlue) {
  sheet.mergeCells(range);
  const topLeft = range.split(":")[0];
  sheet.getRange(topLeft).values = [[text]];
  sheet.getRange(range).format = {
    fill,
    font: { bold: true, color: C.navy, size: 12 },
    verticalAlignment: "center",
  };
}

function header(range) {
  range.format = {
    fill: C.teal,
    font: { bold: true, color: C.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#B7C6CE" },
  };
}

function body(range) {
  range.format = {
    verticalAlignment: "top",
    wrapText: true,
    borders: { insideHorizontal: { style: "thin", color: C.midGray } },
  };
}

function setWidths(sheet, widths) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

function boolCn(value) { return value ? "是" : "否"; }

// 使用说明
{
  const s = wb.worksheets.getItem("使用说明");
  s.showGridLines = false;
  title(s, "蓟州区旅游经济：数据预处理汇总与模型决策", "H");
  s.mergeCells("A2:H2");
  s.getRange("A2").values = [["数据分两层保存：严格年度数据保持官方缺口；建模主表提供透明的情景补全。任何补全值、二手值和约束值都不得在论文中称为官方实测。"]];
  s.getRange("A2:H2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 40, font: { color: C.navy } };

  subtitle(s, "A4:H4", "本轮数据审计结论");
  s.getRange("A5:H5").values = [["年度期数", null, "游客官方缺失", null, "收入官方缺失", null, "异常年份", null]];
  s.getRange("A6:H6").values = [[payload.summary.rows, null, payload.summary.official_missing.visitor, null, payload.summary.official_missing.income, null, payload.summary.anomaly_years.join("、"), null]];
  for (const [range, fill] of [["A5:B6", C.paleBlue], ["C5:D6", C.paleOrange], ["E5:F6", C.paleOrange], ["G5:H6", C.paleRed]]) {
    s.getRange(range).format = { fill, borders: { preset: "outside", style: "thin", color: C.midGray }, horizontalAlignment: "center", verticalAlignment: "center" };
  }
  s.getRange("A5:H5").format.font = { bold: true, color: C.darkGray };
  s.getRange("A6:H6").format.font = { bold: true, color: C.navy, size: 15 };
  s.getRange("A5:H6").format.rowHeight = 28;

  subtitle(s, "A8:H8", "处理原则与最终用途");
  const rows = [
    ["缺失值", "不采用均值填充；疫情期属于非随机缺失。使用累计约束、同口径供给代理、目标隐含基数和二手锚点生成建模情景，并保留上下界。", "输出", "缺失补全、建模主表"],
    ["异常值", "在对数同比尺度同时执行 IQR(1.5倍四分位距) 与 |Z|>3 检测。2019口径断点与2020—2023冲击/恢复值全部保留。", "输出", "异常检测、异常图表"],
    ["转换", "为线性模型、Ridge、SVM等提供 Z-score；为神经网络/距离模型提供Min-Max；为弹性与乘法增长模型提供自然对数。", "输出", "建模主表、转换说明"],
    ["建模", "年度标签极少，主模型采用低参数、可外推且可滚动回测的方法；复杂机器学习只作负对照或敏感性。", "建议", "模型建议"],
  ];
  s.getRange("A9:H12").values = rows.map(r => [r[0], r[1], null, null, null, r[2], r[3], null]);
  for (let r = 9; r <= 12; r++) { s.mergeCells(`B${r}:E${r}`); s.mergeCells(`G${r}:H${r}`); }
  s.getRange("A9:H12").format = { wrapText: true, verticalAlignment: "center", borders: { insideHorizontal: { style: "thin", color: C.midGray } } };
  s.getRange("A9:A12").format.font = { bold: true, color: C.navy };
  s.getRange("F9:F12").format.font = { bold: true, color: C.teal };
  s.getRange("A9:H12").format.rowHeight = 48;

  subtitle(s, "A14:H14", "建议使用顺序");
  s.mergeCells("A15:H15");
  s.getRange("A15").values = [["① 严格年度数据做证据审计 → ② 缺失补全做情景敏感性 → ③ 建模主表用于程序读取 → ④ 异常检测决定断点变量 → ⑤ 模型建议确定主模型与稳健性模型。"]];
  s.getRange("A15:H15").format = { fill: C.paleGreen, wrapText: true, rowHeight: 38, font: { bold: true, color: C.navy } };
  setWidths(s, { A: 16, B: 24, C: 14, D: 18, E: 20, F: 12, G: 24, H: 18 });
  s.freezePanes.freezeRows(2);
}

// 严格年度数据
{
  const s = wb.worksheets.getItem("严格年度数据");
  s.showGridLines = false;
  title(s, "官方优选年度宽表（原值保持不变）", "M");
  s.mergeCells("A2:M2");
  s.getRange("A2").values = [["本表原样摘录 official_annual_summary_2010_2025.csv。空白即未取得可核验的官方年度实际，不在本表回填。"]];
  s.getRange("A2:M2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 34 };
  const keys = ["year", "preferred_visitor_10k_persons", "visitor_status", "preferred_direct_income_100m_cny", "direct_status", "preferred_comprehensive_income_100m_cny", "comprehensive_status", "preferred_gdp_100m_cny", "gdp_status", "preferred_tertiary_100m_cny", "tertiary_status", "source_ids", "quality_note"];
  const headers = ["年份", "游客量(万人次)", "游客状态", "直接收入(亿元)", "直接收入状态", "综合收入(亿元)", "综合收入状态", "GDP(亿元)", "GDP状态", "第三产业(亿元)", "第三产业状态", "来源ID", "质量备注"];
  s.getRange("A4:M4").values = [headers]; header(s.getRange("A4:M4"));
  const rows = payload.annual.map(row => keys.map(key => row[key]));
  s.getRange(`A5:M${4 + rows.length}`).values = rows; body(s.getRange(`A5:M${4 + rows.length}`));
  s.getRange(`A5:A${4 + rows.length}`).format.numberFormat = "0";
  for (const col of ["B", "D", "F", "H", "J"]) s.getRange(`${col}5:${col}${4 + rows.length}`).format.numberFormat = "#,##0.000";
  s.getRange(`A5:M${4 + rows.length}`).format.rowHeight = 34;
  const table = s.tables.add(`A4:M${4 + rows.length}`, true, "StrictAnnualTable"); table.style = "TableStyleMedium2";
  setWidths(s, { A: 9, B: 15, C: 22, D: 16, E: 22, F: 17, G: 30, H: 14, I: 22, J: 17, K: 22, L: 45, M: 55 });
  s.freezePanes.freezeRows(4); s.freezePanes.freezeColumns(1);
}

// 缺失补全（16年完整映射）
{
  const s = wb.worksheets.getItem("缺失补全");
  s.showGridLines = false;
  title(s, "缺失值补全、证据角色与区间", "M");
  s.mergeCells("A2:M2");
  s.getRange("A2").values = [["补全优先级：官方优选值 > 透明代数/累计约束 > 同口径代理情景。所有非官方值都标注“不允许作为官方实测”，并提供低—高敏感性范围。"]];
  s.getRange("A2:M2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 38 };
  const headers = ["年份", "游客官方值", "游客建模值", "游客数据角色", "游客补全方法", "游客低值", "游客高值", "收入官方值", "收入建模值", "收入数据角色", "收入补全方法", "收入低值", "收入高值"];
  s.getRange("A4:M4").values = [headers]; header(s.getRange("A4:M4"));
  const rows = payload.processed.map(r => [
    r.year, r.preferred_visitor_10k_persons, r.visitor_model_10k_persons, r.visitor_data_role, r.visitor_imputation_method,
    r.visitor_low, r.visitor_high, r.preferred_comprehensive_income_100m_cny, r.income_model_100m_cny, r.income_data_role,
    r.income_imputation_method, r.income_low, r.income_high,
  ]);
  s.getRange(`A5:M${4 + rows.length}`).values = rows; body(s.getRange(`A5:M${4 + rows.length}`));
  s.getRange(`B5:C${4 + rows.length}`).format.numberFormat = "#,##0.000";
  s.getRange(`F5:I${4 + rows.length}`).format.numberFormat = "#,##0.000";
  s.getRange(`L5:M${4 + rows.length}`).format.numberFormat = "#,##0.000";
  s.getRange(`D5:D${4 + rows.length}`).conditionalFormats.add("notContainsText", { text: "official_preferred", format: { fill: C.paleOrange, font: { color: "#9A4D16", bold: true } } });
  s.getRange(`J5:J${4 + rows.length}`).conditionalFormats.add("notContainsText", { text: "official_preferred", format: { fill: C.paleOrange, font: { color: "#9A4D16", bold: true } } });
  s.getRange(`A5:M${4 + rows.length}`).format.rowHeight = 45;
  const table = s.tables.add(`A4:M${4 + rows.length}`, true, "ImputationMapTable"); table.style = "TableStyleMedium4";
  setWidths(s, { A: 9, B: 14, C: 14, D: 30, E: 48, F: 13, G: 13, H: 14, I: 14, J: 30, K: 48, L: 13, M: 13 });
  s.freezePanes.freezeRows(4); s.freezePanes.freezeColumns(1);
}

// 建模主表：使用工作表公式保持转换可审计
{
  const s = wb.worksheets.getItem("建模主表");
  s.showGridLines = false;
  title(s, "建模主表（完整核心序列＋公式化转换）", "X");
  s.mergeCells("A2:X2");
  s.getRange("A2").values = [["B—E列由缺失补全与严格年度数据映射；F、M—T列为Excel公式。若使用树模型，可直接使用B—E原尺度；Ridge/SVM优先使用Z-score；弹性模型使用对数。"]];
  s.getRange("A2:X2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 38 };
  const headers = ["年份", "游客量(万人次)", "综合收入(亿元)", "GDP(亿元)", "第三产业(亿元)", "人均次消费(元)", "疫情期", "恢复期", "宏观口径断点", "游客数据角色", "收入数据角色", "异常标记", "ln游客", "ln收入", "Z游客", "Z收入", "MinMax游客", "MinMax收入", "游客log同比", "收入log同比", "游客低", "游客高", "收入低", "收入高"];
  s.getRange("A4:X4").values = [headers]; header(s.getRange("A4:X4"));
  const n = payload.processed.length;
  for (let i = 0; i < n; i++) {
    const r = 5 + i;
    const src = payload.processed[i];
    const imp = 5 + i;
    s.getRange(`A${r}:L${r}`).values = [[src.year, null, null, src.preferred_gdp_100m_cny, src.preferred_tertiary_100m_cny, null, src.pandemic_dummy, src.recovery_dummy, src.macro_scope_break_dummy, src.visitor_data_role, src.income_data_role, src.anomaly_flag]];
    s.getRange(`B${r}:C${r}`).formulas = [[`='缺失补全'!C${imp}`, `='缺失补全'!I${imp}`]];
    s.getRange(`F${r}`).formulas = [[`=C${r}*10000/B${r}`]];
    s.getRange(`M${r}:R${r}`).formulas = [[
      `=LN(B${r})`, `=LN(C${r})`,
      `=(B${r}-AVERAGE($B$5:$B$20))/STDEV.P($B$5:$B$20)`,
      `=(C${r}-AVERAGE($C$5:$C$20))/STDEV.P($C$5:$C$20)`,
      `=(B${r}-MIN($B$5:$B$20))/(MAX($B$5:$B$20)-MIN($B$5:$B$20))`,
      `=(C${r}-MIN($C$5:$C$20))/(MAX($C$5:$C$20)-MIN($C$5:$C$20))`,
    ]];
    s.getRange(`S${r}:T${r}`).formulas = i === 0 ? [["", ""]] : [[`=LN(B${r}/B${r - 1})`, `=LN(C${r}/C${r - 1})`]];
    s.getRange(`U${r}:X${r}`).formulas = [[`='缺失补全'!F${imp}`, `='缺失补全'!G${imp}`, `='缺失补全'!L${imp}`, `='缺失补全'!M${imp}`]];
  }
  body(s.getRange("A5:X20"));
  s.getRange("B5:F20").format.numberFormat = "#,##0.000";
  s.getRange("M5:T20").format.numberFormat = "0.000000";
  s.getRange("U5:X20").format.numberFormat = "#,##0.000";
  s.getRange("L5:L20").conditionalFormats.addCustom("=LEN($L5)>0", { fill: C.paleRed, font: { color: "#9F2D20", bold: true } });
  s.getRange("J5:K20").conditionalFormats.add("notContainsText", { text: "official_preferred", format: { fill: C.paleOrange } });
  s.getRange("A5:X20").format.rowHeight = 34;
  const table = s.tables.add("A4:X20", true, "ModelReadyTable"); table.style = "TableStyleMedium2";
  setWidths(s, { A: 9, B: 14, C: 14, D: 12, E: 15, F: 16, G: 9, H: 9, I: 14, J: 30, K: 30, L: 28, M: 12, N: 12, O: 12, P: 12, Q: 13, R: 13, S: 14, T: 14, U: 12, V: 12, W: 12, X: 12 });
  s.freezePanes.freezeRows(4); s.freezePanes.freezeColumns(1);
}

// 异常检测
{
  const s = wb.worksheets.getItem("异常检测");
  s.showGridLines = false;
  title(s, "IQR 与 Z-score 异常值检测（对数同比尺度）", "L");
  s.mergeCells("A2:L2");
  s.getRange("A2").values = [["趋势型年度序列若直接对水平值做箱线图，会把正常增长误识为异常。因此先计算 ln(x_t/x_{t-1})，再用 IQR 与 Z-score 双重检测。最终异常值不删除，只标记并进入断点/状态建模。"]];
  s.getRange("A2:L2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 42 };
  const headers = ["年份", "指标代码", "指标", "对数同比", "同比(%)", "Z-score", "IQR下界", "IQR上界", "IQR异常", "Z异常", "最终异常", "处理"];
  s.getRange("A4:L4").values = [headers]; header(s.getRange("A4:L4"));
  const rows = payload.anomalies.map(r => [r.year, r.metric, r.metric_cn, r.log_yoy, r.yoy_percent, r.z_score, r.iqr_lower, r.iqr_upper, boolCn(r.iqr_flag), boolCn(r.z_flag), boolCn(r.final_flag), r.treatment]);
  s.getRange(`A5:L${4 + rows.length}`).values = rows; body(s.getRange(`A5:L${4 + rows.length}`));
  s.getRange(`D5:H${4 + rows.length}`).format.numberFormat = "0.000000";
  s.getRange(`E5:E${4 + rows.length}`).format.numberFormat = "0.00";
  s.getRange(`K5:K${4 + rows.length}`).conditionalFormats.add("containsText", { text: "是", format: { fill: C.paleRed, font: { color: "#9F2D20", bold: true } } });
  const table = s.tables.add(`A4:L${4 + rows.length}`, true, "AnomalyAuditTable"); table.style = "TableStyleMedium2";
  setWidths(s, { A: 9, B: 34, C: 18, D: 14, E: 12, F: 12, G: 13, H: 13, I: 11, J: 11, K: 11, L: 34 });
  s.freezePanes.freezeRows(4);
}

// 异常图表：原生图表，且渲染为独立PNG
{
  const s = wb.worksheets.getItem("异常图表");
  s.showGridLines = false;
  title(s, "异常检测可视化：趋势与同比分布", "P");
  s.mergeCells("A2:P2");
  s.getRange("A2").values = [["红色标记年份见异常检测表。箱线图基于对数同比，不对补全值或结构冲击作机械删除。"]];
  s.getRange("A2:P2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 34 };
  s.getRange("A4:F4").values = [["年份", "游客量", "综合收入", "GDP", "第三产业", "人均次消费"]]; header(s.getRange("A4:F4"));
  for (let i = 0; i < 16; i++) {
    const r = 5 + i;
    const m = 5 + i;
    s.getRange(`A${r}:F${r}`).formulas = [[`='建模主表'!A${m}`, `='建模主表'!B${m}`, `='建模主表'!C${m}`, `='建模主表'!D${m}`, `='建模主表'!E${m}`, `='建模主表'!F${m}`]];
  }
  body(s.getRange("A5:F20")); s.getRange("B5:F20").format.numberFormat = "#,##0.00";
  const trendTable = s.tables.add("A4:F20", true, "TrendChartData"); trendTable.style = "TableStyleMedium2";

  const visitorChart = s.charts.add("line", s.getRange("A4:B20"));
  visitorChart.title = "游客量趋势（万人次）"; visitorChart.hasLegend = false; visitorChart.xAxis = { axisType: "textAxis" }; visitorChart.yAxis = { numberFormatCode: "#,##0" }; visitorChart.setPosition("H4", "P16");
  const incomeChart = s.charts.add("line", { chartType: "line", title: "旅游综合收入趋势（亿元）", hasLegend: false });
  const incomeSeries = incomeChart.series.add("综合收入"); incomeSeries.categoryFormula = "'异常图表'!$A$5:$A$20"; incomeSeries.formula = "'异常图表'!$C$5:$C$20"; incomeSeries.fill = C.orange;
  incomeChart.title = "旅游综合收入趋势（亿元）"; incomeChart.xAxis = { axisType: "textAxis" }; incomeChart.yAxis = { numberFormatCode: "#,##0" }; incomeChart.setPosition("H18", "P30");

  s.getRange("A23:F23").values = [["年份", "游客量", "旅游综合收入", "GDP", "第三产业增加值", "人均次消费"]]; header(s.getRange("A23:F23"));
  const anomalyByMetric = new Map();
  for (const row of payload.anomalies) anomalyByMetric.set(`${row.year}|${row.metric}`, row.log_yoy * 100);
  const metrics = ["visitor_model_10k_persons", "income_model_100m_cny", "preferred_gdp_100m_cny", "preferred_tertiary_100m_cny", "per_visit_spend_yuan"];
  const growthRows = [];
  for (let year = 2011; year <= 2025; year++) growthRows.push([year, ...metrics.map(metric => anomalyByMetric.get(`${year}|${metric}`) ?? null)]);
  s.getRange("A24:F38").values = growthRows; body(s.getRange("A24:F38")); s.getRange("B24:F38").format.numberFormat = "0.00";
  const growthChart = s.charts.add("line", s.getRange("A23:F38"));
  growthChart.title = "核心指标对数同比与异常波动（%）"; growthChart.hasLegend = true; growthChart.xAxis = { axisType: "textAxis" }; growthChart.yAxis = { numberFormatCode: "0.0" }; growthChart.setPosition("H32", "P47");
  setWidths(s, { A: 10, B: 14, C: 14, D: 13, E: 15, F: 16, G: 3, H: 12, I: 12, J: 12, K: 12, L: 12, M: 12, N: 12, O: 12, P: 12 });
  s.freezePanes.freezeRows(4);
}

// 转换说明
{
  const s = wb.worksheets.getItem("转换说明");
  s.showGridLines = false;
  title(s, "变量转换公式、适用模型与注意事项", "G");
  s.mergeCells("A2:G2");
  s.getRange("A2").values = [["数据文件同时保留原尺度、自然对数、Z-score 和 Min-Max。不要把标准化后的系数直接解释为原单位效应；树模型通常无需标准化。"]];
  s.getRange("A2:G2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 38 };
  s.getRange("A4:G4").values = [["处理", "公式", "是否采用", "适用模型", "优点", "风险", "本表位置"]]; header(s.getRange("A4:G4"));
  const rows = [
    ["自然对数", "ln(x)", "采用", "对数线性、弹性、Gompertz/指数增长", "减弱异方差并把乘法关系转为加法", "零值不可直接取对数；本题核心指标均为正", "建模主表 M:N"],
    ["Z-score", "z=(x-μ)/σ", "采用", "Ridge、Lasso、SVM、PCA、KNN", "消除量纲，便于正则化和距离计算", "均值和标准差必须只在训练折计算，防止泄漏", "建模主表 O:P；CSV含全部核心指标"],
    ["Min-Max", "x'=(x-min)/(max-min)", "采用", "神经网络、距离模型、可视化评分", "映射到[0,1]，便于比较", "对新极值和异常值敏感；比赛主模型不依赖它", "建模主表 Q:R；CSV含全部核心指标"],
    ["对数同比", "g_t=ln(x_t/x_{t-1})", "采用", "异常检测、增长率比较", "适合趋势型序列，不会把长期增长直接判异常", "补全值会影响相邻两期，需结合数据角色解读", "建模主表 S:T、异常检测"],
    ["删除/缩尾", "—", "不采用", "—", "—", "会抹掉疫情冲击与口径断点，损失题目核心信息", "异常值全部保留"],
  ];
  s.getRange("A5:G9").values = rows; body(s.getRange("A5:G9")); s.getRange("A5:G9").format.rowHeight = 70;
  const table = s.tables.add("A4:G9", true, "TransformGuideTable"); table.style = "TableStyleMedium2";
  subtitle(s, "A11:G11", "建模时的防泄漏规则", C.paleBlue);
  s.mergeCells("A12:G13");
  s.getRange("A12").values = [["滚动回测的每一个训练窗都必须独立计算均值、标准差、最小值和最大值，再转换验证年；不能使用2010—2025全样本统计量回测。工作簿中的全样本转换仅用于探索和论文描述，正式预测脚本应在Pipeline内部拟合转换器。"]];
  s.getRange("A12:G13").format = { fill: C.paleRed, wrapText: true, verticalAlignment: "center", font: { color: "#8D2D21", bold: true } };
  setWidths(s, { A: 18, B: 29, C: 14, D: 34, E: 34, F: 42, G: 28 });
  s.freezePanes.freezeRows(4);
}

// 供给辅助
{
  const s = wb.worksheets.getItem("供给辅助");
  s.showGridLines = false;
  title(s, "旅游供给与相关解释变量长表", "U");
  s.mergeCells("A2:U2");
  s.getRange("A2").values = [["左表为住宿供给，右表为宏观/财政/交通变量。限额以上住宿餐饮业与星级饭店口径不得相加；辅助变量不强行填满，以免制造虚假信息。"]];
  s.getRange("A2:U2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 38 };
  const supplyKeys = ["year", "metric", "value", "unit", "value_status", "metric_scope", "vintage", "source_id", "evidence_tier", "source_url", "notes"];
  const supplyHeaders = ["年份", "指标", "值", "单位", "状态", "口径", "版本", "来源ID", "证据级", "来源链接", "备注"];
  s.getRange("A4:K4").values = [supplyHeaders]; header(s.getRange("A4:K4"));
  const supplyRows = payload.supply.map(r => supplyKeys.map(k => r[k]));
  s.getRange(`A5:K${4 + supplyRows.length}`).values = supplyRows; body(s.getRange(`A5:K${4 + supplyRows.length}`));
  s.getRange(`C5:C${4 + supplyRows.length}`).format.numberFormat = "#,##0.000";
  const supplyTable = s.tables.add(`A4:K${4 + supplyRows.length}`, true, "SupplyLongTable"); supplyTable.style = "TableStyleMedium4";

  const relKeys = ["year", "metric", "value", "unit", "value_status", "source_id", "source_url", "notes"];
  const relHeaders = ["年份", "指标", "值", "单位", "状态", "来源ID", "来源链接", "备注"];
  s.getRange("N4:U4").values = [relHeaders]; header(s.getRange("N4:U4"));
  const relRows = payload.related.map(r => relKeys.map(k => r[k]));
  s.getRange(`N5:U${4 + relRows.length}`).values = relRows; body(s.getRange(`N5:U${4 + relRows.length}`));
  s.getRange(`P5:P${4 + relRows.length}`).format.numberFormat = "#,##0.000";
  const relTable = s.tables.add(`N4:U${4 + relRows.length}`, true, "RelatedLongTable"); relTable.style = "TableStyleMedium2";
  setWidths(s, { A: 9, B: 20, C: 13, D: 12, E: 24, F: 39, G: 22, H: 28, I: 10, J: 42, K: 48, L: 3, M: 3, N: 9, O: 36, P: 13, Q: 13, R: 24, S: 28, T: 42, U: 48 });
  s.freezePanes.freezeRows(4);
}

// 数据补充
{
  const s = wb.worksheets.getItem("数据补充");
  s.showGridLines = false;
  title(s, "必须补充与可选补充数据路线", "E");
  s.mergeCells("A2:E2");
  s.getRange("A2").values = [["“必须补充”决定能否形成可信预测或因果结论；“可选补充”用于提高精度、解释机制或扩展创新模型。Kaggle只适合算法演示，不宜替代蓟州区官方数据。"]];
  s.getRange("A2:E2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 42 };
  const headers = ["优先级", "所需数据", "原因", "推荐获取途径", "假设/限制"];
  s.getRange("A4:E4").values = [headers]; header(s.getRange("A4:E4"));
  const rows = payload.needs.map(r => [r.priority, r.data_needed, r.reason, r.recommended_source, r.assumption_or_limit]);
  s.getRange(`A5:E${4 + rows.length}`).values = rows; body(s.getRange(`A5:E${4 + rows.length}`));
  s.getRange(`A5:A${4 + rows.length}`).conditionalFormats.add("containsText", { text: "必须", format: { fill: C.paleRed, font: { color: "#9F2D20", bold: true } } });
  s.getRange(`A5:A${4 + rows.length}`).conditionalFormats.add("containsText", { text: "可选", format: { fill: C.paleGreen, font: { color: "#176B55", bold: true } } });
  s.getRange(`A5:E${4 + rows.length}`).format.rowHeight = 76;
  const table = s.tables.add(`A4:E${4 + rows.length}`, true, "DataNeedsTable"); table.style = "TableStyleMedium2";
  setWidths(s, { A: 14, B: 45, C: 42, D: 56, E: 50 });
  s.freezePanes.freezeRows(4);
}

// 模型建议
{
  const s = wb.worksheets.getItem("模型建议");
  s.showGridLines = false;
  title(s, "两份建模试验的评价与最终模型建议", "H");
  s.mergeCells("A2:H2");
  s.getRange("A2").values = [["结论：机器学习报告对小样本、外推和前视偏差的判断是正确的；传统报告的统计解释更完整，但对数断点模型延续14%—18%增长率，只能作为乐观上界。最终应采用“低参数主模型＋政策情景＋结构敏感性”的三层架构。"]];
  s.getRange("A2:H2").format = { fill: C.paleYellow, wrapText: true, rowHeight: 48, font: { color: C.navy, bold: true } };

  subtitle(s, "A4:H4", "现有方案对比");
  s.getRange("A5:H5").values = [["方案", "目标尺度/结构", "滚动验证", "2030游客量", "2030收入", "优点", "主要风险", "建议角色"]]; header(s.getRange("A5:H5"));
  const compare = [
    ["原值 Ridge", "绝对增量＋疫情/恢复哑变量", "sMAPE 11.33% / 15.08%", 3941.1, 298.5, "低方差、能线性外推、验证设计较规范", "恢复期只有2个测试点；长期斜率仍不稳定", "数据驱动主基准"],
    ["对数断点回归", "乘法增长＋2023后水平断点", "报告为样本内MAPE，非同口径滚动验证", 5982.6, 619.0, "解释直观、参数少、区间可计算", "把疫情前14%—18%增长延续至2030，偏乐观", "乐观上界/结构敏感性"],
    ["BayesianRidge(对数)", "固定特征＋对数高斯", "条件区间", 5970.8, 615.2, "能表达模型条件不确定性", "区间未覆盖模型结构、目标变换和政策变化", "稳健性模型，不作唯一结论"],
    ["政策锚定情景", "2025目标＋收入8%＋人均消费3%", "非统计回测", 3548.9, 339.4, "适合规划讨论，假设透明", "目标不是实际；不具统计置信含义", "政策基准情景"],
    ["RF / RBF-SVR / 样条", "非外推或边界外推", "负对照", "1172—2594", "80—203", "能揭示算法外推缺陷", "训练域外变常数、回落或反向下降", "仅作负对照"],
  ];
  s.getRange("A6:H10").values = compare; body(s.getRange("A6:H10"));
  s.getRange("D6:E10").format.numberFormat = "#,##0.0"; s.getRange("A6:H10").format.rowHeight = 64;
  const compareTable = s.tables.add("A5:H10", true, "ExistingModelComparison"); compareTable.style = "TableStyleMedium2";

  subtitle(s, "A12:H12", "按小问落地的最终架构", C.paleBlue);
  s.getRange("A13:H13").values = [["小问/任务", "主模型", "稳健性/创新层", "数据层", "验证", "输出", "不建议", "判定理由"]]; header(s.getRange("A13:H13"));
  const finalRows = [
    ["问题1：历史规律", "质量分层的稳健分段对数回归（疫情前/冲击/恢复）", "贝叶斯局部线性趋势或HMM状态切换", "严格观测为主；补全层只做敏感性", "断点检验＋留一/滚动验证＋残差诊断", "增长阶段、冲击幅度、恢复速度和口径断点", "单段指数增长", "能解释2019口径断点与2020—2023结构冲击，参数数量可控"],
    ["问题2：2026—2030预测", "原值 Ridge（时间趋势＋疫情/恢复哑变量）作为点预测基准", "与阻尼趋势/政策锚定情景组成三路径；用模型分歧形成压力包络", "未来未知外生变量不进入点预测；2025二手值只作锚点敏感性", "expanding-origin滚动回测；分阶段报告MAPE/sMAPE", "数据驱动基准、政策基准、乐观上界，不给伪精确唯一曲线", "RF、RBF-SVR、LSTM、Transformer、高阶ARIMAX", "当前每个目标只有12个官方年度标签，低维强正则最稳健"],
    ["问题3：政策与敏感性", "R=V×S/10000恒等分解＋情景树＋LHS Monte Carlo", "若补到投入/客源面板，再升级DID/合成控制或贝叶斯网络", "参数用区间与三角分布；区分可控政策和外生冲击", "全局敏感性（PRCC/Sobol可选）＋约束一致性检查", "关键杠杆排序、情景概率、稳健策略", "直接回归财政投入弹性、用文旅体传媒支出代替旅游专项投入", "现有数据不能识别因果弹性，情景模型更诚实且可答辩"],
  ];
  s.getRange("A14:H16").values = finalRows; body(s.getRange("A14:H16")); s.getRange("A14:H16").format.rowHeight = 115;
  const finalTable = s.tables.add("A13:H16", true, "FinalModelArchitecture"); finalTable.style = "TableStyleMedium4";

  subtitle(s, "A18:H18", "最重要的论文表述边界", C.paleRed);
  s.mergeCells("A19:H21");
  s.getRange("A19").values = [["不要把补全层写成“处理后真实数据”，应写成“用于模型连续性和敏感性分析的约束一致情景值”；不要把BayesianRidge条件区间或小样本残差区间称为总不确定性95%区间；不要用全样本标准化后再滚动回测。主结论应报告模型结构敏感性：2030年游客量约3549—5983万人次、收入约298.5—619亿元的跨模型差异，说明目标尺度与长期增长假设比超参数更重要。"]];
  s.getRange("A19:H21").format = { fill: C.paleRed, wrapText: true, verticalAlignment: "center", font: { color: "#8D2D21", bold: true } };
  setWidths(s, { A: 28, B: 38, C: 42, D: 38, E: 36, F: 40, G: 38, H: 50 });
  s.freezePanes.freezeRows(5);
}

// Formula/error audit and visual verification.
const formulaErrors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(outputDir, "preprocessing_formula_errors.ndjson"), formulaErrors.ndjson, "utf8");

const modelInspect = await wb.inspect({
  kind: "table",
  range: "建模主表!A4:X20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 24,
  maxChars: 30000,
});
await fs.writeFile(path.join(outputDir, "preprocessing_model_table_check.ndjson"), modelInspect.ndjson, "utf8");

for (const name of sheetNames) {
  const preview = await wb.render({ sheetName: name, autoCrop: "all", scale: 0.75, format: "png" });
  const safe = name.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(path.join(outputDir, `preprocessing_preview_${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
  if (name === "异常图表") {
    await fs.writeFile(path.join(outputDir, "蓟州区旅游经济_异常检测图.png"), new Uint8Array(await preview.arrayBuffer()));
  }
}

const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: sheetNames, rows: payload.summary.rows, anomalyFlags: payload.summary.anomaly_flag_count }, null, 2));
