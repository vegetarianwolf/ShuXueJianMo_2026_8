import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/junk mass/小乱七八糟/数学建模/26国赛/shared files/outputs/01a00ea1-ee9b-7fe3-a9c3-5934549ee5ec";
const payload = JSON.parse(await fs.readFile(path.join(outputDir, "analysis_data.json"), "utf8"));
const outputPath = path.join(outputDir, "蓟州区旅游经济_模型选择与预处理.xlsx");

const wb = Workbook.create();
const sheetNames = [
  "使用说明",
  "模型选择",
  "模型流程",
  "原始年度数据",
  "补充假设",
  "预处理结果",
  "异常检测",
  "趋势图",
  "数据补充",
];
for (const name of sheetNames) wb.worksheets.add(name);

const COLORS = {
  navy: "#16324F",
  teal: "#0F6B78",
  blue: "#2A6F97",
  green: "#2A9D8F",
  orange: "#F4A261",
  red: "#E76F51",
  yellow: "#E9C46A",
  paleBlue: "#EAF3F8",
  paleGreen: "#E8F5F1",
  paleOrange: "#FFF2E5",
  paleRed: "#FCEBE8",
  paleYellow: "#FFF8DD",
  gray: "#F3F5F7",
  midGray: "#D7DEE5",
  darkGray: "#53606B",
  white: "#FFFFFF",
};

function title(sheet, text, endCol) {
  sheet.mergeCells(`A1:${endCol}1`);
  const r = sheet.getRange("A1");
  r.values = [[text]];
  r.format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 18 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 34;
}

function subtitle(sheet, range, text, fill = COLORS.paleBlue) {
  sheet.mergeCells(range);
  const topLeft = range.split(":")[0];
  const r = sheet.getRange(topLeft);
  r.values = [[text]];
  r.format = {
    fill,
    font: { bold: true, color: COLORS.navy, size: 12 },
    verticalAlignment: "center",
  };
}

function header(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white },
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
    borders: { insideHorizontal: { style: "thin", color: COLORS.midGray } },
  };
}

function setWidths(sheet, widths) {
  for (const [col, width] of Object.entries(widths)) {
    sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  }
}

// -------------------- 使用说明 --------------------
{
  const s = wb.worksheets.getItem("使用说明");
  s.showGridLines = false;
  title(s, "蓟州区旅游经济：模型选择与数据预处理工作簿", "H");
  s.mergeCells("A2:H2");
  s.getRange("A2").values = [["用途：完成问题1—3的两套模型方案比较、缺失补全、异常检测、转换处理和数据补充规划。所有假设补全值均与官方观测分列保存，不会伪装成官方实际值。"]];
  s.getRange("A2:H2").format = { fill: COLORS.paleYellow, font: { color: COLORS.navy }, wrapText: true, rowHeight: 38 };

  subtitle(s, "A4:H4", "关键数据诊断");
  const kpiLabels = [["年度样本期数", "游客量原始缺失", "综合收入原始缺失", "同比IQR异常年份"]];
  s.getRange("A5:H5").values = [["年度样本期数", null, "游客量原始缺失", null, "综合收入原始缺失", null, "同比IQR异常年份", null]];
  s.getRange("A6:H6").formulas = [["=COUNTA('原始年度数据'!A4:A19)", null, "=COUNTBLANK('原始年度数据'!B4:B19)", null, "=COUNTBLANK('原始年度数据'!F4:F19)", null, '="2020、2022、2023"', null]];
  for (const pair of [["A5:B6", COLORS.paleBlue], ["C5:D6", COLORS.paleOrange], ["E5:F6", COLORS.paleOrange], ["G5:H6", COLORS.paleRed]]) {
    const [range, fill] = pair;
    s.getRange(range).format = { fill, borders: { preset: "outside", style: "thin", color: COLORS.midGray }, horizontalAlignment: "center", verticalAlignment: "center" };
  }
  s.getRange("A5:H5").format.font = { bold: true, color: COLORS.darkGray };
  s.getRange("A6:H6").format.font = { bold: true, color: COLORS.navy, size: 15 };
  s.getRange("A5:H6").format.rowHeight = 28;

  subtitle(s, "A8:H8", "结论先行");
  const conclusions = [
    ["问题1主模型", "分段Gompertz＋疫情干预＋滚动更新", "原因", "短样本、趋势明显、必须解释结构冲击"],
    ["问题2主模型", "干预ARIMAX＋阻尼ETS滚动组合", "原因", "低参数、可检验、可给95%区间且能利用宏观变量"],
    ["问题3主模型", "动态弹性情景＋LHS Monte Carlo", "原因", "能把政策语言转为参数区间并直接生成敏感性排序"],
    ["不建议作为主模型", "LSTM、Transformer、XGBoost", "原因", "仅16期年度样本，深度/树模型极易过拟合且难以形成可信区间"],
  ];
  s.getRange("A9:H12").values = conclusions.map(r => [r[0], r[1], r[2], r[3], null, null, null, null]);
  s.mergeCells("D9:H9"); s.mergeCells("D10:H10"); s.mergeCells("D11:H11"); s.mergeCells("D12:H12");
  s.getRange("A9:H12").format = { wrapText: true, verticalAlignment: "center", borders: { insideHorizontal: { style: "thin", color: COLORS.midGray } } };
  s.getRange("A9:A12").format.font = { bold: true, color: COLORS.navy };
  s.getRange("C9:C12").format.font = { bold: true, color: COLORS.teal };
  s.getRange("A9:H12").format.rowHeight = 34;

  subtitle(s, "A14:H14", "数据文件与复核入口");
  s.getRange("A15:H15").values = [["文件/入口", "作用", "记录数", "时间范围", "口径提醒", "来源", "获取日期", "备注"]];
  header(s.getRange("A15:H15"));
  const sourceRows = [
    ["official_annual_summary_2010_2025.csv", "年度优选宽表", 16, "2010—2025", "不替代长表版本信息", "本地数据目录", "2026-08-17", "本工作簿原始年度数据来源"],
    ["official_tourism_observations_2010_2025.csv", "旅游长表", 46, "2010—2025", "直接收入与综合收入不可拼接", "本地数据目录", "2026-08-17", "含目标值和五年累计值"],
    ["official_macro_observations_2010_2025.csv", "GDP与第三产业长表", 37, "2010—2025", "初值、预计值、年鉴修订值并存", "本地数据目录", "2026-08-17", "优先最终/修订值"],
    ["official_related_observations_2014_2025.csv", "相关解释变量", 37, "2014—2025", "多数指标覆盖不完整", "本地数据目录", "2026-08-17", "用于候选外生变量筛选"],
    ["蓟州区政府数据发布", "官方补数入口", null, null, null, "https://www.tjjz.gov.cn/zwgk/sjfb/", null, null],
    ["天津统计年鉴", "宏观修订值", null, null, null, "https://stats.tj.gov.cn/tjsj_52032/tjnj/", null, null],
    ["天津市文旅局统计信息", "文旅供给与结构校验", null, null, null, "https://whly.tj.gov.cn/ZWGKYXXGK1640/ZFXXGK5456_1/FDZDGKNR5153/TJXX3610/", null, null],
  ];
  s.getRange("A16:H22").values = sourceRows;
  body(s.getRange("A16:H22"));
  s.getRange("C16:C19").format.numberFormat = "0";
  s.getRange("A15:H22").format.rowHeight = 34;
  setWidths(s, { A: 31, B: 23, C: 10, D: 14, E: 30, F: 42, G: 14, H: 24 });
  s.freezePanes.freezeRows(2);
}

// -------------------- 模型选择 --------------------
{
  const s = wb.worksheets.getItem("模型选择");
  s.showGridLines = false;
  title(s, "问题1—3模型方案比较", "G");
  s.mergeCells("A2:G2");
  s.getRange("A2").values = [["选择原则：主方案优先满足短样本稳定性、统计检验和可解释性；创新方案用于不确定性表达、稳健性验证和答辩展示，不以复杂度替代证据。"]];
  s.getRange("A2:G2").format = { fill: COLORS.paleYellow, wrapText: true, rowHeight: 34, font: { color: COLORS.navy } };
  const headers = ["小问", "方案", "推荐模型", "核心原理", "适配性", "创新点", "局限性"];
  s.getRange("A4:G4").values = [headers];
  header(s.getRange("A4:G4"));
  const rows = payload.model_schemes.map(m => [m.question, m.scheme, m.model, m.principle, m.fit, m.innovation, m.limitation]);
  s.getRange("A5:G10").values = rows;
  body(s.getRange("A5:G10"));
  s.getRange("A5:A10").format = { fill: COLORS.paleBlue, font: { bold: true, color: COLORS.navy }, horizontalAlignment: "center", verticalAlignment: "center" };
  s.getRange("B5:B10").format.font = { bold: true, color: COLORS.teal };
  s.getRange("C5:C10").format = { fill: COLORS.gray, font: { bold: true, color: COLORS.navy }, wrapText: true, verticalAlignment: "top" };
  s.getRange("A5:G10").format.rowHeight = 112;
  const modelTable = s.tables.add("A4:G10", true, "ModelSelectionTable");
  modelTable.style = "TableStyleMedium2";
  setWidths(s, { A: 10, B: 23, C: 34, D: 48, E: 44, F: 44, G: 42 });
  s.freezePanes.freezeRows(4);
}

// -------------------- 模型流程 --------------------
{
  const s = wb.worksheets.getItem("模型流程");
  s.showGridLines = false;
  title(s, "模型应用流程与决策节点", "K");
  setWidths(s, { A: 3, B: 22, C: 5, D: 22, E: 5, F: 24, G: 5, H: 24, I: 5, J: 25, K: 3 });

  const makeFlow = (top, q, boxes, note) => {
    subtitle(s, `B${top}:J${top}`, q, COLORS.paleBlue);
    const boxRow = top + 2;
    const endRow = boxRow + 2;
    const cols = ["B", "D", "F", "H", "J"];
    for (let i = 0; i < cols.length; i++) {
      s.mergeCells(`${cols[i]}${boxRow}:${cols[i]}${endRow}`);
      s.getRange(`${cols[i]}${boxRow}`).values = [[boxes[i].text]];
      const fill = boxes[i].decision ? COLORS.paleYellow : (boxes[i].negative ? COLORS.paleRed : COLORS.paleGreen);
      s.getRange(`${cols[i]}${boxRow}:${cols[i]}${endRow}`).format = {
        fill,
        font: { bold: true, color: COLORS.navy },
        horizontalAlignment: "center",
        verticalAlignment: "center",
        wrapText: true,
        borders: { preset: "outside", style: boxes[i].decision ? "medium" : "thin", color: boxes[i].decision ? COLORS.orange : COLORS.green },
      };
      if (i < cols.length - 1) {
        const arrowCol = ["C", "E", "G", "I"][i];
        s.mergeCells(`${arrowCol}${boxRow}:${arrowCol}${endRow}`);
        s.getRange(`${arrowCol}${boxRow}`).values = [["→"]];
        s.getRange(`${arrowCol}${boxRow}:${arrowCol}${endRow}`).format = { font: { bold: true, color: COLORS.blue, size: 20 }, horizontalAlignment: "center", verticalAlignment: "center" };
      }
    }
    s.mergeCells(`B${endRow + 1}:J${endRow + 2}`);
    s.getRange(`B${endRow + 1}`).values = [[note]];
    s.getRange(`B${endRow + 1}:J${endRow + 2}`).format = { fill: COLORS.gray, font: { color: COLORS.darkGray }, wrapText: true, verticalAlignment: "center" };
  };

  makeFlow(3, "问题1：历史规律与基准增长模型", [
    { text: "输入多版本长表\n保留证据等级" },
    { text: "口径是否一致？\n否：分序列/桥接\n是：继续", decision: true },
    { text: "缺失是否位于疫情期？\n是：累计约束补全\n否：插值/比例桥接", decision: true },
    { text: "变点是否显著？\n是：分段Gompertz\n否：单段增长" , decision: true},
    { text: "输出趋势、冲击幅度\n适用区间与误差" },
  ], "创新分支：若需同时表达来源质量和状态概率，则以质量加权贝叶斯局部趋势＋隐马尔可夫状态切换进行稳健性验证。" );

  makeFlow(12, "问题2：游客量与综合收入预测", [
    { text: "输入清洗序列\n对数/平稳性检查" },
    { text: "样本是否少于30期？\n是：限制低阶模型\n否：再考虑复杂模型", decision: true },
    { text: "未来外生变量可得？\n是：ARIMAX\n否：ETS/灰色模型", decision: true },
    { text: "残差是否白噪声？\n否：调阶/变换\n是：滚动验证", decision: true },
    { text: "误差互补？\n是：动态加权组合\n否：选最优单模", decision: true },
  ], "输出：2026—2030点预测、95%预测区间、MAE/RMSE/MAPE，以及与问题1基准模型的统一验证集比较。" );

  makeFlow(21, "问题3：情景、敏感性与量化建议", [
    { text: "承接问题2\n建立基准预测路径" },
    { text: "驱动因素可量化？\n是：估计弹性\n否：标准化指数", decision: true },
    { text: "参数是否在历史/规划边界？\n否：重新标定\n是：三情景赋值", decision: true },
    { text: "非线性反馈显著？\n是：SD＋贝叶斯网络\n否：动态弹性", decision: true },
    { text: "LHS Monte Carlo\n敏感性排序\n鲁棒建议" },
  ], "决策输出必须形成“因素—政策工具—投入强度—实施期—预期客流/收入增量—监测指标”的闭环。" );
  s.freezePanes.freezeRows(1);
}

// -------------------- 原始年度数据 --------------------
{
  const s = wb.worksheets.getItem("原始年度数据");
  s.showGridLines = false;
  title(s, "官方优选年度宽表（原始层，不做覆盖）", "M");
  s.mergeCells("A2:M2");
  s.getRange("A2").values = [["来源：official_annual_summary_2010_2025.csv。空白保持为空；target、inferred、provisional等状态不得当作同等质量的官方实际值。"]];
  s.getRange("A2:M2").format = { fill: COLORS.paleYellow, wrapText: true, rowHeight: 34 };
  const keys = [
    "year", "preferred_visitor_10k_persons", "visitor_status", "preferred_direct_income_100m_cny", "direct_status",
    "preferred_comprehensive_income_100m_cny", "comprehensive_status", "preferred_gdp_100m_cny", "gdp_status",
    "preferred_tertiary_100m_cny", "tertiary_status", "source_ids", "quality_note",
  ];
  const headers = ["年份", "游客量(万人次)", "游客状态", "直接收入(亿元)", "直接收入状态", "综合收入(亿元)", "综合收入状态", "GDP(亿元)", "GDP状态", "第三产业(亿元)", "第三产业状态", "来源ID", "质量说明"];
  s.getRange("A3:M3").values = [headers];
  header(s.getRange("A3:M3"));
  const rows = payload.annual.map(r => keys.map(k => r[k] ?? null));
  s.getRange("A4:M19").values = rows;
  body(s.getRange("A4:M19"));
  s.getRange("A4:A19").format.numberFormat = "0";
  for (const col of ["B", "D", "F", "H", "J"]) s.getRange(`${col}4:${col}19`).format.numberFormat = "#,##0.00";
  s.getRange("A3:M19").format.rowHeight = 31;
  const rawTable = s.tables.add("A3:M19", true, "RawAnnualTable");
  rawTable.style = "TableStyleMedium2";
  setWidths(s, { A: 9, B: 16, C: 20, D: 16, E: 20, F: 17, G: 26, H: 13, I: 27, J: 16, K: 27, L: 48, M: 48 });
  s.freezePanes.freezeRows(3);
  s.freezePanes.freezeColumns(1);
}

// -------------------- 补充假设 --------------------
{
  const s = wb.worksheets.getItem("补充假设");
  s.showGridLines = false;
  title(s, "缺失补全参数与约束（黄色单元格可编辑）", "G");
  s.mergeCells("A2:G2");
  s.getRange("A2").values = [["这些数值是建模假设或官方累计/目标约束，不是新增官方年度实际。修改黄色输入后，预处理结果、异常检测和趋势图会联动更新。"]];
  s.getRange("A2:G2").format = { fill: COLORS.paleRed, font: { color: COLORS.navy }, wrapText: true, rowHeight: 36 };
  s.getRange("A3:G3").values = [["参数", "数值", "单位", "类型", "依据/计算逻辑", "可编辑", "建议敏感性范围"]];
  header(s.getRange("A3:G3"));
  const c = payload.constants;
  const inputRows = [
    ["2021—2025游客累计", c.five_year_visitors, "万人次", "官方累计约束", "政府报告五年累计值，不拆作单年观测", "否", "固定"],
    ["2021—2025综合收入累计", c.five_year_income, "亿元", "官方累计约束", "政府报告五年累计值", "否", "固定"],
    ["2025游客目标替代值", c.target_2025_visitors, "万人次", "假设输入", "无年度实际值时暂以目标代替", "是", "±5%至±10%"],
    ["2025综合收入目标替代值", c.target_2025_income, "亿元", "假设输入", "无年度实际值时暂以目标代替", "是", "±5%至±10%"],
    ["2020游客保留率", 0.55, "%", "假设输入", "相对2019年，用于刻画首年疫情冲击", "是", "45%—65%"],
    ["2020人均消费保留率", 0.90, "%", "假设输入", "相对2019人均旅游消费", "是", "80%—100%"],
    ["综合收入/直接收入桥接倍数", 5.0, "倍", "假设输入", "2011、2013—2015、2017年关系约为5倍", "是", "4.8—5.2"],
    ["2021综合收入观测", 110.0, "亿元", "官方观测", "用于分配2021—2022游客余量", "否", "固定"],
    ["2023游客量观测", 2363.0, "万人次", "官方观测", "五年累计约束已知项", "否", "固定"],
    ["2024游客量观测", 2643.0, "万人次", "官方观测", "五年累计约束已知项", "否", "固定"],
    ["2023综合收入观测", 191.5, "亿元", "官方观测", "五年累计约束已知项", "否", "固定"],
    ["2024综合收入观测", 221.0, "亿元", "官方观测", "五年累计约束已知项", "否", "固定"],
  ];
  s.getRange("A4:G15").values = inputRows;
  s.getRange("A16:G23").values = [
    ["2021—2022游客余量", null, "万人次", "公式输出", "五年累计－2023－2024－2025替代值", "否", null],
    ["2022综合收入", null, "亿元", "公式输出", "五年累计－2021－2023－2024－2025替代值", "否", null],
    ["2021游客量", null, "万人次", "公式输出", "游客余量按2021/2022收入占比分配", "否", null],
    ["2022游客量", null, "万人次", "公式输出", "游客余量按2021/2022收入占比分配", "否", null],
    ["2020游客量", null, "万人次", "公式输出", "2019游客量×2020保留率", "否", null],
    ["2020综合收入", null, "亿元", "公式输出", "2019人均消费×2020游客量×消费保留率", "否", null],
    ["2016综合收入", null, "亿元", "公式输出", "2016直接收入×桥接倍数", "否", null],
    ["2017 GDP", null, "亿元", "公式输出", "2016与2018同口径段线性插值", "否", null],
  ];
  const formulas = [
    "=B4-B6-B12-B13",
    "=B5-B7-B11-B14-B15",
    "=B16*B11/(B11+B17)",
    "=B16*B17/(B11+B17)",
    "='原始年度数据'!B13*B8",
    "='原始年度数据'!F13/'原始年度数据'!B13*B20*B9",
    "='原始年度数据'!D10*B10",
    "=AVERAGE('原始年度数据'!H10,'原始年度数据'!H12)",
  ];
  for (let i = 0; i < formulas.length; i++) s.getRange(`B${16 + i}`).formulas = [[formulas[i]]];
  body(s.getRange("A4:G23"));
  s.getRange("B4:B23").format.numberFormat = "#,##0.00";
  s.getRange("B8:B9").format.numberFormat = "0.0%";
  for (const row of [6, 7, 8, 9, 10]) {
    s.getRange(`B${row}`).format = { fill: COLORS.paleYellow, font: { bold: true, color: COLORS.navy }, numberFormat: row === 8 || row === 9 ? "0.0%" : "#,##0.00" };
  }
  s.getRange("A16:G23").format.fill = COLORS.paleGreen;
  s.getRange("A3:G23").format.rowHeight = 31;
  const assumptionTable = s.tables.add("A3:G23", true, "AssumptionTable");
  assumptionTable.style = "TableStyleMedium4";
  setWidths(s, { A: 31, B: 15, C: 12, D: 18, E: 49, F: 10, G: 22 });
  s.freezePanes.freezeRows(3);
}

// -------------------- 预处理结果 --------------------
{
  const s = wb.worksheets.getItem("预处理结果");
  s.showGridLines = false;
  title(s, "建模主数据：原始值、补全值、异常标记与转换结果", "AD");
  s.mergeCells("A2:AD2");
  s.getRange("A2").values = [["规则：官方值优先；疫情期缺失采用累计约束＋明确假设；常规缺失采用比例桥接或同口径线性插值。IQR异常只标记不删除。Z-score用于回归/组合模型，LN用于稳定方差。"]];
  s.getRange("A2:AD2").format = { fill: COLORS.paleYellow, wrapText: true, rowHeight: 38 };
  const headers = [
    "年份", "游客原始", "直接收入原始", "综合收入原始", "GDP原始", "第三产业原始", "游客状态", "综合收入状态", "GDP状态", "第三产业状态",
    "游客清洗值", "游客处理方法", "综合收入清洗值", "收入处理方法", "GDP清洗值", "GDP处理方法", "第三产业清洗值", "第三产业处理方法",
    "游客同比", "收入同比", "游客IQR判定", "收入IQR判定", "疫情哑变量", "GDP口径断点", "游客Z", "收入Z", "GDP Z", "第三产业Z", "LN游客", "LN收入",
  ];
  s.getRange("A3:AD3").values = [headers];
  header(s.getRange("A3:AD3"));
  for (let i = 0; i < 16; i++) {
    const r = 4 + i;
    const raw = 4 + i;
    s.getRange(`A${r}:J${r}`).formulas = [[
      `='原始年度数据'!A${raw}`,
      `=IF('原始年度数据'!B${raw}="","",'原始年度数据'!B${raw})`,
      `=IF('原始年度数据'!D${raw}="","",'原始年度数据'!D${raw})`,
      `=IF('原始年度数据'!F${raw}="","",'原始年度数据'!F${raw})`,
      `=IF('原始年度数据'!H${raw}="","",'原始年度数据'!H${raw})`,
      `=IF('原始年度数据'!J${raw}="","",'原始年度数据'!J${raw})`,
      `=IF('原始年度数据'!C${raw}="","",'原始年度数据'!C${raw})`,
      `=IF('原始年度数据'!G${raw}="","",'原始年度数据'!G${raw})`,
      `=IF('原始年度数据'!I${raw}="","",'原始年度数据'!I${raw})`,
      `=IF('原始年度数据'!K${raw}="","",'原始年度数据'!K${raw})`,
    ]];
    const year = 2010 + i;
    let visitorFormula = `=B${r}`;
    let visitorMethod = "保留官方优选值";
    if (year === 2020) { visitorFormula = "='补充假设'!$B$20"; visitorMethod = "2019×疫情保留率（假设）"; }
    if (year === 2021) { visitorFormula = "='补充假设'!$B$18"; visitorMethod = "累计余量按收入份额分配"; }
    if (year === 2022) { visitorFormula = "='补充假设'!$B$19"; visitorMethod = "累计余量按收入份额分配"; }
    if (year === 2025) { visitorFormula = "='补充假设'!$B$6"; visitorMethod = "目标值替代，非年度实际"; }
    s.getRange(`K${r}`).formulas = [[visitorFormula]];
    s.getRange(`L${r}`).values = [[visitorMethod]];

    let incomeFormula = `=D${r}`;
    let incomeMethod = "保留官方优选值";
    if (year === 2016) { incomeFormula = "='补充假设'!$B$22"; incomeMethod = "直接收入×历史桥接倍数"; }
    if (year === 2020) { incomeFormula = "='补充假设'!$B$21"; incomeMethod = "2019人均消费×游客量×保留率"; }
    if (year === 2022) { incomeFormula = "='补充假设'!$B$17"; incomeMethod = "五年累计约束反推"; }
    if (year === 2025) { incomeFormula = "='补充假设'!$B$7"; incomeMethod = "目标值替代，非年度实际"; }
    s.getRange(`M${r}`).formulas = [[incomeFormula]];
    s.getRange(`N${r}`).values = [[incomeMethod]];

    let gdpFormula = `=E${r}`;
    let gdpMethod = "保留官方优选值";
    if (year === 2017) { gdpFormula = "='补充假设'!$B$23"; gdpMethod = "同口径段线性插值"; }
    s.getRange(`O${r}`).formulas = [[gdpFormula]];
    s.getRange(`P${r}`).values = [[gdpMethod]];

    let tertiaryFormula = `=F${r}`;
    let tertiaryMethod = "保留官方优选值";
    if (year === 2012) {
      tertiaryFormula = `=O${r}*((Q${r-1}/O${r-1})+((A${r}-A${r-1})/(A${r+1}-A${r-1}))*((Q${r+1}/O${r+1})-(Q${r-1}/O${r-1})))`;
      tertiaryMethod = "第三产业/GDP占比线性插值";
    }
    if (year >= 2015 && year <= 2018) {
      tertiaryFormula = `=O${r}*((Q8/O8)+((A${r}-A8)/(A13-A8))*((Q13/O13)-(Q8/O8)))`;
      tertiaryMethod = "2014—2019占比桥接";
    }
    s.getRange(`Q${r}`).formulas = [[tertiaryFormula]];
    s.getRange(`R${r}`).values = [[tertiaryMethod]];

    if (i === 0) {
      s.getRange(`S${r}:V${r}`).values = [[null, null, null, null]];
    } else {
      s.getRange(`S${r}`).formulas = [[`=K${r}/K${r-1}-1`]];
      s.getRange(`T${r}`).formulas = [[`=M${r}/M${r-1}-1`]];
      s.getRange(`U${r}`).formulas = [[`=IF(OR(S${r}<'异常检测'!$B$7,S${r}>'异常检测'!$B$8),"保留-结构冲击","正常")`]];
      s.getRange(`V${r}`).formulas = [[`=IF(OR(T${r}<'异常检测'!$E$7,T${r}>'异常检测'!$E$8),"保留-结构冲击","正常")`]];
    }
    s.getRange(`W${r}`).formulas = [[`=IF(AND(A${r}>=2020,A${r}<=2022),1,0)`]];
    s.getRange(`X${r}`).formulas = [[`=IF(A${r}>=2019,1,0)`]];
    s.getRange(`Y${r}`).formulas = [[`=(K${r}-AVERAGE($K$4:$K$19))/STDEV.S($K$4:$K$19)`]];
    s.getRange(`Z${r}`).formulas = [[`=(M${r}-AVERAGE($M$4:$M$19))/STDEV.S($M$4:$M$19)`]];
    s.getRange(`AA${r}`).formulas = [[`=(O${r}-AVERAGE($O$4:$O$19))/STDEV.S($O$4:$O$19)`]];
    s.getRange(`AB${r}`).formulas = [[`=(Q${r}-AVERAGE($Q$4:$Q$19))/STDEV.S($Q$4:$Q$19)`]];
    s.getRange(`AC${r}`).formulas = [[`=LN(K${r})`]];
    s.getRange(`AD${r}`).formulas = [[`=LN(M${r})`]];
  }
  body(s.getRange("A4:AD19"));
  s.getRange("A4:A19").format.numberFormat = "0";
  for (const col of ["B", "C", "D", "E", "F", "K", "M", "O", "Q"]) s.getRange(`${col}4:${col}19`).format.numberFormat = "#,##0.00";
  s.getRange("S4:T19").format.numberFormat = "0.0%";
  s.getRange("Y4:AD19").format.numberFormat = "0.000";
  s.getRange("K4:Q19").conditionalFormats.addCustom("=AND(K4<>\"\",B4=\"\")", { fill: COLORS.paleYellow, font: { color: COLORS.navy } });
  s.getRange("U4:V19").conditionalFormats.add("containsText", { text: "结构冲击", format: { fill: COLORS.paleRed, font: { bold: true, color: "#9F2D20" } } });
  s.getRange("W4:X19").conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: COLORS.paleOrange, font: { bold: true } } });
  const processedTable = s.tables.add("A3:AD19", true, "ProcessedModelData");
  processedTable.style = "TableStyleMedium2";
  setWidths(s, {
    A: 9, B: 14, C: 14, D: 14, E: 12, F: 15, G: 19, H: 24, I: 24, J: 24,
    K: 14, L: 28, M: 16, N: 30, O: 13, P: 22, Q: 16, R: 27,
    S: 12, T: 12, U: 18, V: 18, W: 12, X: 14, Y: 10, Z: 10, AA: 10, AB: 12, AC: 11, AD: 11,
  });
  s.getRange("A3:AD19").format.rowHeight = 31;
  s.freezePanes.freezeRows(3);
  s.freezePanes.freezeColumns(1);
}

// -------------------- 异常检测 --------------------
{
  const s = wb.worksheets.getItem("异常检测");
  s.showGridLines = false;
  title(s, "异常检测：在同比序列上使用IQR，不删除真实冲击", "N");
  s.mergeCells("A2:N2");
  s.getRange("A2").values = [["理由：原序列带长期趋势，在水平值上做Z-score会把高增长年份误判为异常；因此以同比变化做IQR检测。|Z|≥3仅作辅助。2020、2022、2023分别对应疫情冲击、再度收缩和恢复反弹，均保留。"]];
  s.getRange("A2:N2").format = { fill: COLORS.paleYellow, wrapText: true, rowHeight: 42 };
  s.getRange("A4:B8").values = [["游客量同比Q1", null], ["游客量同比Q3", null], ["游客量同比IQR", null], ["游客量下界", null], ["游客量上界", null]];
  s.getRange("D4:E8").values = [["收入同比Q1", null], ["收入同比Q3", null], ["收入同比IQR", null], ["收入下界", null], ["收入上界", null]];
  s.getRange("B4").formulas = [["=(SMALL('预处理结果'!S5:S19,4)+SMALL('预处理结果'!S5:S19,5))/2"]];
  s.getRange("B5").formulas = [["=(SMALL('预处理结果'!S5:S19,11)+SMALL('预处理结果'!S5:S19,12))/2"]];
  s.getRange("B6").formulas = [["=B5-B4"]];
  s.getRange("B7").formulas = [["=B4-1.5*B6"]];
  s.getRange("B8").formulas = [["=B5+1.5*B6"]];
  s.getRange("E4").formulas = [["=(SMALL('预处理结果'!T5:T19,4)+SMALL('预处理结果'!T5:T19,5))/2"]];
  s.getRange("E5").formulas = [["=(SMALL('预处理结果'!T5:T19,11)+SMALL('预处理结果'!T5:T19,12))/2"]];
  s.getRange("E6").formulas = [["=E5-E4"]];
  s.getRange("E7").formulas = [["=E4-1.5*E6"]];
  s.getRange("E8").formulas = [["=E5+1.5*E6"]];
  s.getRange("A4:B8").format = { fill: COLORS.paleBlue, borders: { insideHorizontal: { style: "thin", color: COLORS.midGray } } };
  s.getRange("D4:E8").format = { fill: COLORS.paleGreen, borders: { insideHorizontal: { style: "thin", color: COLORS.midGray } } };
  s.getRange("B4:B8").format.numberFormat = "0.0%";
  s.getRange("E4:E8").format.numberFormat = "0.0%";
  s.getRange("A10:E10").values = [["年份", "游客量同比", "综合收入同比", "游客判定", "收入判定"]];
  header(s.getRange("A10:E10"));
  for (let i = 0; i < 16; i++) {
    const r = 11 + i;
    const p = 4 + i;
    s.getRange(`A${r}:E${r}`).formulas = [[
      `='预处理结果'!A${p}`,
      `=IF('预处理结果'!S${p}="","",'预处理结果'!S${p})`,
      `=IF('预处理结果'!T${p}="","",'预处理结果'!T${p})`,
      `=IF('预处理结果'!U${p}="","",'预处理结果'!U${p})`,
      `=IF('预处理结果'!V${p}="","",'预处理结果'!V${p})`,
    ]];
  }
  body(s.getRange("A11:E26"));
  s.getRange("B11:C26").format.numberFormat = "0.0%";
  s.getRange("D11:E26").conditionalFormats.add("containsText", { text: "结构冲击", format: { fill: COLORS.paleRed, font: { bold: true, color: "#9F2D20" } } });
  const anomalyTable = s.tables.add("A10:E26", true, "AnomalyResultTable");
  anomalyTable.style = "TableStyleMedium2";

  s.getRange("A30:D30").values = [["年份", "游客量同比", "IQR下界", "IQR上界"]];
  s.getRange("F30:I30").values = [["年份", "综合收入同比", "IQR下界", "IQR上界"]];
  header(s.getRange("A30:D30"));
  header(s.getRange("F30:I30"));
  for (let i = 0; i < 16; i++) {
    const rr = 31 + i;
    const src = 11 + i;
    s.getRange(`A${rr}:D${rr}`).formulas = [[`=A${src}`, `=B${src}`, "=$B$7", "=$B$8"]];
    s.getRange(`F${rr}:I${rr}`).formulas = [[`=A${src}`, `=C${src}`, "=$E$7", "=$E$8"]];
  }
  body(s.getRange("A31:D46"));
  body(s.getRange("F31:I46"));
  s.getRange("B31:D46").format.numberFormat = "0.0%";
  s.getRange("G31:I46").format.numberFormat = "0.0%";

  const visitorIqr = s.charts.add("line", s.getRange("A30:D46"));
  visitorIqr.title = "游客量同比与IQR判定边界";
  visitorIqr.hasLegend = true;
  visitorIqr.yAxis = { numberFormatCode: "0%" };
  visitorIqr.xAxis = { axisType: "textAxis" };
  visitorIqr.setPosition("G4", "N15");
  const incomeIqr = s.charts.add("line", s.getRange("F30:I46"));
  incomeIqr.title = "综合收入同比与IQR判定边界";
  incomeIqr.hasLegend = true;
  incomeIqr.yAxis = { numberFormatCode: "0%" };
  incomeIqr.xAxis = { axisType: "textAxis" };
  incomeIqr.setPosition("G17", "N28");
  setWidths(s, { A: 10, B: 16, C: 18, D: 20, E: 20, F: 10, G: 12, H: 12, I: 12, J: 12, K: 12, L: 12, M: 12, N: 12 });
  s.freezePanes.freezeRows(2);
}

// -------------------- 趋势图 --------------------
{
  const s = wb.worksheets.getItem("趋势图");
  s.showGridLines = false;
  title(s, "原始序列与清洗序列对照", "N");
  s.mergeCells("A2:N2");
  s.getRange("A2").values = [["图中原始序列保留缺口；清洗序列使用补充假设补全。2020—2022的变化不做平滑删除，正式模型必须加入疫情/恢复状态。"]];
  s.getRange("A2:N2").format = { fill: COLORS.paleYellow, wrapText: true, rowHeight: 34 };
  s.getRange("A4:E4").values = [["年份", "游客原始", "游客清洗", "收入原始", "收入清洗"]];
  header(s.getRange("A4:E4"));
  for (let i = 0; i < 16; i++) {
    const r = 5 + i;
    const p = 4 + i;
    s.getRange(`A${r}:E${r}`).formulas = [[
      `='预处理结果'!A${p}`,
      `=IF('预处理结果'!B${p}="","",'预处理结果'!B${p})`,
      `='预处理结果'!K${p}`,
      `=IF('预处理结果'!D${p}="","",'预处理结果'!D${p})`,
      `='预处理结果'!M${p}`,
    ]];
  }
  body(s.getRange("A5:E20"));
  s.getRange("B5:E20").format.numberFormat = "#,##0.00";
  const trendTable = s.tables.add("A4:E20", true, "TrendHelperTable");
  trendTable.style = "TableStyleMedium2";
  const vchart = s.charts.add("line", s.getRange("A4:C20"));
  vchart.title = "游客量：原始缺口与约束补全（万人次）";
  vchart.hasLegend = true;
  vchart.xAxis = { axisType: "textAxis" };
  vchart.yAxis = { numberFormatCode: "#,##0" };
  vchart.setPosition("G4", "N17");
  const ichart = s.charts.add("line", { chartType: "line", title: "旅游综合收入：原始缺口与约束补全（亿元）", hasLegend: true });
  const is1 = ichart.series.add("综合收入原始");
  is1.categoryFormula = "'趋势图'!$A$5:$A$20";
  is1.formula = "'趋势图'!$D$5:$D$20";
  const is2 = ichart.series.add("综合收入清洗");
  is2.categoryFormula = "'趋势图'!$A$5:$A$20";
  is2.formula = "'趋势图'!$E$5:$E$20";
  ichart.title = "旅游综合收入：原始缺口与约束补全（亿元）";
  ichart.hasLegend = true;
  ichart.xAxis = { axisType: "textAxis" };
  ichart.yAxis = { numberFormatCode: "#,##0" };
  ichart.setPosition("G19", "N32");
  setWidths(s, { A: 10, B: 15, C: 15, D: 15, E: 15, F: 3, G: 12, H: 12, I: 12, J: 12, K: 12, L: 12, M: 12, N: 12 });
  s.freezePanes.freezeRows(4);
}

// Threshold cells are now populated; rewrite the dependent flag formulas once
// so the artifact calculation cache reflects the final IQR bounds.
{
  const s = wb.worksheets.getItem("预处理结果");
  for (let i = 1; i < 16; i++) {
    const r = 4 + i;
    s.getRange(`U${r}`).formulas = [[`=IF(OR(S${r}<'异常检测'!$B$7,S${r}>'异常检测'!$B$8),"保留-结构冲击","正常")`]];
    s.getRange(`V${r}`).formulas = [[`=IF(OR(T${r}<'异常检测'!$E$7,T${r}>'异常检测'!$E$8),"保留-结构冲击","正常")`]];
  }
}

// -------------------- 数据补充 --------------------
{
  const s = wb.worksheets.getItem("数据补充");
  s.showGridLines = false;
  title(s, "必须补充、可选补充与相关指标覆盖", "G");
  s.mergeCells("A2:G2");
  s.getRange("A2").values = [["原则：必须补充决定模型是否可识别；可选补充只在能够稳定提高样本外预测时纳入。无官方依据的数据不生成伪精确值，必要时使用区间或标准化指数。"]];
  s.getRange("A2:G2").format = { fill: COLORS.paleYellow, wrapText: true, rowHeight: 38 };
  const headers = ["类别", "变量/缺口", "当前状态", "为什么需要", "推荐获取途径", "本轮生成假设", "不确定性处理"];
  s.getRange("A4:G4").values = [headers];
  header(s.getRange("A4:G4"));
  const rows = payload.supplementary_needs.map(r => [r.category, r.variable, r.current_gap, r.why, r.recommended_source, r.generated_assumption, r.uncertainty]);
  s.getRange(`A5:G${4 + rows.length}`).values = rows;
  body(s.getRange(`A5:G${4 + rows.length}`));
  s.getRange(`A5:A${4 + rows.length}`).conditionalFormats.add("containsText", { text: "必须", format: { fill: COLORS.paleRed, font: { bold: true, color: "#9F2D20" } } });
  s.getRange(`A5:A${4 + rows.length}`).conditionalFormats.add("containsText", { text: "可选", format: { fill: COLORS.paleGreen, font: { bold: true, color: "#176B55" } } });
  s.getRange(`A5:G${4 + rows.length}`).format.rowHeight = 92;
  const supplementTable = s.tables.add(`A4:G${4 + rows.length}`, true, "SupplementNeedsTable");
  supplementTable.style = "TableStyleMedium2";

  const start = 7 + rows.length;
  subtitle(s, `A${start}:G${start}`, "现有相关指标覆盖（只用于筛选，不代表可直接进入模型）", COLORS.paleBlue);
  s.getRange(`A${start + 1}:E${start + 1}`).values = [["指标", "单位", "观测数", "起始年", "结束年"]];
  header(s.getRange(`A${start + 1}:E${start + 1}`));
  const coverageRows = payload.related_coverage.map(r => [r.metric, r.unit, r.n, r.start_year, r.end_year]);
  s.getRange(`A${start + 2}:E${start + 1 + coverageRows.length}`).values = coverageRows;
  body(s.getRange(`A${start + 2}:E${start + 1 + coverageRows.length}`));
  s.getRange(`A${start + 2}:E${start + 1 + coverageRows.length}`).format.rowHeight = 34;
  s.getRange(`C${start + 2}:E${start + 1 + coverageRows.length}`).format.numberFormat = "0";
  setWidths(s, { A: 34, B: 34, C: 36, D: 36, E: 52, F: 46, G: 35 });
  s.freezePanes.freezeRows(4);
}

// Compact formula/error inspections before export.
const processedCheck = await wb.inspect({
  kind: "table",
  range: "预处理结果!A3:AD19",
  include: "values,formulas",
  tableMaxRows: 19,
  tableMaxCols: 30,
  maxChars: 20000,
});
await fs.writeFile(path.join(outputDir, "processed_check.ndjson"), processedCheck.ndjson, "utf8");

const formulaErrors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(outputDir, "formula_errors.ndjson"), formulaErrors.ndjson, "utf8");

for (const name of sheetNames) {
  const preview = await wb.render({ sheetName: name, autoCrop: "all", scale: 0.8, format: "png" });
  const safe = name.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(path.join(outputDir, `preview_${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);

console.log(JSON.stringify({ outputPath, sheets: sheetNames, modelRows: payload.model_schemes.length, processedRows: 16 }, null, 2));
