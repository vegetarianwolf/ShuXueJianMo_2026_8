import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const dataDir = path.join(repoRoot, "data", "jizhou_tourism_economy");
const metadataDir = path.join(repoRoot, "data", "metadata");
const outputDir = path.join(
  repoRoot,
  "outputs",
  "01a00dfa-9be0-7871-990a-d6863466c203",
);
const renderDir = "/private/tmp/jizhou-spreadsheet-01/rendered-government";
const outputPath = path.join(
  outputDir,
  "蓟州区旅游经济政府数据扩充版_2010-2025.xlsx",
);

const [
  summaryCsv,
  tourismCsv,
  macroCsv,
  relatedCsv,
  supplyCsv,
  gapEvidenceCsv,
  sourcesCsv,
] =
  await Promise.all([
    fs.readFile(
      path.join(dataDir, "official_annual_summary_2010_2025.csv"),
      "utf8",
    ),
    fs.readFile(
      path.join(dataDir, "official_tourism_observations_2010_2025.csv"),
      "utf8",
    ),
    fs.readFile(
      path.join(dataDir, "official_macro_observations_2010_2025.csv"),
      "utf8",
    ),
    fs.readFile(
      path.join(dataDir, "official_related_observations_2014_2025.csv"),
      "utf8",
    ),
    fs.readFile(
      path.join(dataDir, "official_tourism_supply_observations_2012_2024.csv"),
      "utf8",
    ),
    fs.readFile(
      path.join(dataDir, "supplemental_gap_evidence_2016_2025.csv"),
      "utf8",
    ),
    fs.readFile(path.join(metadataDir, "sources.csv"), "utf8"),
  ]);

const workbook = await Workbook.fromCSV(summaryCsv, { sheetName: "年度主表" });
await workbook.fromCSV(tourismCsv, { sheetName: "旅游观测" });
await workbook.fromCSV(macroCsv, { sheetName: "宏观观测" });
await workbook.fromCSV(relatedCsv, { sheetName: "相关指标" });
await workbook.fromCSV(supplyCsv, { sheetName: "供给能力" });
await workbook.fromCSV(gapEvidenceCsv, { sheetName: "补缺线索" });
await workbook.fromCSV(sourcesCsv, { sheetName: "来源清单" });
const notes = workbook.worksheets.add("说明与缺口");

const navy = "#17365D";
const blue = "#D9EAF7";
const paleBlue = "#EAF3F8";
const paleGreen = "#E2F0D9";
const paleYellow = "#FFF2CC";
const paleRed = "#FCE4D6";
const lightGray = "#F2F2F2";
const border = "#C9D2DC";
const textColor = "#1F2937";

function applyBodyStyle(sheet, rangeAddress) {
  const range = sheet.getRange(rangeAddress);
  range.format.font = { name: "Aptos", size: 10, color: textColor };
  range.format.borders = { preset: "all", style: "thin", color: border };
  range.format.verticalAlignment = "center";
}

function applyHeaderStyle(sheet, rangeAddress) {
  const range = sheet.getRange(rangeAddress);
  range.format.fill = navy;
  range.format.font = {
    name: "Aptos",
    size: 10,
    bold: true,
    color: "#FFFFFF",
  };
  range.format.wrapText = true;
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.rowHeight = 34;
}

function setWidth(sheet, rangeAddress, width) {
  sheet.getRange(rangeAddress).format.columnWidth = width;
}

const main = workbook.worksheets.getItem("年度主表");
main.showGridLines = false;
main.freezePanes.freezeRows(1);
main.getRange("A1:O17").values;
main.getRange("A1:M1").values = [[
  "年份",
  "优选游客量（万人次）",
  "游客值状态",
  "优选直接收入（亿元）",
  "直接收入状态",
  "优选综合收入（亿元）",
  "综合收入状态",
  "优选GDP（亿元）",
  "GDP状态",
  "优选第三产业增加值（亿元）",
  "三产状态",
  "来源ID",
  "质量说明",
]];
main.getRange("N1:O1").values = [["综合收入/GDP", "第三产业/GDP"]];
main.getRange("N2").formulas = [["=IF(OR(F2=\"\",H2=\"\"),\"\",F2/H2)"]];
main.getRange("N2:N17").fillDown();
main.getRange("O2").formulas = [["=IF(OR(J2=\"\",H2=\"\"),\"\",J2/H2)"]];
main.getRange("O2:O17").fillDown();
applyBodyStyle(main, "A1:O17");
applyHeaderStyle(main, "A1:O1");
main.getRange("A2:A17").format.numberFormat = "0";
for (const col of ["B", "D", "F", "H", "J"]) {
  main.getRange(`${col}2:${col}17`).format.numberFormat = "#,##0.0000";
}
main.getRange("N2:O17").format.numberFormat = "0.0%";
main.getRange("N2:O17").format.fill = paleGreen;
main.getRange("C2:C17").format.fill = lightGray;
main.getRange("E2:E17").format.fill = lightGray;
main.getRange("G2:G17").format.fill = lightGray;
main.getRange("I2:I17").format.fill = lightGray;
main.getRange("K2:K17").format.fill = lightGray;
for (const row of [12, 13, 14, 17]) {
  main.getRange(`B${row}:G${row}`).format.fill = paleYellow;
}
main.getRange("B12:G12").format.fill = paleRed;
main.getRange("B14:G14").format.fill = paleRed;
main.getRange("B17:G17").format.fill = paleRed;
main.getRange("L2:M17").format.wrapText = true;
main.getRange("A2:K17").format.rowHeight = 23;
setWidth(main, "A1:A17", 9);
setWidth(main, "B1:B17", 18);
setWidth(main, "C1:C17", 20);
setWidth(main, "D1:D17", 18);
setWidth(main, "E1:E17", 21);
setWidth(main, "F1:F17", 19);
setWidth(main, "G1:G17", 23);
setWidth(main, "H1:H17", 16);
setWidth(main, "I1:I17", 20);
setWidth(main, "J1:J17", 24);
setWidth(main, "K1:K17", 21);
setWidth(main, "L1:L17", 42);
setWidth(main, "M1:M17", 58);
setWidth(main, "N1:O17", 16);

const tourism = workbook.worksheets.getItem("旅游观测");
tourism.showGridLines = false;
tourism.freezePanes.freezeRows(1);
tourism.getRange("A1:N1").values = [[
  "起始年",
  "结束年",
  "周期类型",
  "指标",
  "数值",
  "单位",
  "值状态",
  "指标口径",
  "来源ID",
  "证据等级",
  "发布日期",
  "获取日期",
  "来源URL",
  "备注",
]];
applyBodyStyle(tourism, "A1:N47");
applyHeaderStyle(tourism, "A1:N1");
tourism.getRange("A2:B47").format.numberFormat = "0";
tourism.getRange("E2:E47").format.numberFormat = "#,##0.0000";
tourism.getRange("J2:J47").format.numberFormat = "0";
tourism.getRange("G2:G47").format.fill = lightGray;
tourism.getRange("J2:J47").format.fill = paleBlue;
tourism.getRange("M2:N47").format.wrapText = true;
setWidth(tourism, "A1:B47", 10);
setWidth(tourism, "C1:C47", 16);
setWidth(tourism, "D1:D47", 30);
setWidth(tourism, "E1:E47", 14);
setWidth(tourism, "F1:F47", 15);
setWidth(tourism, "G1:G47", 29);
setWidth(tourism, "H1:H47", 23);
setWidth(tourism, "I1:I47", 31);
setWidth(tourism, "J1:J47", 11);
setWidth(tourism, "K1:L47", 14);
setWidth(tourism, "M1:M47", 66);
setWidth(tourism, "N1:N47", 58);

const macro = workbook.worksheets.getItem("宏观观测");
macro.showGridLines = false;
macro.freezePanes.freezeRows(1);
macro.getRange("A1:J1").values = [[
  "年份",
  "指标",
  "数值",
  "单位",
  "值状态",
  "版本",
  "来源ID",
  "证据等级",
  "来源URL",
  "备注",
]];
applyBodyStyle(macro, "A1:J56");
applyHeaderStyle(macro, "A1:J1");
macro.getRange("A2:A56").format.numberFormat = "0";
macro.getRange("C2:C56").format.numberFormat = "#,##0.0000";
macro.getRange("H2:H56").format.numberFormat = "0";
macro.getRange("E2:F56").format.fill = lightGray;
macro.getRange("H2:H56").format.fill = paleBlue;
macro.getRange("I2:J56").format.wrapText = true;
setWidth(macro, "A1:A56", 10);
setWidth(macro, "B1:B56", 29);
setWidth(macro, "C1:C56", 16);
setWidth(macro, "D1:D56", 14);
setWidth(macro, "E1:E56", 31);
setWidth(macro, "F1:F56", 28);
setWidth(macro, "G1:G56", 31);
setWidth(macro, "H1:H56", 11);
setWidth(macro, "I1:I56", 66);
setWidth(macro, "J1:J56", 58);

const related = workbook.worksheets.getItem("相关指标");
related.showGridLines = false;
related.freezePanes.freezeRows(1);
related.getRange("A1:H1").values = [[
  "年份",
  "指标",
  "数值",
  "单位",
  "值状态",
  "来源ID",
  "来源URL",
  "备注",
]];
applyBodyStyle(related, "A1:H45");
applyHeaderStyle(related, "A1:H1");
related.getRange("A2:A45").format.numberFormat = "0";
related.getRange("C2:C45").format.numberFormat = "#,##0.0000";
related.getRange("E2:E45").format.fill = lightGray;
related.getRange("G2:H45").format.wrapText = true;
setWidth(related, "A1:A45", 10);
setWidth(related, "B1:B45", 43);
setWidth(related, "C1:C45", 16);
setWidth(related, "D1:D45", 18);
setWidth(related, "E1:E45", 18);
setWidth(related, "F1:F45", 31);
setWidth(related, "G1:G45", 66);
setWidth(related, "H1:H45", 58);

const supply = workbook.worksheets.getItem("供给能力");
supply.showGridLines = false;
supply.freezePanes.freezeRows(1);
supply.getRange("A1:K1").values = [[
  "年份",
  "指标",
  "数值",
  "单位",
  "值状态",
  "指标范围",
  "版本",
  "来源ID",
  "证据等级",
  "来源URL",
  "备注",
]];
applyBodyStyle(supply, "A1:K42");
applyHeaderStyle(supply, "A1:K1");
supply.getRange("A2:A42").format.numberFormat = "0";
supply.getRange("C2:C42").format.numberFormat = "#,##0.0000";
supply.getRange("I2:I42").format.numberFormat = "0";
supply.getRange("E2:G42").format.fill = lightGray;
supply.getRange("I2:I42").format.fill = paleBlue;
supply.getRange("J2:K42").format.wrapText = true;
setWidth(supply, "A1:A42", 10);
setWidth(supply, "B1:B42", 22);
setWidth(supply, "C1:C42", 16);
setWidth(supply, "D1:D42", 14);
setWidth(supply, "E1:E42", 24);
setWidth(supply, "F1:F42", 43);
setWidth(supply, "G1:G42", 27);
setWidth(supply, "H1:H42", 31);
setWidth(supply, "I1:I42", 11);
setWidth(supply, "J1:J42", 66);
setWidth(supply, "K1:K42", 58);

const gapEvidence = workbook.worksheets.getItem("补缺线索");
gapEvidence.showGridLines = false;
gapEvidence.freezePanes.freezeRows(1);
gapEvidence.getRange("A1:K1").values = [[
  "起始年",
  "结束年",
  "指标",
  "数值",
  "单位",
  "证据角色",
  "来源ID",
  "推导式/原文",
  "模型用途",
  "来源URL",
  "备注",
]];
applyBodyStyle(gapEvidence, "A1:K8");
applyHeaderStyle(gapEvidence, "A1:K1");
gapEvidence.getRange("A2:B8").format.numberFormat = "0";
gapEvidence.getRange("D2:D8").format.numberFormat = "#,##0.0000";
gapEvidence.getRange("F2:F8").format.fill = paleYellow;
gapEvidence.getRange("I2:I8").format.fill = paleRed;
gapEvidence.getRange("H2:K8").format.wrapText = true;
setWidth(gapEvidence, "A1:B8", 10);
setWidth(gapEvidence, "C1:C8", 37);
setWidth(gapEvidence, "D1:D8", 16);
setWidth(gapEvidence, "E1:E8", 17);
setWidth(gapEvidence, "F1:F8", 27);
setWidth(gapEvidence, "G1:G8", 34);
setWidth(gapEvidence, "H1:H8", 30);
setWidth(gapEvidence, "I1:I8", 18);
setWidth(gapEvidence, "J1:J8", 72);
setWidth(gapEvidence, "K1:K8", 64);

const sources = workbook.worksheets.getItem("来源清单");
sources.showGridLines = false;
sources.freezePanes.freezeRows(1);
sources.getRange("A1:G1").values = [[
  "来源ID",
  "主题",
  "来源名称",
  "URL",
  "仓库路径",
  "获取状态",
  "说明",
]];
applyBodyStyle(sources, "A1:G62");
applyHeaderStyle(sources, "A1:G1");
sources.getRange("F2:F62").format.fill = paleBlue;
sources.getRange("C2:G62").format.wrapText = true;
setWidth(sources, "A1:A62", 34);
setWidth(sources, "B1:B62", 24);
setWidth(sources, "C1:C62", 42);
setWidth(sources, "D1:D62", 72);
setWidth(sources, "E1:E62", 62);
setWidth(sources, "F1:F62", 34);
setWidth(sources, "G1:G62", 66);

notes.showGridLines = false;
notes.mergeCells("A1:D1");
notes.mergeCells("A2:D2");
notes.getRange("A1").values = [["蓟州区旅游经济政府数据扩充版"]];
notes.getRange("A2").values = [[
  "整理日期 2026-08-17｜官方原件、授权数据库核验、目标、累计与推导值已分开保存",
]];
notes.getRange("A4:D4").values = [["覆盖概览", "有值年份", "总年份", "说明"]];
notes.getRange("A5:A8").values = [
  ["优选游客量"],
  ["优选综合收入"],
  ["优选GDP"],
  ["优选第三产业增加值"],
];
notes.getRange("B5").formulas = [["=COUNT('年度主表'!B2:B17)"]];
notes.getRange("B5:B8").fillDown();
notes.getRange("B6").formulas = [["=COUNT('年度主表'!F2:F17)"]];
notes.getRange("B7").formulas = [["=COUNT('年度主表'!H2:H17)"]];
notes.getRange("B8").formulas = [["=COUNT('年度主表'!J2:J17)"]];
notes.getRange("C5:C8").values = [[16], [16], [16], [16]];
notes.getRange("D5:D8").values = [
  ["含约数和已下线官方页缓存值"],
  ["含2010反推值和2015辅助附件值"],
  ["16年全覆盖；2012—2020已用天津统计年鉴回列值补齐"],
  ["16年全覆盖；2012—2020已用天津统计年鉴回列值补齐"],
];
notes.getRange("A10:D10").values = [["核心口径", "规则", "影响", "建议"]];
notes.getRange("A11:D15").values = [
  ["直接收入 vs 综合收入", "两个独立指标", "旧口径约1:5", "不得拼成一条收入序列"],
  ["实际 vs 目标", "实际优先", "目标通常更高", "目标仅用于情景假设"],
  ["年度 vs 五年累计", "累计值不拆年", "2021—2025累计792.5亿元", "只作总量约束"],
  ["初值 vs 修订值", "后版年鉴优先", "同年GDP/三产可变化", "长表保留所有版本"],
  ["空白值", "空白不等于0", "疫情期缺口集中", "显式建模缺失机制"],
];
notes.getRange("A17:D17").values = [["未解决缺口", "年份", "当前可用信息", "处理"]];
notes.getRange("A18:D23").values = [
  ["年度游客量/综合收入", "2020", "未找到可靠公开年度总量", "保留空白"],
  ["精确游客量", "2021", "仅找到综合收入110亿元", "保留游客空白"],
  ["年度游客量/实际综合收入", "2022", "仅有累计余量和目标隐含基数", "不得回填推导值"],
  ["官方年度旅游实际", "2025", "二手报道2691万人次且收入突破200亿元", "仅作敏感性证据"],
  ["旅游综合收入", "2016", "仅找到直接收入", "不得按5倍自动冒充实测"],
  ["连续政策/行为/全域供给", "2012—2025", "已有道路、财政和限上住宿锚点", "补齐连续同口径序列"],
];
notes.getRange("A25:D25").values = [["证据等级", "含义", "是否有本地原件", "推荐用途"]];
notes.getRange("A26:D29").values = [
  [1, "可直接复核的政府HTML/PDF/XLS/RAR", "是", "主模型"],
  [2, "授权数据库展示原年鉴值或官方索引复核", "浏览器核验记录", "宏观主序列/稳健性"],
  [3, "政府站辅助附件而非统计公报", "附件当前不可直取", "补充说明"],
  [4, "二手媒体转述或阈值", "无官方原表", "仅敏感性/约束"],
];
notes.getRange("A30:D30").values = [["建模过滤", "保留", "排除/单列", "原因"]];
notes.getRange("A31:D34").values = [
  ["旅游主样本", "annual + observed/revised", "target/aggregate", "避免把计划当实际"],
  ["严格主样本", "evidence_tier=1", "tier 2/3及推导值", "控制证据质量"],
  ["宏观主序列", "final/revised/yearbook", "provisional", "使用最新版核算"],
  ["制度断点", "设置断点变量", "直接连线解释", "2019 GDP和2021社零口径跳变"],
];
notes.getRange("A36:D36").values = [[
  "来源入口",
  "蓟州区政府数据发布",
  "南开大学图书馆数据库目录",
  "中经网统计数据库",
]];
notes.getRange("A37:D37").values = [[
  "URL",
  "https://www.tjjz.gov.cn/zwgk/sjfb/",
  "https://lib.nankai.edu.cn/sjkjs_15469/list.htm",
  "https://ceidata.cei.cn/db",
]];
applyBodyStyle(notes, "A1:D37");
notes.getRange("A1:D1").format.fill = navy;
notes.getRange("A1:D1").format.font = {
  name: "Aptos Display",
  size: 18,
  bold: true,
  color: "#FFFFFF",
};
notes.getRange("A1:D1").format.horizontalAlignment = "center";
notes.getRange("A1:D1").format.rowHeight = 34;
notes.getRange("A2:D2").format.fill = blue;
notes.getRange("A2:D2").format.font = {
  name: "Aptos",
  size: 10,
  italic: true,
  color: navy,
};
notes.getRange("A2:D2").format.horizontalAlignment = "center";
for (const headerRow of [4, 10, 17, 25, 30, 36]) {
  applyHeaderStyle(notes, `A${headerRow}:D${headerRow}`);
}
notes.getRange("B5:B8").format.fill = paleGreen;
notes.getRange("B5:C8").format.numberFormat = "0";
notes.getRange("A18:D23").format.fill = paleYellow;
notes.getRange("A18:D21").format.fill = paleRed;
notes.getRange("A1:D37").format.wrapText = true;
notes.getRange("A1:D37").format.verticalAlignment = "center";
setWidth(notes, "A1:A37", 28);
setWidth(notes, "B1:B37", 35);
setWidth(notes, "C1:C37", 39);
setWidth(notes, "D1:D37", 43);
notes.freezePanes.freezeRows(2);

const mainInspect = await workbook.inspect({
  kind: "table",
  range: "年度主表!A1:O17",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 15,
  maxChars: 10000,
});
console.log(`MAIN_INSPECT\n${mainInspect.ndjson}`);

const supplementalInspect = await workbook.inspect({
  kind: "table",
  range: "供给能力!A1:K42",
  include: "values,formulas",
  tableMaxRows: 45,
  tableMaxCols: 11,
  maxChars: 24000,
});
console.log(`SUPPLY_INSPECT\n${supplementalInspect.ndjson}`);

const gapInspect = await workbook.inspect({
  kind: "table",
  range: "补缺线索!A1:K8",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 11,
  maxChars: 12000,
});
console.log(`GAP_INSPECT\n${gapInspect.ndjson}`);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 4000,
});
console.log(`ERROR_SCAN\n${errorScan.ndjson}`);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(renderDir, { recursive: true });
const renderTargets = [
  ["年度主表", "A1:O17"],
  ["旅游观测", "A1:N22"],
  ["宏观观测", "A1:J30"],
  ["相关指标", "A1:H30"],
  ["供给能力", "A1:K24"],
  ["补缺线索", "A1:K8"],
  ["来源清单", "A1:G20"],
  ["说明与缺口", "A1:D37"],
];
for (const [sheetName, range] of renderTargets) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale: 1.25,
    format: "png",
  });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(renderDir, `${sheetName}.png`), bytes);
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const artifactInspect = await workbook.inspect({
  kind: "workbook,sheet,table,region,formula,definedName,drawing,thread,computedStyle",
  include: "values,formulas,styles",
  tableMaxRows: 100,
  tableMaxCols: 20,
  maxChars: 2000000,
});
await fs.writeFile(`${outputPath}.inspect.ndjson`, `${artifactInspect.ndjson}\n`, "utf8");
console.log(`OUTPUT ${outputPath}`);
