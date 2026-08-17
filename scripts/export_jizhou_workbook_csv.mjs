import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = path.resolve(process.argv[2] ?? process.cwd());
const workbookPath = path.join(
  repoRoot,
  "outputs",
  "01a00dfa-9be0-7871-990a-d6863466c203",
  "蓟州区旅游经济政府数据扩充版_2010-2025.xlsx",
);
const outputDir = path.join(
  repoRoot,
  "data",
  "jizhou_tourism_economy",
  "csv_exports",
);

const exports = [
  {
    sheet: "年度主表",
    range: "A1:O17",
    file: "01_annual_summary_2010_2025.csv",
    description: "2010—2025年度优选宽表，末两列比率以小数保存",
  },
  {
    sheet: "旅游观测",
    range: "A1:N47",
    file: "02_tourism_observations_2010_2025.csv",
    description: "旅游人次、直接收入和综合收入长表，保留口径与版本",
  },
  {
    sheet: "宏观观测",
    range: "A1:J56",
    file: "03_macro_observations_2010_2025.csv",
    description: "GDP与第三产业增加值长表，保留初值和修订值",
  },
  {
    sheet: "相关指标",
    range: "A1:H45",
    file: "04_related_indicators_2014_2025.csv",
    description: "社零、投资、收入、人口、财政等解释变量长表",
  },
  {
    sheet: "来源清单",
    range: "A1:G62",
    file: "05_source_catalog.csv",
    description: "来源ID、政府网址、本地原件路径和获取状态",
  },
  {
    sheet: "说明与缺口",
    range: "A1:D37",
    file: "06_data_notes_and_gaps.csv",
    description: "覆盖概览、口径规则、数据缺口和推荐过滤方式",
  },
  {
    sheet: "供给能力",
    range: "A1:K42",
    file: "07_tourism_supply_2012_2024.csv",
    description: "限额以上住宿餐饮供给长表及单列星级饭店锚点",
  },
  {
    sheet: "补缺线索",
    range: "A1:K8",
    file: "08_supplemental_gap_evidence_2016_2025.csv",
    description: "二手报道、代数约束和目标隐含基数，仅用于敏感性与校验",
  },
];

function escapeCsvCell(value) {
  if (value === null || value === undefined) return "";
  const text = value instanceof Date
    ? value.toISOString().slice(0, 10)
    : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows) {
  return `${rows.map((row) => row.map(escapeCsvCell).join(",")).join("\n")}\n`;
}

const workbook = await SpreadsheetFile.importXlsx(
  await FileBlob.load(workbookPath),
);
await fs.mkdir(outputDir, { recursive: true });

const manifestRows = [
  ["file", "source_sheet", "source_range", "data_rows", "columns", "description"],
];

for (const item of exports) {
  const sheet = workbook.worksheets.getItem(item.sheet);
  const values = sheet.getRange(item.range).values;
  const csv = toCsv(values);

  // Round-trip through artifact-tool so every exported CSV is parseable.
  const parsed = await Workbook.fromCSV(csv, { sheetName: item.sheet });
  const verification = await parsed.inspect({
    kind: "table",
    range: `${item.sheet}!A1:B2`,
    include: "values",
    tableMaxRows: 2,
    tableMaxCols: 2,
    maxChars: 1200,
  });
  if (!verification.ndjson) {
    throw new Error(`CSV round-trip verification failed: ${item.file}`);
  }

  await fs.writeFile(path.join(outputDir, item.file), csv, "utf8");
  manifestRows.push([
    item.file,
    item.sheet,
    item.range,
    values.length - 1,
    values[0]?.length ?? 0,
    item.description,
  ]);
  console.log(
    `${item.file}: ${values.length - 1} data rows x ${values[0]?.length ?? 0} columns`,
  );
}

const manifestCsv = toCsv(manifestRows);
await Workbook.fromCSV(manifestCsv, { sheetName: "manifest" });
await fs.writeFile(path.join(outputDir, "manifest.csv"), manifestCsv, "utf8");
console.log(`OUTPUT ${outputDir}`);
