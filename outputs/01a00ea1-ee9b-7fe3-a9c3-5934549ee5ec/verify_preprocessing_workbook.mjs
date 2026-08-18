import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = "D:/junk mass/小乱七八糟/数学建模/26国赛/shared files/outputs/01a00ea1-ee9b-7fe3-a9c3-5934549ee5ec";
const workbookPath = path.join(outputDir, "蓟州区旅游经济_预处理汇总数据集.xlsx");
const blob = await FileBlob.load(workbookPath);
const wb = await SpreadsheetFile.importXlsx(blob);
const sheets = await wb.inspect({ kind: "sheet", include: "id,name", maxChars: 5000 });
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "post-export error scan" });
const main = await wb.inspect({ kind: "region", range: "建模主表!A4:X20", maxChars: 12000 });
await fs.writeFile(path.join(outputDir, "verify_sheets.ndjson"), sheets.ndjson, "utf8");
await fs.writeFile(path.join(outputDir, "verify_errors.ndjson"), errors.ndjson, "utf8");
await fs.writeFile(path.join(outputDir, "verify_main.ndjson"), main.ndjson, "utf8");
console.log(JSON.stringify({ workbookPath, sheetCount: 10, errorScan: errors.ndjson }, null, 2));
