#!/usr/bin/env python3
"""Build the auditable, branch-unified data layer used by model benchmarks.

The target labels always come from ``main``'s canonical annual CSV.  Data from
other audited tips may provide auxiliary context or sensitivity records, but
never replaces a canonical target and never becomes a test label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PINNED_AUDITED_TIPS = {
    "main": "579333d9746a9cd0e12877bbcfbac4721f3ed33b",
    "origin/111": "71c7ddcd287e3713f05e50651a8d52508daf5b89",
    "origin/邱志烨-数据搜索": "47cc10d6c1333bfee4a121573566250c59374415",
    "codex/jizhou-tourism-modeling": "66e27eb5a29bbf3abd51dc2dc1af4b8e41fc349c",
    "codex/jizhou-tourism-ml": "3709084fc84614223ee00979494aa82b458296fe",
}
AUDITED_REFS = tuple(PINNED_AUDITED_TIPS)
CANONICAL_REF = "main"
CANONICAL_PATH = (
    "data/jizhou_tourism_economy/official_annual_summary_2010_2025.csv"
)
GDP_REF = "origin/111"
GDP_PATH = "data/tianjin_gdp_2010_2025/天津市GDP数据集_2010-2025.xlsx"
TRACEABLE_PATH = "111完整示例/蓟州区旅游经济整合可追溯数据集_2010-2025.xlsx"
QIU_REF = "origin/邱志烨-数据搜索"
QIU_IMPUTATION_PATH = "data/邱志烨-数据处理与搜集/蓟州区旅游经济_缺失补全记录.csv"
DEFAULT_CUTOFF_YEAR = 2019

BENCHMARK_FIELDS = (
    "split_id",
    "split",
    "metric",
    "year",
    "value",
    "unit",
    "status",
    "source_ids",
    "quality_note",
    "is_observed",
    "cutoff_year",
)
METRICS = (
    {
        "metric": "tourist_visits",
        "value_column": "preferred_visitor_10k_persons",
        "status_column": "visitor_status",
        "unit": "10k_persons",
    },
    {
        "metric": "tourism_comprehensive_income",
        "value_column": "preferred_comprehensive_income_100m_cny",
        "status_column": "comprehensive_status",
        "unit": "100m_cny",
    },
)
SENSITIVITY_METRICS = {
    "游客量": "tourist_visits",
    "旅游综合收入": "tourism_comprehensive_income",
}

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF_RE = re.compile(r"^([A-Z]+)([0-9]+)$")


class UnifiedDataError(RuntimeError):
    """Raised when a source asset no longer matches its audited contract."""


def _git(repo_root: Path, args: Sequence[str], *, text: bool = False) -> bytes | str:
    command = ["git", "-C", str(repo_root), *args]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise UnifiedDataError(f"git command failed ({' '.join(args)}): {message}")
    if text:
        return result.stdout.decode("utf-8")
    return result.stdout


def _git_blob(repo_root: Path, ref: str, path: str) -> bytes:
    try:
        commit = PINNED_AUDITED_TIPS[ref]
    except KeyError as error:
        raise UnifiedDataError(f"unrecognized audited ref label: {ref}") from error
    return bytes(_git(repo_root, ["show", f"{commit}:{path}"]))


def _validate_pinned_commits(repo_root: Path) -> dict[str, str]:
    commits = dict(PINNED_AUDITED_TIPS)
    for label, commit in commits.items():
        resolved = str(
            _git(repo_root, ["rev-parse", f"{commit}^{{commit}}"], text=True)
        ).strip()
        if resolved != commit:
            raise UnifiedDataError(
                f"pinned commit for {label} resolved unexpectedly: {resolved}"
            )
    return commits


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_blob(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _column_number(cell_reference: str) -> int:
    match = _CELL_REF_RE.match(cell_reference)
    if not match:
        raise UnifiedDataError(f"invalid worksheet cell reference: {cell_reference!r}")
    number = 0
    for letter in match.group(1):
        number = number * 26 + ord(letter) - ord("A") + 1
    return number


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
    return values


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> str | int | float | bool | None:
    cell_type = cell.get("t", "n")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t"))

    value_node = cell.find(f"{{{_MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def _xlsx_sheet_rows(content: bytes, sheet_name: str) -> list[tuple[int, dict[int, Any]]]:
    """Read worksheet values using only the Python standard library.

    The audited workbooks omit worksheet ``dimension`` nodes, which makes
    openpyxl's read-only dimensions empty.  Parsing the OOXML rows directly is
    both dependency-free and faithful to cached formula values.
    """

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id: str | None = None
        for sheet in workbook.iter(f"{{{_MAIN_NS}}}sheet"):
            if sheet.get("name") == sheet_name:
                relationship_id = sheet.get(f"{{{_OFFICE_REL_NS}}}id")
                break
        if not relationship_id:
            raise UnifiedDataError(f"worksheet {sheet_name!r} not found")

        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target: str | None = None
        for relationship in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship"):
            if relationship.get("Id") == relationship_id:
                target = relationship.get("Target")
                break
        if not target:
            raise UnifiedDataError(f"worksheet relationship missing for {sheet_name!r}")
        worksheet_path = target.lstrip("/")
        if not worksheet_path.startswith("xl/"):
            worksheet_path = f"xl/{worksheet_path}"

        root = ET.fromstring(archive.read(worksheet_path))
        shared = _shared_strings(archive)
        rows: list[tuple[int, dict[int, Any]]] = []
        for row in root.iter(f"{{{_MAIN_NS}}}row"):
            row_number = int(row.get("r", str(len(rows) + 1)))
            values: dict[int, Any] = {}
            for cell in row.findall(f"{{{_MAIN_NS}}}c"):
                reference = cell.get("r")
                if reference:
                    values[_column_number(reference)] = _cell_value(cell, shared)
            rows.append((row_number, values))
        return rows


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def _validate_header(
    rows: Sequence[tuple[int, dict[int, Any]]],
    row_number: int,
    expected: Sequence[str],
) -> None:
    row = next((values for number, values in rows if number == row_number), None)
    actual = tuple(row.get(index, "") if row else "" for index in range(1, len(expected) + 1))
    if actual != tuple(expected):
        raise UnifiedDataError(
            f"worksheet header changed at row {row_number}: expected {tuple(expected)!r}, "
            f"got {actual!r}"
        )


def _canonical_rows(content: bytes) -> list[dict[str, str]]:
    rows = _read_csv_blob(content)
    required = {
        "year",
        "preferred_visitor_10k_persons",
        "visitor_status",
        "preferred_comprehensive_income_100m_cny",
        "comprehensive_status",
        "source_ids",
        "quality_note",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - (set(rows[0]) if rows else set()))
        raise UnifiedDataError(f"canonical annual CSV is missing columns: {missing}")
    years = [int(row["year"]) for row in rows]
    if years != list(range(2010, 2026)):
        raise UnifiedDataError(f"canonical annual years must be 2010-2025, got {years}")
    return rows


def _is_observed(status: str) -> bool:
    return status.strip().lower().startswith("observed")


def _benchmark_rows(
    canonical: Sequence[dict[str, str]],
    cutoff_year: int,
    *,
    split_id: str | None = None,
    test_max_year: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_id = split_id or f"chronological_cutoff_{cutoff_year}"
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for source in canonical:
        year = int(source["year"])
        for metric in METRICS:
            value = source[metric["value_column"]].strip()
            status = source[metric["status_column"]].strip()
            if not value:
                continue
            observed = _is_observed(status)
            split = "train" if year <= cutoff_year else "test"
            if split == "test" and test_max_year is not None and year > test_max_year:
                continue
            if split == "test" and not observed:
                continue
            row = {
                "split_id": split_id,
                "split": split,
                "metric": metric["metric"],
                "year": str(year),
                "value": value,
                "unit": metric["unit"],
                "status": status,
                "source_ids": source["source_ids"],
                "quality_note": source["quality_note"],
                "is_observed": str(observed).lower(),
                "cutoff_year": str(cutoff_year),
            }
            (train if split == "train" else test).append(row)
    return train, test


def _rolling_origin_rows(
    canonical: Sequence[dict[str, str]],
    *,
    min_train_records: int = 5,
    outer_test_max_year: int = 2023,
) -> list[dict[str, str]]:
    split_id = (
        f"rolling_origin_min{min_train_records}_max{outer_test_max_year}"
    )
    output: list[dict[str, str]] = []
    for metric in METRICS:
        records: list[dict[str, str]] = []
        for source in canonical:
            year = int(source["year"])
            value = source[metric["value_column"]].strip()
            status = source[metric["status_column"]].strip()
            if value and year <= outer_test_max_year:
                records.append(
                    {
                        "metric": metric["metric"],
                        "year": str(year),
                        "value": value,
                        "unit": metric["unit"],
                        "status": status,
                        "source_ids": source["source_ids"],
                        "quality_note": source["quality_note"],
                        "is_observed": str(_is_observed(status)).lower(),
                    }
                )
        for test_index in range(min_train_records, len(records)):
            test_record = records[test_index]
            if test_record["is_observed"] != "true":
                continue
            fold_id = f"{metric['metric']}_test_{test_record['year']}"
            for role, record in [
                *(("train", train_record) for train_record in records[:test_index]),
                ("test", test_record),
            ]:
                output.append(
                    {
                        "split_id": split_id,
                        "fold_id": fold_id,
                        "metric": record["metric"],
                        "role": role,
                        "year": record["year"],
                        "value": record["value"],
                        "unit": record["unit"],
                        "status": record["status"],
                        "source_ids": record["source_ids"],
                        "quality_note": record["quality_note"],
                        "is_observed": record["is_observed"],
                        "min_train_records": str(min_train_records),
                        "outer_test_max_year": str(outer_test_max_year),
                    }
                )
    return output


def _extract_gdp(repo_root: Path, commits: dict[str, str]) -> list[dict[str, str]]:
    rows = _xlsx_sheet_rows(_git_blob(repo_root, GDP_REF, GDP_PATH), "GDP数据")
    _validate_header(
        rows,
        4,
        ("年份", "地区生产总值（亿元）", "实际GDP增速", "来源说明"),
    )
    output: list[dict[str, str]] = []
    for row_number, values in rows:
        year = values.get(1)
        if row_number < 5 or not isinstance(year, (int, float)):
            continue
        year_number = int(year)
        if not 2010 <= year_number <= 2025:
            continue
        output.append(
            {
                "year": str(year_number),
                "tianjin_gdp_100m_cny": _format_number(values.get(2)),
                "tianjin_real_gdp_growth_rate": _format_number(values.get(3)),
                "status": "official_observed",
                "is_observed": "true",
                "source_note": _format_number(values.get(4)),
                "source_ref": GDP_REF,
                "source_commit": commits[GDP_REF],
                "source_path": GDP_PATH,
                "source_sheet": "GDP数据",
                "source_row": str(row_number),
            }
        )
    years = [int(row["year"]) for row in output]
    if years != list(range(2010, 2026)):
        raise UnifiedDataError(f"Tianjin GDP workbook years changed: {years}")
    if any(
        not row["tianjin_gdp_100m_cny"]
        or not row["tianjin_real_gdp_growth_rate"]
        for row in output
    ):
        raise UnifiedDataError("Tianjin GDP workbook contains missing GDP or growth values")
    return output


def _extract_tianjin_tourism(
    repo_root: Path, commits: dict[str, str]
) -> list[dict[str, str]]:
    rows = _xlsx_sheet_rows(
        _git_blob(repo_root, GDP_REF, TRACEABLE_PATH), "天津市旅游基准"
    )
    _validate_header(
        rows,
        3,
        (
            "年份",
            "天津国内游客(万人次)",
            "天津国内旅游收入(亿元)",
            "蓟州游客(万人次)",
            "蓟州综合收入(亿元)",
            "蓟州游客占天津",
            "蓟州收入占天津",
        ),
    )
    output: list[dict[str, str]] = []
    for row_number, values in rows:
        year = values.get(1)
        if row_number < 4 or not isinstance(year, (int, float)):
            continue
        output.append(
            {
                "year": str(int(year)),
                "tianjin_domestic_visitors_10k_persons": _format_number(values.get(2)),
                "tianjin_domestic_tourism_income_100m_cny": _format_number(values.get(3)),
                "jizhou_visitors_10k_persons": _format_number(values.get(4)),
                "jizhou_comprehensive_income_100m_cny": _format_number(values.get(5)),
                "jizhou_visitor_share": _format_number(values.get(6)),
                "jizhou_income_share": _format_number(values.get(7)),
                "status": "official_observed",
                "is_observed": "true",
                "source_ref": GDP_REF,
                "source_commit": commits[GDP_REF],
                "source_path": TRACEABLE_PATH,
                "source_sheet": "天津市旅游基准",
                "source_row": str(row_number),
            }
        )
    years = [int(row["year"]) for row in output]
    if years != [2020, 2021, 2023, 2024]:
        raise UnifiedDataError(f"Tianjin tourism benchmark rows changed: {years}")
    if any(
        not row["tianjin_domestic_visitors_10k_persons"]
        or not row["tianjin_domestic_tourism_income_100m_cny"]
        for row in output
    ):
        raise UnifiedDataError("Tianjin tourism benchmark contains missing Tianjin values")
    return output


def _sensitivity_rows(repo_root: Path, commits: dict[str, str]) -> list[dict[str, str]]:
    source_rows = _read_csv_blob(_git_blob(repo_root, QIU_REF, QIU_IMPUTATION_PATH))
    required = {
        "year",
        "target",
        "official_value",
        "model_value",
        "low",
        "high",
        "unit",
        "data_role",
        "method",
        "allow_as_official",
        "recommended_use",
    }
    if not source_rows or not required.issubset(source_rows[0]):
        missing = sorted(required - (set(source_rows[0]) if source_rows else set()))
        raise UnifiedDataError(f"Qiu sensitivity CSV is missing columns: {missing}")
    output: list[dict[str, str]] = []
    for source in source_rows:
        target = source["target"].strip()
        if target not in SENSITIVITY_METRICS:
            raise UnifiedDataError(f"unknown Qiu sensitivity target: {target!r}")
        if source["official_value"].strip() or source["allow_as_official"].strip() != "否":
            raise UnifiedDataError("Qiu sidecar record unexpectedly claims an official value")
        output.append(
            {
                "metric": SENSITIVITY_METRICS[target],
                "original_target": target,
                "year": source["year"].strip(),
                "value": source["model_value"].strip(),
                "low": source["low"].strip(),
                "high": source["high"].strip(),
                "unit": source["unit"].strip(),
                "status": source["data_role"].strip(),
                "method": source["method"].strip(),
                "official_value": source["official_value"].strip(),
                "allow_as_official": source["allow_as_official"].strip(),
                "recommended_use": source["recommended_use"].strip(),
                "is_observed": "false",
                "excluded_from_benchmark": "true",
                "source_ref": QIU_REF,
                "source_commit": commits[QIU_REF],
                "source_path": QIU_IMPUTATION_PATH,
            }
        )
    if len(output) != 8:
        raise UnifiedDataError(f"expected 8 Qiu sensitivity records, got {len(output)}")
    return output


def _is_data_asset(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return (
        path.startswith("data/")
        or (path.startswith("outputs/") and suffix in {".csv", ".json", ".xlsx", ".xls"})
        or (path.startswith("111完整示例/") and suffix in {".csv", ".xlsx", ".xls"})
    )


def _asset_decision(ref: str, path: str) -> tuple[str, str, str]:
    suffix = Path(path).suffix.lower().lstrip(".") or "no_extension"
    if path == CANONICAL_PATH:
        if ref == CANONICAL_REF:
            return (
                "canonical_annual_targets",
                "canonical_target_truth",
                "唯一目标真值；benchmark 标签逐项由此生成。",
            )
        return (
            "canonical_annual_targets",
            "duplicate_of_main_canonical",
            "分支副本不建立第二套真值；仍以 main 对应 blob 为准。",
        )
    if ref == GDP_REF and path == GDP_PATH:
        return (
            "auxiliary_workbook",
            "accepted_auxiliary_covariate",
            "按工作表规范提取天津市 GDP；不得作为蓟州目标标签。",
        )
    if ref == GDP_REF and path == TRACEABLE_PATH:
        return (
            "benchmark_workbook",
            "accepted_benchmark_context",
            "仅提取天津市旅游基准 4 行作为外部比较背景。",
        )
    if ref == QIU_REF and path == QIU_IMPUTATION_PATH:
        return (
            "imputation_sidecar",
            "sensitivity_only",
            "补值显式标记 is_observed=false，禁止进入 benchmark 标签。",
        )
    if path.startswith("outputs/"):
        return (
            "generated_model_output",
            "excluded_generated_output",
            "属于分支模型结果，不是输入数据或真实标签。",
        )
    if path.startswith("data/邱志烨-数据处理与搜集/"):
        return (
            "derived_qiu_dataset",
            "excluded_derived_targets",
            "含补值、变换或诊断派生量；仅 sidecar 中的明确补值记录被保留。",
        )
    if path.startswith("data/raw/"):
        return (
            f"raw_evidence_{suffix}",
            "provenance_evidence_only",
            "保留来源证据，不直接成为统一目标标签。",
        )
    if path.startswith("data/metadata/"):
        return (
            f"source_metadata_{suffix}",
            "provenance_metadata_only",
            "用于来源追踪和口径解释，不直接用于 benchmark。",
        )
    if path.endswith(".inspect.ndjson"):
        return (
            "workbook_inspection_log",
            "provenance_evidence_only",
            "工作簿解析审计记录，不直接用于建模。",
        )
    if suffix in {"png", "jpg", "jpeg"} or Path(path).name == ".DS_Store":
        return (
            "preview_or_system_file",
            "excluded_non_tabular_preview",
            "预览或系统文件，不是建模数据。",
        )
    if path.endswith(".gitkeep") or Path(path).name == "邱志烨-数据搜索":
        return (
            "placeholder",
            "excluded_placeholder",
            "目录占位，不含可用观测。",
        )
    if path.startswith("data/jizhou_tourism_economy/"):
        return (
            f"official_supporting_{suffix}",
            "supporting_evidence_not_target_truth",
            "可作追溯或辅助核对；目标标签仍只取 main canonical annual CSV。",
        )
    if path.startswith("111完整示例/"):
        return (
            f"example_workbook_{suffix}",
            "excluded_derived_example",
            "示例建模/预测工作簿，不作为统一真值。",
        )
    return (
        f"other_data_asset_{suffix}",
        "excluded_unselected_asset",
        "已扫描但未被统一数据契约选用。",
    )


def _tree_assets(repo_root: Path, ref: str) -> list[tuple[str, str, int]]:
    raw = bytes(_git(repo_root, ["ls-tree", "-rz", "--long", ref]))
    assets: list[tuple[str, str, int]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, object_type, oid, size = metadata.decode("ascii").split()
        del mode
        path = path_bytes.decode("utf-8")
        if object_type == "blob" and _is_data_asset(path):
            assets.append((path, oid, int(size)))
    return sorted(assets)


def _inventory_rows(repo_root: Path, commits: dict[str, str]) -> list[dict[str, str]]:
    blob_hashes: dict[str, str] = {}
    output: list[dict[str, str]] = []
    for ref in AUDITED_REFS:
        assets = _tree_assets(repo_root, commits[ref])
        if not assets:
            raise UnifiedDataError(f"audited ref has no data assets: {ref}")
        for path, oid, size in assets:
            if oid not in blob_hashes:
                blob_hashes[oid] = _sha256(bytes(_git(repo_root, ["cat-file", "blob", oid])))
            asset_class, decision, reason = _asset_decision(ref, path)
            output.append(
                {
                    "audited_ref": ref,
                    "commit_sha": commits[ref],
                    "asset_path": path,
                    "asset_type": Path(path).suffix.lower().lstrip(".") or "none",
                    "size_bytes": str(size),
                    "git_blob_oid": oid,
                    "sha256": blob_hashes[oid],
                    "asset_class": asset_class,
                    "decision": decision,
                    "decision_reason": reason,
                }
            )
    return output


def _readme(
    commits: dict[str, str],
    canonical_sha: str,
    primary_train_count: int,
    primary_test_count: int,
    rolling_count: int,
    stress_train_count: int,
    stress_test_count: int,
    inventory_count: int,
    stress_cutoff_year: int,
) -> str:
    refs = "\n".join(f"- `{ref}` → `{commits[ref]}`" for ref in AUDITED_REFS)
    return f"""# 统一分支数据层

此目录由 `scripts/build_unified_branch_data.py` 可重复生成。五个审计 tip 均固定到下列不可变 40 位提交 SHA，脚本不会解析实时分支指针。蓟州游客量与旅游综合收入的**唯一目标真值**是集成前 `main` 固定提交中的 `{CANONICAL_PATH}`；统一副本的 SHA-256 为 `{canonical_sha}`。

## 数据契约

- `canonical_official_annual_2010_2025.csv`：上述 main 文件的逐字节副本，不接受其他分支覆盖。
- `benchmark_observations.csv`：供统一评测使用的完整 canonical 长表；字段为 `{', '.join(BENCHMARK_FIELDS)}`，保留原定 2019 时间切分标记以便复核跨疫情表现。
- `primary_train.csv`：最终训练/交叉验证契约，包含 `year <= 2023` 的全部 {primary_train_count} 条非空 canonical preferred 目标记录。推导记录不被伪装成实测，仍保留 `is_observed=false` 供严格证据敏感性分析。
- `primary_test.csv`：本次重跑中执行隔离的 pseudo-holdout，仅含 2024 年 {primary_test_count} 条实际观测；2024 不进入滚动选模或压力测试。由于模型代码形成时 2024 数据已经存在，它不是真正前瞻的未知样本。
- `rolling_origin_folds.csv`：min-train=5、外层测试年不晚于 2023 的 {rolling_count} 条 fold 记录。训练 role 与 `primary_train.csv` 使用相同 canonical preferred 证据契约；每折训练年份严格早于测试年，测试 role 只允许实际观测。
- `stress_train.csv` / `stress_test.csv`：cutoff={stress_cutoff_year} 的跨疫情压力测试，共 {stress_train_count}/{stress_test_count} 条；stress test 仅覆盖 2020—2023，明确排除 2024 pseudo-holdout。
- `sensitivity_imputations.csv`：邱分支的 8 条补值记录；全部 `is_observed=false`、`excluded_from_benchmark=true`，不得用作测试标签。
- `tianjin_gdp_2010_2025.csv`：从 `origin/111` 的 `GDP数据` 工作表按表头提取的 16 行辅助宏观数据。
- `tianjin_tourism_benchmark.csv`：从 `origin/111` 的 `天津市旅游基准` 工作表按表头提取的 4 行外部比较数据。
- `branch_data_inventory.csv`：5 个审计 tip 的数据资产清单，共 {inventory_count} 行，记录 Git blob、SHA-256 与纳入/排除决策。扫描范围为各 tip 的 `data/**`、表格型 `outputs/**` 和 `111完整示例` 工作簿。

测试标签不会被任何补值、目标值、情景值或模型预测覆盖。天津市 GDP 和旅游基准只作辅助特征/外部背景，也不构成蓟州目标标签。

## 已审计 tip

{refs}
"""


def build_unified_data(
    repo_root: Path, output_dir: Path, cutoff_year: int = DEFAULT_CUTOFF_YEAR
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    commits = _validate_pinned_commits(repo_root)

    canonical_content = _git_blob(repo_root, CANONICAL_REF, CANONICAL_PATH)
    canonical = _canonical_rows(canonical_content)
    canonical_sha = _sha256(canonical_content)
    benchmark_train, benchmark_test = _benchmark_rows(canonical, cutoff_year)
    primary_train, primary_test = _benchmark_rows(
        canonical,
        2023,
        split_id="primary_holdout_2024",
        test_max_year=2024,
    )
    stress_train, stress_test = _benchmark_rows(
        canonical,
        cutoff_year,
        split_id=f"stress_cutoff_{cutoff_year}_through_2023",
        test_max_year=2023,
    )
    rolling = _rolling_origin_rows(canonical)
    if any(row["is_observed"] != "true" for row in primary_test):
        raise UnifiedDataError("primary holdout contains a non-observed target")
    if any(row["is_observed"] != "true" for row in stress_test):
        raise UnifiedDataError("stress test contains a non-observed target")

    gdp = _extract_gdp(repo_root, commits)
    tourism = _extract_tianjin_tourism(repo_root, commits)
    sensitivity = _sensitivity_rows(repo_root, commits)
    inventory = _inventory_rows(repo_root, commits)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canonical_official_annual_2010_2025.csv").write_bytes(canonical_content)
    _write_csv(output_dir / "primary_train.csv", BENCHMARK_FIELDS, primary_train)
    _write_csv(output_dir / "primary_test.csv", BENCHMARK_FIELDS, primary_test)
    _write_csv(
        output_dir / "benchmark_observations.csv",
        BENCHMARK_FIELDS,
        [*benchmark_train, *benchmark_test],
    )
    _write_csv(output_dir / "stress_train.csv", BENCHMARK_FIELDS, stress_train)
    _write_csv(output_dir / "stress_test.csv", BENCHMARK_FIELDS, stress_test)
    _write_csv(
        output_dir / "rolling_origin_folds.csv",
        (
            "split_id",
            "fold_id",
            "metric",
            "role",
            "year",
            "value",
            "unit",
            "status",
            "source_ids",
            "quality_note",
            "is_observed",
            "min_train_records",
            "outer_test_max_year",
        ),
        rolling,
    )
    _write_csv(
        output_dir / "tianjin_gdp_2010_2025.csv",
        (
            "year",
            "tianjin_gdp_100m_cny",
            "tianjin_real_gdp_growth_rate",
            "status",
            "is_observed",
            "source_note",
            "source_ref",
            "source_commit",
            "source_path",
            "source_sheet",
            "source_row",
        ),
        gdp,
    )
    _write_csv(
        output_dir / "tianjin_tourism_benchmark.csv",
        (
            "year",
            "tianjin_domestic_visitors_10k_persons",
            "tianjin_domestic_tourism_income_100m_cny",
            "jizhou_visitors_10k_persons",
            "jizhou_comprehensive_income_100m_cny",
            "jizhou_visitor_share",
            "jizhou_income_share",
            "status",
            "is_observed",
            "source_ref",
            "source_commit",
            "source_path",
            "source_sheet",
            "source_row",
        ),
        tourism,
    )
    _write_csv(
        output_dir / "sensitivity_imputations.csv",
        (
            "metric",
            "original_target",
            "year",
            "value",
            "low",
            "high",
            "unit",
            "status",
            "method",
            "official_value",
            "allow_as_official",
            "recommended_use",
            "is_observed",
            "excluded_from_benchmark",
            "source_ref",
            "source_commit",
            "source_path",
        ),
        sensitivity,
    )
    _write_csv(
        output_dir / "branch_data_inventory.csv",
        (
            "audited_ref",
            "commit_sha",
            "asset_path",
            "asset_type",
            "size_bytes",
            "git_blob_oid",
            "sha256",
            "asset_class",
            "decision",
            "decision_reason",
        ),
        inventory,
    )
    (output_dir / "README.md").write_text(
        _readme(
            commits,
            canonical_sha,
            len(primary_train),
            len(primary_test),
            len(rolling),
            len(stress_train),
            len(stress_test),
            len(inventory),
            cutoff_year,
        ),
        encoding="utf-8",
    )
    return {
        "canonical_ref": CANONICAL_REF,
        "canonical_commit": commits[CANONICAL_REF],
        "canonical_path": CANONICAL_PATH,
        "canonical_sha256": canonical_sha,
        "cutoff_year": cutoff_year,
        "benchmark_rows": len(benchmark_train) + len(benchmark_test),
        "primary_train_rows": len(primary_train),
        "primary_test_rows": len(primary_test),
        "rolling_fold_rows": len(rolling),
        "stress_train_rows": len(stress_train),
        "stress_test_rows": len(stress_test),
        "sensitivity_rows": len(sensitivity),
        "tianjin_gdp_rows": len(gdp),
        "tianjin_tourism_rows": len(tourism),
        "inventory_rows": len(inventory),
        "audited_refs": list(AUDITED_REFS),
    }


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cutoff-year", type=int, default=DEFAULT_CUTOFF_YEAR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or args.repo_root / "data/unified"
    summary = build_unified_data(args.repo_root, output_dir, args.cutoff_year)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
