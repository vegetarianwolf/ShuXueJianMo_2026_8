from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE = Path(r"D:\junk mass\小乱七八糟\数学建模\26国赛\shared files\data\jizhou_tourism_economy")


def pivot_metric(path: Path, scope: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if scope is not None:
        frame = frame.loc[frame["metric_scope"] == scope]
    return frame.pivot_table(index="year", columns="metric", values="value", aggfunc="first")


def main() -> None:
    annual = pd.read_csv(BASE / "official_annual_summary_2010_2025.csv", encoding="utf-8-sig")
    supply = pivot_metric(
        BASE / "official_tourism_supply_observations_2012_2024.csv",
        "above_designated_size_accommodation_and_catering",
    )
    related = pivot_metric(BASE / "official_related_observations_2014_2025.csv")
    panel = annual.set_index("year").join(supply).join(related, rsuffix="_related")
    print("SUPPLY 2019-2024")
    print(supply.loc[2019:2024].to_string())
    print("\nRELATED COVERAGE")
    print(related.notna().sum().sort_values(ascending=False).to_string())
    print("\nRELATED 2019-2025")
    print(related.loc[2019:2025].to_string())
    print("\nCORRELATIONS WITH OBSERVED TOURISM")
    cols = [
        "preferred_visitor_10k_persons",
        "preferred_comprehensive_income_100m_cny",
        "guest_rooms",
        "beds",
        "room_revenue",
        "preferred_gdp_100m_cny",
        "preferred_tertiary_100m_cny",
    ]
    print(panel[cols].corr(min_periods=5).round(3).to_string())
    share_room = supply.loc[2021, "room_revenue"] / supply.loc[[2021, 2022], "room_revenue"].sum()
    share_income = 110 / (110 + 55.7491)
    results = {
        "room_revenue_share_2021": share_room,
        "income_proxy_share_2021": share_income,
        "allocation_room_revenue": [2803 * share_room, 2803 * (1-share_room)],
        "allocation_income_proxy": [2803 * share_income, 2803 * (1-share_income)],
        "2020_room_revenue_ratio": supply.loc[2020, "room_revenue"] / supply.loc[2019, "room_revenue"],
        "2020_room_proxy_visitor": 2800 * supply.loc[2020, "room_revenue"] / supply.loc[2019, "room_revenue"],
        "2020_room_proxy_income": 165 * supply.loc[2020, "room_revenue"] / supply.loc[2019, "room_revenue"],
    }
    print("\nCANDIDATES")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
