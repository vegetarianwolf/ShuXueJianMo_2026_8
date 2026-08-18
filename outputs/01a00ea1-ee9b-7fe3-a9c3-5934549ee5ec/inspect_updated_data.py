from pathlib import Path
import json

import pandas as pd


BASE = Path(r"D:\junk mass\小乱七八糟\数学建模\26国赛\shared files\data\jizhou_tourism_economy")


def main() -> None:
    output = []
    for path in sorted(BASE.glob("*.csv")):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
            output.append(
                {
                    "name": path.name,
                    "shape": frame.shape,
                    "columns": list(frame.columns),
                    "nulls": {
                        key: int(value)
                        for key, value in frame.isna().sum().items()
                        if value
                    },
                    "head": frame.head(3).fillna("").to_dict("records"),
                }
            )
        except Exception as exc:
            output.append({"name": path.name, "error": str(exc)})
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
