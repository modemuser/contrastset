"""Turn cities.csv into a contrastset pair file: per city, a = the true statement, b = the false one.

Usage: python make_pairs.py data/cities.csv > pairs.jsonl
Every city in the set has exactly one true and one false statement, which differ
only in the country, so the pairs can go straight into `contrastset check`. The
CSV columns pass through as extra fields.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict


def main() -> None:
    by_city: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
        by_city[row["city"]]["a" if row["label"] == "1" else "b"] = row
    for city, sides in by_city.items():
        if set(sides) != {"a", "b"}:
            raise SystemExit(f"{city}: expected one true and one false statement")
        print(json.dumps({"a": sides["a"]["statement"], "b": sides["b"]["statement"], "city": city,
                          "country_a": sides["a"]["country"], "country_b": sides["b"]["country"]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
