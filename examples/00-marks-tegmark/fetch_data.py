"""Download Marks and Tegmark's `cities` and `sp_en_trans` statement sets into data/.

Usage: python fetch_data.py
The files come from saprmarks/geometry-of-truth at a pinned commit and are checked
against their sha256, so every run sees the same statements. They are not committed
here: the upstream repository has no licence file.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "https://github.com/saprmarks/geometry-of-truth"
COMMIT = "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb"
FILES = {
    "cities": "f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e",
    "sp_en_trans": "9afa97c1b8af539a05f47655a80a5c7832c6fdb914885fd24cb4bcccbe2795b5",
}
DATA = Path(__file__).parent / "data"


def fetch(name: str) -> Path:
    path = DATA / f"{name}.csv"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == FILES[name]:
        return path
    url = f"https://raw.githubusercontent.com/saprmarks/geometry-of-truth/{COMMIT}/datasets/{name}.csv"
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    got = hashlib.sha256(body).hexdigest()
    if got != FILES[name]:
        raise SystemExit(f"{url}: sha256 {got}, expected {FILES[name]}")
    DATA.mkdir(exist_ok=True)
    path.write_bytes(body)
    return path


if __name__ == "__main__":
    for name in sys.argv[1:] or list(FILES):
        print(fetch(name), file=sys.stderr)
