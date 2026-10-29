"""Pull candidate towns from Wikidata and write a `grade` entity file.

Each town gets three questions: country, first-level region, river. Well-known and
obscure towns are mixed by sampling evenly across sitelink counts (how many
Wikipedia language editions have an article on the town: a cheap fame proxy), so
the graded file can end up with known and unknown towns that look alike on the
surface. Not part of the library.

Pipeline, with "country known" as the first filter:

    uv run python scripts/wikidata_seed.py --per-country 40 -o experiments/wikidata-towns.jsonl
    uv run contrastset grade experiments/wikidata-towns.jsonl --model qwen3:14b --gate country -o runs/towns.jsonl

`--gate country` asks the country question first and stops for towns whose country
the model doesn't know: those are unfamiliar entities, the wrong side of the
"fact unknown, entity familiar" design, so their region and river aren't worth asking.

Answer lists carry every name Wikidata has for the answer: the English and local
labels plus aliases in both languages. The hand-written probe showed that a missing
alias grades a correct answer as wrong ("Wipper" for the Wupper's upper course).
Towns whose name echoes a region or river answer are dropped: the same word (river
Steyr, town Steyr) or the same stem of five or more letters (Chomutovka / Chomutov,
Oslava / Náměšť nad Oslavou). The question would contain its own answer, and
dropping only the echoed answer is worse: the model's echo then grades as wrong
(Musile di Piave -> "Piave", left with only the alias "Fiume Sacro alla Patria").

Wikidata coverage is uneven: many towns have no river statement (P206), so the
sample is "towns whose river Wikidata records", not all towns.
"""

import argparse
import itertools
import json
import random
import re
import sys
import time
from datetime import date

import httpx

from contrastset.grade import normalise

SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "contrastset-seed/0.1 (https://github.com/modemuser/contrastset)"}

# code: (country QID, Wikidata class of its towns, name of its first-level division, local languages)
COUNTRIES = {
    "de": ("Q183", "Q42744322", "federal state", ["de"]),  # urban municipality in Germany
    "at": ("Q40", "Q667509", "federal state", ["de"]),  # municipality of Austria
    "pl": ("Q36", "Q2616791", "voivodeship", ["pl"]),  # urban municipality of Poland
    "cz": ("Q213", "Q15978299", "region", ["cs"]),  # municipality with town privileges in the Czech Republic
    "sk": ("Q214", "Q6784672", "region", ["sk"]),  # municipality of Slovakia
    "fr": ("Q142", "Q484170", "region", ["fr"]),  # commune of France
    "it": ("Q38", "Q747074", "region", ["it"]),  # comune of Italy
    "es": ("Q29", "Q2074737", "autonomous community", ["es"]),  # municipality of Spain
    "pt": ("Q45", "Q13217644", "district", ["pt"]),  # municipality of Portugal
    "se": ("Q34", "Q12813115", "county", ["sv"]),  # urban area in Sweden
}

# Country names as they read inside a sentence ("of the Czech Republic").
IN_SENTENCE = {"Czech Republic": "the Czech Republic"}

# Towns of one country with population, sitelinks, first-level regions (Q10864048:
# first-level administrative country subdivision) and rivers (Q4022: river).
QUERY = """
SELECT ?town (MAX(?p) AS ?pop) (SAMPLE(?l) AS ?links)
  (GROUP_CONCAT(DISTINCT STRAFTER(STR(?region), "entity/"); separator=" ") AS ?regions)
  (GROUP_CONCAT(DISTINCT STRAFTER(STR(?river), "entity/"); separator=" ") AS ?rivers) WHERE {
  ?town wdt:P31 wd:%(town_class)s ; wdt:P17 wd:%(country)s ; wdt:P206 ?river ; wdt:P1082 ?p ; wikibase:sitelinks ?l .
  ?river wdt:P31/wdt:P279* wd:Q4022 .
  ?town wdt:P131+ ?region . ?region wdt:P31/wdt:P279* wd:Q10864048 .
  FILTER NOT EXISTS { ?region wdt:P576 ?dissolved }  # current regions only ("Centre-Val de Loire", not "Centre")
} GROUP BY ?town
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def get(client: httpx.Client, url: str, params: dict, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                if "error" not in data:
                    return data
            log(f"  retry {attempt + 1}: HTTP {r.status_code}")
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log(f"  retry {attempt + 1}: {exc}")
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def towns_of(client: httpx.Client, code: str, min_pop: int, max_pop: int) -> list[dict]:
    country, town_class, _, _ = COUNTRIES[code]
    rows = get(client, SPARQL, {"query": QUERY % {"town_class": town_class, "country": country},
                                "format": "json"})["results"]["bindings"]
    out = []
    for row in rows:
        regions = row["regions"]["value"].split()
        pop = int(float(row["pop"]["value"]))
        if len(regions) != 1 or not min_pop <= pop <= max_pop:  # several regions: historic or ambiguous
            continue
        out.append({"qid": row["town"]["value"].rsplit("/", 1)[1], "population": pop,
                    "sitelinks": int(row["links"]["value"]), "region": regions[0],
                    "rivers": row["rivers"]["value"].split(), "country_code": code})
    return out


def stratified(towns: list[dict], k: int, rng: random.Random, bands: int = 4) -> list[dict]:
    """k towns spread evenly over sitelink quantile bands, obscure to well-known."""
    ordered = sorted(towns, key=lambda t: t["sitelinks"])
    picked = []
    for b in range(bands):
        band = ordered[b * len(ordered) // bands:(b + 1) * len(ordered) // bands]
        picked += rng.sample(band, min(len(band), k // bands + (b < k % bands)))
    return picked


def names(client: httpx.Client, qids: set[str], languages: list[str]) -> dict[str, dict]:
    """Labels and aliases per QID, in the given languages."""
    out: dict[str, dict] = {}
    ids = sorted(qids)
    for i in range(0, len(ids), 50):
        data = get(client, API, {"action": "wbgetentities", "ids": "|".join(ids[i:i + 50]),
                                 "props": "labels|aliases", "languages": "|".join(languages), "format": "json"})
        for qid, ent in data["entities"].items():
            out[qid] = {"labels": {lang: v["value"] for lang, v in ent.get("labels", {}).items()},
                        "aliases": {lang: [a["value"] for a in v] for lang, v in ent.get("aliases", {}).items()}}
    return out


def answer_list(entry: dict | None, languages: list[str]) -> list[str]:
    """English label first, then local labels, then aliases; deduplicated after
    normalisation; one- and two-letter codes dropped (too ambiguous to match)."""
    if not entry:
        return []
    raw = [entry["labels"].get("en")] + [entry["labels"].get(lang) for lang in languages]
    raw += entry["aliases"].get("en", []) + [a for lang in languages for a in entry["aliases"].get(lang, [])]
    out, seen = [], set()
    for name in raw:
        key = normalise(name or "", ("en", "de"))
        junk = name is None or re.search(r"[\d():]", name) or re.fullmatch(r"[A-Z]{2}-[A-Z]+", name)
        if not junk and len(key) > 2 and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _echoes(a: str, b: str, stem: int = 5) -> bool:
    if a == b:
        return True
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n >= stem


def drop_echoes(answers: list[str], town: str) -> list[str]:
    """Remove answers sharing a word or a 5+ letter stem with the town's name."""
    town_words = normalise(town).split()
    return [a for a in answers
            if not any(_echoes(w, t) for w in normalise(a).split() for t in town_words)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", default=",".join(COUNTRIES), help=f"comma-separated, from {','.join(COUNTRIES)}")
    ap.add_argument("--per-country", type=int, default=40)
    ap.add_argument("--min-pop", type=int, default=3000)
    ap.add_argument("--max-pop", type=int, default=200000, help="0 = no cap")
    ap.add_argument("--min-sitelinks", type=int, default=0,
                    help="draw only from towns with at least this many sitelinks (a fame band)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="JSONL file(s) whose towns (meta.qid) must not be drawn again; repeatable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    excluded = {json.loads(line)["meta"]["qid"] for f in args.exclude for line in open(f, encoding="utf-8")}
    max_pop = args.max_pop or 10**12
    client = httpx.Client(headers=HEADERS, timeout=90)
    codes = args.countries.split(",")
    picked: list[dict] = []
    for code in codes:
        start = time.time()
        towns = towns_of(client, code, args.min_pop, max_pop)
        eligible = [t for t in towns if t["sitelinks"] >= args.min_sitelinks and t["qid"] not in excluded]
        chosen = stratified(eligible, args.per_country, rng)
        picked += chosen
        log(f"{code}: {len(towns)} towns with one region and a river, {len(eligible)} eligible "
            f"(sitelinks >= {args.min_sitelinks}, not excluded), picked {len(chosen)} ({time.time() - start:.0f}s)")

    languages = sorted({"en", *(lang for c in codes for lang in COUNTRIES[c][3])})
    country_qids = {COUNTRIES[c][0] for c in codes}
    qids = {t["qid"] for t in picked} | {t["region"] for t in picked} | country_qids
    qids |= {r for t in picked for r in t["rivers"]}
    log(f"fetching labels and aliases for {len(qids)} items")
    info = names(client, qids, languages)

    # Interleave countries (de, at, pl, ..., de, at, ...) so that any prefix of the file,
    # e.g. `grade --limit 40`, is a cross-country sample rather than 40 German towns.
    by_country = [[t for t in picked if t["country_code"] == c] for c in codes]
    picked = [t for round_ in itertools.zip_longest(*by_country) for t in round_ if t is not None]

    written = dropped = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for t in picked:
            country_qid, _, division, local = COUNTRIES[t["country_code"]]
            town_names = answer_list(info.get(t["qid"]), local)
            if not town_names:
                dropped += 1
                continue
            town = town_names[0].split("/")[0].strip()  # bilingual "Sangüesa/Zangoza": first name
            country = answer_list(info.get(country_qid), local)
            region = answer_list(info.get(t["region"]), local)
            rivers = [a for r in t["rivers"] for a in answer_list(info.get(r), local)]
            # If the town's name gives away any answer to a question (Musile di Piave),
            # drop the town: dropping just that answer would grade the echo as wrong.
            if not (country and region and rivers) or any(
                    drop_echoes(answers, town) != answers for answers in (region, rivers)):
                dropped += 1
                continue
            record = {
                "entity": town, "type": "town",
                "questions": [
                    {"id": "country", "q": f"Which country is {town} in?", "answers": country},
                    {"id": "region", "q": f"Which {division} of {IN_SENTENCE.get(country[0], country[0])} is {town} in?",
                     "answers": region},
                    {"id": "river", "q": f"Which river flows through {town}?", "answers": rivers},
                ],
                "meta": {"source": "wikidata", "retrieved": date.today().isoformat(), "qid": t["qid"],
                         "country_qid": country_qid, "region_qid": t["region"], "river_qids": t["rivers"],
                         "population": t["population"], "sitelinks": t["sitelinks"]},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    log(f"wrote {written} towns to {args.output} ({dropped} dropped: no names, or the name echoes an answer)")


if __name__ == "__main__":
    main()
