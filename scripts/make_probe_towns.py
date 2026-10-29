"""Write examples/03-two-clause-rivers/entities.jsonl: 29 hand-picked real towns, well-known and obscure
alternating, each with a country, region and river question.

A stopgap until the Wikidata seed script (MVP step 3). Not part of the library.

v2: the region question names the country and that country's own word for its
first-level division. v1 asked "Which state or region is X in?", and the model
answered with the country ("Verona" -> "Italy" 9 of 9 times), which graded as
confidently_wrong although it was a misread question.
"""

import json
from pathlib import Path

NRW = ["North Rhine-Westphalia", "Nordrhein-Westfalen", "NRW"]
RP = ["Rhineland-Palatinate", "Rheinland-Pfalz"]
UA = ["Upper Austria", "Oberösterreich"]
SB = ["South Bohemian Region", "South Bohemia", "Jihočeský kraj"]
VIS = ["Vistula", "Wisła", "Wisla"]
MOS = ["Moselle", "Mosel"]

# Country -> name of its first-level division, as used in the region question.
DIVISION = {
    "Germany": "federal state", "Austria": "federal state", "Poland": "voivodeship",
    "Czech Republic": "region", "Slovakia": "region", "France": "region", "Italy": "region",
    "Spain": "autonomous community", "Portugal": "region", "Sweden": "county",
    "Slovenia": "traditional region",
}

# (town, country answers, region answers, river answers)
TOWNS = [
    ("Heidelberg", ["Germany"], ["Baden-Württemberg"], ["Neckar"]),
    ("Bad Münstereifel", ["Germany"], NRW, ["Erft"]),
    ("Avignon", ["France"], ["Provence-Alpes-Côte d'Azur", "Provence", "Vaucluse", "PACA"], ["Rhône"]),
    ("Kežmarok", ["Slovakia"], ["Prešov Region", "Prešov"], ["Poprad"]),
    ("Kraków", ["Poland"], ["Lesser Poland", "Lesser Poland Voivodeship", "Małopolska", "Małopolskie"], VIS),
    ("Wipperfürth", ["Germany"], NRW, ["Wupper"]),
    ("Verona", ["Italy"], ["Veneto"], ["Adige", "Etsch"]),
    ("Tolmin", ["Slovenia"], ["Slovenian Littoral", "Littoral", "Goriška", "Primorska"], ["Soča", "Isonzo"]),
    ("Koblenz", ["Germany"], RP, ["Rhine", "Rhein", *MOS]),
    ("Altena", ["Germany"], NRW, ["Lenne"]),
    ("Coimbra", ["Portugal"], ["Centro", "Centro Region", "Coimbra District"], ["Mondego"]),
    ("Mértola", ["Portugal"], ["Alentejo", "Beja District", "Beja"], ["Guadiana"]),
    ("Düren", ["Germany"], NRW, ["Rur", "Roer"]),
    ("Písek", ["Czech Republic", "Czechia"], SB, ["Otava"]),
    ("Uppsala", ["Sweden"], ["Uppsala County", "Uppland", "Svealand"], ["Fyris", "Fyrisån"]),
    ("Brantôme", ["France"], ["Nouvelle-Aquitaine", "Dordogne", "Périgord"], ["Dronne"]),
    ("Wetzlar", ["Germany"], ["Hesse", "Hessen"], ["Lahn", "Dill"]),
    ("Sandomierz", ["Poland"], ["Świętokrzyskie", "Świętokrzyskie Voivodeship", "Holy Cross"], VIS),
    ("Wels", ["Austria"], UA, ["Traun"]),
    ("Idar-Oberstein", ["Germany"], RP, ["Nahe"]),
    ("Ronda", ["Spain"], ["Andalusia", "Andalucía", "Málaga", "Province of Málaga"], ["Guadalevín"]),
    ("Monschau", ["Germany"], NRW, ["Rur", "Roer"]),
    ("Ptuj", ["Slovenia"], ["Styria", "Lower Styria", "Štajerska", "Drava Statistical Region", "Podravska"], ["Drava", "Drau"]),
    ("Cochem", ["Germany"], RP, MOS),
    ("Tábor", ["Czech Republic", "Czechia"], SB, ["Lužnice"]),
    ("Tczew", ["Poland"], ["Pomeranian Voivodeship", "Pomeranian", "Pomerania", "Pomorskie"], VIS),
    ("Bernkastel-Kues", ["Germany"], RP, MOS),
    ("Steyr", ["Austria"], UA, ["Enns"]),  # river Steyr left out: it would match any echo of the town name
    ("Bad Kreuznach", ["Germany"], RP, ["Nahe"]),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "examples" / "03-two-clause-rivers" / "entities.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for town, country, region, river in TOWNS:
            questions = [
                {"id": "country", "q": f"Which country is {town} in?", "answers": country},
                {"id": "region", "q": f"Which {DIVISION[country[0]]} of {country[0]} is {town} in?", "answers": region},
                {"id": "river", "q": f"Which river flows through {town}?", "answers": river},
            ]
            record = {"entity": town, "type": "town", "questions": questions,
                      "meta": {"source": "hand-written probe", "probe_version": 2}}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(TOWNS)} towns to {out}")


if __name__ == "__main__":
    main()
