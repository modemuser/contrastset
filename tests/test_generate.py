import json
from pathlib import Path

import pytest

from contrastset import cli, generate
from contrastset.adapters import FakeAdapter
from contrastset.records import Pair

SPEC = Path(__file__).parent.parent / "examples" / "03-two-clause-rivers" / "spec.yaml"
FRAMES = [
    "Which river flows through [[X1]], and which river flows through [[X2]]?",
    "What river runs through [[X1]], and what river runs through [[X2]]?",
    "Which river flows through [[X1]]?",  # missing a placeholder: rejected
    "Which river flows through [[X1]], and which river flows past Passau to [[X2]]?",  # example name: rejected
    "Which rivers flow through [[X1]] and [[X2]]?",  # one clause for two placeholders: rejected
]


def graded_file(tmp_path: Path) -> Path:
    rows = []
    for town, river_bin in [("Heidelberg", "known"), ("Koblenz", "known"), ("Wetzlar", "known"),
                            ("Kežmarok", "unstable"), ("Wels", "unstable")]:
        rows.append({
            "entity": town, "type": "town", "bin": "mixed", "bin_counts": {}, "thresholds": {},
            "questions": [{"id": "river", "q": f"Which river flows through {town}?", "answers": ["x"],
                           "samples": [], "n": 8, "n_correct": 0, "n_refused": 0,
                           "correct_rate": 1.0 if river_bin == "known" else 0.0,
                           "refusal_rate": 0, "consistency": None, "modal_answer": None,
                           "greedy_correct": None, "bin": river_bin}],
            "provenance": {"verb": "grade"},
        })
    path = tmp_path / "graded.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def fake_generator() -> FakeAdapter:
    return FakeAdapter({"JSON array": ["```json\n" + json.dumps(FRAMES) + "\n```"]})


def test_frame_prompt_names_no_candidate_entity_and_no_bins(tmp_path: Path) -> None:
    fake = fake_generator()
    result, _ = generate.run(fake, SPEC, str(graded_file(tmp_path)), n_frames=2)
    prompt = fake.calls[0]["prompt"]
    for town in ["Heidelberg", "Koblenz", "Wetzlar", "Kežmarok", "Wels"]:
        assert town not in prompt
    positions = prompt.split("in this order:")[1].split("Example pairs")[0]
    assert "unstable" not in prompt and "known" not in positions
    assert '"Which river flows through [[X1]]?"' in prompt
    assert result.frames == FRAMES[:2]
    assert [reason for _, reason in result.rejected] == ["placeholders [1], expected [1, 2] once each in order",
                                                         "reuses example names ['Passau']",
                                                         "placeholders share a clause (only 'and' between them)"]


def test_pairs_differ_only_in_the_swapped_slot(tmp_path: Path) -> None:
    _, pairs = generate.run(fake_generator(), SPEC, str(graded_file(tmp_path)), n_frames=2, limit=40)
    out = list(pairs)
    assert out and len({p.pair_id for p in out}) == len(out)
    for pair in out:
        assert pair.frame in FRAMES[:2]
        a_vals = [f.value for f in pair.fills["a"]]
        b_vals = [f.value for f in pair.fills["b"]]
        assert generate.fill(pair.frame, a_vals) == pair.a and generate.fill(pair.frame, b_vals) == pair.b
        assert [f.bin for f in pair.fills["a"]] in (["known", "known"],)
        assert sorted(f.bin for f in pair.fills["b"]) == ["known", "unstable"]  # type: ignore[type-var]
        shared = [f.value for f in pair.fills["a"] if f.slot == "shared"]
        assert shared == [f.value for f in pair.fills["b"] if f.slot == "shared"]
        assert len(set(a_vals)) == 2 and len(set(b_vals)) == 2
        assert pair.provenance.inputs.keys() == {"spec", "graded"}
    # order: both -> the unknown clause appears first and second
    positions = {[f.slot for f in p.fills["b"]].index("unknown") for p in out}
    assert positions == {0, 1}


def test_same_seed_same_pairs(tmp_path: Path) -> None:
    graded = str(graded_file(tmp_path))
    first = [p.pair_id for p in generate.run(fake_generator(), SPEC, graded, n_frames=2, limit=10, seed=3)[1]]
    again = [p.pair_id for p in generate.run(fake_generator(), SPEC, graded, n_frames=2, limit=10, seed=3)[1]]
    assert first == again


def test_graded_spec_needs_graded_file() -> None:
    with pytest.raises(ValueError, match="--graded"):
        generate.run(fake_generator(), SPEC, None)


def test_inline_sources_need_no_graded_file(tmp_path: Path) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("""
contrast: nested belief versus flat statement
examples: [{a: x1, b: y1}, {a: x2, b: y2}, {a: x3, b: y3}]
slots:
  select:
    a: {source: inline, values: ["Anna thinks that it rains"]}
    b: {source: inline, values: ["it rains, Anna thinks"]}
""")
    fake = FakeAdapter({"JSON array": ['["She said: [[X1]]."]']})
    _, pairs = generate.run(fake, spec, None, n_frames=1)
    [pair] = list(pairs)
    assert (pair.a, pair.b) == ("She said: Anna thinks that it rains.", "She said: it rains, Anna thinks.")
    assert pair.fills["a"][0].source == "inline" and "graded" not in pair.provenance.inputs


def test_cli_generate_streams_pairs(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "make_adapter", lambda *a, **k: fake_generator())
    out = tmp_path / "pairs.jsonl"
    assert cli.main(["generate", str(SPEC), "--graded", str(graded_file(tmp_path)), "--model", "fake",
                     "--adapter", "anthropic", "--frames", "2", "--limit", "4", "-o", str(out)]) == 0
    lines = [Pair.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(lines) == 4
    err = capsys.readouterr().err
    assert "frame 1:" in err and "rejected" in err and "4 pairs from 2 frames" in err


def _graded_row(town: str, bins: dict[str, tuple[str, float]]) -> dict:  # type: ignore[type-arg]
    return {
        "entity": town, "type": "town", "bin": "mixed", "bin_counts": {}, "thresholds": {},
        "questions": [{"id": qid, "q": f"{qid} of {town}?", "answers": ["x"], "samples": [], "n": 8,
                       "n_correct": round(rate * 8), "n_refused": 0, "correct_rate": rate, "refusal_rate": 0,
                       "consistency": None, "modal_answer": None, "greedy_correct": None, "bin": b}
                      for qid, (b, rate) in bins.items()],
        "provenance": {"verb": "grade"},
    }


def test_familiar_default_and_range_filters() -> None:
    from contrastset.records import GradedEntity, GradedSource

    graded = generate.graded_candidates([GradedEntity.model_validate(r) for r in [
        _graded_row("Kežmarok", {"country": ("known", 1.0), "region": ("known", 0.88), "river": ("unstable", 0.0)}),
        _graded_row("Coimbra", {"country": ("known", 1.0), "region": ("known", 1.0), "river": ("unstable", 0.62)}),
        _graded_row("Nowhere", {"country": ("unstable", 0.25), "region": ("refused", 0.0), "river": ("refused", 0.0)}),
    ]])
    from contrastset.records import SlotDefaults

    unknown_river = GradedSource(source="graded", where={"question": "river", "bin": ["unstable", "refused"]})
    # no library default: familiar is off unless the spec says otherwise
    assert len(generate.slot_candidates(unknown_river, graded, Path("."))) == 3
    familiar = SlotDefaults(familiar=True)
    assert [c["value"] for c in generate.slot_candidates(unknown_river, graded, Path("."), familiar)] == \
        ["Kežmarok", "Coimbra"]
    strict = GradedSource.model_validate({"source": "graded", "where": {"question": "river", "correct_rate": {"max": 0.125}}})
    assert [c["value"] for c in generate.slot_candidates(strict, graded, Path("."), familiar)] == ["Kežmarok"]
    slot_overrides = unknown_river.model_copy(update={"familiar": False})
    assert len(generate.slot_candidates(slot_overrides, graded, Path("."), familiar)) == 3
    only_country = unknown_river.model_copy(update={"familiar": ["country"]})
    assert len(generate.slot_candidates(only_country, graded, Path("."))) == 2


def test_factor_balances_same_and_different_country() -> None:
    from collections import Counter

    from contrastset.records import Factor
    import random as _random

    def cand(town: str, country: str, known: bool) -> dict:  # type: ignore[type-arg]
        return {"value": town, "answer.country": country, "question": "river", "known": known}

    pools = {
        "shared": [cand("Heidelberg", "Germany", True), cand("Verona", "Italy", True)],
        "known": [cand("Wetzlar", "Germany", True), cand("Koblenz", "Germany", True), cand("Pisa", "Italy", True)],
        "unknown": [cand("Bad Münstereifel", "Germany", False), cand("Ronda", "Spain", False)],
    }
    factor = Factor(field="answer.country", between=[("shared", "known"), ("shared", "unknown")])
    out = list(generate.combos(pools, [], _random.Random(0), None, {"nc": factor}))
    for combo, levels in out:
        rel = [combo["shared"]["answer.country"] == combo[s]["answer.country"] for s in ("known", "unknown")]
        assert rel[0] == rel[1], "mixed combinations must be dropped"
        assert levels["nc"] == ("same" if rel[0] else "different")
    counts = Counter(levels["nc"] for _, levels in out)
    assert counts["same"] == 2  # Heidelberg + {Wetzlar, Koblenz} + Bad Münstereifel
    # round-robin: the first 4 alternate between cells
    assert [lv["nc"] for _, lv in out[:4]] in (["different", "same"] * 2, ["same", "different"] * 2)


def test_pairs_record_factor_level(tmp_path: Path) -> None:
    _, pairs = generate.run(fake_generator(), SPEC, str(graded_file(tmp_path)), n_frames=2, limit=6)
    assert all("neighbour_country" in p.factors for p in pairs)


def test_rotations_put_the_swapped_slot_in_every_position(tmp_path: Path) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("""
contrast: c
examples: [{a: x1, b: y1}, {a: x2, b: y2}, {a: x3, b: y3}]
slots:
  select:
    k1: {source: inline, values: [A]}
    k2: {source: inline, values: [B]}
    k3: {source: inline, values: [C]}
    u: {source: inline, values: [U]}
  sides: {a: [k1, k2, k3], b: [k1, u, k3]}
  order: rotations
""")
    fake = FakeAdapter({"JSON array": ['["first [[X1]], then ask [[X2]], then ask [[X3]]?"]']})
    _, pairs = generate.run(fake, spec, None, n_frames=1)
    out = [(p.a, p.b, p.factors["clause_order"]) for p in pairs]
    frame = "first {}, then ask {}, then ask {}?"
    assert out == [(frame.format("A", "B", "C"), frame.format("A", "U", "C"), "rotation_0"),
                   (frame.format("B", "C", "A"), frame.format("U", "C", "A"), "rotation_1"),
                   (frame.format("C", "A", "B"), frame.format("C", "A", "U"), "rotation_2")]
