import json

import pytest

from contrastset import cli, grade
from contrastset.adapters import FakeAdapter
from contrastset.records import Bin, Entity, EntityBin, Thresholds

T = grade.Template("t", "Q: {{question}}\nA:")


def entity(name: str, *questions: tuple[str, str, list[str]]) -> Entity:
    return Entity.model_validate({"entity": name, "type": "town",
                                  "questions": [{"id": i, "q": q, "answers": a} for i, q, a in questions]})


def test_normalise_folds_case_accents_punctuation_articles() -> None:
    assert grade.normalise("  The Rhône!  ") == "rhone"
    assert grade.normalise("Wisła") == grade.normalise("wisla")
    assert grade.normalise("Nordrhein-Westfalen.") == "nordrhein westfalen"


def test_match_answer_exact_and_containment() -> None:
    assert grade.match_answer("Germany.", ["Germany"]) == "Germany"
    assert grade.match_answer("It is in the Federal Republic of Germany", ["Germany"]) == "Germany"
    assert grade.match_answer("Germanyland", ["Germany"]) is None
    assert grade.match_answer("Czechia", ["Czech Republic", "Czechia"]) == "Czechia"


def test_refusal_detection_reports_phrase() -> None:
    phrases = grade.REFUSAL_PHRASES["en"]
    assert grade.detect_refusal("I don't know.", phrases) == "i don't know"
    assert grade.detect_refusal("Germany", phrases) is None


def test_bins() -> None:
    cfg = grade.GradeConfig(n=8)
    fake = FakeAdapter({
        "Düren": ["Germany"],
        "Atlantis": ["I don't know."],
        "Xyz": ["Mars"],  # always the same wrong answer
        "Qrs": ["Mars", "Venus", "Pluto", "Ceres"],  # scattered wrong answers
        "Mid": ["Germany", "France"],  # half right: not known, not consistent
    })
    def g(name: str) -> Bin:
        return grade.grade_question(fake, T, entity(name, ("0", f"Where is {name}?", ["Germany"])),
                                    entity(name, ("0", f"Where is {name}?", ["Germany"])).questions[0], cfg).bin
    assert g("Düren") is Bin.known
    assert g("Atlantis") is Bin.refused
    assert g("Xyz") is Bin.confidently_wrong
    assert g("Qrs") is Bin.unstable
    assert g("Mid") is Bin.unstable


def test_question_grade_keeps_every_sample_and_counts() -> None:
    fake = FakeAdapter({"Düren": ["Germany", "Germany", "Germany", "France"]})
    e = entity("Düren", ("country", "Which country is Düren in?", ["Germany"]))
    qg = grade.grade_question(fake, T, e, e.questions[0], grade.GradeConfig(n=8, seed=0))
    assert len(qg.samples) == 9 and qg.samples[-1].greedy and qg.samples[-1].seed is None
    first = grade.question_seed(0, "Düren", "country")
    assert [s.seed for s in qg.samples[:8]] == list(range(first, first + 8))
    expected_correct = sum((first + i) % 4 != 3 for i in range(8))  # the fake answers France on seed % 4 == 3
    assert qg.n_correct == expected_correct and qg.correct_rate == expected_correct / 8
    assert qg.greedy_correct
    assert fake.calls[0]["prompt"] == "Q: Which country is Düren in?\nA:"


def test_aggregate_uniform_else_mixed() -> None:
    assert grade.aggregate([Bin.known, Bin.known]) is EntityBin.known
    assert grade.aggregate([Bin.refused, Bin.refused]) is EntityBin.refused
    assert grade.aggregate([Bin.known, Bin.confidently_wrong]) is EntityBin.mixed


def test_thresholds_are_configurable() -> None:
    fake = FakeAdapter({"Mid": ["Germany", "France"]})
    e = entity("Mid", ("0", "Where is Mid?", ["Germany"]))
    lenient = grade.GradeConfig(n=8, thresholds=Thresholds(correct_min=0.5))
    assert grade.grade_question(fake, T, e, e.questions[0], lenient).bin is Bin.known


def test_worst_question_report_finds_the_odd_question() -> None:
    fake = FakeAdapter({"country": ["Germany"], "region": ["NRW"], "river": ["I don't know"]})
    entities = [entity(f"Town{i}", ("country", f"country of Town{i}?", ["Germany"]),
                       ("region", f"region of Town{i}?", ["NRW"]),
                       ("river", f"river of Town{i}?", ["Rur"])) for i in range(3)]
    graded = list(grade.grade(fake, T, entities, grade.GradeConfig(n=4)))
    assert all(g.bin is EntityBin.mixed for g in graded)
    report = grade.summary(graded)
    assert "3 of 3 would be uniform" in report
    assert "drop river" in report and "refused->known x3" in report


def test_cli_grade_streams_jsonl_with_provenance(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "e.jsonl"
    path.write_text("\n".join(json.dumps({"entity": n, "type": "town", "questions": [
        {"q": f"Which country is {n} in?", "answers": ["Germany"]}]}) for n in ["Düren", "Atlantis", "Aachen"]))
    fake = FakeAdapter({"Düren": ["Germany"]})
    monkeypatch.setattr(cli, "make_adapter", lambda *a, **k: fake)
    assert cli.main(["grade", str(path), "--model", "fake", "--template", "qwen3-nothink", "--limit", "2", "-n", "2"]) == 0
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert [line["entity"] for line in lines] == ["Düren", "Atlantis"]
    assert [line["bin"] for line in lines] == ["known", "refused"]
    prov = lines[0]["provenance"]
    assert prov["template"]["name"] == "builtin:qwen3-nothink" and prov["inputs"]["entities"]
    assert prov["config"]["matcher"] == grade.MATCHER
    assert "[    1/2] Düren" in captured.err and "entity bins:" in captured.err


def test_template_needs_question_placeholder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "t.txt"
    bad.write_text("no placeholder")
    with pytest.raises(ValueError, match="placeholder"):
        grade.Template.load(str(bad))


def test_gate_stops_after_unknown_gate_question() -> None:
    fake = FakeAdapter({"country of Düren": ["Germany"], "river of Düren": ["Rur"],
                        "country of Nowhere": ["I don't know"], "river of Nowhere": ["Rur"]})
    entities = [entity(n, ("river", f"river of {n}?", ["Rur"]), ("country", f"country of {n}?", ["Germany"]))
                for n in ["Düren", "Nowhere"]]
    duren, nowhere = grade.grade(fake, T, entities, grade.GradeConfig(n=4, gate="country"))
    assert [q.id for q in duren.questions] == ["country", "river"] and duren.gate and duren.gate.passed
    assert [q.id for q in nowhere.questions] == ["country"] and nowhere.gate and not nowhere.gate.passed
    assert not any("river of Nowhere" in c["prompt"] for c in fake.calls)
    assert "gate country: 1 passed, 1 stopped" in grade.summary([duren, nowhere])


def test_priors_mark_shared_wrong_answers_within_a_group() -> None:
    def town(name: str, country: str, answers: list[str], said: str) -> Entity:
        return Entity.model_validate({"entity": name, "type": "town", "questions": [
            {"id": "country", "q": f"country of {name}?", "answers": [country]},
            {"id": "river", "q": f"river of {name}?", "answers": answers}]})
    towns = [town(f"SK{i}", "Slovakia", [f"R{i}"], "Váh") for i in range(5)] + \
            [town("SK-known", "Slovakia", ["Váh"], "Váh"), town("CZ0", "Czechia", ["Otava"], "Váh")]
    fake = FakeAdapter(lambda p, t, s: "Slovakia" if "country of SK" in p else "Czechia" if "country of CZ" in p
                       else "Váh")
    graded = list(grade.grade(fake, T, towns, grade.GradeConfig(n=4)))
    marked = grade.annotate_priors(graded, min_entities=4, group="answer.country")
    river = {g.entity: next(q for q in g.questions if q.id == "river") for g in marked}
    assert all(river[f"SK{i}"].prior_guess and river[f"SK{i}"].prior_answer == "vah" for i in range(5))
    assert river["SK-known"].prior_guess is False  # Váh is right for it: knowledge, not a default
    assert river["CZ0"].prior_guess is False  # a different group, and alone there
    assert grade.annotate_priors(graded, min_entities=6, group="answer.country")[0].questions[1].prior_guess is False
    assert marked[0].provenance.config["priors"] == {"min_entities": 4, "group": "answer.country", "source": "samples"}


def test_bool_selector_matches_prior_guess() -> None:
    from pathlib import Path

    from contrastset import generate
    from contrastset.records import GradedSource, GradedView

    views = [GradedView.model_validate({"entity": e, "questions": [{"id": "river", "bin": "confidently_wrong",
                                                                     "prior_guess": p}]})
             for e, p in [("A", True), ("B", False), ("C", None)]]
    cands = generate.graded_candidates(views)
    src = GradedSource.model_validate({"source": "graded", "where": {"prior_guess": True}})
    assert [c["value"] for c in generate.slot_candidates(src, cands, Path("."))] == ["A"]
    src = GradedSource.model_validate({"source": "graded", "where": {"prior_guess": False}})
    assert [c["value"] for c in generate.slot_candidates(src, cands, Path("."))] == ["B"]  # unset never matches


def test_logprob_mode_bins_from_untempered_masses() -> None:
    import math
    from types import SimpleNamespace

    probs = {"Mur": 0.55, "Mura": 0.30, "Drava": 0.30, "I don't know": 0.10}

    class FakeScorer(FakeAdapter):
        def score(self, prompt: str, continuation: str) -> SimpleNamespace:
            return SimpleNamespace(logprob=math.log(probs.get(continuation, 1e-6)))

    fake = FakeScorer({"Graz": ["Mur"]})
    e = entity("Graz", ("river", "Which river flows through Graz?", ["Mur", "Mura"]))
    cfg = grade.GradeConfig(n=4, logprob=True, refusal_phrases=("i don't know",))
    qg = grade.grade_question(fake, T, e, e.questions[0], cfg)
    lp = qg.logprob
    assert lp is not None and lp.forms == ["Mur"]  # "Mura" starts with "Mur": counted once
    assert lp.p_correct == pytest.approx(0.55, abs=1e-4) and lp.p_refusal == pytest.approx(0.10, abs=1e-4)
    assert lp.modal_answer == "Mur" and lp.modal_correct
    assert lp.consistency == pytest.approx(0.55 / 0.90, abs=1e-4)
    assert lp.bin is Bin.unstable  # 0.55 < correct_min 0.75, and the modal answer is correct
    plain = grade.grade_question(fake, T, e, e.questions[0], grade.GradeConfig(n=4))
    assert plain.logprob is None and "logprob" not in plain.model_dump()


def test_prefix_free_compares_tokens_not_strings() -> None:
    tokens = {"South Bohemia": (1, 2, 3), "South Bohemian Region": (1, 2, 4, 5), "Mur": (7,), "Mura": (7, 8)}
    assert grade.prefix_free(list(tokens), tokens.__getitem__) == ["South Bohemia", "South Bohemian Region", "Mur"]
    assert grade.prefix_free(list(tokens)) == ["South Bohemia", "Mur"]  # the string fallback would drop the full form


def test_logprob_only_grade_and_logprob_priors() -> None:
    import math
    from types import SimpleNamespace

    class FakeScorer(FakeAdapter):
        def score(self, prompt: str, continuation: str) -> SimpleNamespace:
            p = {"Váh": 0.7, "R0": 0.05, "R1": 0.05, "R2": 0.05, "R3": 0.05}.get(continuation, 1e-6)
            return SimpleNamespace(logprob=math.log(p))

    towns = [entity(f"SK{i}", ("country", f"country of SK{i}?", ["Slovakia"]), ("river", f"river of SK{i}?", [f"R{i}"]))
             for i in range(4)]
    fake = FakeScorer(lambda p, t, s: "Slovakia" if "country" in p else "Váh")
    cfg = grade.GradeConfig(n=0, logprob=True, refusal_phrases=("i don't know",))
    graded = list(grade.grade(fake, T, towns, cfg))
    river = graded[0].questions[1]
    assert river.basis == "logprob" and river.n == 0 and river.samples[0].greedy
    assert river.correct_rate == pytest.approx(0.05, abs=1e-3) and river.bin.value == "unstable"
    assert river.logprob is not None and river.logprob.modal_answer == "Váh" and not river.logprob.modal_correct
    assert "basis" not in grade.grade_question(fake, T, towns[0], towns[0].questions[1],
                                               grade.GradeConfig(n=2)).model_dump()  # default left out
    marked = grade.annotate_priors(graded, min_entities=4, group="answer.country", source="logprob")
    assert all(next(q for q in g.questions if q.id == "river").prior_guess for g in marked)
    with pytest.raises(ValueError, match="--logprob"):
        grade.grade_question(fake, T, towns[0], towns[0].questions[1], grade.GradeConfig(n=0))


def test_logprob_only_keeps_refusals_the_prefix_list_misses() -> None:
    import math
    from types import SimpleNamespace

    class FakeScorer(FakeAdapter):
        def score(self, prompt: str, continuation: str) -> SimpleNamespace:
            p = {"Sorry, I don't know": 0.7, "I don't know": 0.05}.get(continuation, 1e-6)
            return SimpleNamespace(logprob=math.log(p))

    fake = FakeScorer(lambda p, t, s: "Sorry, I don't know.")
    e = entity("Zyx", ("river", "river of Zyx?", ["Nil"]))
    q = grade.grade_question(fake, T, e, e.questions[0], grade.GradeConfig(n=0, logprob=True, refusal_phrases=("i don't know",)))
    assert q.samples[0].refusal_phrase == "i don't know"  # the detector ran on the greedy answer
    assert q.logprob is not None and q.logprob.greedy_refusal_phrase == "i don't know"
    assert q.refusal_rate == pytest.approx(0.75, abs=1e-3) and q.bin.value == "refused"


def test_questions_get_their_own_random_streams() -> None:
    a = grade.question_seed(0, "Ronda", "river")
    assert a != grade.question_seed(0, "Písek", "river") != grade.question_seed(0, "Ronda", "region")
    assert grade.question_seed(5, "Ronda", "river") == a + 5  # the base seed still shifts everything


def test_logprob_forms_include_accent_folded_spellings() -> None:
    assert grade._variants("Soča") == ["Soča", "Soca"]
    assert grade._variants("świętokrzyskie") == ["świętokrzyskie", "swietokrzyskie", "Świętokrzyskie", "Swietokrzyskie"]
    assert grade.fold_accents("Wisła, Łódź") == "Wisla, Lodz"
