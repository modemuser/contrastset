import io
import json

import pytest

from contrastset import jsonl
from contrastset.records import Entity, FileSource, GradedSource, InlineSource, Pair, Provenance, Spec


def test_entity_from_plan_example_gets_default_question_ids() -> None:
    line = '{"entity": "Düren", "type": "town", "questions": [{"q": "Which country is Düren in?", "answers": ["Germany"]}]}'
    entity = Entity.model_validate_json(line)
    assert entity.questions[0].id == "0"


def test_typo_in_field_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="answers"):
        Entity.model_validate({"entity": "Düren", "type": "town", "questions": [{"q": "?", "answer": ["Germany"]}]})


def test_duplicate_question_ids_rejected() -> None:
    q = {"id": "country", "q": "?", "answers": ["x"]}
    with pytest.raises(ValueError, match="duplicate"):
        Entity.model_validate({"entity": "E", "type": "t", "questions": [q, q]})


def test_jsonl_reports_bad_line_number(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "entities.jsonl"
    good = '{"entity": "A", "type": "t", "questions": [{"q": "?", "answers": ["x"]}]}'
    path.write_text(good + "\n\n" + '{"entity": "B"}\n')
    with pytest.raises(ValueError, match=r"entities.jsonl:3"):
        list(jsonl.read(path, Entity))


def test_pair_round_trips_through_jsonl() -> None:
    pair = Pair(
        pair_id="p0",
        a="The Rur flows through Düren.",
        b="The Rur flows through Monschau.",
        fills={"a": [{"slot": "a", "value": "Düren", "source": "graded", "question": "river", "bin": "known",
                      "entity_bin": "mixed", "type": "town"}],
               "b": [{"slot": "b", "value": "Monschau", "source": "graded", "question": "river", "bin": "unstable",
                      "entity_bin": "mixed", "type": "town"}]},
        contrast="whether the subject model knows the entity",
        spec_sha256="abc",
        provenance=Provenance(verb="generate", inputs={"spec": "abc", "graded": "def"}),
    )
    buf = io.StringIO()
    jsonl.write(pair, buf)
    data = json.loads(buf.getvalue())
    assert data["record"] == "contrastset.pair/1"
    assert Pair.model_validate(data) == pair


EXAMPLES = [{"a": f"a{i}", "b": f"b{i}"} for i in range(3)]


def test_spec_with_graded_selectors() -> None:
    spec = Spec.model_validate({
        "contrast": "whether the subject model knows the entity",
        "examples": EXAMPLES,
        "slots": {
            "select": {"a": {"source": "graded", "where": {"question": "river", "bin": "known"}},
                       "b": {"source": "graded", "where": {"question": "river", "bin": "unstable", "q.country": "known"}}},
            "same": ["type", "question"],
            "order": "both",
        },
    })
    b = spec.slots.select["b"]
    assert isinstance(b, GradedSource) and b.where["q.country"] == "known"
    assert spec.slots.needs_graded()
    assert spec.constraints.forbidden_words == [] and spec.checks is None


def test_spec_without_graded_file_mixes_inline_and_file_sources() -> None:
    spec = Spec.model_validate({
        "contrast": "nested belief versus flat statement over the same words",
        "examples": EXAMPLES,
        "slots": {
            "select": {"a": {"source": "inline", "values": ["Anna thinks that", {"value": "Anna knows that", "verb": "know"}]},
                       "b": {"source": "file", "path": "flat.jsonl", "where": {"verb": "know"}}},
            "same": ["verb"],
        },
        "checks": [{"use": "length", "severity": "fail", "options": {"max_word_diff": 2}},
                   {"use": "mychecks.blind_judge"}],
    })
    assert isinstance(spec.slots.select["a"], InlineSource) and isinstance(spec.slots.select["b"], FileSource)
    assert not spec.slots.needs_graded()
    assert spec.checks is not None and spec.checks[1].severity is None


def test_slot_source_must_be_named() -> None:
    with pytest.raises(ValueError, match="source"):
        Spec.model_validate({"contrast": "x", "examples": EXAMPLES, "slots": {"select": {"a": {"where": {}}}}})


def test_spec_needs_three_to_five_examples() -> None:
    with pytest.raises(ValueError):
        Spec.model_validate({"contrast": "x", "examples": [{"a": "a", "b": "b"}], "slots": {"select": {}}})
