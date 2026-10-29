"""Record schemas: the file format shared by every verb.

Every file contrastset reads or writes is JSONL (JSON Lines): one JSON object per
line, so it can be read by eye, streamed, and filtered with `jq`. Each output record
carries a `record` tag naming its kind and schema version, e.g. "contrastset.graded/1",
so a file explains itself without the command that produced it.

Record kinds:
    Entity        input to `grade`, usually hand-written or scripted
    GradedEntity  output of `grade`: what the subject model knows, with all raw evidence
    Spec          input to `generate` and `check` (YAML or JSON, not JSONL): what a contrast is
    Pair          output of `generate`: one matched pair of texts
    CheckFlag     output of `check`: per-pair verdicts from every active check

Terms:
    subject model    the small local model being studied; graded behaviourally
    generator model  a capable model that writes sentences; never decides what the
                     subject model knows
    bin              the knowledge category an entity or question is sorted into
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contrastset import __version__


class _Config(BaseModel):
    """Authored by hand (specs): an unknown field is an error, so a typo like
    "familar" fails loudly instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid")


class _Data(BaseModel):
    """A data record (entity, graded, pair, check lines). Files may come from other
    tools, so unknown fields are kept and passed through unchanged; jsonl.read warns
    once per file which ones it saw. Strictness applies only to the fields a verb
    actually uses: those are required and type-checked."""

    model_config = ConfigDict(extra="allow")


def now() -> datetime:
    """The current time, or SOURCE_DATE_EPOCH if set (the reproducible-builds
    convention), so regenerated files can match committed ones byte for byte."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    return datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- provenance -------------------------------------------------------------------


class Provenance(_Data):
    """Where a record came from. With the inputs and the models, a run is reproducible."""

    tool: Literal["contrastset"] = "contrastset"
    tool_version: str = __version__
    verb: Literal["grade", "generate", "check"]
    model: dict[str, Any] | None = None  # adapter.describe(); None for `check`
    template: dict[str, str] | None = None  # {"name": ..., "sha256": ...}
    sampling: dict[str, Any] = Field(default_factory=dict)  # n, temperature, max_tokens, ...
    seed: int | None = None
    inputs: dict[str, str] = Field(default_factory=dict)  # role -> sha256, e.g. {"graded": "ab12..."}
    config: dict[str, Any] = Field(default_factory=dict)  # verb-specific settings, e.g. matcher, thresholds
    created: datetime = Field(default_factory=now)


# --- grade input --------------------------------------------------------------------


class Question(_Data):
    q: str
    answers: list[str] = Field(min_length=1)  # any of these counts as correct
    # Stable id shared by the same kind of question across entities (e.g. "country"),
    # so badly worded questions can be spotted across the whole file. Defaults to the
    # question's index within its entity.
    id: str | None = None


class Entity(_Data):
    """One line of a `grade` input file."""

    entity: str
    type: str
    questions: list[Question] = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)  # free-form, e.g. a Wikidata id

    @model_validator(mode="after")
    def _default_question_ids(self) -> Entity:
        for i, question in enumerate(self.questions):
            if question.id is None:
                question.id = str(i)
        ids = [question.id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate question ids in entity {self.entity!r}: {ids}")
        return self


# --- grade output -------------------------------------------------------------------


class Bin(StrEnum):
    """Knowledge category of one question for one entity."""

    known = "known"  # mostly correct
    confidently_wrong = "confidently_wrong"  # mostly the same wrong answer
    unstable = "unstable"  # answers scatter: guessing
    refused = "refused"  # declines or hedges


class EntityBin(StrEnum):
    """Knowledge category of a whole entity: a Bin shared by all its questions, or mixed."""

    known = "known"
    confidently_wrong = "confidently_wrong"
    unstable = "unstable"
    refused = "refused"
    mixed = "mixed"  # questions disagree; per-question bins are kept in the record


class Thresholds(_Data):
    """How samples become bins. Precedence: refused, then known, then
    confidently_wrong, else unstable. Rates are fractions of sampled answers
    (the greedy answer is reported separately and never counts)."""

    refusal_min: float = 0.5  # refused if at least this share of samples refuse
    correct_min: float = 0.75  # known if at least this share is correct (6 of 8)
    consistency_min: float = 0.75  # confidently_wrong if the modal wrong answer has this share


class Sample(_Data):
    text: str  # raw completion, untouched
    temperature: float
    seed: int | None
    greedy: bool = False
    normalised: str  # after lowercasing, stripping punctuation, articles, whitespace
    correct: bool
    matched_answer: str | None = None  # which reference answer matched, if any
    # Crude phrase-list refusal detection; the matched phrase is kept so you can audit it.
    refusal_phrase: str | None = None


class LogprobGrade(_Data):
    """`grade --logprob`: the same question judged from the model's untempered
    probabilities instead of from samples. Masses are of completions that START with
    the given text (log P of the prefix), so "Rhine" also counts "Rhine River"."""

    p_correct: float  # mass of completions starting with a reference answer (forms made prefix-free)
    p_refusal: float  # the same for the template's refusal phrases
    p_modal: float  # mass of the greedy answer (the mode under untempered decoding)
    modal_answer: str  # greedy answer, first line, trailing punctuation removed
    modal_correct: bool
    consistency: float | None  # p_modal / (1 - p_refusal): modal share among non-refused mass
    bin: Bin  # from bin_question with p_correct, p_refusal, consistency and the same thresholds
    forms: list[str]  # answer forms scored
    refusal_forms: list[str]
    # The refusal phrase found in the greedy answer by the phrase detector, if any. A
    # greedy refusal that does not start with a listed phrase ("Sorry, I don't know.")
    # adds its own mass to p_refusal, so the refused bin cannot vanish for want of
    # samples to detect refusals in.
    greedy_refusal_phrase: str | None = None
    # Upper bounds on the mass that pruning in the segmentation expansion could have
    # dropped (HFAdapter.prefix_mass): the true value is in [p, p + pruned_max].
    p_correct_pruned_max: float | None = None
    p_modal_pruned_max: float | None = None
    # Which distribution the masses are under: 1.0 = untempered logits; otherwise the
    # sampling distribution (top-k/top-p on untempered, then this temperature), i.e.
    # the expected value of the sampled rates.
    temperature: float = 1.0


class QuestionGrade(_Data):
    id: str
    q: str
    answers: list[str]
    samples: list[Sample]  # every sampled completion plus the greedy one
    n: int  # number of sampled (non-greedy) completions
    n_correct: int
    n_refused: int
    correct_rate: float
    refusal_rate: float
    # Share of non-refused samples giving the modal (most common) answer. Correct
    # samples count as their matched reference answer, so "Germany" and
    # "Federal Republic of Germany" agree. None if every sample refused.
    consistency: float | None
    modal_answer: str | None
    greedy_correct: bool | None
    bin: Bin
    # Set by `grade --priors` (None until then): this question's wrong modal answer is
    # the most common wrong answer to it among at least N entities of the same group
    # (e.g. country): a default guess like "the region's big river", not a belief
    # about this entity.
    prior_guess: bool | None = None
    prior_answer: str | None = None  # the shared default guess, when prior_guess
    # `grade --logprob` only; left out of the output otherwise. `logprob` is under the
    # untempered logits; `logprob_sampler` under the sampling distribution.
    logprob: LogprobGrade | None = Field(default=None, exclude_if=lambda v: v is None)
    logprob_sampler: LogprobGrade | None = Field(default=None, exclude_if=lambda v: v is None)
    # "logprob" when graded with -n 0: no samples; bin, correct_rate, refusal_rate and
    # consistency are then the untempered log-prob values. Left out when "samples".
    basis: Literal["samples", "logprob"] = Field(default="samples", exclude_if=lambda v: v == "samples")


class SelfReport(_Data):
    """What the model says when asked directly whether it knows the entity.

    The self-report is the variable under study, so it is NEVER used for scoring;
    it is stored only for later comparison against the behavioural grade."""

    prompt: str
    text: str
    used_for_scoring: Literal[False] = False


class GradedEntity(_Data):
    """One line of `grade` output."""

    record: Literal["contrastset.graded/1"] = "contrastset.graded/1"
    entity: str
    type: str
    meta: dict[str, Any] = Field(default_factory=dict)
    bin: EntityBin
    # Rule turning per-question bins into the entity bin. "uniform_else_mixed": an
    # entity is in bin X only if every question is in X; anything else is mixed.
    aggregation: Literal["uniform_else_mixed"] = "uniform_else_mixed"
    bin_counts: dict[Bin, int]  # how many questions landed in each bin
    questions: list[QuestionGrade]  # only the questions actually asked (see gate)
    thresholds: Thresholds
    # With `grade --gate ID`, question ID is asked first and the rest only if it lands
    # in known. A gated-out entity carries just that question; its bin covers only it.
    gate: Gate | None = None
    self_report: SelfReport | None = None
    provenance: Provenance


class Gate(_Data):
    question: str
    passed: bool


# --- read views: the input contracts ------------------------------------------------
#
# What a verb READS is a view with only the fields it uses, so files from other tools
# work. What a verb WRITES is the full record above or below. Everything else in an
# input line is passed through unchanged.


def _same_kind(record: str | None, kind: str) -> None:
    if record is not None and not record.startswith(f"contrastset.{kind}/"):
        raise ValueError(f"this is a {record!r} record, expected contrastset.{kind}/...")


class QuestionView(_Data):
    """A graded question as `generate` reads it. Only `id` and `bin` are required;
    the rest is needed only if a spec uses it (e.g. correct_rate in a Range filter,
    answers for answer.<id>, q for describing frame positions)."""

    id: str
    bin: Bin
    q: str | None = None
    answers: list[str] = Field(default_factory=list)
    correct_rate: float | None = None
    consistency: float | None = None
    refusal_rate: float | None = None
    n_correct: int | None = None
    n: int | None = None
    prior_guess: bool | None = None


class GradedView(_Data):
    """A graded line as `generate` reads it: the contract for graded input. A full
    `grade` output line satisfies it; so does {"entity": "Düren", "questions":
    [{"id": "river", "bin": "known"}]} from any other tool."""

    # Fields of the full graded record that this view reads past: not reported as unknown.
    also_known: ClassVar[frozenset[str]] = frozenset({"aggregation", "bin_counts", "thresholds", "gate",
                                                      "self_report", "provenance"})

    record: str | None = None
    entity: str
    type: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    bin: EntityBin | None = None  # if absent: uniform_else_mixed over the questions
    questions: list[QuestionView] = Field(min_length=1)

    @model_validator(mode="after")
    def _kind(self) -> GradedView:
        _same_kind(self.record, "graded")
        if self.bin is None:
            bins = {q.bin for q in self.questions}
            self.bin = EntityBin(bins.pop().value) if len(bins) == 1 else EntityBin.mixed
        return self


class FillView(_Data):
    """A fill as `check` reads it; every field optional except the value."""

    value: str
    slot: str | None = None
    question: str | None = None
    bin: Bin | None = None


class PairView(_Data):
    """A pair as `check` reads it: the contract for pair input. Only `a` and `b` are
    required. `pair_id` defaults to the first 16 hex digits of sha256(a + newline + b),
    so check output can still be joined back to the input. Checks that need more
    (frame, fills) declare it and are skipped, not failed, where it is missing."""

    # Fields of the full pair record that this view reads past: not reported as unknown.
    also_known: ClassVar[frozenset[str]] = frozenset({"spec_sha256", "provenance"})

    record: str | None = None
    pair_id: str = ""
    a: str
    b: str
    frame: str | None = None
    fills: dict[str, list[FillView]] = Field(default_factory=dict)
    factors: dict[str, str] = Field(default_factory=dict)
    contrast: str | None = None

    @model_validator(mode="after")
    def _id(self) -> PairView:
        _same_kind(self.record, "pair")
        if not self.pair_id:
            self.pair_id = sha256_text(f"{self.a}\n{self.b}")[:16]
        return self


# --- generate input -----------------------------------------------------------------


class ExamplePair(_Config):
    """A hand-written pair. Style guidance for the generator only: its entities are
    never used as slot values, and `check` warns if a generated pair reuses them."""

    a: str
    b: str


class Range(_Config):
    """Inclusive numeric bounds for a `where` filter, e.g. {"max": 0.125}."""

    min: float | None = None
    max: float | None = None

    def contains(self, value: Any) -> bool:
        if not isinstance(value, int | float) or isinstance(value, bool):
            return False
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


Where = dict[str, bool | str | list[str] | Range]
"""Filter on candidate fields: field -> allowed value, list of values (any-of), or a
numeric Range."""


class GradedSource(_Config):
    """Slot values from the graded file passed with `--graded`.

    The unit is an (entity, question) pair, not an entity: knowledge is relational, so
    a town can be known for its country and unknown for its river. Each graded entity
    with k questions yields k candidates. Candidate fields for `where` and `same`:
        value, entity          the entity name
        question               the question id, e.g. "river"
        bin                    the bin of THIS question for this entity
        entity_bin             the entity-level bin (uniform_else_mixed)
        correct_rate, consistency, refusal_rate, n_correct, n
                               this question's numbers; filter with a Range,
                               e.g. {"correct_rate": {"max": 0.125}}
        type                   the entity type
        meta.<key>             a key inside the entity's meta
        q.<question id>        the bin of another question of the same entity
        answer.<question id>   that question's first reference answer, e.g.
                               answer.country = "Germany"

    E.g. {"question": "river", "correct_rate": {"max": 0.125}} selects towns whose
    river the model almost never gets right.

    The middle correct_rate band (half-known facts, e.g. {"min": 0.25, "max": 0.75})
    is a resource, not junk: keep it out of clean known-versus-unknown contrast sets,
    but select it when studying graded confidence, since facts near the decision
    boundary are where a continuous readout shows itself. At 8 samples the middle
    band is noisy (FINDINGS.md); grade with more samples before relying on it.

    `familiar` keeps only candidates whose entity the model recognises: every OTHER
    question of the entity is in bin known. Then only the
    selected fact varies, not whether the entity is familiar at all: "fact unknown,
    entity familiar", the harder and cleaner side of the known-entity versus
    unknown-entity distinction in the 2024 entity-recognition interpretability work.
    A list of question ids requires just those to be known; false turns it off.
    Unset: the spec's `slot_defaults.familiar`, which is false unless the spec says so.
    """

    source: Literal["graded"]
    where: Where = Field(default_factory=dict)
    familiar: bool | list[str] | None = None


class InlineSource(_Config):
    """Slot values written in the spec itself, e.g. hand-written wordings.

    Each value is a string or an object with at least a "value" key; other keys are
    candidate fields for `where` and `same`."""

    source: Literal["inline"]
    values: list[str | dict[str, Any]] = Field(min_length=1)
    where: Where = Field(default_factory=dict)


class FileSource(_Config):
    """Slot values from a plain JSONL file, one object per line with at least a
    "value" key. A relative path is resolved against the spec file's folder."""

    source: Literal["file"]
    path: str
    where: Where = Field(default_factory=dict)


SlotSource = Annotated[GradedSource | InlineSource | FileSource, Field(discriminator="source")]


Level = Literal["same", "different"]


def _both_levels() -> list[Level]:
    return ["same", "different"]


class Factor(_Config):
    """A relation between slots that the design controls instead of leaving to chance.

    For each slot pair in `between`, `field` is either equal ("same") or not
    ("different"). A combination is at level "same" if every listed pair is equal,
    "different" if none is; combinations in between are dropped, since their two
    sides would differ in the factor and confound the contrast. Pairs are balanced
    across levels and record their level in Pair.factors.

    E.g. whether the neighbour town is in the same country:
        field: answer.country
        between: [[shared, known], [shared, unknown]]
    """

    field: str
    between: list[tuple[str, str]] = Field(min_length=1)
    levels: list[Level] = Field(default_factory=_both_levels)


class Slots(_Config):
    """Where slot values come from, which side each slot appears on, and how values
    are combined.

    `same` and `order` work the same way whatever the sources are: every candidate,
    graded or not, is seen as a flat set of fields. Within one pair every slot gets a
    different value.

    Example, "two known clauses" versus "one known, one unknown":
        select: {shared: <known>, known: <known>, unknown: <unstable>}
        sides:  {a: [shared, known], b: [shared, unknown]}
        order:  both
    """

    select: dict[str, SlotSource]  # slot name -> source, e.g. {"a": ..., "b": ...}
    # Slots on each side, in clause order. A slot on both sides is shared: the same
    # value on both. Default: side a holds slot "a", side b holds slot "b".
    sides: dict[Literal["a", "b"], list[str]] | None = None
    same: list[str] = Field(default_factory=list)  # fields that must be equal across slots, e.g. ["question"]
    # "both": also emit every pair with the clause order reversed on both sides.
    # "rotations": every cyclic rotation of the clause order on both sides; with three
    # clauses each slot is once first, middle and last, same frame and same values.
    # The order used is recorded as Pair.factors["clause_order"].
    order: Literal["fixed", "both", "rotations"] = "fixed"
    # Controlled factors, crossed and balanced; see Factor.
    factors: dict[str, Factor] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_sides(self) -> Slots:
        sides = self.resolved_sides()
        for side, names in sides.items():
            unknown = [n for n in names if n not in self.select]
            if unknown:
                raise ValueError(f"side {side} uses slots {unknown} that are not in select")
        for name, factor in self.factors.items():
            unknown = [n for pair in factor.between for n in pair if n not in self.select]
            if unknown:
                raise ValueError(f"factor {name} relates slots {unknown} that are not in select")
        if len(sides["a"]) != len(sides["b"]):
            raise ValueError("sides a and b must have the same number of slots, so one frame fits both")
        return self

    def resolved_sides(self) -> dict[Literal["a", "b"], list[str]]:
        if self.sides is not None:
            return {"a": self.sides.get("a", []), "b": self.sides.get("b", [])}
        if "a" not in self.select or "b" not in self.select:
            raise ValueError('without `sides`, select must name slots "a" and "b"')
        return {"a": ["a"], "b": ["b"]}

    def needs_graded(self) -> bool:
        return any(isinstance(src, GradedSource) for src in self.select.values())


class SlotDefaults(_Config):
    """Defaults for every graded slot of a spec, so a rule like "only familiar
    entities" is stated once and not remembered per slot."""

    familiar: bool | list[str] = False


class Constraints(_Config):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    forbidden_words: list[str] = Field(default_factory=list)
    min_words: int | None = None
    max_words: int | None = None
    # Written "register" in files; renamed in Python because pydantic models already
    # have a `register` method.
    register_: str | None = Field(default=None, alias="register")  # e.g. "neutral encyclopaedic"
    language: str | None = None  # e.g. "en"
    # Every placeholder in its own clause, no lists, no extra task. For multi-clause
    # question designs where clause position is a controlled variable; wrong for
    # statement-style contrasts ("She said: [[X1]].").
    one_clause_per_slot: bool = False


Severity = Literal["warn", "fail"]


class CheckRef(_Config):
    """One active check. `use` is a built-in name (e.g. "length") or a dotted path to
    your own check (e.g. "mychecks.blind_judge"); see contrastset.checks."""

    use: str
    severity: Severity | None = None  # overrides the severity the check reports
    options: dict[str, Any] = Field(default_factory=dict)  # passed to the check as keyword arguments


class Spec(_Config):
    """Input to `generate` and `check`. Says what a contrast is, never what a model
    knows: a graded file is passed separately, so one spec works against any
    subject model."""

    contrast: str  # the single variable allowed to differ, in plain words
    # 3 to 5 hand-written example pairs: style guidance for the generator model. Not
    # needed when `frames` is given; `check` uses them for similarity checks.
    examples: list[ExamplePair] = Field(default_factory=list, max_length=5)
    # Where slot values come from. Only `generate` needs it: a spec that only
    # configures `check` for hand-written pairs has no slots.
    slots: Slots | None = None
    slot_defaults: SlotDefaults = Field(default_factory=SlotDefaults)
    # Hand-written frames with [[X1]], [[X2]], ... placeholders. Given: no generator
    # model is called at all.
    frames: list[str] | None = None
    constraints: Constraints = Field(default_factory=Constraints)
    # Checks `check` runs on pairs made from this spec. None: the built-in defaults.
    checks: list[CheckRef] | None = None

    @model_validator(mode="after")
    def _frames_or_examples(self) -> Spec:
        if self.slots is not None and self.frames is None and len(self.examples) < 3:
            raise ValueError("give 3 to 5 `examples` (the generator model writes frames from them) "
                             "or hand-written `frames`")
        return self


# --- generate output ----------------------------------------------------------------


class Fill(_Data):
    """A value placed into a slot. Graded fills are (entity, question) pairs and also
    carry the question, its bin, and the entity's type and entity-level bin."""

    slot: str
    value: str
    source: Literal["graded", "inline", "file"]
    question: str | None = None  # question id, for graded fills
    question_text: str | None = None
    bin: Bin | None = None  # bin of `question` for this entity
    entity_bin: EntityBin | None = None
    type: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)  # remaining candidate fields


class Pair(_Data):
    """One line of `generate` output."""

    record: Literal["contrastset.pair/1"] = "contrastset.pair/1"
    pair_id: str
    a: str
    b: str
    # The text both sides were filled from, with [[X1]], [[X2]], ... placeholders, so
    # a and b differ only where the placeholders are. None if not made from a frame.
    frame: str | None = None
    factors: dict[str, str] = Field(default_factory=dict)  # level of each controlled factor
    # Values used in each side, in order of appearance in the text.
    fills: dict[Literal["a", "b"], list[Fill]]
    contrast: str
    spec_sha256: str
    # Generator model, prompt version, and the hashes of the spec and graded file
    # (provenance.inputs), so a pair always says whose knowledge it assumed.
    provenance: Provenance


# --- check output -------------------------------------------------------------------


class Verdict(_Data):
    """One finding of one check on one pair. A check that finds nothing returns None."""

    check_id: str
    severity: Severity
    reason: str
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)


class CheckFlag(_Data):
    """One line of `check` output: does this pair differ only in the contrast?"""

    record: Literal["contrastset.check/1"] = "contrastset.check/1"
    pair_id: str
    verdict: Literal["ok", "warn", "fail"]  # worst severity among verdicts, "ok" if none
    verdicts: list[Verdict] = Field(default_factory=list)
    # Checks that could not run on this pair (a field or resource they need is
    # missing). A skip is not a pass: see `check --require`.
    skipped: dict[str, str] = Field(default_factory=dict)  # check id -> why
    provenance: Provenance
