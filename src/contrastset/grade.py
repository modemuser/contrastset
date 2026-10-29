"""`grade`: a behavioural census of what the subject model knows.

The model is never asked whether it knows something; that self-report is the
variable under study, so using it to score would be circular. Instead each question
is asked N times with sampling plus once greedily, and the answers are compared with
the reference answers.

Per question:
    correct_rate   share of sampled answers that match a reference answer
    refusal_rate   share of sampled answers that decline or hedge (crude phrase list)
    consistency    share of non-refused sampled answers equal to the modal (most
                   common) answer; correct answers count as the reference they matched
    bin            refused, then known, then confidently_wrong, else unstable
                   (see records.Thresholds for the cut-offs)

Per entity: bin X if every question is in bin X, else mixed.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from contrastset.adapters import Adapter
from contrastset.records import (
    Bin,
    Entity,
    EntityBin,
    Gate,
    GradedEntity,
    LogprobGrade,
    Provenance,
    Question,
    QuestionGrade,
    Sample,
    Thresholds,
    now,
    sha256_text,
)

MATCHER = "normalised-exact-or-contains/v1"
REFUSAL_DETECTOR = "phrase-list/v1 (crude: substring match on a fixed list, misses paraphrases)"

# --- normalisation and matching -----------------------------------------------------

ARTICLES = {"en": {"the", "a", "an"}, "de": {"der", "die", "das", "ein", "eine"}}
# Letters Unicode decomposition doesn't split into base letter + accent.
_FOLD = str.maketrans({"ł": "l", "ø": "o", "đ": "d", "æ": "ae", "œ": "oe", "ı": "i"})


def normalise(text: str, languages: Sequence[str] = ("en",)) -> str:
    """Lowercase, fold accents (Düren -> duren), turn punctuation into spaces, drop
    articles, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.casefold().translate(_FOLD))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\W_]+", " ", text)
    articles = set().union(*(ARTICLES.get(lang, set()) for lang in languages))
    return " ".join(word for word in text.split() if word not in articles)


def first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""


def match_answer(text: str, answers: Sequence[str], languages: Sequence[str] = ("en",)) -> str | None:
    """Return the reference answer the completion matches, or None.

    Matches if the normalised first line equals a normalised answer, or the normalised
    completion contains it as whole words. Known weaknesses: paraphrases and partial
    answers are missed, and an answer that is also part of the question (e.g. a river
    named like the town) matches whenever the model repeats the question."""
    line = normalise(first_line(text), languages)
    whole = f" {normalise(text, languages)} "
    for answer in answers:
        target = normalise(answer, languages)
        if target and (line == target or f" {target} " in whole):
            return answer
    return None


REFUSAL_PHRASES = {
    "en": [
        "i don't know", "i do not know", "i'm not sure", "i am not sure", "not sure",
        "no information", "don't have information", "do not have information",
        "don't have enough", "not familiar", "no idea", "not aware", "i cannot", "i can't",
        "unable to", "i couldn't find", "uncertain", "unknown", "no reliable",
    ],
    "de": ["ich weiß nicht", "keine ahnung", "nicht sicher", "keine informationen", "unbekannt"],
}


def detect_refusal(text: str, phrases: Sequence[str], languages: Sequence[str] = ("en",)) -> str | None:
    """Return the first refusal phrase found in the completion, or None."""
    whole = f" {normalise(text, languages)} "
    for phrase in phrases:
        if f" {normalise(phrase, languages)} " in whole:
            return phrase
    return None


# --- prompt templates ---------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """A prompt template: the exact text the subject model sees, with {{question}}
    (and optionally {{entity}}) filled in. It must render the model's full chat format,
    the same way the downstream probing experiment will.

    A template file may start with a YAML front-matter block that carries the
    refusal phrases written for ITS wording, since what a refusal looks like depends
    on what the prompt invites:

        ---
        refusal_phrases: ["i don't know", "not sure"]
        ---
        <|im_start|>user
        {{question}} If you do not know, say "I don't know".<|im_end|> ...

    The hash covers the prompt body only; the phrases are hashed separately."""

    name: str
    text: str
    refusal_phrases: tuple[str, ...] | None = None

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    def render(self, question: str, entity: str) -> str:
        return self.text.replace("{{question}}", question).replace("{{entity}}", entity)

    @classmethod
    def load(cls, name_or_path: str) -> Template:
        """A built-in name (see contrastset/templates/) or a path to a template file."""
        path = Path(name_or_path)
        if path.is_file():
            text, name = path.read_text(encoding="utf-8"), str(path)
        else:
            builtin = resources.files("contrastset") / "templates" / f"{name_or_path}.txt"
            if not builtin.is_file():
                raise FileNotFoundError(f"no template file or built-in template named {name_or_path!r}")
            text, name = builtin.read_text(encoding="utf-8"), f"builtin:{name_or_path}"
        phrases = None
        if text.startswith("---\n"):
            head, sep, body = text[4:].partition("\n---\n")
            if not sep:
                raise ValueError(f"template {name}: front matter starts with --- but never ends")
            meta = yaml.safe_load(head) or {}
            unknown = set(meta) - {"refusal_phrases"}
            if unknown:
                raise ValueError(f"template {name}: unknown front-matter keys {sorted(unknown)}")
            if "refusal_phrases" in meta:
                phrases = tuple(str(x) for x in meta["refusal_phrases"])
            text = body
        if "{{question}}" not in text:
            raise ValueError(f"template {name} has no {{{{question}}}} placeholder")
        return cls(name, text, phrases)


def graded_template_hashes(records: Iterable[Any]) -> set[str]:
    """The template hashes recorded in graded lines (full records or views)."""
    out = set()
    for r in records:
        prov = getattr(r, "provenance", None)
        if prov is None:
            prov = (getattr(r, "model_extra", None) or {}).get("provenance")
        if prov is None:
            continue
        template = (prov.template if hasattr(prov, "template") else prov.get("template")) or {}
        if template.get("sha256"):
            out.add(template["sha256"])
    return out


def require_same_template(template: Template, records: Iterable[Any]) -> None:
    """The rule: the template a model is probed or checked with must be the exact
    string it was graded with. Bins graded under one wording don't transfer to
    activations or answers collected under another (FINDINGS.md: a refusal
    invitation in the instruction changes what the model says first)."""
    hashes = graded_template_hashes(records)
    if not hashes:
        raise ValueError("the graded file records no template hash; cannot verify it matches")
    if len(hashes) > 1:
        raise ValueError(f"the graded file mixes {len(hashes)} templates; grade with one template")
    if template.sha256 not in hashes:
        raise ValueError(f"template {template.name} (sha256 {template.sha256[:12]}) is not the one the graded file "
                         f"was made with ({next(iter(hashes))[:12]}): grade and probe with the same string")


# --- grading ------------------------------------------------------------------------


def question_seed(base: int, entity: str, question_id: str) -> int:
    """The first seed for one question: the base seed plus a hash of (entity,
    question id). Sample i then uses question_seed + i. Every question gets its own
    random streams; with one shared set of seeds (base + i for every question, as
    before 2026-10-28) a single lucky or unlucky set of streams moves every bin in a
    file the same way (FINDINGS.md, parity)."""
    digest = hashlib.sha256(f"{entity}\x00{question_id}".encode()).hexdigest()
    return base + int(digest[:8], 16) % 100_000_000


@dataclass
class GradeConfig:
    n: int = 8  # sampled completions per question
    temperature: float = 0.7
    seed: int = 0  # base seed; sample i of a question uses question_seed(seed, entity, question) + i
    max_tokens: int = 32
    thresholds: Thresholds = field(default_factory=Thresholds)
    languages: tuple[str, ...] = ("en",)
    refusal_phrases: tuple[str, ...] | None = None  # None: built-in list for `languages`
    refusal_source: str = "built-in"  # where the phrases came from, for provenance
    # Ask this question id first; grade the rest only if it lands in known. Cheap
    # first filter for large candidate lists, e.g. "country" for towns: an entity
    # whose country the model doesn't know is not a familiar entity.
    gate: str | None = None
    # Also judge every question from the untempered log-probabilities (needs an
    # adapter with score(), i.e. hf). Stored beside the sampled numbers.
    logprob: bool = False

    def phrases(self) -> list[str]:
        if self.refusal_phrases is not None:
            return list(self.refusal_phrases)
        return [p for lang in self.languages for p in REFUSAL_PHRASES.get(lang, [])]


def bin_question(correct_rate: float, refusal_rate: float, consistency: float | None,
                 modal_is_correct: bool, t: Thresholds) -> Bin:
    if refusal_rate >= t.refusal_min:
        return Bin.refused
    if correct_rate >= t.correct_min:
        return Bin.known
    if consistency is not None and consistency >= t.consistency_min and not modal_is_correct:
        return Bin.confidently_wrong
    return Bin.unstable


def score_question(question: Question, sampled: Sequence[tuple[str, float, int | None]], greedy_text: str,
                   cfg: GradeConfig) -> QuestionGrade:
    """Turn raw completions into a QuestionGrade. Pure: no model, so a graded file can
    be re-scored from its stored samples (see rescore)."""
    phrases = cfg.phrases()

    def sample(text: str, temperature: float, seed: int | None, greedy: bool) -> Sample:
        matched = match_answer(text, question.answers, cfg.languages)
        return Sample(text=text, temperature=temperature, seed=seed, greedy=greedy,
                      normalised=normalise(first_line(text), cfg.languages), correct=matched is not None,
                      matched_answer=matched, refusal_phrase=detect_refusal(text, phrases, cfg.languages))

    samples = [sample(text, t, seed, False) for text, t, seed in sampled]
    greedy = sample(greedy_text, 0.0, None, True)
    n = len(samples)
    n_correct = sum(s.correct for s in samples)
    n_refused = sum(s.refusal_phrase is not None for s in samples)
    # Answer key for consistency: the matched reference if correct, else the
    # normalised first line; refusals that aren't also correct are left out.
    keys = [s.matched_answer if s.correct else s.normalised
            for s in samples if s.correct or s.refusal_phrase is None]
    modal, modal_count = Counter(keys).most_common(1)[0] if keys else (None, 0)
    consistency = modal_count / len(keys) if keys else None
    correct_rate, refusal_rate = n_correct / n, n_refused / n
    assert question.id is not None
    return QuestionGrade(
        id=question.id, q=question.q, answers=question.answers, samples=[*samples, greedy], n=n,
        n_correct=n_correct, n_refused=n_refused, correct_rate=correct_rate, refusal_rate=refusal_rate,
        consistency=consistency, modal_answer=modal, greedy_correct=greedy.correct,
        bin=bin_question(correct_rate, refusal_rate, consistency, modal in question.answers, cfg.thresholds),
    )


def prefix_free(forms: Sequence[str], tokenize: Any = None) -> list[str]:
    """Distinct forms, dropping any whose mass is already inside another's: one whose
    TOKEN sequence starts with another form's tokens. With a tokenizer, prefixes are
    compared as token ids: "South Bohemia" is a string prefix of "South Bohemian
    Region" but not a token prefix ("...emia" vs "...emian"), so both are kept.
    Without one (tests), plain string prefixes are used."""
    distinct = list(dict.fromkeys(f for f in forms if f))
    if tokenize is None:
        return [f for f in distinct if not any(o != f and f.startswith(o) for o in distinct)]
    toks = {f: tuple(tokenize(f)) for f in distinct}
    return [f for f in distinct
            if not any(o != f and len(toks[o]) < len(toks[f]) and toks[f][:len(toks[o])] == toks[o]
                       for o in distinct)]


def fold_accents(text: str) -> str:
    """Remove diacritics, keep case and punctuation: "Soča" -> "Soca", "Wisła" -> "Wisla"."""
    text = unicodedata.normalize("NFKD", text.translate(_FOLD_SURFACE))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


_FOLD_SURFACE = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ı": "i"})


def _variants(text: str) -> list[str]:
    """Surface forms the sampled matcher would also accept, for prefix scoring: as
    written, capitalised, and without diacritics (the matcher folds accents, so
    "Soca River" counts as Soča; the log-prob mass must count it too)."""
    out = [text, fold_accents(text)]
    out += [t[:1].upper() + t[1:] for t in out if t[:1].islower()]
    return list(dict.fromkeys(out))


def logprob_question(scorer: Any, prompt: str, question: Question, greedy_text: str,
                     cfg: GradeConfig) -> tuple[LogprobGrade, LogprobGrade | None]:
    """Judge a question from the model's probabilities. p_correct is a PREFIX mass:
    the probability that the answer BEGINS with a reference form (as written,
    capitalised or without diacritics). It undercounts wordy answers that state the
    name later ("River Fyris", "Yes, the river ... is the Fyrisån"), which the
    sampled matcher, matching anywhere, counts. Mass of the reference answers,
    of the refusal phrases and of the greedy answer, binned with the same thresholds
    as the sampled grade. Returns the grade under the untempered logits (T = 1) and,
    where the scorer supports it, under the sampling distribution (top-k/top-p, then
    cfg.temperature): the expected value of what sampling would show."""
    tokenize = getattr(scorer, "tokenize", None)
    marginal = hasattr(scorer, "prefix_mass")
    # With marginal prefix masses, string prefixes nest exactly ("Mur" contains
    # every "Mura..." completion); with single-segmentation scores, compare tokens.
    forms = prefix_free([v for a in question.answers for v in _variants(a)], None if marginal else tokenize)
    refusal_forms = prefix_free([v for p in cfg.phrases() for v in _variants(p)], tokenize)
    modal = first_line(greedy_text).rstrip(" .!,;:")
    both = hasattr(scorer, "score_many")

    def names(texts: Sequence[str]) -> tuple[float, float, float, float]:
        """Answers and the greedy answer: summed over every token segmentation, with
        the bound on what pruning dropped (raw, sampler, raw bound, sampler bound)."""
        if not marginal:
            r, s_ = masses(texts)
            return r, s_, 0.0, 0.0
        raw = samp = raw_lost = samp_lost = 0.0
        for text in texts:
            m = scorer.prefix_mass(prompt, text, sampler_temperature=cfg.temperature)
            raw += m["raw"]
            samp += m["sampler"] or 0.0
            raw_lost += m["raw_pruned_max"] or 0.0
            samp_lost += m["sampler_pruned_max"] or 0.0
        return min(1.0, raw), min(1.0, samp), raw_lost, samp_lost

    def masses(texts: Sequence[str]) -> tuple[float, float]:
        """Refusal phrases are scored in the canonical segmentation only, on purpose:
        they are everyday English ("I don't know"), where the canonical tokenization
        is almost certainly the one the model generates, and summing over
        segmentations for ~40 phrases per question would cost far more than it can
        change. Rare names are where segmentations diverge, and those (answers, the
        greedy answer) go through `names`."""
        if not texts:
            return 0.0, 0.0
        if both:
            scores = scorer.score_many(prompt, list(texts), sampler_temperature=cfg.temperature)
            return (min(1.0, sum(math.exp(sc.logprob) for sc in scores)),
                    min(1.0, sum(math.exp(sc.sampler_logprob) for sc in scores)))
        raw = min(1.0, sum(math.exp(scorer.score(prompt, t).logprob) for t in texts))
        return raw, raw

    c1, cs, c1_lost, cs_lost = names(forms)
    m1, ms, m1_lost, ms_lost = names([modal] if modal else [])
    r1, rs = masses(refusal_forms)
    modal_correct = match_answer(modal, question.answers, cfg.languages) is not None
    greedy_refusal = detect_refusal(greedy_text, cfg.phrases(), cfg.languages)
    if greedy_refusal and modal and not any(modal.startswith(f) for f in refusal_forms):
        r1, rs = min(1.0, r1 + m1), min(1.0, rs + ms)  # the greedy refusal's own mass was not counted yet

    def grade_at(p_correct: float, p_refusal: float, p_modal: float, temperature: float,
                 correct_lost: float, modal_lost: float) -> LogprobGrade:
        rest = 1.0 - p_refusal
        consistency = min(1.0, p_modal / rest) if rest > 1e-9 else None
        return LogprobGrade(p_correct=p_correct, p_refusal=p_refusal, p_modal=p_modal, modal_answer=modal,
                            modal_correct=modal_correct, consistency=consistency,
                            bin=bin_question(p_correct, p_refusal, consistency, modal_correct, cfg.thresholds),
                            forms=forms, refusal_forms=refusal_forms, temperature=temperature,
                            greedy_refusal_phrase=greedy_refusal,
                            p_correct_pruned_max=correct_lost if marginal else None,
                            p_modal_pruned_max=modal_lost if marginal else None)

    return (grade_at(c1, r1, m1, 1.0, c1_lost, m1_lost),
            grade_at(cs, rs, ms, cfg.temperature, cs_lost, ms_lost) if both else None)


def grade_question(adapter: Adapter, template: Template, entity: Entity, question: Question,
                   cfg: GradeConfig) -> QuestionGrade:
    prompt = template.render(question.q, entity.entity)
    [greedy_text] = adapter.complete(prompt, n=1, temperature=0.0, seed=None, max_tokens=cfg.max_tokens)
    if cfg.n == 0:  # log-prob only: one greedy decode plus teacher-forced scoring
        if not cfg.logprob:
            raise ValueError("-n 0 grades from log-probabilities alone: it needs --logprob")
        lp, lp_sampler = logprob_question(adapter, prompt, question, greedy_text, cfg)
        greedy = score_question(question, [(greedy_text, 0.0, None)], greedy_text, cfg).samples[-1]
        assert question.id is not None
        return QuestionGrade(
            id=question.id, q=question.q, answers=question.answers, samples=[greedy], n=0, n_correct=0,
            n_refused=0, correct_rate=lp.p_correct, refusal_rate=lp.p_refusal, consistency=lp.consistency,
            modal_answer=lp.modal_answer, greedy_correct=greedy.correct, bin=lp.bin, logprob=lp,
            logprob_sampler=lp_sampler, basis="logprob")
    assert question.id is not None
    seed = question_seed(cfg.seed, entity.entity, question.id)
    texts = adapter.complete(prompt, n=cfg.n, temperature=cfg.temperature, seed=seed, max_tokens=cfg.max_tokens)
    sampled = [(text, cfg.temperature, seed + i) for i, text in enumerate(texts)]
    graded = score_question(question, sampled, greedy_text, cfg)
    if cfg.logprob:
        graded.logprob, graded.logprob_sampler = logprob_question(adapter, prompt, question, greedy_text, cfg)
    return graded


def rescore(record: GradedEntity, phrases: Sequence[str], languages: Sequence[str] = ("en",)) -> GradedEntity:
    """Re-score a graded line from its stored raw samples with the current matcher,
    refusal detector and the line's own thresholds; everything else is kept. With
    unchanged code and phrases the result equals the input, byte for byte when
    written, so a committed graded file can be verified without the model."""
    cfg = GradeConfig(thresholds=record.thresholds, languages=tuple(languages), refusal_phrases=tuple(phrases))
    questions = []
    for qg in record.questions:
        sampled = [(s.text, s.temperature, s.seed) for s in qg.samples if not s.greedy]
        greedy = next(s.text for s in qg.samples if s.greedy)
        rescored = score_question(Question(id=qg.id, q=qg.q, answers=qg.answers), sampled, greedy, cfg)
        if qg.basis == "logprob":  # no samples to re-score from
            questions.append(qg)
            continue
        kept = {"prior_guess": qg.prior_guess, "prior_answer": qg.prior_answer, "logprob": qg.logprob,
                "logprob_sampler": qg.logprob_sampler}
        questions.append(rescored.model_copy(update={**kept, **(qg.model_extra or {})}))
    counts = Counter(q.bin for q in questions)
    return record.model_copy(update={"questions": questions, "bin": aggregate(counts),
                                     "bin_counts": {b: counts.get(b, 0) for b in Bin}})


def aggregate(bins: Iterable[Bin]) -> EntityBin:
    """Bin X if every question is in bin X, else mixed. Bins are not ordered, so there
    is no "minimum": known and confidently_wrong are different kinds of answer."""
    distinct = set(bins)
    return EntityBin(distinct.pop().value) if len(distinct) == 1 else EntityBin.mixed


def provenance(adapter: Adapter, template: Template, cfg: GradeConfig, entities_sha256: str | None) -> Provenance:
    return Provenance(
        verb="grade",
        model=adapter.describe(),
        template={"name": template.name, "sha256": template.sha256},
        sampling={"n": cfg.n, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "greedy_extra": True},
        seed=cfg.seed,
        inputs={"entities": entities_sha256} if entities_sha256 else {},
        config={
            "matcher": MATCHER,
            "refusal_detector": REFUSAL_DETECTOR,
            "refusal_phrases_sha256": sha256_text("\n".join(cfg.phrases())),
            "refusal_phrases_source": cfg.refusal_source,
            "languages": list(cfg.languages),
            "thresholds": cfg.thresholds.model_dump(),
            "gate": cfg.gate,
            "logprob": cfg.logprob,
            "seeding": "per question: seed + sha256(entity, question id) mod 1e8, then + i",
        },
    )


def grade_entity(adapter: Adapter, template: Template, entity: Entity, cfg: GradeConfig,
                 prov: Provenance) -> GradedEntity:
    gate = None
    ordered = list(entity.questions)
    if cfg.gate is not None:
        first = next((q for q in ordered if q.id == cfg.gate), None)
        if first is None:
            raise ValueError(f"entity {entity.entity!r} has no gate question {cfg.gate!r}")
        ordered = [first, *(q for q in ordered if q is not first)]
    questions = []
    for q in ordered:
        questions.append(grade_question(adapter, template, entity, q, cfg))
        if q.id == cfg.gate:
            gate = Gate(question=q.id, passed=questions[-1].bin is Bin.known)
            if not gate.passed:
                break
    counts = Counter(q.bin for q in questions)
    return GradedEntity(
        entity=entity.entity, type=entity.type, meta=entity.meta, bin=aggregate(counts),
        bin_counts={b: counts.get(b, 0) for b in Bin}, questions=questions, thresholds=cfg.thresholds,
        gate=gate, provenance=prov.model_copy(update={"created": now()}),  # when this line finished
        **(entity.model_extra or {}),  # unknown input fields pass through unchanged
    )


def grade(adapter: Adapter, template: Template, entities: Iterable[Entity], cfg: GradeConfig,
          entities_sha256: str | None = None) -> Iterator[GradedEntity]:
    """Grade entities one at a time, yielding each as soon as it is done."""
    prov = provenance(adapter, template, cfg, entities_sha256)
    for entity in entities:
        yield grade_entity(adapter, template, entity, cfg, prov)


# --- prior guesses ------------------------------------------------------------------


def group_key(record: GradedEntity, field: str | None) -> str:
    """The group an entity belongs to for prior-guess counting: "answer.<question id>"
    (that question's first reference answer, e.g. answer.country), "meta.<key>",
    "type", or None for one group."""
    if field is None:
        return ""
    if field == "type":
        return record.type
    if field.startswith("meta."):
        return str(record.meta.get(field[5:], ""))
    if field.startswith("answer."):
        qid = field[7:]
        q = next((q for q in record.questions if q.id == qid), None)
        return q.answers[0] if q and q.answers else ""
    raise ValueError(f"unknown group field {field!r}: use answer.<id>, meta.<key> or type")


def _modal(q: QuestionGrade, source: str) -> tuple[str | None, bool]:
    """(normalised modal answer, is it wrong) from the samples or from the log-probs."""
    if source == "logprob":
        if q.logprob is None:
            return None, False
        return normalise(q.logprob.modal_answer), not q.logprob.modal_correct
    return q.modal_answer, q.bin is not Bin.known and q.modal_answer not in q.answers


def annotate_priors(records: Sequence[GradedEntity], *, min_entities: int = 4,
                    group: str | None = None, source: str = "samples") -> list[GradedEntity]:
    """Mark default guesses. Within each group and question id, take the most common
    wrong modal answer; if at least `min_entities` entities share it, each of them
    gets prior_guess = True on that question (prior_answer = the guess). Every other
    question gets prior_guess = False. Needs the whole file, so it is a pass over a
    finished grade, not part of the streaming grade.

    source "samples": the modal sampled answer; "logprob": the greedy answer, the mode
    of the untempered distribution (needs a --logprob grade). The cell the sampler
    shifts most, so for anything probed on HF, define it from the log-probs."""
    wrong: dict[tuple[str, str], Counter[str]] = {}
    for r in records:
        key = group_key(r, group)
        for q in r.questions:
            modal, is_wrong = _modal(q, source)
            if modal and is_wrong:
                wrong.setdefault((key, q.id), Counter())[modal] += 1
    defaults = {k: c.most_common(1)[0] for k, c in wrong.items()}
    out = []
    for r in records:
        key = group_key(r, group)
        questions = []
        for q in r.questions:
            guess, count = defaults.get((key, q.id), (None, 0))
            modal, is_wrong = _modal(q, source)
            is_prior = count >= min_entities and is_wrong and modal == guess
            questions.append(q.model_copy(update={"prior_guess": is_prior, "prior_answer": guess if is_prior else None}))
        prov = r.provenance.model_copy(update={"config": {**r.provenance.config,
                                                          "priors": {"min_entities": min_entities, "group": group,
                                                                     "source": source}}})
        out.append(r.model_copy(update={"questions": questions, "provenance": prov}))
    return out


# --- reports ------------------------------------------------------------------------


def progress_line(i: int, total: int | None, g: GradedEntity) -> str:
    parts = [f"{q.id}={q.bin.value}(p {q.correct_rate:.2f} ok, {q.refusal_rate:.2f} ref)" if q.basis == "logprob"
             else f"{q.id}={q.bin.value}({q.n_correct}/{q.n} ok, {q.n_refused} ref)" for q in g.questions]
    count = f"{i}/{total}" if total else str(i)
    return f"[{count:>7}] {g.entity[:24]:<24} {g.bin.value:<17} " + "  ".join(parts)


def summary(graded: Sequence[GradedEntity]) -> str:
    """Bin counts, per-question bins, and the worst-question report.

    Worst-question report: for each mixed entity, would it become uniform (all
    questions in one bin) if a single question were dropped? If many entities hinge on
    the same question id, that question is probably badly worded or badly answered
    (missing aliases), and the problem is the question, not the model."""
    by_question: dict[str, Counter[Bin]] = {}
    for g in graded:
        for q in g.questions:
            by_question.setdefault(q.id, Counter())[q.bin] += 1
    total = Counter(q.bin for g in graded for q in g.questions)
    n_units = sum(total.values())
    lines = [f"graded {len(graded)} entities, {n_units} (entity, question) pairs"]
    gates = [g.gate for g in graded if g.gate is not None]
    if gates:
        stopped = sum(not gate.passed for gate in gates)
        lines.append(f"gate {gates[0].question}: {len(gates) - stopped} passed, {stopped} stopped after the gate "
                     "question (counted in question bins, excluded from entity bins and the report)")
        graded = [g for g in graded if g.gate is None or g.gate.passed]
    lines.append("question bins (the slot unit):")
    for qid, counts in by_question.items():
        lines.append(f"  {qid:<12} " + "  ".join(f"{b.value}={counts.get(b, 0)}" for b in Bin))
    lines.append(f"  {'all':<12} " + "  ".join(f"{b.value}={total.get(b, 0)}" for b in Bin))
    lines.append("correct_rate bands (<= 1/8 | middle | >= 7/8): middle = half-known, keep for graded-confidence work")
    for qid in by_question:
        rates = [q.correct_rate for g in graded for q in g.questions if q.id == qid]
        low = sum(r <= 0.125 for r in rates)
        high = sum(r >= 0.875 for r in rates)
        lines.append(f"  {qid:<12} {low} | {len(rates) - low - high} | {high}")
    entity_bins = Counter(g.bin for g in graded)
    lines.append("entity bins:  " + "  ".join(f"{b.value}={entity_bins.get(b, 0)}" for b in EntityBin))

    mixed = [g for g in graded if g.bin is EntityBin.mixed]
    lines.append(f"worst-question report ({len(mixed)} mixed):")
    if not mixed:
        return "\n".join(lines)
    small = [g for g in mixed if len(g.questions) < 3]
    if small:
        lines.append(f"  {len(small)} mixed entities have < 3 questions: dropping one always leaves a uniform"
                     " rest, so they say nothing about which question is at fault (skipped)")
    culprits: Counter[str] = Counter()
    destination: dict[str, Counter[str]] = {}
    flippable = 0
    for g in mixed:
        if len(g.questions) < 3:
            continue
        odd = [q for q in g.questions if len({o.bin for o in g.questions if o is not q}) == 1]
        if odd:
            flippable += 1
            q = odd[0]  # with >= 3 questions at most one question can be the odd one out
            rest_bin = next(o.bin for o in g.questions if o is not q)
            culprits[q.id] += 1
            destination.setdefault(q.id, Counter())[f"{q.bin.value}->{rest_bin.value}"] += 1
    considered = len(mixed) - len(small)
    lines.append(f"  {flippable} of {considered} would be uniform without their single odd question")
    for qid, count in culprits.most_common():
        detail = ", ".join(f"{k} x{v}" for k, v in destination[qid].most_common())
        lines.append(f"    drop {qid:<10} {count:>3} entities flip   ({detail})")
    return "\n".join(lines)
