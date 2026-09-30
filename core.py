"""The part of this program that holds no opinion about any particular job.

Reads only its arguments and interface.json, and does not import app — so the Plan is out
of reach by the import graph, not by remembering. Same arguments, same answer, for ever,
which is what lets a Replay run any of it against a plan that is not the deployed one.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


# ── the closed vocabulary: data in interface.json, read here and not held ─
Json = dict[str, Any]
INTERFACE: Json = json.loads(Path("interface.json").read_text(encoding="utf-8"))
WIDGETS: Json = INTERFACE["widgets"]
OPTION_WIDGETS = {name: w["when"] for name, w in WIDGETS.items() if w.get("opt")}
BOUNDS: Json = INTERFACE["bounds"]
VOCABULARY: Json = INTERFACE["vocabulary"]

# Two kinds of request, each with its own noise floor: they measure 10x apart.
NOISE_KINDS = ("choice", "score")
LANGUAGE_CHOICES = {"en": "written in English", "ja": "written in Japanese",
                    "other": "written in some other language"}
WRITTEN_POLICY = ("per_dimension", "reserve_early", "min_q", "max_q",
                  "done_threshold", "gap_k")
INTEGER_POLICY = ("per_dimension", "reserve_early", "min_q", "max_q", "samples")

# ── plan validation: the one place a human-written constant survives ──────

def in_bounds(name: str, value: object, where: str = "") -> str | None:
    low, high = BOUNDS[name]
    where = where or f"policy.{name}"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"{where}: expected a number, got {value!r}"
    if name in INTEGER_POLICY and not isinstance(value, int):
        return f"{where}: expected a whole number, got {value!r}"
    if not low <= value <= high:
        return f"{where}: {value} is outside the allowed range [{low}, {high}]"
    return None

def check_plan(plan: Json) -> list[str]:
    """Every way a generated plan can be unusable, named so the writer can fix it.

    A generated plan arrives against a strict schema, so the shape checks come first for
    the hand-edited one `--check` exists to read: it has to name the problem rather than
    traceback out of the middle of a rule about something else.
    """
    if not isinstance(plan, dict):
        return [f"a plan is an object, and this is a {type(plan).__name__}"]
    errors: list[str] = []
    for key in ("from", "lang", "policy", "done", "total", "tie_break", "coverage",
                "dimensions", "fit", "questions"):
        if key not in plan:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    errors = [f"{key}: expected {kind.__name__}, got {type(plan[key]).__name__}"
              for key, kind in (("policy", dict), ("done", dict), ("fit", dict),
                                ("dimensions", list), ("questions", list))
              if not isinstance(plan[key], kind)]
    if errors:
        return errors

    policy, dimensions = plan["policy"], plan["dimensions"]
    weights = {d["name"]: d.get("weight") for d in dimensions
               if isinstance(d, dict) and "name" in d}
    for name in WRITTEN_POLICY + ("samples",):
        if name not in policy:
            errors.append(f"policy.{name}: missing")
        elif error := in_bounds(name, policy[name]):
            errors.append(error)
    floors = policy.get("noise_floor")
    if not isinstance(floors, dict):
        errors.append(f"policy.noise_floor: expected one floor per question kind"
                      f" {list(NOISE_KINDS)}, got {floors!r}")
    else:
        for kind in NOISE_KINDS:
            if kind not in floors:
                errors.append(f"policy.noise_floor.{kind}: missing")
            elif error := in_bounds("noise_floor", floors[kind],
                                    f"policy.noise_floor.{kind}"):
                errors.append(error)
    # JSON Schema cannot say this, so it lives here rather than in the bounds table.
    if isinstance(policy.get("min_q"), int) and isinstance(policy.get("max_q"), int):
        if policy["max_q"] <= policy["min_q"]:
            errors.append(f"policy.max_q ({policy['max_q']}) must exceed"
                          f" policy.min_q ({policy['min_q']}); otherwise `done` never decides")
    # Nor these: the turn budget and the coverage debt are written independently.
    owed = policy.get("per_dimension")
    if isinstance(owed, int) and isinstance(policy.get("max_q"), int) and dimensions:
        debt, head = owed * len(dimensions), policy.get("reserve_early")
        if debt > policy["max_q"]:
            errors.append(f"policy.per_dimension ({owed}) owes {debt} answers"
                          f" across {len(dimensions)} dimensions, more than policy.max_q"
                          f" ({policy['max_q']}); coverage could never be met")
        elif plan.get("coverage") == "reserve_debt" and isinstance(head, int) \
                and debt + head >= policy["max_q"]:
            errors.append(f"policy: {debt} answers owed plus reserve_early ({head}) fills"
                          f" policy.max_q ({policy['max_q']}), so every turn is reserved for"
                          f" coverage and the judge picks nothing. Raise max_q above"
                          f" {debt + head}, or lower per_dimension or reserve_early")

    if not dimensions:
        errors.append("dimensions: at least one is required")
    for index, dimension in enumerate(dimensions):
        where = f"dimensions[{index}]"
        if not isinstance(dimension, dict):
            errors.append(f"{where}: expected an object, got {dimension!r}")
            continue
        missing = [k for k in ("name", "weight", "aggregate") if k not in dimension]
        if missing:
            errors.append(f"{where}: missing {missing}")
            continue
        where = f"dimensions[{dimension['name']!r}]"
        if error := in_bounds("weight", dimension["weight"], f"{where}.weight"):
            errors.append(error)
        if dimension["aggregate"] not in VOCABULARY["aggregate"]:
            errors.append(f"{where}.aggregate: {dimension['aggregate']!r} is not in the"
                          f" vocabulary {sorted(VOCABULARY['aggregate'])}")
    if len(weights) != len(dimensions):
        errors.append("dimensions: two share a name")
    for key in ("total", "tie_break", "coverage"):
        if plan[key] not in VOCABULARY[key]:
            errors.append(f"{key}: {plan[key]!r} is not in the vocabulary"
                          f" {sorted(VOCABULARY[key])}")

    # `done` is never scored, so it stays out of a Replay and out of the flip rate.
    done = plan["done"]
    for when in ("true", "false"):
        if not isinstance(done.get(when), str) or not done[when].strip():
            errors.append(f"done.{when}: needs a sentence saying, in the Brief's own terms,"
                          f" when the interview has heard enough")

    low, high = BOUNDS["rubric_levels"]
    fit_rubric = plan["fit"].get("rubric")
    if not isinstance(fit_rubric, list) or not low <= len(fit_rubric) <= high:
        errors.append(f"fit.rubric: needs between {low} and {high} levels")

    questions = plan["questions"]
    if not questions:
        errors.append("questions: at least one is required")
    seen: set[str] = set()
    worded: set[str] = set()
    for index, question in enumerate(questions):
        where, missing = question_shape(question, index)
        if missing:
            errors += missing
            continue
        errors += same_words(question, worded, where)
        if question["id"].startswith("_"):
            errors.append(f"{where}: ids starting with _ are reserved for the scoring pass")
        if question["id"] in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(question["id"])
        if question["dimension"] not in weights:
            errors.append(f"{where}.dimension: {question['dimension']!r} has no weight")
        if not low <= len(question["rubric"]) <= high:
            errors.append(f"{where}.rubric: needs between {low} and {high} levels")
        errors += widget_problems(question, where)
    shaped = [q for q in questions if isinstance(q, dict)]
    lengths = {len(q["rubric"]) for q in shaped if isinstance(q.get("rubric"), list)}
    if len(lengths) > 1:
        errors.append(f"questions: rubrics have {sorted(lengths)} levels;"
                      f" scores from different length rubrics cannot be combined")

    covering = {d: sum(1 for q in shaped if q.get("dimension") == d) for d in weights}
    if uncovered := [d for d in covering if not covering[d]]:
        errors.append(f"dimensions: no question covers {uncovered}")
    if thin := [d for d in covering if covering[d] == 1]:
        errors.append(f"dimensions: one question each covers {thin}; two keeps a single"
                      f" weak answer from being the whole of a dimension")
    return errors


def same_words(question: Json, worded: set[str], where: str) -> list[str]:
    """Whether this question is worded like one already seen. Shared by both checkers,
    a whole Plan and one applicant's questions.

    An answer is kept under the words it was given, in `answers_so_far` and again when it
    is scored, so two questions worded alike are one key: the second answer overwrites the
    first and both questions are then scored against whichever survived.
    """
    if question["text"] in worded:
        return [f"{where}.text: another question is worded identically; answers are kept"
                f" under the question's own words, so one would overwrite the other"]
    worded.add(question["text"])
    return []


def widget_choices(question: Json) -> list[str]:
    """Which inputs this question may be put in front of somebody with.

    Fixed per question, so a runtime choice stays out of the ruler: everybody meets the
    same question with the same inputs available, and rubric, dimension and arithmetic are
    the same whichever one collected the words.
    """
    named = dict.fromkeys([question["ui"]] + list(question.get("ui_options") or []))
    return [name for name in named if name in WIDGETS]

# ── the schema a model writes a plan against ──────────────────────────────

def numeric(name: str) -> Json:
    low, high = BOUNDS[name]
    return {"type": "integer" if name in INTEGER_POLICY else "number",
            "minimum": low, "maximum": high}

def named(options: Json) -> Json:
    return {"type": "string", "enum": sorted(options),
            "description": "; ".join(f"{k}: {v['when']}" for k, v in options.items())}

def obj(properties: Json) -> Json:
    """Strict mode wants every key required and nothing extra."""
    return {"type": "object", "additionalProperties": False,
            "required": sorted(properties), "properties": properties}

QID = "^[a-z][a-z0-9_]{1,23}$"
EXTRA_QID = "^x_[a-z0-9_]{1,21}$"


def question_schema(rubric: Json, pattern: str) -> Json:
    """One shape for a question, whether a whole Plan is being written or the two to four
    for one applicant — so no field can reach one and not the other."""
    return obj({
        "id": {"type": "string", "pattern": pattern},
        "dimension": {"type": "string"},
        "text": {"type": "string"},
        "context": {"type": "string"},
        "options": {"type": ["array", "null"], "items": {"type": "string"}},
        "rubric": rubric,
        "ui": named(WIDGETS),
        # Jev picks among these per candidate; empty means everybody gets `ui`.
        "ui_options": {"type": "array", "items": named(WIDGETS)}})


def supplement_schema(levels: int) -> Json:
    """Questions for one applicant. Same shape and rubric length as the plan's, so they
    read on the same scale — but they never enter a dimension's score."""
    exact = {"type": "array", "items": {"type": "string"},
             "minItems": levels, "maxItems": levels}
    return obj({"questions": {"type": "array", "minItems": BOUNDS["supplement"][0],
                              "maxItems": BOUNDS["supplement"][1],
                              "items": question_schema(exact, EXTRA_QID)}})


QUESTION_FIELDS = ("id", "dimension", "text", "options", "context", "rubric", "ui")


def question_shape(question: Json, index: int) -> tuple[str, list[str]]:
    """Where to report a question's problems, and whether it has the fields to have any.

    Shared by both checkers, a whole Plan and one applicant's questions, so the field
    list cannot reach one and not the other. What each of them then requires of a
    question differs, and stays with them.
    """
    where = f"questions[{index}]"
    if not isinstance(question, dict):
        return where, [f"{where}: expected an object, got {question!r}"]
    if missing := [k for k in QUESTION_FIELDS if k not in question]:
        return where, [f"{where}: missing {missing}"]
    if not all(isinstance(question[k], str) for k in ("id", "text")) \
            or not isinstance(question["rubric"], list):
        return where, [f"{where}: id and text are strings, rubric a list of levels"]
    return f"questions[{question['id']!r}]", []


def widget_problems(question: Json, where: str) -> list[str]:
    """Whether a question's inputs are ones this code can draw. Shared by both checkers,
    a whole Plan and one applicant's questions."""
    problems = []
    for field, name in ([("ui", question["ui"])]
                        + [("ui_options", spare) for spare in question.get("ui_options") or []]):
        if name not in WIDGETS:
            problems.append(f"{where}.{field}: {name!r} is not in the vocabulary"
                            f" {sorted(WIDGETS)}")
        elif bool(question["options"]) != (name in OPTION_WIDGETS):
            problems.append(f"{where}.{field}: {name!r} and options="
                            f"{question['options']!r} disagree")
    return problems


def plan_schema() -> Json:
    levels = {"type": "array", "items": {"type": "string"},
              "minItems": BOUNDS["rubric_levels"][0], "maxItems": BOUNDS["rubric_levels"][1]}
    return obj({
        "lang": {"type": "string", "enum": sorted(LANGUAGE_CHOICES)},
        "policy": obj({name: numeric(name) for name in WRITTEN_POLICY}),
        "done": obj({"true": {"type": "string"}, "false": {"type": "string"}}),
        "total": named(VOCABULARY["total"]),
        "tie_break": named(VOCABULARY["tie_break"]),
        "coverage": named(VOCABULARY["coverage"]),
        "dimensions": {"type": "array", "minItems": BOUNDS["dimensions"][0],
                       "maxItems": BOUNDS["dimensions"][1], "items": obj({
            "name": {"type": "string"},
            "weight": numeric("weight"),
            "aggregate": named(VOCABULARY["aggregate"])})},
        "fit": obj({"rubric": levels}),
        "questions": {"type": "array", "minItems": BOUNDS["questions"][0],
                      "maxItems": BOUNDS["questions"][1],
                      "items": question_schema(levels, QID)}})

# ── scoring: Jev supplies the numbers, the plan picks how they combine ────
AGGREGATORS = {"mean": lambda values: sum(values) / len(values), "min": min, "max": max}

# What to do about a dimension still owed answers; the Plan names one.
COVERAGE: Json = {
    "reserve_debt": lambda remaining, asking, debt: (
        remaining if not debt["short"]
        or debt["owed"] + debt["head_start"] < debt["turns_left"]
        else [qid for qid in remaining if not asking[qid].get("extra")
              and asking[qid]["dimension"] in debt["short"]] or remaining),
    "free": lambda remaining, asking, debt: remaining,
}


# What to do when Jev says the options are level. Nothing to escalate to, so code settles
# it — but which way is the Plan's to name.
TIE_BREAKS: Json = {
    "cover_first": lambda remaining, asking, short: next(
        (qid for qid in remaining
         if not asking[qid].get("extra") and asking[qid]["dimension"] in short),
        remaining[0]),
    "plan_order": lambda remaining, asking, short: remaining[0],
}


# How the dimensions become one number. Which one a company wants is its promise, not this
# code's: the Plan names one, code only implements the names.
TOTALS: Json = {
    "weighted_mean": lambda scores, weights: (
        sum(weights[name] * score for name, score in scores.items())
        / (sum(weights[name] for name in scores) or 1)),
    # Weights deliberately unread: the point of this one is that no axis carries another.
    "min_dimension": lambda scores, weights: min(scores.values()),
}
FIT = "_fit"

# One wording for the interview and for a Replay of it; drift would compare two questions.
SCORE_QUESTION = "Given `job_context`, which level does `candidate_answer` reach?"
FIT_QUESTION = ("Given `job_context`, how well does the person described by"
                " `answers_so_far` fit this company? Judge the whole interview,"
                " not any single answer.")

def score_questions(questions: list[Json], answers: dict[str, str], fit_rubric: list[str]) -> Json:
    asked = {q["id"]: {"type": "score", "criteria": q["rubric"],
                       "instructions": {"question_asked": q["text"],
                                        "candidate_answer": answers[q["text"]],
                                        "question": SCORE_QUESTION}}
             for q in questions if q["text"] in answers}
    return asked | {FIT: {"type": "score", "criteria": fit_rubric,
                          "instructions": FIT_QUESTION}}

def combine_dimensions(rows: list[Json], dimensions: list[Json]) -> tuple[Json, list[Json]]:
    """The per-dimension maths, free of module state so a Replay can run another plan's."""
    combined: Json = {}
    flags: list[Json] = []
    for dimension in dimensions:
        name = dimension["name"]
        scored = [r for r in rows
                  if r["dimension"] == name and r["score"] is not None and not r.get("extra")]
        if not scored:
            flags.append({"kind": "uncovered", "dimension": name})
            continue
        combine = AGGREGATORS[dimension["aggregate"]]
        combined[name] = {"score": round(combine([r["score"] for r in scored]), 2),
                          "n": len(scored), "how": dimension["aggregate"],
                          "confidence": round(min(r["confidence"] for r in scored), 2)}
    return combined, flags

def plan_total(combined: Json, dimensions: list[Json], how: str) -> float | None:
    """The one number, by whichever method the Plan named. None, not zero, when nothing was
    scored: an outage is not a low mark."""
    if not combined:
        return None
    weights = {d["name"]: d["weight"] for d in dimensions}
    scores = {name: row["score"] for name, row in combined.items()}
    return round(TOTALS[how](scores, weights), 2)

# ── reading Jev's answers, and how far they move between identical asks ───

def average_answers(runs: list[Json]) -> Json:
    """Average the numbers across identical requests and re-pick the winner.

    Jev's distribution moves between byte-identical requests. Averaging shrinks that spread
    by roughly the square root of the sample count, and costs only tokens: the calls are
    independent.
    """
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    merged: Json = {}
    for key, value in runs[0].items():
        # Only the runs that carry the key: Jev is across a network, and a field one call
        # left out must not take the whole batch down with it.
        held = [run[key] for run in runs if key in run]
        if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
            merged[key] = {name: sum(one.get(name, 0.0) for one in held) / len(held)
                           for name in value}
        elif number(value):
            merged[key] = sum(held) / len(held)
        else:
            merged[key] = value
    if isinstance(odds := merged.get("probabilities"), dict) and odds and "choice" in merged:
        merged["choice"] = max(odds, key=odds.get)
    return merged

def gap(answer: Json) -> float:
    """How far the winner leads the runner-up. Below the noise floor it means nothing."""
    ranked = sorted((answer.get("probabilities") or {}).values(), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else 1.0

def landed_on(answer: Json) -> tuple[int | None, float]:
    """Which rubric level the mass actually favours, and by how much.

    `score` is a weighted average and can sit between levels nobody voted for, and
    `confidence` collapses the distribution by an undocumented rule. These two are what
    survive reading.
    """
    ranked = sorted(((int(k), v) for k, v in (answer.get("probabilities") or {}).items()),
                    key=lambda kv: -kv[1])
    if not ranked:
        return None, 1.0
    lead = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
    return ranked[0][0], round(lead, 3)

def noise_floor(runs: list[Json], samples: int) -> Json:
    """How far `gap` moves across byte-identical requests — the scale every threshold in a
    Plan is expressed against.

    Measured on the basis it is spent on: the gate reads an answer already averaged over
    `samples` calls, and averaging shrinks the spread, so a floor taken from single calls is
    too wide a ruler for it. A floor that does not say its `samples` says nothing.
    """
    whole = max(1, samples)
    groups = [runs[at:at + whole] for at in range(0, len(runs) - whole + 1, whole)]
    gaps = [gap(average_answers(group)) for group in groups]
    return {"n": len(gaps),
            "mean_gap": round(statistics.fmean(gaps), 4) if gaps else None,
            # The gate is `gap < gap_k * floor`: a standard deviation, and how many to clear.
            "spread": round(statistics.stdev(gaps), 4) if len(gaps) > 1 else None}

def floor_from(runs: list[Json], slots: list[str], samples: int) -> Json:
    """One floor out of however many slots a request carries.

    A Choice carries one slot, so its spread *is* the floor. A Score carries one per answer:
    the floor is their mean, and every slot is kept so a single wild rubric stays visible
    instead of being averaged into the number that gates the others.
    """
    each = {slot: noise_floor([r[slot] for r in runs if slot in r], samples)
            for slot in slots}
    spreads = [m["spread"] for m in each.values() if m["spread"] is not None]
    # Says whether the floor means anything: a distribution pinned on one level has spread
    # zero however noisy the judge is, and a mean gap near 1 is that pin.
    means = [m["mean_gap"] for m in each.values() if m["mean_gap"] is not None]
    return {"floor": round(statistics.fmean(spreads), 4) if spreads else None,
            "mean_gap": round(statistics.fmean(means), 3) if means else None,
            "groups": min((m["n"] for m in each.values()), default=0),
            "slots": {slot: m["spread"] for slot, m in each.items()}}

# ── the bar: fitted to decisions, scored on the ones it did not see ───────

def side(total: float | None, boundary: float) -> str:
    return "?" if total is None else ("pass" if total >= boundary else "fail")

def best_boundary(decided: list[tuple[float, str]],
                  boundary: float) -> tuple[float, int, int, tuple[float, float]]:
    """The cut that disagrees with the fewest hiring decisions, how the current one does
    beside it, and the span of every cut that does as well — the bar's real resolution, and
    usually wider than the one number suggests.

    Decides nothing: a Setter who believes it edits `measurement_boundary` and commits.
    """
    totals = sorted({total for total, _ in decided})
    # A cut changes nothing until it crosses a total, so one midpoint stands for each band
    # between adjacent totals — and the band, not the midpoint, is what `span` reports.
    # The open bands are bands too — a history that all went one way is often best
    # described by passing everybody or nobody — and their midpoints are the infinities,
    # which is what `side` already reads them as.
    bands = list(zip([-math.inf] + totals, totals + [math.inf]))
    missed = lambda cut: sum(side(total, cut) != what for total, what in decided)
    best = min([boundary] + [(low + high) / 2 for low, high in bands], key=missed)
    equal = [band for band in bands if missed((band[0] + band[1]) / 2) == missed(best)]
    # Outermost, not the band `best` sits in: when tying bands are not adjacent this
    # overstates the width, and claiming less precision than there is is the safe direction.
    return (best, missed(best), missed(boundary),
            (min(low for low, _ in equal), max(high for _, high in equal)) if equal
            else (best, best))

# A boundary fitted on fifteen records containing three hires is noise, so below this
# `calibrate` reports the shape and refuses the number.
CALIBRATION_MIN = (10, 3)

def isotonic(observed: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    """The closest non-decreasing fit to the observations, by pooling adjacent violators.

    Monotone by construction, which is the point: a higher total may never come out less
    likely to have been hired than a lower one. Runs of equal rate are pooled too, so a
    total reads the same whoever else shares it.
    """
    blocks: list[list[float]] = []
    for position, value in sorted(observed):
        blocks.append([position, position, value, 1.0])
        while len(blocks) > 1 and blocks[-2][2] >= blocks[-1][2]:
            left, right = blocks.pop(-2), blocks.pop()
            weight = left[3] + right[3]
            blocks.append([left[0], right[1],
                           (left[2] * left[3] + right[2] * right[3]) / weight, weight])
    return [(left, right, round(rate, 3)) for left, right, rate, _ in blocks]

def fitted(curve: list[tuple[float, float, float]], total: float | None) -> float | None:
    """Read a total off the curve. The same total reads the same for everybody: this is the
    bar with a shape, not an adjustment to one person's score.

    Between two blocks it interpolates rather than stepping — the gap between the last
    rejection and the first hire is exactly the case nobody decided, and snapping it to
    whichever end is nearer would report certainty the decisions do not contain.
    """
    if total is None or not curve:
        return None
    if total <= curve[0][1]:
        return curve[0][2]
    for (_, below, under), (above, top, over) in zip(curve, curve[1:]):
        if total <= top:
            if total >= above or above == below:
                return over
            return round(under + (over - under) * (total - below) / (above - below), 3)
    return curve[-1][2]

def calibrate(decided: list[tuple[float, str]], boundary: float) -> Json:
    """What this company's own decisions say a total means. Decides nothing: a Setter who
    believes it writes the bar into interface.json and commits.

    `usable` stays false until there are enough decisions of both kinds, because a curve
    fitted on three hires reproduces those three hires and nothing else.
    """
    hired = sum(1 for _, what in decided if what == "pass")
    least, least_each = CALIBRATION_MIN
    if not decided:
        return {"n": 0, "hired": 0, "usable": False, "bar": boundary, "span": None,
                "wrong": None, "at_current": None, "curve": []}
    bar, wrong, at_current, span = best_boundary(decided, boundary)
    return {"n": len(decided), "hired": hired,
            "usable": len(decided) >= least and min(hired, len(decided) - hired) >= least_each,
            "bar": bar, "span": span, "wrong": wrong, "at_current": at_current,
            "curve": isotonic([(total, 1.0 if what == "pass" else 0.0)
                               for total, what in decided])}

def accuracy(decided: list[tuple[float, str]], boundary: float) -> Json:
    """How often the fitted bar is right about a decision it did not see.

    Leave one out: fit on every decision but one, read that one off, count. `calibrate`'s
    in-sample `wrong` always flatters — a cut may land between any two totals, so separable
    decisions come out perfect by construction. `chance` is what always answering the
    commoner way would score, not 0.5; a bar that cannot beat it has found nothing.
    """
    if len(decided) < 3:
        return {"n": len(decided), "right": None, "rate": None, "chance": None,
                "beats_chance": None}
    right = sum(side(total, best_boundary(decided[:at] + decided[at + 1:], boundary)[0]) == what
                for at, (total, what) in enumerate(decided))
    hired = sum(1 for _, what in decided if what == "pass")
    chance = max(hired, len(decided) - hired) / len(decided)
    rate = right / len(decided)
    return {"n": len(decided), "right": right, "rate": round(rate, 3),
            "chance": round(chance, 3), "beats_chance": rate > chance}
