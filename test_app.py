"""Checks for app.py. No API key needed — Jev is replaced by a stub.

    PLAN=<path.json> python3 test_app.py

PLAN names the plan they run against, the way the server takes one. This repository
ships none, so there is nothing to leave it unset for.

Every check runs even when an earlier one fails, and each starts from a clean session.
"""
from __future__ import annotations

import ast
import base64
import builtins
import io
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

import app
import core

app.boot()          # nothing is loaded on import; this is the plan under test

KOREAN = re.compile(r"[\uac00-\ud7a3]")
DIMENSIONS = list(app.WEIGHTS)
BY_DIMENSION = {dim: [qid for qid, q in app.QUESTIONS.items() if q["dimension"] == dim]
                for dim in DIMENSIONS}
REAL_ASK_JEV = app.ask_jev
REAL_SAVE_REPORT = app.save_report
REAL_NOTIFY = app.notify
saved: list[dict[str, Any]] = []


# ── harness ───────────────────────────────────────────────────────────────
CHECKS: list[Any] = []


def check(fn: Any) -> Any:
    """A check runs by being defined. Nothing lists them, so nothing can leave one out."""
    CHECKS.append(fn)
    return fn


class FakeHandler(app.Handler):
    def __init__(self, body: dict[str, Any], path: str = "/answer"):
        # The handler reads a Cookie header now, because the interviewer's session
        # lives in one rather than in a JS variable that a reload throws away.
        self._body, self.path, self.sent, self.headers = body, path, [], {}

    def body(self) -> dict[str, Any]:
        return self._body

    def send(self, body: Any, ctype: str | None = None, code: int = 200) -> None:
        self.sent.append((code, body))


def post(body: dict[str, Any], path: str = "/answer") -> tuple[int, Any]:
    handler = FakeHandler(body, path)
    app.Handler.do_POST(handler)
    return handler.sent[0]


def fake_jev(next_id: str | None = None, done: float = 0.1, score: float = 2.0,
             confidence: float = 0.9) -> list[dict[str, Any]]:
    """One stub for both shapes: a turn asks done/next, the closing pass asks for scores."""
    calls: list[dict[str, Any]] = []

    def spy(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        calls.append({"state": state, "questions": questions})
        if app.FIT not in questions:                    # a turn, not the scoring pass
            answer: dict[str, Any] = {}
            if "done" in questions:                     # only asked when it could stop
                answer["done"] = {"noul": done}
            if next_id and "next" in questions:
                answer["next"] = {"choice": next_id}
            return answer
        return {qid: {"score": score, "confidence": confidence} for qid in questions}

    app.ask_jev = spy
    return calls


def turns(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in calls if app.FIT not in c["questions"]]


def scorings(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in calls if app.FIT in c["questions"]]


def reaches_in(where: str) -> Any:
    """What a function in `where` can call, transitively. Two checks ask whether a path still
    leads somewhere it should not, and this is how they ask."""
    held = {node.name: node for node in
            ast.parse(pathlib.Path(where).read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)}

    def reaches(name: str, seen: tuple[str, ...] = ()) -> set[str]:
        if name in seen or name not in held:
            return set()
        found = {name}
        for call in ast.walk(held[name]):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                found |= reaches(call.func.id, seen + (name,))
        return found
    return reaches


def fresh(pending: str | None = None) -> dict[str, Any]:
    state = app.new_state("jti-test")
    if pending:
        state["pending"] = pending
    return state


def answer(state: dict[str, Any], question_id: str | None, a: Any) -> tuple[int, Any]:
    return post({"state": app.sign(state), "id": question_id, "a": a})


def state_of(payload: dict[str, Any]) -> dict[str, Any]:
    return app.unsign(payload["state"])


def text_of(question_id: str) -> str:
    return app.QUESTIONS[question_id]["text"]


# ── the plan ──────────────────────────────────────────────────────────────
@check
def the_shipped_plan_passes_its_own_checker() -> None:
    assert app.check_plan(app.plan) == [], app.check_plan(app.plan)
    assert abs(sum(app.WEIGHTS.values()) - 1) < 1e-9, "weights do not sum to 1"
    assert app.plan["source"]["brief_sha256"], "the plan does not say which Brief it came from"
    for question in app.QUESTION_LIST:
        where, rubric = question["id"], question["rubric"]
        assert len(rubric) == len(set(rubric)), f"{where}: rubric levels are not distinct"
        assert min(len(level) for level in rubric) >= 15, f"{where}: rubric level too short to judge"
        assert question["options"] is None or len(question["options"]) >= 2, f"{where}: lone option"


@check
def the_checker_catches_every_way_a_plan_can_break() -> None:
    """A plan is written by a model; this is the only thing standing behind it."""
    import copy
    cases = {
        "max_q <= min_q":        lambda p: p["policy"].update(max_q=p["policy"]["min_q"]),
        "min_q out of bounds":   lambda p: p["policy"].update(min_q=99),
        "samples out of bounds": lambda p: p["policy"].update(samples=50),
        "per_dimension out of bounds": lambda p: p["policy"].update(per_dimension=99),
        "coverage the budget cannot pay": lambda p: p["policy"].update(per_dimension=4),
        "done not worded":       lambda p: p["done"].update(true=" "),
        "reserve_early out of bounds": lambda p: p["policy"].update(reserve_early=99),
        "unknown aggregate":     lambda p: p["dimensions"][0].update(
                                     aggregate="harmonic_mean"),
        "weight out of bounds":  lambda p: p["dimensions"][0].update(weight=9.9),
        "dimension missing a key": lambda p: p["dimensions"][0].pop("weight"),
        "two dimensions share a name": lambda p: p["dimensions"][1].update(
                                     name=p["dimensions"][0]["name"]),
        "dimension with no weight": lambda p: p["questions"][0].update(dimension="_nope_"),
        "unknown widget":        lambda p: p["questions"][0].update(ui="hologram"),
        "widget/options disagree": lambda p: p["questions"][0].update(ui="radio", options=None),
        "duplicate id":          lambda p: p["questions"][1].update(id=p["questions"][0]["id"]),
        "rubric too short":      lambda p: p["questions"][0].update(rubric=["one"]),
        "rubrics of two lengths": lambda p: p["questions"][0].update(
                                     rubric=p["questions"][0]["rubric"] + ["an extra level"]),
        "missing top-level key": lambda p: p.pop("dimensions"),
        "dimension uncovered":   lambda p: p.update(questions=[
                                     q for q in p["questions"]
                                     if q["dimension"] != DIMENSIONS[-1]]),
        # Keep exactly one question of the first dimension, whatever the plan's shape:
        # popping the first question leaves a valid plan when that dimension has more.
        "dimension on one question": lambda p: p.update(questions=
                                     [q for q in p["questions"]
                                      if q["dimension"] == DIMENSIONS[0]][:1]
                                     + [q for q in p["questions"]
                                        if q["dimension"] != DIMENSIONS[0]]),
        "every turn reserved for coverage": lambda p: p["policy"].update(reserve_early=4),
        # A plan is edited by hand as often as it is generated, and `--check` is what
        # reads that one: it has to name the problem rather than traceback.
        "two questions worded alike": lambda p: p["questions"][1].update(
                                     text=p["questions"][0]["text"]),
        "a fraction where the schema says integer": lambda p: p["policy"].update(min_q=3.5),
        "a question that is not an object": lambda p: p["questions"].__setitem__(0, "q1"),
        "a dimension that is not an object": lambda p: p["dimensions"].__setitem__(0, 3),
        "a dimension with no name": lambda p: p["dimensions"][0].pop("name"),
        "questions that are not a list": lambda p: p.update(questions="q1"),
        "policy that is not an object": lambda p: p.update(policy="fast"),
        "a rubric that is not a list": lambda p: p["questions"][0].update(rubric=4),
    }
    for name, mutate in cases.items():
        broken = copy.deepcopy(app.plan)
        mutate(broken)
        assert app.check_plan(broken), f"not caught: {name}"
    assert app.check_plan(["not a plan at all"]), "a JSON array was read as a plan"


@check
def widget_spec_returns_a_surveyjs_question() -> None:
    for options, ui in ((None, "textarea"), (None, "zzz"), (["a", "b"], "radio"), (["a", "b"], "zzz")):
        name, spec = app.widget_spec(options, ui)
        assert name in app.WIDGETS, f"{ui!r} fell back to an unknown widget"
        assert spec["name"] == "a" and spec["isRequired"], spec
        assert spec["type"] == app.WIDGETS[name]["sj"]["type"], spec
        assert ("choices" in spec) == bool(options), spec
        json.dumps(spec)


# ── one turn ──────────────────────────────────────────────────────────────
@check
def an_answer_is_stored_and_followed() -> None:
    next_id = BY_DIMENSION[DIMENSIONS[1]][0]
    fake_jev(next_id)

    code, payload = answer(fresh(), app.FIRST_QUESTION, "<b>answer</b>")
    state = state_of(payload)

    assert code == 200, code
    assert state["answers"][app.FIRST_QUESTION] == "<b>answer</b>", \
        "the server stores answers verbatim; escaping belongs to the client"
    assert payload["id"] == next_id == state["pending"]
    assert payload["question"]["title"] == text_of(next_id)
    assert payload["history"][0]["a"] == "<b>answer</b>"
    assert payload["lang"] == app.LANG


@check
def the_candidate_is_never_handed_a_mark() -> None:
    """Q61: scores exist for the Interviewer, and must not reach the browser."""
    fake_jev(BY_DIMENSION[DIMENSIONS[1]][0])
    _, payload = answer(fresh(), app.FIRST_QUESTION, "answer")
    blob = json.dumps(payload, ensure_ascii=False)
    for word in ("score", "confidence", "rubric", "\"conf\""):
        assert word not in blob, f"the payload leaks {word!r} to the candidate"
    assert "score" not in json.dumps(state_of(payload)), "the signed state carries a mark"


@check
def nothing_rides_along_but_the_brief_and_what_was_said_here() -> None:
    """Their account was read once, at application, and what it was for
    a person has approved. It must not reach the interview a second time, unreviewed."""
    calls = fake_jev(None, done=0.99)
    answer(fresh(), app.FIRST_QUESTION, "answer")

    assert turns(calls), "no turn call was made"
    for call in calls:
        assert set(call["state"]) == {"job_context", "answers_so_far"}, \
            f"the state carries more than the brief and the answers: {sorted(call['state'])}"


@check
def the_brief_rides_every_turn_and_the_rubric_rides_the_scoring() -> None:
    calls = fake_jev(BY_DIMENSION[DIMENSIONS[1]][0])
    answer(fresh(), app.FIRST_QUESTION, "answer")
    turn = turns(calls)[0]

    assert app.BRIEF[:20] in turn["state"]["job_context"], "the brief is missing from the state"
    assert turn["state"]["answers_so_far"][text_of(app.FIRST_QUESTION)] == "answer"
    assert app.FIRST_QUESTION not in turn["questions"]["next"]["criteria"], \
        "an answered question is still a candidate"

    scores, _ = app.score_all(fresh() | {"asked": [app.FIRST_QUESTION],
                                         "answers": {app.FIRST_QUESTION: "answer"}})
    assert scores[app.FIRST_QUESTION] == {"score": 2.0, "confidence": 0.9,
                                          "level": None, "gap": 1.0}
    assert scorings(calls)[-1]["questions"][app.FIRST_QUESTION]["criteria"] == \
        app.QUESTIONS[app.FIRST_QUESTION]["rubric"]


@check
def forged_answers_and_states_are_rejected() -> None:
    next_id = BY_DIMENSION[DIMENSIONS[1]][0]
    fake_jev(next_id)
    state = fresh()

    assert answer(state, next_id, "x")[0] == 400, "id other than pending"
    assert answer(state, app.FIRST_QUESTION, "  ")[0] == 400, "blank answer"
    assert answer(state, app.FIRST_QUESTION, None)[0] == 400, "null answer"
    assert post({"state": "not.a.token", "id": app.FIRST_QUESTION, "a": "x"})[0] == 400
    assert post({"id": app.FIRST_QUESTION, "a": "x"})[0] == 400, "missing state"

    signed = app.sign(state)
    tampered = signed[:-1] + ("A" if signed[-1] != "A" else "B")
    assert post({"state": tampered, "id": app.FIRST_QUESTION, "a": "x"})[0] == 400, \
        "a re-signed state was accepted"

    with_options = next((qid for qid, q in app.QUESTIONS.items() if q["options"]), None)
    if with_options:
        assert answer(fresh(with_options), with_options, "not-an-option")[0] == 400


# ── policy ────────────────────────────────────────────────────────────────
@check
def a_dimension_still_owed_an_answer_takes_the_last_turns() -> None:
    """What the plan owes each dimension decides who competes, so pin it rather than
    inherit it."""
    first = BY_DIMENSION[DIMENSIONS[0]][0]
    cap, owed = app.POLICY["max_q"], app.POLICY["per_dimension"]
    try:
        app.POLICY["per_dimension"] = 1
        app.POLICY["max_q"] = 1 + len(DIMENSIONS) - 1
        calls = fake_jev(BY_DIMENSION[DIMENSIONS[1]][0])
        answer(fresh(first), first, "answer")
        offered = turns(calls)[0]["questions"]["next"]["criteria"]
        assert offered, "no candidates were offered"
        assert all(app.QUESTIONS[qid]["dimension"] != DIMENSIONS[0] for qid in offered), \
            "a dimension already owed nothing is still competing for the tight budget"

        app.POLICY["per_dimension"] = 2
        app.POLICY["max_q"] = 1 + 2 * len(DIMENSIONS) - 1
        calls = fake_jev(BY_DIMENSION[DIMENSIONS[1]][0])
        answer(fresh(first), first, "answer")
        offered = turns(calls)[0]["questions"]["next"]["criteria"]
        assert any(app.QUESTIONS[qid]["dimension"] == DIMENSIONS[0] for qid in offered), \
            "a dimension still owed a second answer was dropped"
    finally:
        app.POLICY["max_q"], app.POLICY["per_dimension"] = cap, owed


@check
def the_hard_cap_ends_the_interview() -> None:
    def spy(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if app.FIT in questions:                        # the scoring pass, not a turn
            return {qid: {"score": 2.0, "confidence": 0.9} for qid in questions}
        offered = questions.get("next", {}).get("criteria", {})
        return {**({"done": {"noul": 0.0}} if "done" in questions else {}),
                **({"next": {"choice": next(iter(offered))}} if offered else {})}

    app.ask_jev = spy
    limit = min(app.POLICY["max_q"], len(app.QUESTIONS))
    state, payload = fresh(), None

    for index in range(limit):
        pending = state["pending"]
        options = app.QUESTIONS[pending]["options"]
        _, payload = answer(state, pending, options[0] if options else f"answer {index}")
        if payload.get("finished"):
            break
        state = state_of(payload)

    assert payload.get("finished"), "the interview ran past the cap"
    assert len(saved) == 1, f"expected one saved report, got {len(saved)}"
    assert len(saved[0]["answers"]) == limit


@check
def a_failed_api_call_does_not_end_the_interview() -> None:
    app.ask_jev = lambda state, questions: None
    _, payload = answer(fresh(), app.FIRST_QUESTION, "answer")

    assert payload["id"] == next(qid for qid in app.QUESTIONS if qid != app.FIRST_QUESTION), \
        "the fallback must offer the next question in the plan's own order"
    assert payload["num"] == 2
    assert payload["question"]["title"] == text_of(payload["id"]), "the title is the question verbatim"


# ── scoring ───────────────────────────────────────────────────────────────
def scored(*pairs: tuple[str, float, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judgements as score_all now returns them: a level, its lead, and the number."""
    state = fresh() | {"asked": [qid for qid, *_ in pairs],
                       "answers": {qid: "answer" for qid, *_ in pairs}}
    return state, {qid: {"score": score, "confidence": conf,
                         "level": round(score), "gap": 0.9}
                   for qid, score, conf in pairs}


@check
def weights_are_applied_in_code_and_gaps_are_flagged() -> None:
    high, low = BY_DIMENSION[DIMENSIONS[0]][:2]
    report = app.score_report(*scored((high, 2.0, 0.9), (low, 1.0, 0.3)))

    assert report["dimensions"][DIMENSIONS[0]] == \
        {"score": 1.5, "n": 2, "how": "mean", "confidence": 0.3}
    assert abs(report["total"] - 1.5) < 0.01, \
        "with one dimension scored, normalisation makes it the total"
    kinds = [flag["kind"] for flag in report["flags"]]
    assert "tied_levels" not in kinds, "a clear lead must not be flagged"
    tied = app.score_report(*scored((high, 2.0, 0.9)))["flags"]
    assert "tied_levels" not in [f["kind"] for f in tied]

    state, judged = scored((high, 1.5, 0.9))
    judged[high]["gap"] = app.POLICY["gap_k"] * app.POLICY["noise_floor"]["score"] / 2
    close = [f["kind"] for f in app.score_report(state, judged)["flags"]]
    assert "tied_levels" in close, \
        "two levels inside the noise floor were shown as if one had won"
    assert all({"kind": "uncovered", "dimension": dim} in report["flags"] for dim in DIMENSIONS[1:])
    assert report["flags"][-1] == {"kind": "human_review"}
    assert all(app.flag_text(flag) for flag in report["flags"]), "a flag kind has no translation"
    assert set(report) >= {"answers", "dimensions", "weights", "total", "lang", "context_file"}


@check
def the_plan_picks_how_a_dimension_aggregates() -> None:
    """The name is in the plan; the function is here. Changing the name changes the score."""
    dimension = DIMENSIONS[0]
    high, low = BY_DIMENSION[dimension][:2]
    state, scores = scored((high, 2.0, 0.9), (low, 1.0, 0.9))
    entry = next(d for d in app.plan["dimensions"] if d["name"] == dimension)
    original = entry["aggregate"]
    try:
        seen = {}
        for how, expected in (("mean", 1.5), ("min", 1.0), ("max", 2.0)):
            entry["aggregate"] = how
            report = app.score_report(state, scores)
            seen[how] = report["dimensions"][dimension]["score"]
            assert seen[how] == expected, f"{how} gave {seen[how]}, expected {expected}"
            assert report["aggregate"][dimension] == how, "the record disagrees with the plan"
    finally:
        entry["aggregate"] = original
    assert len(set(seen.values())) == 3, "the aggregate name made no difference"


@check
def an_applicants_own_question_does_not_cover_a_dimension() -> None:
    """It cannot score into one, so it must not stand in for one either."""
    dimension = DIMENSIONS[0]
    plan_q = BY_DIMENSION[dimension][0]
    extra = {"id": "x_probe", "dimension": dimension, "text": "written for them",
             "options": None, "context": "when", "ui": "textarea",
             "rubric": app.QUESTIONS[plan_q]["rubric"]}
    directory = pathlib.Path(tempfile.mkdtemp())
    (directory / "cand.json").write_text(json.dumps(
        {"id": "cand", "approved": True, "questions": [extra]}), encoding="utf-8")
    real, app.CANDIDATES_DIR = app.CANDIDATES_DIR, directory
    try:
        # Their questions are copied into the state when the interview begins; nothing
        # reads their record again after that.
        theirs = app.new_state("jti-test", "cand") | {
            "asked": ["x_probe"], "answers": {"x_probe": "said something"}}
        assert "x_probe" in app.questions_for(theirs), "the extra was not copied in"

        # And the record changing underneath cannot change an interview in progress.
        (directory / "cand.json").unlink()
        assert "x_probe" in app.questions_for(theirs), \
            "the interview followed their record instead of its own copy"
        assert app.scored_per_dimension(theirs)[dimension] == 0, \
            "an unscorable question was counted towards the dimension"
        assert dimension in app.under_measured(theirs, 1)

        plan_only = fresh() | {"asked": [plan_q], "answers": {plan_q: "said something"}}
        assert app.scored_per_dimension(plan_only)[dimension] == 1
        assert dimension not in app.under_measured(plan_only, 1)
    finally:
        app.CANDIDATES_DIR = real


@check
def the_plan_picks_how_many_answers_a_dimension_needs() -> None:
    """The plan writes the number outright: how many answers it owes each dimension."""
    first = BY_DIMENSION[DIMENSIONS[0]][0]
    everything = [qid for qid in app.QUESTIONS if qid != first]
    state = fresh() | {"asked": [first], "answers": {first: "answer"}}
    cap, owed = app.POLICY["max_q"], app.POLICY["per_dimension"]
    try:
        app.POLICY["max_q"] = 1 + len(DIMENSIONS) - 1      # only just enough turns
        app.POLICY["per_dimension"] = 0
        assert app.narrow(state, list(everything)) == everything, "0 per dimension narrowed"

        app.POLICY["per_dimension"] = 1
        once = app.narrow(state, list(everything))
        assert once != everything, "1 per dimension did not narrow a tight budget"
        assert all(app.QUESTIONS[qid]["dimension"] != DIMENSIONS[0] for qid in once), \
            "a dimension already asked about is still competing"

        # 2 still owes the first dimension a second answer, so it stays in
        app.POLICY["max_q"] = 1 + 2 * len(DIMENSIONS) - 1
        app.POLICY["per_dimension"] = 2
        twice = app.narrow(state, list(everything))
        assert any(app.QUESTIONS[qid]["dimension"] == DIMENSIONS[0] for qid in twice), \
            "2 per dimension dropped a dimension answered only once"
        assert app.under_measured(state) == DIMENSIONS, \
            "every dimension is still owed a second answer"
    finally:
        app.POLICY["max_q"], app.POLICY["per_dimension"] = cap, owed


@check
def the_plan_says_how_early_to_reserve_turns_for_coverage() -> None:
    """A loose budget holds nothing back until the plan says to start holding."""
    first = BY_DIMENSION[DIMENSIONS[0]][0]
    everything = [qid for qid in app.QUESTIONS if qid != first]
    state = fresh() | {"asked": [first], "answers": {first: "answer"}}
    keep = {name: app.POLICY[name] for name in ("max_q", "reserve_early", "per_dimension")}
    try:
        app.POLICY["per_dimension"] = 1                  # so the first dimension is paid
        app.POLICY["max_q"] = len(app.QUESTIONS) + 4     # and the budget is not tight
        app.POLICY["reserve_early"] = 0
        assert app.narrow(state, list(everything)) == everything, \
            "a loose budget reserved turns anyway"

        app.POLICY["reserve_early"] = app.POLICY["max_q"]
        held = app.narrow(state, list(everything))
        assert held != everything, "the plan asked to reserve early and nothing was held"
        short = app.under_measured(state)
        assert all(app.QUESTIONS[qid]["dimension"] in short for qid in held), \
            "a dimension that is owed nothing is still competing"
    finally:
        app.POLICY.update(keep)


@check
def the_plan_words_the_judgment_that_ends_an_interview() -> None:
    """`done` carries the plan's own words, and is only asked on a turn that could stop."""
    remaining = [BY_DIMENSION[DIMENSIONS[1]][0]]
    asked = app.turn_questions(remaining, app.QUESTIONS, True)
    assert asked["done"]["criteria"] == app.plan["done"], "the words are not the plan's"
    assert set(asked) == {"done", "next"}, asked
    assert set(app.turn_questions(remaining, app.QUESTIONS, False)) == {"next"}, \
        "done was asked on a turn that cannot end the interview"
    assert app.turn_questions([], app.QUESTIONS, False) == {}, \
        "an empty request would have been sent to Jev"

    # Early: the plan owes every dimension two answers, so nothing can stop the interview
    # yet and `next_turn` would throw the judgment away.
    calls = fake_jev(BY_DIMENSION[DIMENSIONS[1]][0])
    answer(fresh(), app.FIRST_QUESTION, "answer")
    assert "done" not in turns(calls)[0]["questions"], \
        "Jev was paid for a judgment next_turn discards"

    # Late, with every dimension paid: now it is worth asking.
    everything = list(app.QUESTIONS)
    late = fresh() | {"asked": everything, "answers": {qid: "answer" for qid in everything}}
    assert app.under_measured(late) == [], "the shipped plan cannot pay its own coverage"
    calls = fake_jev(None, done=0.99)
    app.next_turn(late)
    assert "done" in turns(calls)[0]["questions"], "done was withheld on a turn that could stop"


@check
def identical_requests_are_asked_more_than_once_and_averaged() -> None:
    """Jev wobbles, so the loop samples it and averages the numbers."""
    spread = [0.20, 0.30, 0.40]
    runs: list[int] = []

    def spy(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        value = spread[len(runs) % len(spread)]
        runs.append(1)
        return {"done": {"noul": value},
                "next": {"choice": app.FIRST_QUESTION,
                         "probabilities": {app.FIRST_QUESTION: value, "other": 1 - value}}}

    app.ask_jev = spy
    samples = app.POLICY["samples"]
    averaged = app.ask_jev_repeatedly("state", {"done": {}, "next": {}})

    assert len(runs) == samples, f"asked {len(runs)} times, expected {samples}"
    assert abs(averaged["done"]["noul"] - sum(spread[:samples]) / samples) < 1e-9, averaged
    assert averaged["next"]["choice"] == "other", "the winner is re-picked from the average"


@check
def averaging_survives_an_answer_that_came_back_short() -> None:
    """Jev is across a network. A field one call left out, or a distribution that came back
    empty, must not take the whole batch down: the averaged answer is what the gate reads,
    and there is no answer at all without it."""
    short = app.average_answers([{"score": 2.0, "confidence": 0.9}, {"confidence": 0.5}])
    assert short == {"score": 2.0, "confidence": 0.7}, short

    empty = app.average_answers([{"choice": "a", "probabilities": {}}] * 2)
    assert empty["choice"] == "a", "an empty distribution re-picked a winner from nothing"


@check
def the_noise_floor_is_measured_on_the_basis_the_gate_spends_it_on() -> None:
    """A floor is only a ruler if it was measured the way it is read. `trusted_choice`
    gates on an answer `ask_jev_repeatedly` has already averaged over `samples` calls, so
    a floor taken from single calls is too wide — and by a factor nothing in the plan
    records. Averaging has to come out smaller, or the number says nothing."""
    jitter = [0.30, -0.22, 0.14, -0.06, 0.24, -0.30, 0.08, -0.18, 0.26, -0.26,
              0.12, -0.12, 0.20, -0.20, 0.04, -0.04, 0.28, -0.28, 0.16, -0.16]
    runs = [{"choice": "a", "probabilities": {"a": 0.5 + j, "b": 0.5 - j, "c": 0.2}}
            for j in jitter * 3]
    averaged, single = app.noise_floor(runs, 3), app.noise_floor(runs, 1)

    assert single["n"] == 60 and averaged["n"] == 20, (single, averaged)  # whole groups only
    assert averaged["spread"] < single["spread"], (averaged, single)
    # One call is not a spread. Reading it as zero would trust every pick there is.
    assert app.noise_floor(runs[:1], 3)["spread"] is None, "one run read as a zero floor"


@check
def each_kind_of_question_carries_its_own_measured_floor() -> None:
    """A Choice among questions and a Score against a rubric are different requests, so
    one number cannot be the floor for both — `--noise` measures them apart. A plan
    holding a single scalar is therefore unusable, and each gate has to read its own."""
    assert app.check_plan(json.loads(json.dumps(app.plan))) == [], "the shipped plan"
    for broken, wanted in (
            (0.07, "noise_floor"),                            # one scalar for both kinds
            ({"choice": 0.02}, "noise_floor.score"),          # a kind left out
            ({"choice": 9.0, "score": 0.02}, "noise_floor.choice")):   # out of bounds
        plan = json.loads(json.dumps(app.plan))
        plan["policy"]["noise_floor"] = broken
        problems = app.check_plan(plan)
        assert any(wanted in problem for problem in problems), (broken, problems)


@check
def neither_gate_reads_the_other_kinds_floor() -> None:
    """The two measured floors sit close together on this judge, so every other check here
    would pass with them swapped. Set them far apart, and each gate has to move with its
    own: a mark is flagged against `score`, a question is picked against `choice`."""
    held, gap_k = app.POLICY["noise_floor"], app.POLICY["gap_k"]
    high, other = app.FIRST_QUESTION, BY_DIMENSION[DIMENSIONS[1]][0]
    try:
        # score gate = 0, choice gate = 0.9. A lead of 0.1 clears its own and fails the other.
        app.POLICY = dict(app.POLICY, noise_floor={"choice": 0.45, "score": 0.0})
        state, judged = scored((high, 2.0, 0.9))
        judged[high]["gap"] = 0.1
        flags = [f["kind"] for f in app.score_report(state, judged)["flags"]]
        assert "tied_levels" not in flags, "a mark was flagged against the choice floor"

        app.POLICY = dict(app.POLICY, noise_floor={"choice": 0.0, "score": 0.45})
        picking = fresh() | {"asked": [high], "answers": {high: "answer"}}
        remaining = [qid for qid in app.QUESTIONS if qid != high]
        answered = {"choice": other, "probabilities": {other: 0.55, "x": 0.45}}
        assert app.trusted_choice(picking, remaining, answered) == other, \
            "a pick was overruled against the score floor"
    finally:
        app.POLICY = dict(app.POLICY, noise_floor=held, gap_k=gap_k)


@check
def every_widget_the_vocabulary_names_renders_itself() -> None:
    """A question carries a widget name and the browser gets that widget: nothing wires a
    question to an input by hand. What makes that safe is agreement — `check_plan` refuses
    a `ui` outside the vocabulary, and every name inside it has to render as itself.

    The failure this catches is silent. `widget_html` ends in a plain text input, so a name
    added to interface.json with no branch behind it passes validation and renders a text
    box that looks deliberate. So each name must render something a text box does not.
    """
    plain = app.widget_html("text", None, "label")
    for name, widget in app.WIDGETS.items():
        options = ["Alpha", "Beta", "Gamma"] if widget.get("opt") else None
        html = app.widget_html(name, options, "label")

        if name != "text":
            assert html != plain, f"{name} fell through to the text input"
        chosen, spec = app.widget_spec(options, name)
        assert chosen == name, f"{name} was rendered as {chosen}"
        assert spec["type"] == widget["sj"]["type"], (name, spec)
        for option in options or []:
            assert option in html, f"{name} dropped the option {option}"

        posted = ({"rank": ["2", "1", "3"], "opt": options} if name == "ranking"
                  else {"a": options[:1] if options else ["said"]})
        carried = app.answer_values(posted, name)
        assert carried, f"{name} posted a value and nothing came back"
    assert app.answer_values({"rank": ["2", "1", "3"], "opt": ["A", "B", "C"]},
                             "ranking") == ["B", "A", "C"], "ranking lost its order"




@check
def a_question_has_one_shape_wherever_it_is_written() -> None:
    """A question the Plan owns and one written for a single applicant are the same thing with
    a different id. The shape was defined twice and had already drifted: `ui_options` reached
    the Plan's copy and not the applicant's, so Jev could choose an input for one kind of
    question and not the other, which nobody had decided.

    One definition now, and only the id and how tightly the rubric is pinned may differ.
    """
    mine = app.plan_schema()["properties"]["questions"]["items"]["properties"]
    theirs = app.supplement_schema(3)["properties"]["questions"]["items"]["properties"]
    assert set(mine) == set(theirs), (sorted(mine), sorted(theirs))
    assert "ui_options" in mine, "nothing can be offered a choice of input at all"
    assert {key for key in mine if mine[key] != theirs[key]} == {"id", "rubric"}, \
        "the two question shapes have parted again"

    # The same goes for the checks, which each of them also used to carry its own copy of.
    for check in (app.check_plan, app.check_supplement):
        for shared in ("widget_problems", "same_words"):
            assert shared in check.__code__.co_names, \
                f"{check.__name__} grew its own copy of {shared}"

    # An answer is kept under the question's own words, in `answers_so_far` and again when
    # it is scored. So an applicant's question worded like one the plan already asks is the
    # same key, and the second answer would quietly take the first one's place.
    echo = dict(app.QUESTION_LIST[0], id="x_echo")
    assert [e for e in app.check_supplement({"questions": [echo]}) if ".text:" in e], \
        app.check_supplement({"questions": [echo]})

@check
def jev_chooses_the_input_and_the_plan_bounds_what_it_may_choose() -> None:
    """The name Jev returns is the table index that selects the render function — nothing
    reads its answer and branches on it. What it may return is the Plan's own list, so
    nobody is offered a way of answering that another candidate was not.
    """
    first = app.FIRST_QUESTION
    later = next(qid for qid in app.QUESTIONS
                 if qid != first and len(app.widget_choices(app.QUESTIONS[qid])) > 1)
    asked: list[Any] = []

    def picks(name: str) -> Any:
        def spy(where: Any, questions: Any) -> dict[str, Any]:
            asked.append(questions)
            return {"widget": {"choice": name}}
        return spy

    held = app.ask_jev
    try:
        state = fresh() | {"asked": [first], "answers": {first: "said something"}}
        app.ask_jev = picks("textarea")
        assert app.choose_widget(state, later) == "textarea"
        assert "textarea" in asked[0]["widget"]["criteria"], "it was not offered the choice"
        payload = app.question_payload(state, later)
        assert payload["ui"] == "textarea", payload
        assert "<textarea" in app.turn_html(app.client_payload(state, payload)), \
            "the chosen name did not reach the renderer"

        spent = len(asked)                      # decided once: a reload must not move it
        app.ask_jev = picks("text")
        assert app.choose_widget(state, later) == "textarea", "it asked again on a reload"
        assert len(asked) == spent, "a reload cost a Jev call"

        # An answer outside what the Plan offered, or no answer at all, leaves the Plan's
        # own ui — so the worst case of asking is what this did before anyone asked.
        for answer in (picks("dropdown"), lambda where, questions: None):
            app.ask_jev = answer
            plain = fresh() | {"asked": [first], "answers": {first: "x"}}
            assert app.choose_widget(plain, later) == app.QUESTIONS[later]["ui"]

        # And the first question is never asked about: nothing said yet, nothing to judge.
        app.ask_jev = picks("textarea")
        spent = len(asked)
        app.choose_widget(fresh(), first)
        assert len(asked) == spent, "the first question cost a Jev call"
    finally:
        app.ask_jev = held


@check
def which_input_they_were_given_cannot_move_a_score_or_a_replay() -> None:
    """The constraint that makes choosing at runtime safe at all. Two people who answered the
    same words get the same marks whichever input collected them, and a Replay never reads the
    widget — so the shared ruler and the Golden set are untouched by any of this."""
    qids = list(app.QUESTIONS)
    judged = {qid: {"score": 2.0, "confidence": 0.9, "level": 2, "gap": 0.9} for qid in qids}
    base = fresh() | {"asked": qids, "answers": {qid: f"answer to {qid}" for qid in qids}}
    one = app.score_report(base | {"widgets": {qid: "text" for qid in qids}}, judged)
    two = app.score_report(base | {"widgets": {qid: "textarea" for qid in qids}}, judged)

    assert one["total"] == two["total"] is not None, (one["total"], two["total"])
    assert one["dimensions"] == two["dimensions"], "the widget moved a dimension"
    assert [r["ui"] for r in one["answers"]] != [r["ui"] for r in two["answers"]], \
        "the record does not say which input the person was given"
    assert [dict(r, ui=None) for r in one["answers"]] \
        == [dict(r, ui=None) for r in two["answers"]], "the widget changed a row"

    held = app.ask_jev_repeatedly
    try:
        app.ask_jev_repeatedly = lambda where, asked: (
            {qid: {"score": 2.0, "confidence": 0.9} for qid in qids}
            | {app.FIT: {"score": 2.0, "confidence": 0.9, "probabilities": {"2": 1.0}}})
        replayed = [app.replay(report, app.plan, app.BRIEF)["total"] for report in (one, two)]
    finally:
        app.ask_jev_repeatedly = held
    assert replayed[0] == replayed[1] == one["total"], replayed


@check
def a_question_cannot_offer_an_input_outside_what_the_plan_may_offer() -> None:
    """`ui_options` is held to the same bar as `ui`: in the vocabulary, and agreeing with
    whether the question has options. An unchecked list here would be a KeyError mid-turn."""
    for spare, wanted in ((["dropdown"], "vocabulary"), (["radio"], "disagree")):
        plan = json.loads(json.dumps(app.plan))
        plan["questions"][0] = plan["questions"][0] | {"options": None, "ui_options": spare}
        problems = app.check_plan(plan)
        assert any("ui_options" in problem and wanted in problem for problem in problems), \
            (spare, problems)

@check
def a_question_cannot_name_a_widget_that_does_not_exist() -> None:
    """The other half of the same agreement: the vocabulary is closed, so a plan naming
    something outside it is refused before anyone is asked a question — and a widget that
    takes options may not arrive without them, nor one that takes none arrive with them."""
    for ui, options, wanted in (("dropdown", ["A"], "vocabulary"),
                                ("radio", [], "options"),
                                ("textarea", ["A"], "options")):
        plan = json.loads(json.dumps(app.plan))
        plan["questions"][0] = plan["questions"][0] | {"ui": ui, "options": options}
        problems = app.check_plan(plan)
        assert any(wanted in problem and ".ui" in problem for problem in problems), \
            (ui, options, problems)


@check
def a_level_choice_is_settled_by_code_not_by_a_model() -> None:
    """Below the noise floor the pick carries no information, so code decides."""
    first, other = app.FIRST_QUESTION, BY_DIMENSION[DIMENSIONS[1]][0]
    state = fresh() | {"asked": [first], "answers": {first: "answer"}}
    remaining = [qid for qid in app.QUESTIONS if qid != first]
    floor = app.POLICY["gap_k"] * app.POLICY["noise_floor"]["choice"]

    clear = {"choice": other, "probabilities": {other: 0.5 + floor, "x": 0.5 - floor}}
    assert app.trusted_choice(state, remaining, clear) == other, "a clear lead was overruled"

    level = {"choice": other, "probabilities": {other: 0.5, "x": 0.5 - floor / 4}}
    settled = app.trusted_choice(state, remaining, level)
    assert settled == app.tie_break(state, remaining), "a coin flip was taken at face value"
    assert app.QUESTIONS[settled]["dimension"] in app.under_measured(state), \
        "the tie-break ignored the dimensions still owed an answer"
    assert app.trusted_choice(state, remaining, None) is None
    assert app.gap({"probabilities": {}}) == 1.0, "no distribution must not read as a tie"


@check
def the_report_carries_a_fit_score_beside_the_dimensions() -> None:
    """Q11/Q15: a second axis, scored against the Brief, not a summary of the first."""
    calls: list[dict[str, Any]] = []

    def spy(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        calls.append({"state": state, "questions": questions})
        return {qid: {"score": 1.0 if qid == app.FIT else 2.0, "confidence": 0.8}
                for qid in questions}

    app.ask_jev = spy
    state = fresh() | {"asked": [app.FIRST_QUESTION],
                       "answers": {app.FIRST_QUESTION: "answer"}}
    scores, fit = app.score_all(state)

    asked = scorings(calls)[-1]["questions"]
    assert app.FIT in asked, "the fit question was never asked"
    assert asked[app.FIT]["criteria"] == app.plan["fit"]["rubric"]
    assert app.FIT not in scores, "the fit score leaked into the per-question scores"
    assert fit["score"] == 1.0 and fit["max"] == len(app.plan["fit"]["rubric"]) - 1
    assert scores[app.FIRST_QUESTION]["score"] == 2.0

    report = app.score_report(state, scores, fit)
    assert report["fit"] == fit
    kinds = [f["kind"] for f in report["flags"]]
    assert "tied_levels" not in kinds, "a clear lead must not be flagged"

    shaky = dict(fit, gap=app.POLICY["gap_k"] * app.POLICY["noise_floor"]["score"] / 2)
    flagged = app.score_report(state, scores, shaky)["flags"]
    assert any(f["kind"] == "tied_levels" and f["qid"] == "fit" for f in flagged), \
        "the second axis is not held to the same bar as the first"
    assert all(app.flag_text(f) for f in flagged), "a flag kind has no translation"
    assert report["total"] == 2.0 and report["fit"]["score"] == 1.0, \
        "the two axes collapsed into one number"


@check
def a_link_is_needed_to_start_and_it_expires() -> None:
    """A stateless server cannot burn a link, so it signs one that expires."""
    real_password, app.ADMIN_PASSWORD = app.ADMIN_PASSWORD, "hunter2"
    try:
        assert post({"password": "wrong"}, "/link")[0] == 403
        code, issued = post({"password": "hunter2"}, "/link")
        assert code == 200 and issued["token"], issued
    finally:
        app.ADMIN_PASSWORD = real_password

    fake_jev(BY_DIMENSION[DIMENSIONS[1]][0])
    code, payload = post({"token": issued["token"]}, "/start")
    assert code == 200 and payload["id"] == app.FIRST_QUESTION, payload
    assert "resume" not in state_of(payload), "the interview state carries an account"

    assert post({"token": "forged"}, "/start")[0] == 403
    stale = app.sign({"jti": "old", "exp": app.now() - 1})
    assert post({"token": stale}, "/start")[0] == 403, "expired link accepted"


# ── keeping the result ────────────────────────────────────────────────────
class FakeGithub:
    """Stands in for api.github.com so the shape of the request is checked, not the network."""
    sent: list[dict[str, Any]] = []

    def __init__(self, status: int = 201,
                 replies: list[tuple[int, bytes]] | None = None):
        self.status, self.replies = status, list(replies or [])

    def __call__(self, host: str, **kwargs: Any) -> "FakeGithub":
        self.host = host
        return self

    def request(self, method: str, path: str, body: bytes | None,
                headers: dict[str, str]) -> None:
        FakeGithub.sent.append({"host": self.host, "method": method, "path": path,
                                "body": json.loads(body) if body else None,
                                "headers": headers})

    def getresponse(self) -> FakeResponse:
        if self.replies:
            return FakeResponse(*self.replies.pop(0))
        return FakeResponse(self.status, b'{"content":{}}')

    def close(self) -> None:
        pass


def with_github(status: int = 201) -> Any:
    FakeGithub.sent.clear()
    app.http.client.HTTPSConnection = FakeGithub(status)
    return FakeGithub.sent


@check
def a_finished_interview_is_committed_to_the_operators_repository() -> None:
    real = app.http.client.HTTPSConnection
    settings = (app.DATA_REPO, app.GITHUB_TOKEN, app.save_report, app.RESULTS_DIR)
    app.DATA_REPO, app.GITHUB_TOKEN = "someone/their-private-notes", "ghp_fake"
    app.save_report = REAL_SAVE_REPORT
    app.RESULTS_DIR = pathlib.Path(tempfile.mkdtemp())   # a copy of the repository,
    sent = with_github()                                 # never this working tree
    try:
        state = fresh() | {"asked": [app.FIRST_QUESTION],
                           "answers": {app.FIRST_QUESTION: "answer"}}
        report = app.score_report(state, {app.FIRST_QUESTION: {
            "score": 2.0, "confidence": 0.9, "level": 2, "gap": 0.9}})
        where = app.save_report(report)
    finally:
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN, app.save_report, app.RESULTS_DIR = settings

    assert len(sent) == 1, f"expected one commit, got {len(sent)}"
    call = sent[0]
    assert call["host"] == "api.github.com" and call["method"] == "PUT", call
    assert call["path"].startswith("/repos/someone/their-private-notes/contents/results/"), call
    assert call["headers"]["Authorization"] == "Bearer ghp_fake"
    assert call["body"]["branch"] == app.GITHUB_BRANCH and call["body"]["message"]
    assert where.startswith("someone/their-private-notes/results/"), where

    committed = json.loads(base64.b64decode(call["body"]["content"]).decode())
    assert "resume" not in committed, "the record still carries an account of the applicant"
    assert committed["jti"] == "jti-test", "the link id is missing from the record"


@check
def without_a_repository_the_result_lands_where_a_replay_will_find_it() -> None:
    settings = (app.DATA_REPO, app.GITHUB_TOKEN, app.save_report, app.RESULTS_DIR)
    app.DATA_REPO, app.GITHUB_TOKEN = "", ""
    app.save_report = REAL_SAVE_REPORT
    app.RESULTS_DIR = pathlib.Path(tempfile.mkdtemp())
    sent, real = with_github(), app.http.client.HTTPSConnection
    try:
        report = app.score_report(*scored((app.FIRST_QUESTION, 2.0, 0.9)))
        where = pathlib.Path(app.save_report(report))
        written = json.loads(where.read_text(encoding="utf-8"))
        found = sorted(app.RESULTS_DIR.glob("*.json"))
    finally:
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN, app.save_report, app.RESULTS_DIR = settings

    assert not sent, "an unconfigured deployment still called GitHub"
    assert written == report, "the file is not the report"
    assert found == [where], "regression() globs *.json here and would not see it"


@check
def an_outage_is_not_a_zero() -> None:
    """Jev was unreachable for a whole interview once. The record read 0.00/3."""
    app.ask_jev = lambda state, questions: None
    state = fresh() | {"asked": [app.FIRST_QUESTION],
                       "answers": {app.FIRST_QUESTION: "answer"}}
    scores, fit = app.score_all(state)
    assert scores == {} and fit is None, (scores, fit)

    report = app.score_report(state, scores, fit)
    assert report["total"] is None, "an outage was reported as a score of zero"
    assert report["scored"] is False
    kinds = [f["kind"] for f in report["flags"]]
    assert kinds[0] == "not_scored", f"the outage is not the first thing said: {kinds}"
    assert app.flag_text({"kind": "not_scored"}), "no translation for an outage"

    worked = app.score_report(*scored((app.FIRST_QUESTION, 2.0, 0.9)))
    assert worked["total"] == 2.0 and worked["scored"] is True
    assert "not_scored" not in [f["kind"] for f in worked["flags"]]


@check
def an_outage_can_be_scored_again_from_the_record() -> None:
    """The answers are what cannot be recovered later. The marks can."""
    app.ask_jev = lambda state, questions: None
    state = fresh() | {"asked": [app.FIRST_QUESTION],
                       "answers": {app.FIRST_QUESTION: "what they actually said"}}
    lost = app.score_report(state, *app.score_all(state))
    assert lost["total"] is None and lost["scored"] is False

    directory = pathlib.Path(tempfile.mkdtemp())
    path = directory / "outage.json"
    filed = {"decision": "hired", "at": "2026-10-20T09:00:00+00:00"}
    path.write_text(json.dumps(lost | {"outcome": filed}, ensure_ascii=False),
                    encoding="utf-8")

    fake_jev(None, score=2.0, confidence=0.9)          # Jev is back
    fresh_report = app.rescore(path)

    assert fresh_report is not None and fresh_report["scored"] is True
    assert fresh_report["total"] == 2.0, fresh_report["total"]
    assert fresh_report["finished_at"] == lost["finished_at"], \
        "the interview happened when it happened"
    assert fresh_report["rescored_at"], "a second judgement must say it was one"
    assert fresh_report["answers"][0]["answer"] == "what they actually said"
    assert json.loads(path.read_text(encoding="utf-8"))["total"] == 2.0, \
        "the record on disk was not replaced"
    assert fresh_report.get("outcome") == filed, \
        "scoring again threw away what a person had decided"




# ── the repository is the record, and this process holds a copy of it ────
@check
def a_record_is_replaced_and_not_only_created() -> None:
    """GitHub refuses to overwrite a file unless told which version it is replacing.

    Without the second attempt nothing could ever be written onto a record already
    committed — which is every approval and every hiring decision.
    """
    real = app.http.client.HTTPSConnection
    settings = (app.DATA_REPO, app.GITHUB_TOKEN)
    app.DATA_REPO, app.GITHUB_TOKEN = "someone/theirs", "ghp_fake"
    FakeGithub.sent.clear()
    app.http.client.HTTPSConnection = FakeGithub(replies=[
        (422, b'{"message":"sha wasn\'t supplied"}'),   # PUT, refused
        (200, b'{"sha":"abc123"}'),                     # GET, the version that is there
        (200, b'{"content":{}}')])                      # PUT again, accepted
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        where = app.github_put("results/x.json", b"{}", "again")
    finally:
        sys.stderr = real_stderr
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN = settings

    calls = [(c["method"], c["path"]) for c in FakeGithub.sent]
    assert where == "someone/theirs/results/x.json", where
    assert len(calls) == 3 and calls[1][0] == "GET", calls
    assert FakeGithub.sent[2]["body"].get("sha") == "abc123", \
        "the second attempt did not say which version it was replacing"


@check
def an_approval_is_committed_and_not_only_written_here() -> None:
    """A host that wipes its disk loses an approval kept only here, and then
    questions_for falls back to the shared plan without saying so."""
    settings = (app.DATA_REPO, app.GITHUB_TOKEN, app.CANDIDATES_DIR)
    app.DATA_REPO, app.GITHUB_TOKEN = "someone/theirs", "ghp_fake"
    app.CANDIDATES_DIR = pathlib.Path(tempfile.mkdtemp())
    here = app.CANDIDATES_DIR / "cand.json"
    sent, real = with_github(), app.http.client.HTTPSConnection
    try:
        app.save_candidate({"id": "cand", "approved": True, "questions": []}, "Approved")
        written = json.loads(here.read_text(encoding="utf-8"))
    finally:
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN, app.CANDIDATES_DIR = settings

    assert written["approved"] is True, written
    assert len(sent) == 1, f"the approval was not committed ({len(sent)} calls)"
    assert sent[0]["path"].endswith("contents/candidates/cand.json"), sent[0]["path"]


@check
def an_applicant_nobody_could_write_questions_for_can_still_be_interviewed() -> None:
    """The generator is the one part of this that reaches outside, and it fails — an
    expired key, no credits, a bad gateway. The applicant has already been thanked and
    left. Without an approval offered on that record there is no link that can ever be
    made for them, and the Plan's own questions are a whole interview without the extras.
    """
    failed = {"id": "nobody", "at": app.stamp(), "approved": False, "failed": True,
              "questions": []}
    pane = app.applicant_html(failed)
    assert "/admin/approve" in pane, "a failed applicant is offered no way to be approved"
    assert app.ADMIN_TEXT["unwritable"] in pane, "the failure is not said on the record"

    # and approving it hands the interview the plan's questions, with no extras
    state = app.new_state("jti", "nobody")
    assert state["extra"] == [], state


@check
def the_working_copy_is_filled_from_the_repository() -> None:
    """Otherwise the Interviewer's list empties on every restart while the records
    sit safely in the repository the server wrote them to."""
    settings = (app.DATA_REPO, app.GITHUB_TOKEN, app.RESULTS_DIR, app.CANDIDATES_DIR)
    app.DATA_REPO, app.GITHUB_TOKEN = "someone/theirs", "ghp_fake"
    results = app.RESULTS_DIR = pathlib.Path(tempfile.mkdtemp())
    candidates = app.CANDIDATES_DIR = pathlib.Path(tempfile.mkdtemp())
    one = json.dumps({"content": base64.b64encode(b'{"jti": "abc"}').decode()}).encode()
    FakeGithub.sent.clear()
    real = app.http.client.HTTPSConnection
    app.http.client.HTTPSConnection = FakeGithub(replies=[
        (200, b'[{"type":"file","name":"one.json"},{"type":"dir","name":"old"},'
              b'{"type":"file","name":"notes.md"}]'),
        (200, one),
        (200, b'[{"type":"file","name":"cand.json"}]'),
        (200, one)])
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        app.fill_from_repository()
    finally:
        sys.stderr = real_stderr
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN, app.RESULTS_DIR, app.CANDIDATES_DIR = settings

    assert sorted(p.name for p in results.glob("*")) == ["one.json"], list(results.glob("*"))
    assert sorted(p.name for p in candidates.glob("*")) == ["cand.json"]
    assert json.loads((results / "one.json").read_text(encoding="utf-8")) == {"jti": "abc"}
    fetched = [c["path"] for c in FakeGithub.sent]
    assert not any("old" in path or "notes.md" in path for path in fetched), fetched


@check
def the_brief_and_the_plan_come_from_the_operators_repository() -> None:
    """They are the operator's, not the deployment's — and a server that
    cannot read them must stop, not fall back to the sample it shipped with."""
    settings = (app.repo_file, app.DATA_REPO, app.PLAN_PATH)
    app.DATA_REPO = "someone/theirs"
    try:
        app.PLAN_PATH = pathlib.Path("")
        try:
            app.deployed_plan(from_repo=False)
        except SystemExit as refusal:
            assert "PLAN is unset" in str(refusal), refusal
        else:
            raise AssertionError("it started on a plan nobody named")
        app.PLAN_PATH = settings[2]

        deployed, where = app.deployed_plan(from_repo=False)
        assert where == "this checkout" and json.loads(deployed)["questions"]

        app.repo_file = lambda path: b'{"from": "brief.md", "questions": []}'
        theirs, where = app.deployed_plan(from_repo=True)
        assert where == "someone/theirs", where
        assert json.loads(theirs)["from"] == "brief.md", theirs

        app.repo_file = lambda path: None
        try:
            app.deployed_plan(from_repo=True)
        except SystemExit as refusal:
            assert "someone/theirs" in str(refusal), refusal
        else:
            raise AssertionError("it started anyway, on whatever plan was deployed")
    finally:
        app.repo_file, app.DATA_REPO, app.PLAN_PATH = settings

    # `--plan` is run before a plan exists, so it cannot fall back to naming one.
    refused = subprocess.run(
        [sys.executable, "app.py", "--plan", "brief.md"], capture_output=True, text=True,
        cwd=pathlib.Path(__file__).parent, env=os.environ | {"PLAN": ""})
    assert refused.returncode == 1 and "--out" in refused.stderr, refused.stderr


@check
def importing_the_module_reads_nothing() -> None:
    """A plan read on import is a plan nobody chose. It ties what the deployment runs
    to whatever argv and the environment happened to be, and it puts the operator's
    repository behind `python3 test_app.py`, which is supposed to need no key."""
    asked = subprocess.run(
        [sys.executable, "-c", "import app; print(bool(app.plan or app.QUESTIONS or app.SHELL))"],
        capture_output=True, text=True, cwd=pathlib.Path(__file__).parent,
        env=os.environ | {"DATA_REPO": "someone/theirs", "GITHUB_TOKEN": "not-a-token"})
    assert asked.returncode == 0, f"importing it did something: {asked.stderr}"
    assert asked.stdout.strip() == "False", \
        f"a plan was loaded on import: {asked.stdout}{asked.stderr}"


@check
def a_deployment_can_be_asked_whether_it_will_keep_anything() -> None:
    """A token that cannot write fails the way a token that can looks, so there has
    to be a way to ask before an interview is the thing that finds out."""
    settings = (app.DATA_REPO, app.GITHUB_TOKEN)
    app.DATA_REPO, app.GITHUB_TOKEN = "", ""
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        unset = app.check_repo()
        complaint = sys.stderr.getvalue()
    finally:
        sys.stderr = real_stderr
        app.DATA_REPO, app.GITHUB_TOKEN = settings
    assert unset == 1 and "DATA_REPO and GITHUB_TOKEN" in complaint, complaint

    app.DATA_REPO, app.GITHUB_TOKEN = "someone/theirs", "ghp_fake"
    real = app.http.client.HTTPSConnection
    sent = FakeGithub.sent
    sent.clear()
    brief = "the job, and who we want\n".encode()
    their_plan = json.dumps({"from": "brief.md", "questions": [{"id": "one"}],
                             "source": {"brief_sha256": hashlib.sha256(brief).hexdigest()}})
    holding = lambda raw: json.dumps({"content": base64.b64encode(raw).decode()}).encode()
    app.http.client.HTTPSConnection = FakeGithub(replies=[
        (200, b"[]"),                     # the read probe
        (201, b'{"content":{}}'),         # .keep created
        (200, b'{"content":{}}'),         # .keep replaced
        (200, holding(their_plan.encode())),
        (200, holding(brief))])
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        fine = app.check_repo()
    finally:
        sys.stderr = real_stderr
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN = settings

    writes = [c for c in sent if c["method"] == "PUT"]
    assert fine == 0, "a repository it can read and write was reported as broken"
    assert len(writes) == 2, "the replacing write is the half nothing else exercises"
    assert all(c["path"].endswith(f"contents/{app.KEEP_PROBE}") for c in writes), writes
    assert writes[0]["body"]["content"] != writes[1]["body"]["content"], \
        "identical bytes: GitHub makes no commit, so this proves nothing"


@check
def a_hiring_decision_is_filed_onto_the_record_and_committed() -> None:
    """The one field in a Report a human writes, onto a record already committed."""
    settings = (app.DATA_REPO, app.GITHUB_TOKEN, app.RESULTS_DIR)
    app.DATA_REPO, app.GITHUB_TOKEN = "someone/theirs", "ghp_fake"
    results = app.RESULTS_DIR = pathlib.Path(tempfile.mkdtemp())
    name = "2026-01-01T000000+0000-abc.json"
    (results / name).write_text(json.dumps({"jti": "abc", "total": 2.0}), encoding="utf-8")
    sent, real = with_github(), app.http.client.HTTPSConnection
    try:
        filed = app.record_outcome(name, "hired")
        on_disk = json.loads((results / name).read_text(encoding="utf-8"))
        nonsense = app.record_outcome(name, "probably")
        escaping = app.record_outcome("../../../etc/passwd", "hired")
    finally:
        app.http.client.HTTPSConnection = real
        app.DATA_REPO, app.GITHUB_TOKEN, app.RESULTS_DIR = settings

    assert filed and filed["outcome"]["decision"] == "hired" and filed["outcome"]["at"]
    assert on_disk["outcome"] == filed["outcome"], "the copy this page reads was not updated"
    assert on_disk["total"] == 2.0, "filing a decision moved a mark"
    assert nonsense is None, "any word at all was accepted as a decision"
    assert escaping is None, "a path escaped RESULTS_DIR"
    assert len(sent) == 1 and sent[0]["path"].endswith(f"contents/results/{name}"), sent

    # The control has to post back the record it is drawn under, and a dimension name
    # once shadowed it: the buttons offered to file a decision onto "情報共有・相談の速さ".
    drawn = app.report_html(filed, name)
    assert drawn.count(f"&quot;file&quot;: &quot;{name}&quot;") == 2, drawn[-700:]


@check
def the_regression_counts_against_the_decision_where_there_is_one() -> None:
    """A record that carries an Outcome is measured against what a person decided,
    not against the earlier plan's own total."""
    boundary = app.MEASURED.get("measurement_boundary", 1.5)
    results = pathlib.Path(tempfile.mkdtemp())
    whole = [{"text": text_of(BY_DIMENSION[dim][0]), "answer": "answer"}
             for dim in DIMENSIONS]
    # It scored above the bar and was not hired. A replay that now scores it 0 agrees
    # with the person — where its own old total would have called that a flip.
    outcome = {"total": boundary + 0.5, "outcome": {"decision": "not_hired", "at": "then"}}
    (results / "a.json").write_text(json.dumps(
        {"answers": whole, "fit": {"score": 2.0}} | outcome), encoding="utf-8")
    # The same decision on a record only one dimension carried over to. Its total was
    # renormalised over what survived, so it is not on the scale the bar is fitted in.
    (results / "b.json").write_text(json.dumps(
        {"answers": whole[:1], "fit": {"score": 2.0}} | outcome), encoding="utf-8")

    fake_jev(None, score=0.0, confidence=0.9)
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        flips = app.regression(app.plan, app.BRIEF, results)
        said = sys.stderr.getvalue()
    finally:
        sys.stderr = real_stderr

    assert flips == 0, "a replay that agrees with the decision was counted as a flip"
    assert "1 of 2 were measured against a hiring decision" in said, said
    assert f"uncovered {DIMENSIONS[1]}" in said, said


@check
def the_bar_is_fitted_to_the_decisions_and_not_guessed() -> None:
    """The midpoint of the scale is a convention. Where decisions exist, the cut that
    disagrees with fewest of them is a measurement."""
    decided = [(2.4, "pass"), (2.2, "pass"), (1.4, "fail"), (0.9, "fail"), (1.6, "fail")]
    cut, wrong, wrong_at_midpoint, span = app.best_boundary(decided, 1.5)
    assert wrong == 0, (cut, wrong)
    assert 1.6 < cut <= 2.2, f"{cut} does not separate them"
    assert wrong_at_midpoint == 1, "the midpoint gets the 1.6 one wrong, and should say so"
    # The span is the band the cut sits in — every value above 1.6 and up to 2.2 separates
    # these five the same way, and the midpoint is only the one that stands for it.
    assert span == (1.6, 2.2) and span[0] < cut <= span[1], (span, cut)

    one = [(2.0, "pass")]
    cut, wrong, _, span = app.best_boundary(one, 1.5)
    assert (cut, wrong) == (1.5, 0), "one record moved the bar"
    # Every cut at or below 2.0 agrees with it, and passing everybody is one of them:
    # a history that all went one way is often best described by a cut outside it.
    assert span == (-math.inf, 2.0), span
    assert app.best_boundary([(1.0, "pass"), (2.0, "pass")], 2.5)[1] == 0, \
        "no cut below the lowest total was tried, so passing everybody was unreachable"


@check
def a_record_says_which_judge_left_the_marks() -> None:
    """There is nothing to pin — every published Jev name is a moving alias — so the
    record carries the judge instead, or a Replay cannot tell drift from noise."""
    fake_jev(None, score=2.0, confidence=0.9)
    state = fresh() | {"asked": [app.FIRST_QUESTION],
                       "answers": {app.FIRST_QUESTION: "what they said"}}
    app.finish(state)
    assert saved, "the interview did not finish"
    assert saved[0]["judge"]["model"], saved[0].get("judge")


@check
def what_a_person_decided_fits_a_curve_and_never_one_persons_score() -> None:
    """The decisions are what say whether this company would hire someone, so they are what
    a total has to be read against. What they may not do is enter the arithmetic that makes
    a total: a score that moved with hiring history would mean one thing for the
    tenth applicant and another for the eleventh, and the shared ruler is the product.

    So the fit is a curve, and the same total reads the same off it for everybody — however
    many decisions have landed since, and in whatever order they arrived.
    """
    decided = [(0.5, "fail"), (1.0, "fail"), (1.5, "fail"), (1.6, "pass"), (2.0, "pass"),
               (2.5, "pass"), (1.2, "fail"), (2.2, "pass"), (0.8, "fail"), (2.8, "pass"),
               (1.9, "pass"), (0.3, "fail")]
    fit = app.calibrate(decided, 1.5)
    assert fit["usable"] and fit["n"] == 12 and fit["hired"] == 6, fit

    rates = [rate for _, _, rate in fit["curve"]]
    assert rates == sorted(rates), f"a higher total read as less likely to be hired: {fit}"
    assert app.calibrate(list(reversed(decided)), 1.5)["curve"] == fit["curve"], \
        "the curve depended on the order the decisions arrived in"
    assert app.fitted(fit["curve"], 0.5) <= app.fitted(fit["curve"], 2.5), "not monotone"
    assert app.fitted(fit["curve"], 2.0) == app.fitted(fit["curve"], 2.0), "not a function"
    assert app.fitted(fit["curve"], None) is None, "no total must not read as a probability"
    # A violator is pooled into its neighbour rather than left to invert the curve,
    # and a run of equal rates collapses, so the curve does not count the decisions.
    assert len(app.calibrate([(1.0, "fail")] * 5 + [(2.0, "pass")] * 5, 1.5)["curve"]) == 2
    assert app.isotonic([(1.0, 1.0), (2.0, 0.0)]) == [(1.0, 2.0, 0.5)]

    # And the gap between the last rejection and the first hire is the case nobody
    # decided. Reading a total there as a confident 1.0 was the bug this pins.
    gap = app.calibrate([(1.0, "fail"), (1.1, "fail"), (2.0, "pass"), (2.1, "pass")], 1.5)
    middle = app.fitted(gap["curve"], 1.55)
    assert 0.0 < middle < 1.0, f"an undecided total read as settled: {middle}"
    assert app.fitted(gap["curve"], 1.1) == 0.0 and app.fitted(gap["curve"], 2.0) == 1.0
    assert app.fitted(gap["curve"], 0.2) == 0.0, "below everything decided"
    assert app.fitted(gap["curve"], 9.9) == 1.0, "above everything decided"


@check
def a_bar_fitted_on_too_few_decisions_refuses_to_be_a_number() -> None:
    """A boundary fitted on fifteen records containing three hires is noise.
    A curve fitted on three hires reproduces those three hires, so below the minimum this
    reports the shape and withholds the number rather than dressing noise as a score."""
    assert app.calibrate([], 1.5)["usable"] is False, "nothing fitted a bar"
    assert app.calibrate([], 1.5)["bar"] == 1.5, "an empty fit moved the bar"

    thin = app.calibrate([(1.0, "fail"), (2.0, "pass"), (2.2, "pass")], 1.5)
    assert thin["usable"] is False and thin["n"] == 3, thin
    # And a history with only one kind in it is unusable however long it grows.
    lopsided = app.calibrate([(1.0 + n / 10, "pass") for n in range(20)], 1.5)
    assert lopsided["n"] == 20 and lopsided["usable"] is False, \
        "twenty decisions that all went one way fitted a bar"


@check
def the_fitted_bar_is_a_range_and_says_so() -> None:
    """Decisions that separate cleanly leave a gap, and every cut inside it disagrees with
    none of them. Reporting the single number alone would overstate what the decisions
    settled, so the span every equally-good cut spans is reported beside it."""
    fit = app.calibrate([(1.0, "fail"), (1.1, "fail"), (2.0, "pass"), (2.1, "pass")], 1.5)
    assert fit["wrong"] == 0, fit
    low, high = fit["span"]
    assert low <= fit["bar"] <= high, fit
    # The gap itself, not the one midpoint inside it: every cut above 1.1 and up to 2.0
    # disagrees with none of them, and naming 1.55 alone would claim the decisions settled
    # the bar four times more precisely than they did.
    assert (low, high) == (1.1, 2.0), f"the span is not the gap it was fitted in: {fit}"


@check
def a_fitted_bar_is_scored_on_decisions_it_did_not_see() -> None:
    """`calibrate` reports how many decisions the fitted cut disagrees with, and that number
    always flatters: a cut may be placed between any two totals, so on decisions that
    separate it is perfect by construction and says nothing about the next applicant.

    Leave one out is the honest one, and the bar to beat is not a coin — it is always
    answering the commoner way. Measured here: a total that carries the signal scores 1.0,
    and a total that carries none scores below chance and has to say so.
    """
    separable = [(0.5, "fail"), (1.0, "fail"), (1.2, "fail"), (0.8, "fail"), (0.3, "fail"),
                 (1.1, "fail"), (1.9, "pass"), (2.0, "pass"), (2.5, "pass"), (2.2, "pass"),
                 (2.8, "pass"), (1.95, "pass")]
    good = app.accuracy(separable, 1.5)
    assert good["rate"] == 1.0 and good["beats_chance"] is True, good
    assert good["chance"] == 0.5, good

    # The same totals, decided alternately: the total predicts nothing at all.
    noise = [(float(n), "pass" if n % 2 == 0 else "fail") for n in range(1, 13)]
    blind = app.accuracy(noise, 1.5)
    assert blind["rate"] < blind["chance"], blind
    assert blind["beats_chance"] is False, "noise was reported as a working bar"
    # And in-sample it would have looked respectable, which is the trap this avoids.
    assert app.calibrate(noise, 1.5)["wrong"] < len(noise) - blind["right"], \
        "the in-sample count was not the more flattering one"

    assert app.accuracy([(1.0, "fail"), (2.0, "pass")], 1.5)["rate"] is None, \
        "two decisions cannot score a bar"


@check
def the_decision_a_person_files_reaches_the_fit() -> None:
    """End to end for the one field a human writes: an `outcome` on a committed record is
    read back by a Replay, fitted, and scored on decisions the fit did not see. Nothing here
    touches the real Golden set — the records are built for this check."""
    boundary = app.MEASURED.get("measurement_boundary", 1.5)
    results = pathlib.Path(tempfile.mkdtemp())
    rows = [{"text": app.QUESTIONS[qid]["text"], "answer": f"answer to {qid}"}
            for qid in app.QUESTIONS]
    for index in range(12):
        decision = "not_hired" if index < 6 else "hired"
        (results / f"{index:02d}.json").write_text(json.dumps(
            {"answers": rows, "total": boundary, "fit": {"score": 2.0},
             "outcome": {"decision": decision, "at": "then"}}), encoding="utf-8")

    calls, held = [], app.ask_jev_repeatedly
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        def varying(where: Any, asked: Any) -> dict[str, Any]:
            calls.append(1)                     # the first six score 0, the rest score 3
            mark = 0.0 if len(calls) <= 6 else 3.0
            return {qid: {"score": mark, "confidence": 0.9} for qid in app.QUESTIONS} | {
                app.FIT: {"score": mark, "confidence": 0.9, "probabilities": {"2": 1.0}}}
        app.ask_jev_repeatedly = varying
        app.regression(app.plan, app.BRIEF, results)
        said = sys.stderr.getvalue()
    finally:
        app.ask_jev_repeatedly = held
        sys.stderr = real_stderr

    assert "12 of 12 were measured against a hiring decision" in said, said
    assert "which is the bar's resolution on 12 decision(s)" in said, said
    assert "on decisions it did not see it is right 12/12" in said, said
    assert "ahead of it." in said and "NOT ahead" not in said, said
    assert "fitted: [(0.0, 0.0, 0.0), (3.0, 3.0, 1.0)]" in said, said





PURE_DIGEST = """
import hashlib, json, sys
import app, core
app.boot()
dims = [{"name": n, "weight": 1.0, "aggregate": "mean"} for n in app.WEIGHTS]
rows = [{"dimension": n, "score": 1.5, "confidence": 0.9} for n in app.WEIGHTS]
decided = [(n / 10, "pass" if n > 15 else "fail") for n in range(1, 26)]
runs = [{"choice": "a", "probabilities": {"a": 0.5 + n / 100, "b": 0.5 - n / 100}}
        for n in range(-12, 13)]
asked = list(app.QUESTIONS)[:3]
state = app.new_state("j") | {"asked": asked, "answers": {q: "a" for q in asked}}
judged = {q: {"score": 2.0, "confidence": 0.9, "level": 2, "gap": 0.9} for q in asked}
combined = core.combine_dimensions(rows, dims)
out = [core.check_plan(json.loads(json.dumps(app.plan))), combined,
       [core.plan_total(combined[0], dims, how) for how in core.TOTALS],
       core.calibrate(decided, 1.5), core.accuracy(decided, 1.5), core.noise_floor(runs, 3),
       core.plan_schema(), core.supplement_schema(4),
       [core.widget_choices(q) for q in app.QUESTION_LIST],
       sorted(app.questions_for(state)), app.scored_per_dimension(state),
       app.under_measured(state), app.narrow(state, list(app.QUESTIONS)),
       app.tie_break(state, list(app.QUESTIONS)),
       app.score_report(state, judged, at="2026-01-01T00:00:00+00:00")]
print(hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest())
"""


@check
def the_same_input_gives_the_same_output_in_any_process() -> None:
    """What `pure` has to mean here, checked the way it can actually break.

    Python randomises string hashing per process, so a function that iterates a set answers
    in a different order the next time it runs — and no amount of calling it twice inside one
    process would show that. So the whole readable half runs in three processes with different
    hash seeds and the digests have to match.

    `at` is passed to `score_report` because a clock read inside it would be the one thing
    there that answered differently to the same question. That is why it is an argument.
    """
    digests = set()
    for seed in ("0", "1", "524287"):
        run = subprocess.run(
            [sys.executable, "-c", PURE_DIGEST], capture_output=True, text=True,
            cwd=pathlib.Path(__file__).parent, env=os.environ | {"PYTHONHASHSEED": seed})
        assert run.returncode == 0, run.stderr
        digests.add(run.stdout.strip())
    assert len(digests) == 1, f"the same input answered differently per process: {digests}"


@check
def nothing_that_reads_a_state_reads_a_file() -> None:
    """Made true rather than written down. What an interview asks used to be fetched from the
    applicant's record on every turn, so six functions answered differently once that file
    moved, and a finished interview could raise KeyError when it had gone.

    Their questions are copied into the state when it is built. Building a state is where one
    file may be read; reading a state is not.
    """
    reaches = reaches_in("app.py")

    disk = {"load_candidate", "candidate_path", "repo_file", "github_get", "github_call"}
    for reader in ("questions_for", "score_report", "narrow", "tie_break", "under_measured",
                   "scored_per_dimension", "trusted_choice", "score_all"):
        assert not reaches(reader) & disk, \
            f"{reader} reaches {sorted(reaches(reader) & disk)} — what is asked belongs in the state"
    for builder in ("new_state", "state_from"):
        assert "load_candidate" in reaches(builder), \
            f"{builder} stopped copying the applicant's questions into the state"


@check
def an_answer_whose_question_is_gone_is_reported_and_not_a_crash() -> None:
    """The answers are the part of a record that cannot be recovered later, so a report has to
    come out even when the question one was given to cannot be found — in the plan or in the
    state's own copy. It says so instead of raising."""
    first = app.FIRST_QUESTION
    state = fresh() | {"asked": [first, "x_vanished"],
                       "answers": {first: "said", "x_vanished": "also said"}}
    judged = {first: {"score": 2.0, "confidence": 0.9, "level": 2, "gap": 0.9}}
    report = app.score_report(state, judged)
    assert report["total"] is not None, "one lost question threw the whole report away"
    lost = [flag for flag in report["flags"] if flag["kind"] == "no_question"]
    assert lost == [{"kind": "no_question", "qid": "x_vanished"}], report["flags"]
    assert all(app.flag_text(flag) for flag in report["flags"]), "a flag has no translation"
    assert [row["qid"] for row in report["answers"]] == [first]

@check
def every_dispatch_in_the_program_is_a_named_table() -> None:
    """The one rule, checked instead of believed, in the two halves it actually has.

    Nothing branches by hand: no function compares a value against three or more string
    literals, which is the shape `widget_html` had before it became a table. And every table
    is accounted for — either a Plan or Jev names the entry, or it is routing, where which
    function `/admin/login` reaches is the same for every company and so belongs in code.

    A new table appearing is a decision about this program's one idea, and it fails here
    until somebody says which kind it is.
    """
    judged = {"AGGREGATORS": "aggregate", "TOTALS": "total",
              "TIE_BREAKS": "tie_break", "COVERAGE": "coverage"}
    routing = {"FORM_ROUTES", "ROUTES"}
    tables = set()
    for where in ("core.py", "app.py"):
        tree = ast.parse(pathlib.Path(where).read_text(encoding="utf-8"))
        held = list(tree.body)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                held += node.body
        for node in held:
            named = None
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                named = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                named = node.target.id
            if named and isinstance(node.value, ast.Dict) and node.value.values and all(
                    isinstance(v, (ast.Lambda, ast.Name, ast.Attribute)) for v in node.value.values):
                tables.add(named)
    assert tables == set(judged) | routing | {"WIDGET_HTML"}, \
        f"a dispatch table appeared or vanished without this check: {sorted(tables)}"

    # The tables live in core.py, beside the rest of the half that holds no Plan.
    for table, vocabulary in judged.items():
        assert set(getattr(core, table)) == set(core.VOCABULARY[vocabulary]), \
            f"{table} and the vocabulary it is indexed by have parted"
    assert set(app.WIDGET_HTML) == set(app.WIDGETS), "the inputs and their renderers parted"

    for where in ("core.py", "app.py"):
        tree = ast.parse(pathlib.Path(where).read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            literals: list[str] = []
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Compare)
                        and isinstance(node.ops[0], (ast.Eq, ast.In))):
                    continue
                for side in node.comparators:
                    if isinstance(side, ast.Constant) and isinstance(side.value, str):
                        literals.append(side.value)
                    elif isinstance(side, (ast.Tuple, ast.List)):
                        literals += [e.value for e in side.elts if isinstance(e, ast.Constant)
                                     and isinstance(e.value, str)]
            assert len(literals) < 3, \
                f"{where}:{fn.name} dispatches on {literals} by hand; make it a table"

@check
def the_core_is_closed_and_cannot_reach_the_plan() -> None:
    """`core.py` holds the part of this program with no opinion about any particular job,
    and what keeps it that way is not care — it is the import graph. It imports the standard
    library and nothing else, so there is no name through which the Plan could reach it.

    This is the check that makes the file boundary worth having. Measuring purity once says
    nothing about tomorrow; a module that holds no path to the Plan cannot drift into one.
    """
    tree = ast.parse(pathlib.Path("core.py").read_text(encoding="utf-8"))
    outside = {a.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for a in node.names}
    outside |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert outside <= {"__future__", "hashlib", "json", "math", "pathlib", "statistics",
                       "typing"}, \
        f"core.py reached past the standard library: {sorted(outside)}"

    held = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            held.add(node.name)
        elif isinstance(node, ast.Assign):
            held |= {x.id for target in node.targets for x in ast.walk(target)
                     if isinstance(x, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            held.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            held |= {(a.asname or a.name).split(".")[0] for a in node.names}
    assert not held & {"plan", "POLICY", "QUESTIONS", "WEIGHTS", "BRIEF", "FIRST_QUESTION"}, \
        "core.py grew a Plan of its own"

    known = held | set(dir(builtins))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        bound = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        for nested in ast.walk(node):                 # lambdas and inner defs alike
            if isinstance(nested, (ast.Lambda, ast.FunctionDef)) and nested is not node:
                spec = nested.args
                bound |= {a.arg for a in spec.posonlyargs + spec.args + spec.kwonlyargs}
        bound |= {t.id for t in ast.walk(node)
                  if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        bound |= {f.name for f in ast.walk(node)
                  if isinstance(f, ast.FunctionDef) and f is not node}
        bound |= {h.name for h in ast.walk(node)
                  if isinstance(h, ast.ExceptHandler) and h.name}
        loose = {t.id for t in ast.walk(node)
                 if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load)} - bound - known
        assert not loose, f"core.{node.name} reads {sorted(loose)}, which core.py does not hold"

@check
def the_plan_names_how_its_dimensions_become_one_number() -> None:
    """Whether a strong axis may fairly carry a weak one is a fact about what a company
    promises, so the Plan names the method and code only implements the names. The
    vocabulary and the implementations must agree in both directions — otherwise a plan the
    schema accepts is a KeyError at the end of somebody's interview — and the names have to
    be different choices, or the vocabulary is ceremony.
    """
    assert set(app.TOTALS) == set(app.VOCABULARY["total"]), \
        (sorted(app.TOTALS), sorted(app.VOCABULARY["total"]))

    dimensions = [{"name": "a", "weight": 1.0, "aggregate": "mean"},
                  {"name": "b", "weight": 1.0, "aggregate": "mean"}]
    combined = {"a": {"score": 3.0}, "b": {"score": 1.0}}
    got = {how: app.plan_total(combined, dimensions, how) for how in app.TOTALS}
    assert got["weighted_mean"] == 2.0, got
    assert got["min_dimension"] == 1.0, got            # a weak axis is not carried
    assert len(set(got.values())) == len(got), f"two names, one behaviour: {got}"
    assert all(app.plan_total({}, dimensions, how) is None for how in app.TOTALS), \
        "nothing scored has to read as no mark, never as zero"

    for broken in ("average", None):
        plan = json.loads(json.dumps(app.plan))
        plan["total"] = broken
        assert any("total" in problem for problem in app.check_plan(plan)), broken
    gone = json.loads(json.dumps(app.plan))
    del gone["total"]
    assert any("total" in problem for problem in app.check_plan(gone)), "no total passed"

    # And a record has to say which method made its total, or it cannot be read later.
    state, judged = scored((app.FIRST_QUESTION, 2.0, 0.9))
    assert app.score_report(state, judged)["total_by"] == app.plan["total"]



@check
def the_plan_names_how_a_level_pick_and_a_coverage_debt_are_settled() -> None:
    """Jev returning a level choice, and a dimension still owed answers, were both settled by
    a preference written into this program that no Plan could say otherwise about. Both are
    named now, and the same two things must hold as for every other name: the vocabulary and
    the implementations agree in both directions, and the names are different choices rather
    than ceremony.
    """
    for table, vocabulary in ((app.TIE_BREAKS, "tie_break"), (app.COVERAGE, "coverage")):
        assert set(table) == set(app.VOCABULARY[vocabulary]), \
            (vocabulary, sorted(table), sorted(app.VOCABULARY[vocabulary]))

    # A level pick where what is owed is not what comes first.
    asking = {"a1": {"dimension": "A"}, "b1": {"dimension": "B"}}
    settled = {name: fn(["a1", "b1"], asking, ["B"]) for name, fn in app.TIE_BREAKS.items()}
    assert settled == {"cover_first": "b1", "plan_order": "a1"}, settled
    # An applicant's own question never stands in for coverage.
    theirs = {"x_1": {"dimension": "B", "extra": True}, "b1": {"dimension": "B"}}
    assert app.TIE_BREAKS["cover_first"](["x_1", "b1"], theirs, ["B"]) == "b1"

    # A debt the remaining budget can only just absorb: held back for it, or not.
    debt = {"short": ["B"], "owed": 2, "turns_left": 2, "head_start": 0}
    held = {name: fn(["a1", "b1"], asking, debt) for name, fn in app.COVERAGE.items()}
    assert held == {"reserve_debt": ["b1"], "free": ["a1", "b1"]}, held
    # With turns to spare, reserving changes nothing — that is what `reserve_early` moves.
    assert app.COVERAGE["reserve_debt"](
        ["a1", "b1"], asking, dict(debt, turns_left=9)) == ["a1", "b1"]

    for field in ("tie_break", "coverage"):
        for broken in ("whatever", None):
            plan = json.loads(json.dumps(app.plan))
            plan[field] = broken
            assert any(field in problem for problem in app.check_plan(plan)), (field, broken)
        gone = json.loads(json.dumps(app.plan))
        del gone[field]
        assert any(field in problem for problem in app.check_plan(gone)), field


@check
def a_turn_rule_changing_is_invisible_to_the_flip_rate() -> None:
    """Worth knowing before either name is changed in a live plan. A Replay scores recorded
    answers; it never re-runs the turn loop, so neither of these names reaches it and the
    number a Setter reads at approval time cannot see them move. Same as `noise_floor`:
    adopt them on the argument, not on a flip rate that would report zero either way."""
    reaches = reaches_in("app.py")

    loop = {"narrow", "tie_break", "next_turn", "trusted_choice", "choose_widget"}
    for entry in ("replay", "regression", "rescore"):
        assert not reaches(entry) & loop, \
            f"{entry} now reaches {sorted(reaches(entry) & loop)}, so this note is stale"

@check
def the_interview_and_a_replay_do_the_same_arithmetic() -> None:
    """Two paths produce a `total`: `score_report` for a live interview, and `replay` for
    the Golden set. A bar is fitted against replayed totals and then spent on live ones, so
    if the two ever disagree the flip rate measures a number no interview produces.

    They share `combine_dimensions` and `plan_total` and must keep sharing them: this
    fails if either path grows arithmetic of its own.
    """
    qids = list(app.QUESTIONS)
    marks = {qid: {"score": 1.0 + (index % 3) * 0.5, "confidence": 0.9}
             for index, qid in enumerate(qids)}
    state = fresh() | {"asked": qids,
                       "answers": {qid: f"answer to {qid}" for qid in qids}}
    live = app.score_report(state, {qid: dict(mark, level=round(mark["score"]), gap=0.9)
                                    for qid, mark in marks.items()})

    record = {"answers": [{"text": app.QUESTIONS[qid]["text"],
                           "answer": state["answers"][qid]} for qid in qids]}
    held = app.ask_jev_repeatedly
    try:
        app.ask_jev_repeatedly = lambda where, asked: dict(marks) | {
            app.FIT: {"score": 2.0, "confidence": 0.9, "probabilities": {"2": 1.0}}}
        again = app.replay(record, app.plan, app.BRIEF)
    finally:
        app.ask_jev_repeatedly = held

    assert again["carried"] == len(qids), (again["carried"], len(qids))
    assert again["total"] == live["total"], \
        f"replay totalled {again['total']}, the interview {live['total']}"


@check
def a_replay_matches_questions_by_text_not_by_id() -> None:
    """A regenerated plan renumbers everything; the question text is what survives."""
    kept, dropped = app.QUESTION_LIST[0], app.QUESTION_LIST[1]
    record = {"answers": [{"text": kept["text"], "answer": "answer one"},
                          {"text": dropped["text"], "answer": "answer two"},
                          {"text": "a question this plan no longer asks", "answer": "three"}],
              "total": 2.0, "fit": {"score": 2.0}}
    candidate = json.loads(json.dumps(app.plan))
    candidate["questions"] = [dict(kept, id="renumbered_1"), dict(dropped, id="renumbered_2")]

    calls = fake_jev(None, score=1.0, confidence=0.9)
    after = app.replay(record, candidate, app.BRIEF)

    asked = scorings(calls)[-1]["questions"]
    assert set(asked) == {"renumbered_1", "renumbered_2", app.FIT}, sorted(asked)
    assert asked["renumbered_1"]["instructions"]["candidate_answer"] == "answer one", \
        "the answer was matched to the wrong question"
    assert after["carried"] == 2 and after["recorded"] == 3, after
    assert after["total"] == 1.0 and after["fit"] == 1.0, after


@check
def a_replay_asks_the_same_question_as_the_interview_did() -> None:
    """If the wording drifted, approval would be comparing two different judgements."""
    state = fresh() | {"asked": [app.FIRST_QUESTION],
                       "answers": {app.FIRST_QUESTION: "answer"}}
    live = fake_jev(None)
    app.score_all(state)
    during = scorings(live)[-1]["questions"]

    record = {"answers": [{"text": text_of(app.FIRST_QUESTION), "answer": "answer"}],
              "total": 2.0, "fit": {"score": 2.0}}
    again = fake_jev(None)
    app.replay(record, app.plan, app.BRIEF)
    replayed = scorings(again)[-1]["questions"]

    assert during[app.FIRST_QUESTION] == replayed[app.FIRST_QUESTION], \
        "the interview and its replay asked differently worded questions"
    assert during[app.FIT]["instructions"] == replayed[app.FIT]["instructions"]


@check
def the_regression_counts_who_changes_sides() -> None:
    boundary = app.MEASURED.get("measurement_boundary", 1.5)
    assert app.side(boundary, boundary) == "pass", "the boundary itself must pass"
    assert app.side(boundary - 0.01, boundary) == "fail"
    assert app.side(None, boundary) == "?", "a failed replay must not read as a verdict"

    results = pathlib.Path(tempfile.mkdtemp())
    for name, total in (("a", boundary + 0.5), ("b", boundary - 0.5)):
        (results / f"{name}.json").write_text(json.dumps(
            {"answers": [{"text": text_of(app.FIRST_QUESTION), "answer": "answer"}],
             "total": total, "fit": {"score": 2.0}}), encoding="utf-8")

    fake_jev(None, score=0.0, confidence=0.9)          # every replay now scores 0
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        flips = app.regression(app.plan, app.BRIEF, results)
        empty = app.regression(app.plan, app.BRIEF, pathlib.Path(tempfile.mkdtemp()))
    finally:
        sys.stderr = real_stderr

    assert flips == 1, f"expected the passing one to flip, got {flips}"
    assert empty == 0, "an empty Golden set must not report flips"


@check
def the_notice_says_an_interview_ended_and_nowhere_near_what_it_scored() -> None:
    """A number in a chat message invites deciding from the chat."""
    sent: list[str] = []
    real_url, app.NOTIFY_URL, app.notify = app.NOTIFY_URL, "", REAL_NOTIFY
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        report = app.score_report(*scored((app.FIRST_QUESTION, 2.0, 0.9)),
                                  {"score": 1.0, "confidence": 0.2, "max": 3,
                                   "level": 1, "gap": 0.8, "says": "a level's sentence"})
        app.notify(report, "owner/repo/results/x.json")
        sent.append(sys.stderr.getvalue())
    finally:
        sys.stderr, app.NOTIFY_URL = real_stderr, real_url
        app.notify = lambda r, w: None

    line = sent[0]
    assert report["jti"] in line and "owner/repo/results/x.json" in line, line
    assert "1 questions" in line and "flag" in line, line
    for leak in (str(report["total"]), "2.0", "1.0", "0.2", "confidence",
                 "a level's sentence"):
        assert leak not in line, f"the notice leaks {leak!r}: {line}"


# ── language ──────────────────────────────────────────────────────────────
@check
def ui_text_follows_the_brief_language() -> None:
    assert set(app.ALL_TEXT["en"]) == set(app.ALL_TEXT["ja"]), "the dictionaries have different keys"
    assert app.ALL_TEXT.get("fr", app.ALL_TEXT["en"]) is app.ALL_TEXT["en"], \
        "an unknown language must fall back to English"
    assert app.LANG in app.LANGUAGE_CHOICES, app.LANG


@check
def model_instructions_carry_no_korean() -> None:
    state = fresh() | {"asked": [app.FIRST_QUESTION],
                       "answers": {app.FIRST_QUESTION: "answer"}}
    assert not KOREAN.search(" ".join(app.interview_state(state))), "a state key is in Korean"
    turn = app.turn_questions([BY_DIMENSION[DIMENSIONS[1]][0]], app.QUESTIONS, True)
    for qid, question in turn.items():
        instructions = question["instructions"]
        prompt = instructions if isinstance(instructions, str) else instructions["question"]
        assert not KOREAN.search(prompt), f"{qid}: instructions are in Korean"
    for criteria in (app.WIDGETS, app.OPTION_WIDGETS, app.LANGUAGE_CHOICES,
                     app.turn_questions([], app.QUESTIONS, True)["done"]["criteria"]):
        assert not KOREAN.search(json.dumps(criteria, ensure_ascii=False)), criteria


@check
def payloads_carry_no_korean() -> None:
    last_dimension = BY_DIMENSION[DIMENSIONS[-1]][0]
    fake_jev(last_dimension)
    _, mid = answer(fresh(), app.FIRST_QUESTION, "answer")

    # Finishing needs a turn where `done` is asked at all, so: every dimension paid but
    # this one, and this the last question left.
    rest = [qid for qid in app.QUESTIONS if qid != last_dimension]
    full = fresh(last_dimension) | {"asked": rest, "answers": {q: "answer" for q in rest}}
    fake_jev(None, done=0.99)
    _, final = answer(full, last_dimension, "answer")

    assert final.get("finished"), "a response without `next` must finish, not raise"
    for name, payload in (("mid-interview", mid), ("finish", final)):
        leaked = KOREAN.findall(json.dumps(payload, ensure_ascii=False))
        assert not leaked, f"{name} payload leaks Korean: {''.join(sorted(set(leaked)))}"


@check
def the_server_escapes_every_hole_it_draws() -> None:
    """The markup moved to the server, so the escaping check moved with it.

    Every hole is a plan's text or a candidate's own words. This feeds each widget
    something that would break out if it were not escaped, rather than counting calls,
    so a new widget that forgets `esc` fails here without the test being updated.
    """
    nasty = '<script>alert("x")</script>'
    for ui in app.WIDGETS:
        markup = app.widget_html(ui, [nasty, "b"], nasty)
        assert "<script>" not in markup, f"{ui} renders unescaped markup"
        assert "&lt;script&gt;" in markup, f"{ui} drops the text instead of escaping it"

    drawn = app.turn_html({"question": {"title": nasty, "choices": [nasty]},
                           "dim": nasty, "num": 1, "max": 8,
                           "state": nasty, "id": nasty, "ui": "textarea"})
    assert "<script>" not in drawn, "the turn renders unescaped markup"
    for page in (app.apply_html(nasty, nasty), app.closing_html(nasty)):
        assert "<script>" not in page, "a page renders unescaped markup"


@check
def every_widget_draws_its_own_control() -> None:
    """`radio` once fell through to the default and rendered an invisible text box.

    Escaping checks did not catch it, because a text box escapes its label just fine.
    The vocabulary is compared as a set, so a new name cannot join it untested.
    """
    expected = {"text": 'type="text"', "textarea": "<textarea", "date": 'type="date"',
                "number": 'type="number"', "boolean": 'type="radio"',
                "rating": 'type="radio"', "radio": 'type="radio"',
                "checkbox": 'type="checkbox"', "ranking": 'type="number"',
                "select": "<select"}
    assert set(expected) == set(app.WIDGETS), "the vocabulary changed without this test"
    options = ["one", "two"]
    for ui, marker in expected.items():
        takes = app.WIDGETS[ui].get("opt")
        markup = app.widget_html(ui, options if takes else None, ui)
        assert marker in markup, f"{ui} did not draw its own control"
        if takes:
            for option in options:
                assert option in markup, f"{ui} dropped one of its options"


@check
def the_candidate_is_never_shown_the_words_for_a_mark() -> None:
    """Not the mark, and not the vocabulary either — the page must not know them (Q61)."""
    for word in ("result", "weight", "conf", "saved", "human_review",
                 "rail", "record", "covered"):
        assert word not in app.CLIENT_STRINGS, f"{word} belongs to the report, not the candidate"
    assert set(app.CLIENT_STRINGS) <= set(app.TEXT), "a client string has no translation"

    state = app.new_state("jti-test")
    state["pending"] = app.FIRST_QUESTION
    drawn = app.turn_html(app.client_payload(state, app.question_payload(state, app.FIRST_QUESTION)))
    for kind in ("rail", "record", "covered", "not_scored", "human_review", "tied_levels"):
        leaked = max(app.TEXT[kind].split("%s"), key=len).strip()
        assert leaked and leaked not in drawn, f"the turn shows the candidate {kind!r}"


# ── Jev client ────────────────────────────────────────────────────────────
class FakeResponse:
    def __init__(self, status: int, body: bytes = b'{"answers":{"ok":1}}'):
        self.status, self._body = status, body

    def read(self) -> bytes:
        return self._body


class FakeConnection:
    def __init__(self, statuses: list[int]):
        self.statuses, self.calls = list(statuses), 0

    def request(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1

    def getresponse(self) -> FakeResponse:
        return FakeResponse(self.statuses.pop(0))

    def close(self) -> None:
        pass


def call_jev(statuses: list[int]) -> tuple[Any, int, list[float]]:
    slept: list[float] = []
    connection = FakeConnection(statuses)
    app._connection = connection
    real_sleep, app.time.sleep = app.time.sleep, slept.append
    real_stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        return REAL_ASK_JEV("state", {}), connection.calls, slept
    finally:
        app.time.sleep, sys.stderr, app._connection = real_sleep, real_stderr, None


@check
def rate_limits_are_retried_with_backoff() -> None:
    backoff = app.JEV_BACKOFF
    assert call_jev([200]) == ({"ok": 1}, 1, []), "a success must not retry"
    assert call_jev([429, 200])[:2] == ({"ok": 1}, 2), "429 must retry"
    assert call_jev([429, 200])[2] == [backoff], "the first wait is one backoff"
    assert call_jev([529, 429, 200])[:2] == ({"ok": 1}, 3), "529 must retry too"
    assert call_jev([429, 429, 429]) == (None, 3, [backoff, backoff * 2]), \
        "attempts run out after exponential waits"
    assert call_jev([401, 200])[:2] == (None, 1), "401 must not retry"


def main() -> int:
    failed = []
    for check in CHECKS:
        saved.clear()
        app.ask_jev = REAL_ASK_JEV
        app.save_report = lambda report: saved.append(report) or "somewhere#1"
        app.notify = lambda report, where: None
        # No network in here: which judge it is comes off the wire in real use.
        app.judge_version = lambda: {"model": "jev-test", "released": "2026-09-10"}
        try:
            check()
        except AssertionError as err:
            failed.append(check.__name__)
            print(f"FAIL  {check.__name__}\n      {err}", file=sys.stderr)
        except Exception as err:
            failed.append(check.__name__)
            print(f"ERROR {check.__name__}\n      {type(err).__name__}: {err}", file=sys.stderr)
    print("ok" if not failed else f"{len(failed)} of {len(CHECKS)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
