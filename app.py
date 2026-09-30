"""Adaptive interview server.

    python3 app.py                serve on PORT (default 8000)
    python3 app.py --check <plan> validate a plan against the bounds and vocabulary
    python3 app.py --check-repo   prove DATA_REPO and GITHUB_TOKEN can keep a record
    python3 app.py --report [file] read a finished interview (default: the latest)
    python3 app.py --rescore [file] judge a finished interview again (default: the unscored)
    python3 app.py --noise [n]    measure the noise floor: same turn n times (default 60)

Domain words live in interface.json and in the Plan, not here. An interview is a Brief
(Markdown) and a Plan (JSON) derived from it, and neither is in this repository:
PLAN=<path.json> names the one this deployment runs, and unset is an error. The Plan
owns policy — question counts, thresholds, weights, how each dimension aggregates, how
many answers each dimension is owed. Code owns only the arithmetic.
"""
from __future__ import annotations

import base64
import datetime
import functools
import hashlib
import hmac
import http.client
import json
import os
import secrets
import ssl
import statistics
import sys
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit
from typing import Any

from core import (widget_choices, widget_problems, question_shape, same_words, BOUNDS, TIE_BREAKS, COVERAGE, Json, INTERFACE, WIDGETS, OPTION_WIDGETS, VOCABULARY, NOISE_KINDS,
    LANGUAGE_CHOICES, check_plan, supplement_schema, plan_schema, TOTALS,
    FIT, score_questions, combine_dimensions, plan_total, average_answers, gap, landed_on,
    noise_floor, floor_from, side, best_boundary, CALIBRATION_MIN, isotonic, fitted,
    calibrate, accuracy)


# ── environment ───────────────────────────────────────────────────────────
env_file = Path(".env")
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

OPENROUTER_HOST, OPENROUTER_PATH = "openrouter.ai", "/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
PLAN_MAX_TOKENS = int(os.environ.get("PLAN_MAX_TOKENS", 16000))

GITHUB_HOST = "api.github.com"
DATA_REPO = os.environ.get("DATA_REPO", "")        # "owner/name" — the operator's, never this one
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
# How many records this process copies. Results are named by time so these are the
# newest; candidate ids are random, so there it is 200 arbitrary ones — which only
# matters past 200, where the deeper history is the operator's clone's job.
FETCH_MAX = int(os.environ.get("FETCH_MAX", 200))

JEV_HOST, JEV_PATH = "api.typesafe.ai", "/v1/systemone"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
JEV_RETRY_STATUS, JEV_ATTEMPTS, JEV_BACKOFF = (429, 529), 3, 0.4
# No default: the Brief and the Plan are the operator's, and this repository keeps
# neither.
PLAN_PATH = Path(os.environ.get("PLAN", ""))
RESULTS_DIR = Path(os.environ.get("RESULTS", "results"))
CANDIDATES_DIR = Path(os.environ.get("CANDIDATES", "candidates"))
NOTIFY_URL = os.environ.get("NOTIFY_URL", "")


# ── one outbound request ──────────────────────────────────────────────────
def https(host: str, method: str, path: str, body: bytes | None = None,
          headers: Json | None = None, timeout: int = 20) -> tuple[int, bytes]:
    """One request over one connection, which is closed afterwards.

    A transport failure is `(0, b"")`, so a caller reads a status either way. Jev is the
    one caller that does not come through here: it holds its connection open across turns
    and retries, and both are the point of it.
    """
    connection = http.client.HTTPSConnection(host, timeout=timeout,
                                             context=ssl.create_default_context())
    try:
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    except (http.client.HTTPException, OSError) as err:
        print(f"{host} unreachable: {err}", file=sys.stderr)
        return 0, b""
    finally:
        connection.close()


# ── reading the operator's repository ────────────────────────────────────
# Up here because the Brief and the Plan come through it, before anything reads them.
GITHUB_HEADERS = {"Accept": "application/vnd.github+json",
                  "User-Agent": "adaptive-interview"}


def github_call(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    return https(GITHUB_HOST, method, f"/repos/{DATA_REPO}/contents/{path}", body,
                 GITHUB_HEADERS | {"Authorization": f"Bearer {GITHUB_TOKEN}",
                                   "Content-Type": "application/json"})


def github_get(path: str) -> Json | list[Json] | None:
    """One file, or one directory listing, out of the operator's repository."""
    status, payload = github_call("GET", f"{path}?ref={GITHUB_BRANCH}")
    if status == 200:
        try:
            return json.loads(payload)
        except ValueError:
            pass
    elif status not in (0, 404):
        print(f"github {status} reading {path}: {payload[:200]!r}", file=sys.stderr)
    return None


def repo_file(path: str) -> bytes | None:
    entry = github_get(path)
    return (base64.b64decode(entry["content"])
            if isinstance(entry, dict) and entry.get("content") else None)


# ── the rest of interface.json: strings, prompts, the measured numbers ────
# The vocabulary and the bounds a plan is checked against live in core.py, which holds no
# Plan and so cannot be talked into bending them. What is left here is what the server
# renders and what it pays for: one cost knob, and one figure measured against the
# deployed Jev version.
MEASURED: Json = {k: v for k, v in INTERFACE["measured"].items() if not k.startswith("_")}
INTAKE: Json = INTERFACE["intake"]
TURN: Json = INTERFACE["turn"]
GENERATION: Json = INTERFACE["generation"]
ADMIN_ALL: Json = {k: v for k, v in INTERFACE["admin"].items() if not k.startswith("_")}
ALL_TEXT: dict[str, dict[str, str]] = INTERFACE["strings"]
# Only what the shell renders. The report and flag strings stay server-side, so the
# candidate's page never even learns the words for the marks it must not see (Q61).
CLIENT_STRINGS = ("title", "meta", "answer", "required", "start", "finished",
                  "sendFailed", "apply", "applyHint", "applyEmpty", "applySent")


# ── the Plan this deployment runs ─────────────────────────────────────────
# The Brief and the Plan are the operator's, so they come out of the operator's
# repository and not out of whatever happens to have been deployed. Only the server
# does this: a CLI stands in a clone and reads the files in front of it.
def deployed_plan(from_repo: bool) -> tuple[bytes, str]:
    """The Plan this deployment is to run, and where it was taken from."""
    if not PLAN_PATH.name:
        raise SystemExit(
            "PLAN is unset, and there is no plan in this repository to fall back on.\n"
            "  Write your brief in a repository of your own, then from this checkout:\n"
            "    python3 app.py --plan ../your-records/brief.md"
            " --out ../your-records/plans/interview.json\n"
            "  Read what it wrote, commit it — that commit is the approval — and point\n"
            "  PLAN at it.")
    if not from_repo:
        # The same refusal the repository branch gives. Without it, pointing PLAN at a name
        # that is not here — which is what `--plan` tells you to do for your own — comes out
        # as a traceback from the middle of boot.
        if not PLAN_PATH.is_file():
            raise SystemExit(
                f"no {PLAN_PATH} in this checkout — nothing to interview anyone with.\n"
                "  PLAN names the plan to run, relative to DATA_REPO when that is set and to\n"
                "  this directory when it is not.")
        return PLAN_PATH.read_bytes(), "this checkout"
    written = repo_file(str(PLAN_PATH))
    if written is None:
        raise SystemExit(
            f"no {PLAN_PATH} in {DATA_REPO} — nothing to interview anyone with.\n"
            f"  Write {DATA_REPO}/brief.md, run `python3 app.py --plan brief.md`,\n"
            "  read what it wrote, and commit it. That commit is the approval.\n"
            "  `python3 app.py --check-repo` says whether this end can reach that one.")
    return written, DATA_REPO


# Empty until boot() fills them. Importing this module reads no plan, opens no socket
# and exits nobody: what the deployment runs is a decision its caller makes out loud.
plan: Json = {}
plan_source = ""
QUESTION_LIST: list[Json] = []
QUESTIONS: dict[str, Json] = {}
FIRST_QUESTION = ""
BRIEF = ""
# Policy is written into the plan, never decided here.
POLICY: Json = {}
# Weight and aggregate are kept together in the Plan so they cannot disagree.
# Only the weights are cached here; how a dimension combines is read from the Plan
# at the point of use, so there is no second copy to fall out of date.
WEIGHTS: dict[str, float] = {}
LANG = "en"
TEXT: dict[str, str] = {}
ADMIN_TEXT: Json = {}
CLIENT_TEXT: dict[str, str] = {}
SHELL = b""
ADMIN_SHELL = b""


def boot(from_repo: bool = False) -> None:
    """Read the Plan and the Brief and hold them. Every start-up read is in here, so
    there is exactly one moment this program learns what it is interviewing for."""
    global plan, plan_source, QUESTION_LIST, QUESTIONS, FIRST_QUESTION, BRIEF
    global POLICY, WEIGHTS, LANG, TEXT, ADMIN_TEXT, CLIENT_TEXT, SHELL, ADMIN_SHELL
    plan_bytes, plan_source = deployed_plan(from_repo)
    plan = json.loads(plan_bytes)
    if problems := check_plan(plan):
        raise SystemExit(f"{PLAN_PATH} from {plan_source} is not a usable plan:\n  "
                         + "\n  ".join(problems))

    QUESTION_LIST = plan["questions"]
    QUESTIONS = {q["id"]: q for q in QUESTION_LIST}
    FIRST_QUESTION = QUESTION_LIST[0]["id"]
    # `from` is repo-relative: the Brief is the Setter's input, not part of the
    # generated Plan, so it does not live beside it. Off disk the root it is relative
    # to can only come from the plan's own path: resolving it against the working
    # directory is what let this checkout's brief stand in for the operator's.
    near = (PLAN_PATH.parent / plan["from"], PLAN_PATH.parent.parent / plan["from"])
    found = next((path for path in near if path.is_file()), None)
    brief_bytes = repo_file(plan["from"]) if from_repo else found and found.read_bytes()
    if brief_bytes is None:
        where = DATA_REPO if from_repo else " or ".join(str(path) for path in near)
        raise SystemExit(f"this plan is written from {plan['from']},"
                         f" and it is not in {where}")
    BRIEF = brief_bytes.decode("utf-8")
    # Nothing ever checked this, and the Brief is the one file a person edits by hand: a
    # Plan written from an older one still runs, and asks about a job that has moved.
    if (fresh := hashlib.sha256(brief_bytes).hexdigest()) != (
            recorded := plan.get("source", {}).get("brief_sha256", fresh)):
        print(f"\n  {plan['from']} has changed since the plan was written from it."
              f"\n    the plan was written from {recorded[:12]}; this is {fresh[:12]}"
              f"\n  Every question is the old brief's. Regenerate with"
              f" `--plan {plan['from']}`, read it, commit it.\n", file=sys.stderr)

    POLICY = plan["policy"]
    WEIGHTS = {d["name"]: d["weight"] for d in plan["dimensions"]}
    LANG = plan.get("lang", "en")
    TEXT = ALL_TEXT.get(LANG, ALL_TEXT["en"])
    ADMIN_TEXT = ADMIN_ALL.get(LANG, ADMIN_ALL["en"])
    CLIENT_TEXT = {key: TEXT[key] for key in CLIENT_STRINGS}
    SHELL = Path("index.html").read_bytes()
    ADMIN_SHELL = Path("admin.html").read_bytes()


# ── secrets: deployment config, not interview policy, so not in the Plan ──
# ponytail: a random secret per boot invalidates links on restart. Set SECRET to keep them.
SECRET = (os.environ.get("SECRET") or secrets.token_urlsafe(32)).encode()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
LINK_TTL_SECONDS = int(os.environ.get("LINK_TTL", 60 * 60 * 24))


def sign(payload: Json) -> str:
    """The server keeps nothing between requests; the browser carries it signed."""
    raw = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()).rstrip(b"=")
    mac = base64.urlsafe_b64encode(hmac.new(SECRET, raw, hashlib.sha256).digest()).rstrip(b"=")
    return f"{raw.decode()}.{mac.decode()}"


def unsign(token: object) -> Json | None:
    if not isinstance(token, str) or token.count(".") != 1:
        return None
    raw, _, mac = token.partition(".")
    expected = base64.urlsafe_b64encode(
        hmac.new(SECRET, raw.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        return json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    except (ValueError, UnicodeDecodeError):
        return None


def now() -> int:
    return int(time.time())


# ── telling an open Interviewer page that something finished ──────────────
# A relay, not a store: each watcher is one queue that lives as long as
# its connection. Nothing survives a restart, and nothing needs to — the record is
# written before anyone is told, and the page reloads its list on reconnect.
WATCHERS: set[queue.SimpleQueue[str]] = set()
WATCHERS_LOCK = threading.Lock()
SESSION_COOKIE = "interviewer"
SESSION_TTL_SECONDS = 60 * 60 * 8
HEARTBEAT_SECONDS = 25


def consider(candidate: str, resume: str) -> None:
    """Turn one applicant's account into questions, then forget the account.

    Runs off the request thread: writing a question set takes about two minutes and
    nobody should sit through it.
    """
    questions = write_supplement(resume, os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL))
    del resume
    # A failure has to be visible: the applicant has already been thanked and left,
    # so if nobody records this, nobody ever finds out they are waiting on nothing.
    save_candidate({"id": candidate, "at": stamp(),
                    "approved": False, "failed": questions is None,
                    "plan": plan.get("source", {}).get("brief_sha256", ""),
                    "questions": questions or []}, f"Applicant {candidate} applied")
    announce(candidate, "applied")
    line = (f"Applicant {candidate} is waiting: "
            + ("no questions could be written for them" if questions is None
               else f"{len(questions)} question(s) written for them")
            + ", none asked yet. Approve at /admin")
    print(line, file=sys.stderr)
    post_notice(line)


def announce(jti: str, kind: str = "finished") -> None:
    with WATCHERS_LOCK:
        for watcher in list(WATCHERS):
            watcher.put(f"event: {kind}\ndata: {json.dumps({'jti': jti})}\n\n")


RESUME_MAX_CHARS = 20_000  # Jev takes 32k tokens of state; the Brief and answers share it.


def new_state(jti: str, candidate: str = "") -> Json:
    """No account of themselves: the applicant gave one once, and what it was for —
    the questions written from it — a person has already read and approved.

    Their questions are copied in here rather than read back from their record on every
    turn. The interview then cannot change under them while they are answering it, and
    everything that reads what is being asked becomes a function of its arguments and the
    Plan, with no file underneath. Reading one file is what building a state is for.
    """
    record = load_candidate(candidate)
    return {"asked": [], "answers": {}, "jti": jti, "candidate": candidate,
            "extra": record["questions"] if record and record.get("approved") else [],
            "pending": FIRST_QUESTION}


# ── one applicant's own questions ─────────────────────────────────────────
def candidate_path(candidate: str) -> Path:
    return CANDIDATES_DIR / f"{Path(candidate).name}.json"


def save_candidate(record: Json, message: str) -> None:
    """Both events that change one — the questions being written, and a person
    approving them — are commits, which is what gives the approval its trail."""
    path = candidate_path(record["id"])
    keep(path, f"candidates/{path.name}",
         json.dumps(record, ensure_ascii=False, indent=1).encode(), message)


def load_candidate(candidate: str) -> Json | None:
    path = candidate_path(candidate)
    if not candidate or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def questions_for(state: Json) -> dict[str, Json]:
    """The plan's questions, plus the ones written for this applicant — as they stood when
    their interview began.

    Theirs are marked so the report can show them and still leave them out of a dimension's
    score: the totals have to be the same ruler for everyone. Nothing is read from disk here,
    which is what makes every caller a function of its arguments and the Plan.
    """
    return QUESTIONS | {q["id"]: q | {"extra": True} for q in state.get("extra") or []}


# ── the Golden set: the only thing that catches a plan getting worse ──────
def replay(record: Json, candidate: Json, brief_text: str) -> Json | None:
    """Score one committed interview's answers against a candidate plan.

    Questions are matched by their text, not their id: a regenerated plan renumbers
    everything, and what a Replay can honestly compare is the answers that still have
    a question to be scored against — plus Fit, which reads the whole interview and
    therefore carries over whatever the questions did.
    """
    answers = {row["text"]: row["answer"] for row in record["answers"]}
    questions = score_questions(candidate["questions"], answers, candidate["fit"]["rubric"])
    judged = ask_jev_repeatedly({"job_context": brief_text, "answers_so_far": answers},
                                questions)
    if judged is None:
        return None
    by_id = {q["id"]: q for q in candidate["questions"]}
    rows = [{"qid": qid, "dimension": by_id[qid]["dimension"],
             "score": judged[qid]["score"], "confidence": judged[qid]["confidence"]}
            for qid in judged if qid in by_id]
    combined, flags = combine_dimensions(rows, candidate["dimensions"])
    return {"total": plan_total(combined, candidate["dimensions"], candidate["total"]),
            "fit": round(judged[FIT]["score"], 2) if FIT in judged else None,
            # A dimension no carried question covers is not a low score on it: the total
            # renormalises over whatever is left, and that is a number on another ruler.
            "uncovered": [flag["dimension"] for flag in flags],
            "carried": len(rows), "recorded": len(record["answers"])}


DECIDED = {"hired": "pass", "not_hired": "fail"}


def regression(candidate: Json, brief_text: str, results: Path) -> int:
    """What a Setter reads before approving: how many past people change sides.

    Where a record carries an Outcome, the side it is compared against is the decision
    a person actually made rather than the earlier Plan's own total — the same count,
    measuring whether the Plan is right instead of whether it is unchanged.
    """
    records = sorted(results.glob("*.json")) if results.is_dir() else []
    if not records:
        print("\n  no Golden set yet — this plan is being approved unmeasured.",
              file=sys.stderr)
        return 0

    boundary = MEASURED.get("measurement_boundary", 1.5)
    flips, decided_by_a_person = 0, []
    print(f"\n  replaying {len(records)} past interview(s) against this plan"
          f" (boundary {boundary}):", file=sys.stderr)
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        after = replay(record, candidate, brief_text)
        if after is None:
            print(f"    {path.name}  could not be replayed", file=sys.stderr)
            continue
        decided = DECIDED.get((record.get("outcome") or {}).get("decision", ""))
        was, now = decided or side(record.get("total"), boundary), side(after["total"], boundary)
        # Comparable totals only. A total the other dimensions were renormalised over
        # would move the fitted bar by how much of the plan the replay lost, not by
        # anything the person who decided was looking at.
        if decided and after["total"] is not None and not after["uncovered"]:
            decided_by_a_person.append((after["total"], decided))
        flips += was != now
        mark = "  <-- CHANGES SIDE" if was != now else ""
        print(f"    {path.name}  {record.get('total')} -> {after['total']}"
              f"  fit {record.get('fit', {}).get('score')} -> {after['fit']}"
              f"  [{after['carried']}/{after['recorded']} carried"
              f"{', uncovered ' + ','.join(after['uncovered']) if after['uncovered'] else ''}]"
              f"  {was}{' (decided)' if decided else ''} -> {now}{mark}", file=sys.stderr)
    print(f"\n  {flips} of {len(records)} change sides"
          f" ({flips / len(records):.0%}).", file=sys.stderr)
    # Until enough records carry a decision, the boundary stays the midpoint of the
    # scale and the count above is self-consistency, not accuracy.
    print(f"  {len(decided_by_a_person)} of {len(records)} were measured against a"
          f" hiring decision.", file=sys.stderr)
    fit = calibrate(decided_by_a_person, boundary)
    if fit["n"]:
        low, high = fit["span"]
        print(f"  a boundary of {fit['bar']} disagrees with {fit['wrong']} of them;"
              f" {boundary} disagrees with {fit['at_current']}."
              f"\n  every cut above {low} and up to {high} does as well, which is the"
              f" bar's resolution on {fit['n']} decision(s).", file=sys.stderr)
        held_out = accuracy(decided_by_a_person, boundary)
        if held_out["rate"] is not None:
            print(f"  on decisions it did not see it is right {held_out['right']}"
                  f"/{held_out['n']} ({held_out['rate']}), against {held_out['chance']}"
                  f" for always answering the commoner way"
                  f" — {'ahead' if held_out['beats_chance'] else 'NOT ahead'} of it.",
                  file=sys.stderr)
        if fit["usable"]:
            print(f"  fitted: {fit['curve']}", file=sys.stderr)
        else:
            least, each = CALIBRATION_MIN
            print(f"  not enough to read a total off yet: {least} decisions and {each} of"
                  f" each kind, against {fit['n']} and {min(fit['hired'], fit['n'] - fit['hired'])}"
                  f" here. The shape is printed; the number is not.", file=sys.stderr)
    return flips


# ── writing a plan: the slow, expensive half, and it runs in the CLI ──────
def ask_openrouter(prompt: str, schema: Json, model: str) -> Json | None:
    """The only generative call in the system.

    It writes a Plan in the CLI, and one applicant's own questions on the server when
    they apply — which is before an interview exists, never inside one.
    """
    # A reasoning model left uncapped will think for tens of minutes before writing a
    # plan this size. Capped, it spends ~80 reasoning tokens and finishes in about two.
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": "low", "exclude": True},
        "max_tokens": PLAN_MAX_TOKENS,
        # This model has 33 providers and they are not equally fast. Writing a plan is
        # almost all output tokens, so route to the quickest one. It may also be a
        # differently quantised one — which costs nothing here, because what this call
        # produces is read and approved by a person before it is used.
        "provider": {"sort": "throughput"},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "plan", "strict": True, "schema": schema}},
    }).encode()
    status, payload = https(OPENROUTER_HOST, "POST", OPENROUTER_PATH, body, {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json"}, timeout=300)
    if status != 200:
        if status:                  # https() has already said so if it could not connect
            print(f"openrouter {status}: {payload[:400]!r}", file=sys.stderr)
        return None
    try:
        answer = json.loads(payload)
        choice = answer["choices"][0]
        usage = answer.get("usage", {})
        print(f"  {answer.get('model', model)}  {usage.get('completion_tokens')} tokens"
              f"  ${usage.get('cost')}  finish={choice.get('finish_reason')}", file=sys.stderr)
        written = choice["message"].get("content")
        if not written:
            print(f"  the model returned no content (finish={choice.get('finish_reason')});"
                  f" raise PLAN_MAX_TOKENS or pick another model", file=sys.stderr)
            return None
        return json.loads(written)
    except (KeyError, ValueError, IndexError) as err:
        print(f"openrouter failed: {err}", file=sys.stderr)
        return None


def intake_problems(brief_text: str) -> list[str]:
    """Jev screens the Brief before a writing model is paid to work from it (Q43)."""
    checks = INTAKE["checks"]
    answers = ask_jev({"document": brief_text}, {
        name: {"type": "noul", "criteria": criteria,
               "instructions": f"Judging by `document`: {criteria['true']}?"}
        for name, criteria in checks.items()})
    if answers is None:
        return ["could not reach Jev to screen the document"]
    return [f"{name} ({answers[name]['noul']:.2f}): {checks[name]['false']}"
            for name in checks
            if answers[name]["noul"] < INTAKE["threshold"]]


def write_plan(brief_path: Path, model: str, attempts: int = 2) -> Json | None:
    brief_text = brief_path.read_text(encoding="utf-8")
    if problems := intake_problems(brief_text):
        print("this document cannot carry a plan:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return None

    prompt = GENERATION["prompt"].format(brief=brief_text)
    schema = plan_schema()
    for attempt in range(attempts):
        written = ask_openrouter(prompt, schema, model)
        if written is None:
            continue
        candidate = {"from": brief_path.name, **written,
                     "policy": {**written["policy"], **MEASURED},
                     "provisional": sorted(MEASURED) + ["done_threshold", "gap_k"],
                     "source": {"model": model,
                                "brief_sha256": hashlib.sha256(
                                    brief_path.read_bytes()).hexdigest()}}
        if not (problems := check_plan(candidate)):
            return candidate
        print(f"  attempt {attempt + 1} failed validation:", file=sys.stderr)
        for problem in problems:
            print(f"    {problem}", file=sys.stderr)
    return None


def levels_in_plan() -> int:
    return len(QUESTION_LIST[0]["rubric"])


def check_supplement(written: Json) -> list[str]:
    """The same bar as a plan, for the part of one that this is."""
    errors: list[str] = []
    questions = written.get("questions") or []
    if not questions:
        return ["the model returned no questions"]
    levels, seen = levels_in_plan(), set()
    # Seeded with the plan's wording, not just theirs: an applicant's question worded like
    # one already in the plan is the same key, and the second answer wins.
    worded = {q["text"] for q in QUESTION_LIST}
    for question in questions:
        if question.get("options") == []:
            question["options"] = None
    for index, question in enumerate(questions):
        where, missing = question_shape(question, index)
        if missing:
            errors += missing
            continue
        errors += same_words(question, worded, where)
        if not question["id"].startswith("x_"):
            errors.append(f"{where}: an applicant's question id must start with x_")
        if question["id"] in seen or question["id"] in QUESTIONS:
            errors.append(f"{where}: id already in use")
        seen.add(question["id"])
        if question["dimension"] not in WEIGHTS:
            errors.append(f"{where}.dimension: {question['dimension']!r} is not a dimension"
                          f" of this plan {sorted(WEIGHTS)}")
        if len(question["rubric"]) != levels:
            errors.append(f"{where}.rubric: needs exactly {levels} levels to read on the"
                          f" same scale as the plan's")
        errors += widget_problems(question, where)
    return errors


def write_supplement(resume: str, model: str) -> list[Json] | None:
    """Read the applicant's account once, turn it into questions, and let it go.

    The account is never stored. What survives is the questions it
    provoked, which is what the interviewer approves anyway.
    """
    prompt = GENERATION["supplement"].format(
        brief=BRIEF, resume=resume, dimensions=", ".join(sorted(WEIGHTS)),
        levels=levels_in_plan(), most=BOUNDS["supplement"][1],
        asked="\n".join(f"- [{q['dimension']}] {q['text']}" for q in QUESTION_LIST))
    schema = supplement_schema(levels_in_plan())
    for attempt in range(2):
        written = ask_openrouter(prompt, schema, model)
        if written is None:
            continue
        if not (problems := check_supplement(written)):
            return written["questions"]
        print(f"  supplement attempt {attempt + 1} rejected:", file=sys.stderr)
        for problem in problems:
            print(f"    {problem}", file=sys.stderr)
    return None


# ── Jev client ────────────────────────────────────────────────────────────
# ponytail: one connection behind a lock. Connection pool when concurrent.
_connection: http.client.HTTPSConnection | None = None
_connection_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def judge_version() -> Json:
    """Which Jev judged this, recorded beside the marks it left.

    There is nothing to pin: the service offers `jev-latest` and `jev-preview`, and
    both are moving aliases. So the record carries the release date instead, which is
    what lets a Replay tell a new judge from the scatter of the same one.
    Read once per process, which is as often as it can usefully change here.
    """
    stamped = {"model": JEV_MODEL}
    if not os.environ.get("TYPESAFE_API_KEY"):
        return stamped
    status, payload = https(JEV_HOST, "GET", "/v1/models", timeout=10, headers={
        "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})
    if status:                      # https() has already said so if it could not connect
        try:
            for model in json.loads(payload)["models"]:
                if model.get("name") == JEV_MODEL:
                    return stamped | {"released": model.get("release_date", "")}
        except (KeyError, ValueError) as err:
            print(f"could not read which judge this is: {err}", file=sys.stderr)
    return stamped


def ask_jev(state: Any, questions: Json) -> Json | None:
    body = json.dumps({"state": state, "model": JEV_MODEL, "questions": questions}).encode()
    headers = {"Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}",
               "Content-Type": "application/json"}
    global _connection
    for attempt in range(JEV_ATTEMPTS):
        is_last = attempt == JEV_ATTEMPTS - 1
        try:
            with _connection_lock:
                if _connection is None:
                    _connection = http.client.HTTPSConnection(
                        JEV_HOST, timeout=20, context=ssl.create_default_context())
                _connection.request("POST", JEV_PATH, body, headers)
                response = _connection.getresponse()
                payload, status = response.read(), response.status
            if status == 200:
                return json.loads(payload)["answers"]
            if status in JEV_RETRY_STATUS and not is_last:
                time.sleep(JEV_BACKOFF * 2**attempt)
                continue
            print(f"jev {status}: {payload[:200]!r}", file=sys.stderr)
            return None
        except (http.client.HTTPException, OSError, KeyError, ValueError) as err:
            with _connection_lock:
                if _connection:
                    _connection.close()
                _connection = None
            if is_last:
                print(f"jev unreachable: {err}", file=sys.stderr)
                return None
    return None


def ask_jev_repeatedly(state: Any, questions: Json) -> Json | None:
    runs = [answer for answer in
            (ask_jev(state, questions) for _ in range(max(1, POLICY["samples"]))) if answer]
    if not runs:
        return None
    return {key: average_answers([run[key] for run in runs if key in run])
            for key in runs[0]}


# ── what Jev is asked, once per turn ──────────────────────────────────────
def answers_so_far(state: Json) -> dict[str, str]:
    asking = questions_for(state)
    return {asking[qid]["text"]: state["answers"][qid] for qid in state["asked"]}


def interview_state(state: Json) -> Json:
    """The brief and what has been said. Nothing about the applicant that was not
    said in this interview — a claim on paper must not lift a score."""
    return {"job_context": BRIEF, "answers_so_far": answers_so_far(state)}


def turn_questions(remaining: list[str], asking: dict[str, Json], decidable: bool) -> Json:
    """The two slots are the loop's structure, so they stay here. Their words do not:
    `done`'s criteria are the Plan's, the rest are `interface.json`'s.

    `done` is asked only on a turn where it could end the interview. Below `min_q`, or
    while a dimension is still owed an answer, `next_turn` discards it — and Jev is not
    paid to judge something nobody reads. Both bounds are the Plan's, so what
    is asked each turn follows the Plan too.
    """
    questions: Json = {}
    if decidable:
        questions["done"] = {"type": "noul", "instructions": TURN["done"],
                             "criteria": plan["done"]}
    if remaining:
        # ponytail: every candidate goes in, up to the 255-option Choice limit.
        # Past that, shortlist with BM25 and rerank.
        questions["next"] = {
            "type": "choice",
            "instructions": TURN["next"],
            # Only these three fields, named: the template is edited by hand, and a typo
            # in it would otherwise be a KeyError on every turn of a live interview.
            "criteria": {qid: TURN["option"].format(dimension=asking[qid]["dimension"],
                                                    text=asking[qid]["text"],
                                                    context=asking[qid]["context"])
                         for qid in remaining}}
    return questions


# ── scoring: Jev supplies the numbers, the plan picks how they combine ────


def score_all(state: Json) -> tuple[Json, Json | None]:
    """Every answer scored in one batched call, after the interview, never during it.

    Scores are not read by the loop, so keeping them out of the browser's
    state costs nothing and stops a candidate from reading their own marks.
    """
    if not state["asked"]:
        return {}, None
    asking = questions_for(state)
    questions = score_questions([asking[qid] for qid in state["asked"]],
                                answers_so_far(state), plan["fit"]["rubric"])
    answers = ask_jev_repeatedly(interview_state(state), questions)
    if answers is None:
        return {}, None
    scores = {}
    for qid in state["asked"]:
        if qid not in answers:
            continue
        level, lead = landed_on(answers[qid])
        scores[qid] = {"score": round(answers[qid]["score"], 2),
                       "confidence": round(answers[qid]["confidence"], 2),
                       "level": level, "gap": lead}
    fit = None
    if FIT in answers:
        level, lead = landed_on(answers[FIT])
        rubric = plan["fit"]["rubric"]
        fit = {"score": round(answers[FIT]["score"], 2),
               "confidence": round(answers[FIT]["confidence"], 2),
               "max": len(rubric) - 1, "level": level, "gap": lead,
               "says": rubric[level] if level is not None and level < len(rubric) else None}
    return scores, fit


def score_report(state: Json, scores: Json, fit: Json | None = None,
                 at: str | None = None) -> Json:
    """The record, from the answers and what Jev made of them.

    `at` is when the interview finished. It is an argument because a clock read in here would
    be the one thing in this function that answered differently to the same question, and
    because scoring an old interview again has to keep the time it actually ended.
    """
    asking = questions_for(state)
    rows, lost = [], []
    for qid in state["asked"]:
        if qid not in asking:
            # Not in the Plan, and not in this state's own copy of the applicant's. Without
            # the question there is no rubric to score against and no text to show, so the row
            # is dropped and the flag carries the id — the answer itself survives in the
            # signed state the browser holds. A 500 here would lose the whole report.
            lost.append(qid)
            continue
        judged = scores.get(qid, {"score": None, "confidence": None, "level": None, "gap": None})
        rubric = asking[qid]["rubric"]
        level = judged.get("level")
        rows.append({"qid": qid, "dimension": asking[qid]["dimension"],
                     "extra": bool(asking[qid].get("extra")),
                     # Which input they were actually given. It changes no number here, and
                     # recording it is how that stays checkable rather than asserted.
                     "ui": state.get("widgets", {}).get(qid, asking[qid]["ui"]),
                     "text": asking[qid]["text"], "answer": state["answers"][qid],
                     # The sentence the rubric puts at that level: what a reader agrees
                     # or disagrees with. The number is its lossy compression.
                     "says": rubric[level] if level is not None and level < len(rubric) else None,
                     **judged})

    dimensions, flags = combine_dimensions(rows, plan["dimensions"])
    flags += [{"kind": "no_question", "qid": qid} for qid in lost]
    # Two levels within the noise floor means the sentence shown could as easily have
    # been the other one. That, not `confidence`, is what makes a mark unreadable.
    tied = POLICY["gap_k"] * POLICY["noise_floor"]["score"]
    flags += [{"kind": "tied_levels", "qid": r["qid"]} for r in rows
              if r["gap"] is not None and r["gap"] < tied]
    if fit and fit["gap"] is not None and fit["gap"] < tied:
        flags.append({"kind": "tied_levels", "qid": "fit"})
    # Jev can be unreachable, and then every answer is unjudged. Saying that outright
    # matters more than any other flag: a reader must not take an outage for a zero.
    if not dimensions:
        flags = [{"kind": "not_scored"}] + flags
    flags.append({"kind": "human_review"})

    return {"context_file": plan["from"], "lang": LANG, "policy": POLICY,
            "jti": state.get("jti", "unknown"), "candidate": state.get("candidate", ""),
            "source": plan.get("source", {}),
            "aggregate": {d["name"]: d["aggregate"] for d in plan["dimensions"]},
            # How the total was made, beside how each dimension was: a record that does not
            # say cannot be read years later, and the method is the Plan's to choose now.
            "total_by": plan["total"],
            "finished_at": at or datetime.datetime.now(
                datetime.UTC).isoformat(timespec="seconds"),
            "answers": rows, "dimensions": dimensions, "weights": WEIGHTS, "fit": fit,
            "total": plan_total(dimensions, plan["dimensions"], plan["total"]),
            "scored": bool(dimensions),
            "max": len(QUESTIONS[FIRST_QUESTION]["rubric"]) - 1, "flags": flags}


# ── keeping a record: a commit to the operator's repository ─────────────
# The reading half of this client is above, where the Plan needs it.
def github_put(path: str, content: bytes, message: str, sha: str = "") -> str | None:
    """Create the path, or replace what is there.

    GitHub refuses to replace a file unless told which version is being replaced, and
    a record this program writes twice — a candidate on approval, a result when someone
    is hired — is always a replacement the second time. Asking for the sha only after
    it is refused keeps the common case at one request.
    """
    body = json.dumps({"message": message, "branch": GITHUB_BRANCH,
                       "content": base64.b64encode(content).decode(),
                       **({"sha": sha} if sha else {})}).encode()
    status, payload = github_call("PUT", path, body)
    if status in (200, 201):
        return f"{DATA_REPO}/{path}"
    if status in (409, 422) and not sha:
        existing = github_get(path)
        if isinstance(existing, dict) and existing.get("sha"):
            return github_put(path, content, message, existing["sha"])
    if status:
        print(f"github {status}: {payload[:200]!r}", file=sys.stderr)
    return None


def keep(local: Path, repo_path: str, content: bytes, message: str) -> str:
    """Write a record to the repository, and to the copy of it this process reads.

    The repository is where the record lives. The local file is what
    `/admin` lists and what a Replay globs, and it is filled from the repository at
    boot — writing both is what makes a restart cheap instead of forgetful.
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(content)
    if DATA_REPO and GITHUB_TOKEN:
        if committed := github_put(repo_path, content, message):
            return committed
        print(f"{repo_path} is on this server only and may not survive a restart",
              file=sys.stderr)
    return str(local)


def fill_from_repository() -> None:
    """Fill the working copy from the repository the records actually live in.

    Without this the Interviewer's list empties on every restart while the records sit
    safely in the repository the server wrote them to, and an approval a person made
    stops existing — `questions_for` then falls back to the shared Plan without saying
    so, and the applicant answers an interview nobody promised them.
    """
    if not (DATA_REPO and GITHUB_TOKEN):
        return
    for folder, local in (("results", RESULTS_DIR), ("candidates", CANDIDATES_DIR)):
        listing = github_get(folder)      # github_get says so if it is a real failure;
        if not isinstance(listing, list):  # a repository with nothing in it yet is not
            continue
        names = sorted((entry["name"] for entry in listing
                        if entry.get("type") == "file" and entry["name"].endswith(".json")),
                       reverse=True)[:FETCH_MAX]
        local.mkdir(parents=True, exist_ok=True)
        # ponytail: a name already here is left alone, so a record edited in the
        # repository does not reach a machine that already holds it — delete it to
        # refetch. On the host this is written for, the disk starts empty every time.
        fetched = 0
        for name in names:
            if (local / name).is_file():
                continue
            if content := repo_file(f"{folder}/{name}"):
                (local / name).write_bytes(content)
                fetched += 1
        print(f"{folder}: {fetched} fetched, {len(names)} of them in {DATA_REPO}",
              file=sys.stderr)


KEEP_PROBE = "results/.keep"


def check_repo() -> int:
    """Prove this deployment can keep a record, before there is one to lose.

    A token that cannot write fails the way a token that can looks: the interview
    runs, every answer is recorded here, and the line saying none of it was committed
    goes to a log nobody reads. So it is worth one command when the token is made, and
    one more on the day it quietly expires.
    """
    if not (DATA_REPO and GITHUB_TOKEN):
        missing = " and ".join(name for name, value in
                               (("DATA_REPO", DATA_REPO), ("GITHUB_TOKEN", GITHUB_TOKEN))
                               if not value)
        print(f"  {missing} unset — records are kept on this server only, which the"
              f" free\n  tiers wipe. README, 'For real: two repositories', says how.", file=sys.stderr)
        return 1

    print(f"  repository  {DATA_REPO}   branch {GITHUB_BRANCH}", file=sys.stderr)
    status, payload = github_call("GET", f"?ref={GITHUB_BRANCH}")
    trouble = {
        0: "could not reach api.github.com at all.",
        401: "the token was rejected — mistyped, or expired.",
        403: "the token is valid and not allowed here. Check it grants Contents, and"
             "\n              that SSO is authorised if an organisation owns this.",
        404: f"nothing this token can see at {DATA_REPO}. A private repository reads as"
             "\n              missing, so this is the wrong name, the wrong branch, or a"
             "\n              fine-grained token never given this repository.",
    }.get(status, f"github said {status}: {payload[:120]!r}")
    if status != 200:
        print(f"  read        FAILED — {trouble}", file=sys.stderr)
        return 1
    print("  read        ok", file=sys.stderr)

    # Twice, with different bytes each time: the second write is the one that has to
    # ask which version it is replacing, and that is the half nothing else exercises.
    note = "Finished interviews are committed here by the interview server.\n"
    first = github_put(KEEP_PROBE, f"{note}{stamp()}\n".encode(),
                       "Check this deployment can write here")
    second = github_put(KEEP_PROBE, f"{note}{stamp()} (again)\n".encode(),
                        "Check this deployment can replace what it wrote")
    if not (first and second):
        print("  write       FAILED — it reads but cannot write. A fine-grained token"
              "\n              needs Contents: Read and write; a classic one needs 'repo'.",
              file=sys.stderr)
        return 1
    print(f"  write       ok — {second}, twice, so a record can be replaced",
          file=sys.stderr)

    # A repository that cannot be written to keeps nothing, and one with no Brief
    # in it has nothing to ask anybody.
    if not PLAN_PATH.name:
        print("  plan        PLAN unset — nothing names the plan this deployment runs.",
              file=sys.stderr)
        return 1
    written = repo_file(str(PLAN_PATH))
    if written is None:
        print(f"  plan        not there — no {PLAN_PATH} in this repository yet."
              f"\n              Write brief.md, `--plan brief.md`, read it, commit it.",
              file=sys.stderr)
        return 1
    their_plan = json.loads(written)
    brief = repo_file(their_plan["from"])
    if brief is None:
        print(f"  brief       missing — the plan is written from {their_plan['from']},"
              f"\n              which is not in this repository.", file=sys.stderr)
        return 1
    recorded = their_plan.get("source", {}).get("brief_sha256", "")
    if recorded and recorded != hashlib.sha256(brief).hexdigest():
        print(f"  brief       {their_plan['from']} has moved on since the plan was"
              f" written\n              from it. The questions are the old one's.",
              file=sys.stderr)
        return 1
    print(f"  plan        ok — {len(their_plan['questions'])} questions from"
          f" {their_plan['from']}, and they match it", file=sys.stderr)
    print("\n  This deployment has something to ask, and will keep what it hears.",
          file=sys.stderr)
    return 0


def state_from(report: Json) -> Json:
    """Rebuild just enough of an interview to score it again.

    The other place a state is built, and so the other place one file may be read: the
    applicant's own questions are not in the record, only the answers to them.
    """
    candidate = report.get("candidate", "")
    record = load_candidate(candidate)
    return {"asked": [row["qid"] for row in report["answers"]],
            "answers": {row["qid"]: row["answer"] for row in report["answers"]},
            "jti": report.get("jti", "unknown"), "candidate": candidate,
            "extra": record["questions"] if record and record.get("approved") else []}


def rescore(path: Path) -> Json | None:
    """Score a finished interview again.

    Jev can be unreachable for a whole interview, and then every answer is recorded
    and none of them is judged. The answers are the part that cannot be recovered
    later; the marks can, so this recovers them.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    state = state_from(report)
    asking = questions_for(state)
    missing = [qid for qid in state["asked"] if qid not in asking]
    if missing:
        print(f"  {path.name}: the plan no longer has {missing} — skipping", file=sys.stderr)
        return None
    scores, fit = score_all(state)
    if not scores:
        print(f"  {path.name}: still nothing came back", file=sys.stderr)
        return None
    fresh = score_report(state, scores, fit, at=report["finished_at"]) | {
        "judge": judge_version(),
        "rescored_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    if "outcome" in report:
        fresh["outcome"] = report["outcome"]   # what a person filed outlives a re-score
    keep(path, f"results/{path.name}",
         json.dumps(fresh, ensure_ascii=False, indent=1).encode(),
         f"Scored again: {fresh.get('jti', path.name)}")
    return fresh


# How many averaged groups a reported floor has to rest on. At ten the spread still
# carries about a quarter relative error; it is a floor on nonsense, not a target.
NOISE_GROUPS = 10


def same_request(state: Json, questions: Json, repeats: int) -> list[Json]:
    """One byte-identical request, sent `repeats` times. Nothing here averages: the
    spread between these answers is the thing being measured."""
    runs = []
    for attempt in range(repeats):
        if answer := ask_jev(interview_state(state), questions):
            runs.append(answer)
        print(f"\r    {len(runs)}/{attempt + 1} answered", end="", file=sys.stderr)
    print(file=sys.stderr)
    return runs


# Above this the marks were pinned, and a spread measured there is not a floor.
SATURATED = 0.95


def nearest_the_gate() -> Path | None:
    """The record whose marks sit closest to the gate, which is where a floor has to be
    measured. The newest record is the wrong one to reach for: an interview Jev scored
    with total certainty has nothing to wobble, and measuring there reports a floor of
    roughly zero that would then trust every close mark the gate exists to catch. With no
    record close enough to measure on, there is nothing to measure, and the caller says so.
    """
    ranked = []
    for path in RESULTS_DIR.glob("*.json"):
        gaps = [row["gap"] for row in
                (json.loads(path.read_text(encoding="utf-8")).get("answers") or [])
                if row.get("gap") is not None]
        if len(gaps) >= 4:
            ranked.append((statistics.median(gaps), path.name, path))
    return min(ranked)[2] if ranked else None


def measure_noise(repeats: int) -> int:
    """Measure a noise floor for each kind of question Jev is asked, so the numbers in
    interface.json are measurements and not numbers somebody liked.

    Both come off real requests taken from the record whose marks sit nearest the gate —
    rewound to halfway for the Choice, whole for the Score — because a floor belongs to
    the deployed judge, to the shape of what it is asked, and to how close the call was.
    It writes nothing: a measurement is recorded by the person who read it, the same way
    a Plan is approved.
    """
    latest = nearest_the_gate()
    if latest is None:
        print("no record to measure against — run one interview first", file=sys.stderr)
        return 1
    whole = state_from(json.loads(latest.read_text(encoding="utf-8")))
    kept = whole["asked"][:len(whole["asked"]) // 2]
    half = whole | {"asked": kept, "answers": {q: whole["answers"][q] for q in kept}}
    asking = questions_for(whole)
    remaining = [qid for qid in asking if qid not in half["answers"]]
    if len(remaining) < 2 or not whole["asked"]:
        print(f"{latest.name} is too short to take either request from", file=sys.stderr)
        return 1

    judge = judge_version()
    print(f"  judge  {judge.get('model')} released {judge.get('released', 'unknown')}\n"
          f"  from   {latest.name}\n"
          f"  cost   {2 * repeats} calls, and nothing is written\n"
          f"  choice {len(remaining)} options, rewound to {len(kept)} answer(s)",
          file=sys.stderr)
    measured: Json = {"choice": floor_from(
        same_request(half, turn_questions(remaining, asking, decidable=False), repeats),
        ["next"], POLICY["samples"])}

    scored = score_questions([asking[qid] for qid in whole["asked"]],
                             answers_so_far(whole), plan["fit"]["rubric"])
    print(f"  score  {len(scored)} rubric(s) over {len(whole['asked'])} answer(s)",
          file=sys.stderr)
    measured["score"] = floor_from(same_request(whole, scored, repeats),
                                   sorted(scored), POLICY["samples"])

    held = MEASURED.get("noise_floor")
    for kind in NOISE_KINDS:
        row = measured[kind]
        mine = held.get(kind) if isinstance(held, dict) else held
        print(f"\n  {kind:6s} floor {row['floor']}   mean gap {row['mean_gap']}"
              f"   {row['groups']} group(s) x {len(row['slots'])} slot(s)"
              f"   (it holds {mine})", file=sys.stderr)
        if len(row["slots"]) > 1:
            print("         per slot " + "  ".join(
                f"{slot}={value}" for slot, value in row["slots"].items()), file=sys.stderr)

    pinned = [k for k in NOISE_KINDS
              if (measured[k]["mean_gap"] or 0) > SATURATED]
    if pinned:
        print(f"\n  {pinned}: mean gap over {SATURATED}, so these marks were pinned and"
              f" their spread is not a floor. Measure against a record with closer"
              f" calls in it.", file=sys.stderr)
        return 1
    thin = [k for k in NOISE_KINDS
            if measured[k]["floor"] is None or measured[k]["groups"] < NOISE_GROUPS]
    if thin:
        short = (NOISE_GROUPS - min(measured[k]["groups"] for k in thin)) \
            * max(1, POLICY["samples"])
        print(f"\n  {thin} rest on too few groups to call a floor."
              f" Ask for {short} more: `--noise {repeats + short}`.", file=sys.stderr)
        return 1

    floors = ", ".join(f'"{k}": {measured[k]["floor"]}' for k in NOISE_KINDS)
    print(f"\n  interface.json  \"noise_floor\": {{{floors}}}"
          f"\n  gates at gap_k x floor:  "
          + ",  ".join(f"{k} {round(POLICY['gap_k'] * measured[k]['floor'], 4)}"
                       for k in NOISE_KINDS)
          + "\n  Measured on the averaged basis, which is what reads them. Record them"
            " with the judge's release date: they are that judge's numbers.\n",
          file=sys.stderr)
    return 0


def save_report(report: Json) -> str:
    name = f"{report['finished_at'].replace(':', '')}-{report['jti']}.json"
    return keep(RESULTS_DIR / name, f"results/{name}",
                json.dumps(report, ensure_ascii=False, indent=1).encode(),
                f"Interview result {report['jti']}")


OUTCOMES = ("hired", "not_hired")


def record_outcome(name: str, decision: str) -> Json | None:
    """File what a person decided, onto a record that was written weeks ago.

    The only field in a Report a human writes. It is read when the bar every applicant
    is measured against is fitted, and never when one of them is scored.
    There is no `who`: this page has one password and no idea who is behind it.
    """
    path = RESULTS_DIR / Path(name).name          # nothing escapes RESULTS_DIR
    if decision not in OUTCOMES or not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    report["outcome"] = {"decision": decision, "at": stamp()}
    keep(path, f"results/{path.name}",
         json.dumps(report, ensure_ascii=False, indent=1).encode(),
         f"{decision}: {report.get('jti', path.name)}")
    return report


def notify(report: Json, where: str) -> None:  # noqa: D401 — see below
    """Say that an interview finished and where to read it — never what it scored.

    A number in a chat message invites deciding from the chat. The decision is made
    by a person reading the whole record, and a channel is usually wider
    than the one person entitled to read it.
    """
    announce(report["jti"])
    line = (f"Interview {report['jti']} finished {report['finished_at']}"
            f" — {len(report['answers'])} questions,"
            f" {len(report['flags'])} flag(s). Read it at {where}")
    print(line, file=sys.stderr)
    post_notice(line)


def post_notice(line: str) -> None:
    if not NOTIFY_URL:
        return
    url = urlsplit(NOTIFY_URL)
    status, body = https(url.netloc, "POST",
                         url.path + (f"?{url.query}" if url.query else ""),
                         json.dumps({"text": line}).encode(),
                         {"Content-Type": "application/json"}, timeout=10)
    if status >= 300:
        print(f"notify {status}: {body[:200]!r}", file=sys.stderr)


def print_report(path: Path) -> None:
    """Read a committed record the way the person deciding needs to see it."""
    report = json.loads(path.read_text(encoding="utf-8"))
    width = max((len(d) for d in report["dimensions"]), default=10)
    top = report["max"]
    print(f"\n  {path.name}   {report['finished_at']}   link {report['jti']}")
    print(f"  from {report['context_file']}  ·  plan by {report['source'].get('model', '?')}"
          f"  ·  {len(report['answers'])} questions\n")
    if filed := report.get("outcome"):
        print(f"  {filed['decision']}, recorded {filed['at']}\n")
    for name, value in report["dimensions"].items():
        filled = round(value["score"] / top * 24) if top else 0
        print(f"  {name:<{width}}  {value['score']:>5.2f}/{top}  "
              f"{'#' * filled}{'·' * (24 - filled)}  "
              f"{value['how']}, n={value['n']}, conf {value['confidence']:.2f}")
    total = report["total"]
    print(f"\n  {'total':<{width}}  "
          + (f"{total:>5.2f}/{top}   ({report.get('total_by', 'weighted_mean')})"
             if total is not None else "  --   not scored"))
    if fit := report.get("fit"):
        print(f"  {'fit':<{width}}  {fit['score']:>5.2f}/{fit['max']}"
              f"   level {fit.get('level')}  lead {fit.get('gap')}")
        # The number is a compression of this sentence; the sentence is what a person
        # can agree or disagree with, and disagreeing is the only label we will get.
        if fit.get("says"):
            print(f"  {'':<{width}}  {fit['says']}")
    print()
    for flag in report["flags"]:
        print(f"  ! {flag_text(flag)}")
    print("\n  answers")
    for row in report["answers"]:
        mark = "  --" if row["score"] is None else f"{row['score']:>5.2f}"
        lead = "" if row.get("gap") is None else f"  lead {row['gap']}"
        print(f"\n  [{row['dimension']}] {mark}{lead}")
        if row.get("says"):
            print(f"    = {row['says']}")
        print(f"    Q {row['text']}")
        print(f"    A {row['answer']}")


# ── payloads the browser renders ──────────────────────────────────────────
def widget_spec(options: list[str] | None, ui: str | None) -> tuple[str, Json]:
    name = ui if ui in WIDGETS else ("radio" if options else "text")
    spec = WIDGETS[name]["sj"] | {"name": "a", "isRequired": True}
    return name, (spec | {"choices": options} if options else spec)


def flag_text(flag: Json) -> str:
    subject = flag.get("dimension") or flag.get("qid")
    return TEXT[flag["kind"]] % subject if subject else TEXT[flag["kind"]]


def choose_widget(state: Json, question_id: str) -> str:
    """Which input to put in front of them — chosen now rather than when the Plan was
    written, and chosen once.

    Once, because a reload must not change the widget under somebody's hands, so the pick
    is kept in their state beside their answers and is what a record reports afterwards.

    It cannot reach the score. The rubric, the dimension and the arithmetic are identical
    whichever input collected the words, and the set Jev picks from is the Plan's own, so
    two candidates meet the same question with the same inputs available. That is the whole
    reason this is safe to decide at runtime while the ordering already was.
    """
    question = questions_for(state)[question_id]
    picked = state.setdefault("widgets", {})
    if question_id in picked:
        return picked[question_id]
    allowed = widget_choices(question)
    # Nothing said yet means nothing to judge, so the first question is the Plan's own —
    # the same reason `next` is not asked for it either.
    if len(allowed) < 2 or not state["asked"]:
        picked[question_id] = allowed[0] if allowed else question["ui"]
        return picked[question_id]
    # One call, not `samples` of them, and no gap gate. A tie here means Jev is torn between
    # two inputs the Plan already approved for this question, and then either will do — which
    # is not true of the `next` choice, and is why that one is averaged and gated.
    # The question's own words go in the instructions, never into the state: what rides along
    # to Jev stays the brief and what was said in this interview.
    said = ask_jev(interview_state(state),
                   {"widget": {"type": "choice",
                               "instructions": {"question_asked": question["text"],
                                                "options_offered": len(question["options"] or []),
                                                "question": TURN["widget"]},
                               "criteria": {name: WIDGETS[name]["when"] for name in allowed}}})
    chose = (said or {}).get("widget", {}).get("choice")
    # Unreachable, or an answer outside what was offered, leaves the Plan's own ui: the
    # worst case of asking is exactly what this did before anyone asked.
    picked[question_id] = chose if chose in allowed else question["ui"]
    return picked[question_id]


def question_payload(state: Json, question_id: str) -> Json:
    asking = questions_for(state)
    question = asking[question_id]
    name, spec = widget_spec(question["options"], choose_widget(state, question_id))
    return {"id": question_id, "question": spec | {"title": question["text"]}, "ui": name,
            "num": len(state["asked"]) + 1, "max": min(POLICY["max_q"], len(asking)),
            "dim": question["dimension"]}


def client_payload(state: Json, current: Json) -> Json:
    """What the candidate's browser is allowed to know: their own answers, never a mark."""
    asking = questions_for(state)
    return current | {
        "state": sign(state), "t": CLIENT_TEXT, "lang": LANG,
        "history": [{"q": asking[qid]["text"], "a": state["answers"][qid],
                     "dim": asking[qid]["dimension"]}
                    for qid in state["asked"]]}


def finish(state: Json) -> tuple[None, Json]:
    report = score_report(state, *score_all(state)) | {"judge": judge_version()}
    notify(report, save_report(report))
    return None, {"finished": True}


# ── the turn ──────────────────────────────────────────────────────────────
def scored_per_dimension(state: Json) -> dict[str, int]:
    """Only a question that can score into a dimension counts towards it.

    An applicant's own questions are deliberately left out of the arithmetic, so one
    of them standing in for coverage would leave that dimension
    resting on whatever single plan question happened to fit.
    """
    asking = questions_for(state)
    counts = {name: 0 for name in WEIGHTS}
    for qid in state["asked"]:
        question = asking.get(qid)
        if question and not question.get("extra") and question["dimension"] in counts:
            counts[question["dimension"]] += 1
    return counts


def under_measured(state: Json, want: int | None = None) -> list[str]:
    """Which dimensions are still owed answers. How many they are owed is a number the
    plan writes, not a name this file keeps a number behind."""
    if want is None:
        want = POLICY["per_dimension"]
    counts = scored_per_dimension(state)
    return [name for name in WEIGHTS if counts[name] < want]


def narrow(state: Json, remaining: list[str]) -> list[str]:
    """Hold turns back for whatever a dimension is still owed — or do not, if the Plan says
    not to. Which it wants is a fact about the Brief, so the Plan names it.

    `reserve_early` is how many turns of head start the debt gets under `reserve_debt`. At 0
    the judge picks freely until the budget only just covers what is owed — the last possible
    moment. At `max_q` every turn is reserved from the first, which is breadth before depth.
    """
    want = POLICY["per_dimension"]
    short = under_measured(state, want)
    counts = scored_per_dimension(state)
    return COVERAGE[plan["coverage"]](remaining, questions_for(state), {
        "short": short,
        "owed": sum(want - counts[name] for name in short),
        "turns_left": POLICY["max_q"] - len(state["asked"]),
        "head_start": POLICY["reserve_early"]})


def tie_break(state: Json, remaining: list[str]) -> str | None:
    """Jev says the options are level, so code settles it — there is no System Two to
    escalate to. Which way it settles is the Plan's to name: covering an
    untouched dimension and following the written order are both defensible, and which one
    a company wants is a fact about its Brief rather than about this file.
    """
    if not remaining:
        return None
    return TIE_BREAKS[plan["tie_break"]](remaining, questions_for(state),
                                         under_measured(state))


def trusted_choice(state: Json, remaining: list[str], answer: Json | None) -> str | None:
    if not answer or not answer.get("choice"):
        return None
    # Confidence is low whenever several questions are equally good, which is harmless.
    # The lead over the runner-up is what says whether the pick carries information.
    if gap(answer) < POLICY["gap_k"] * POLICY["noise_floor"]["choice"]:
        return tie_break(state, remaining)
    return answer["choice"]


def next_turn(state: Json) -> tuple[str | None, Json]:
    asking = questions_for(state)
    remaining = [qid for qid in asking if qid not in state["answers"]]
    remaining = narrow(state, remaining)
    asked_count = len(state["asked"])
    # Whether `done` can end the interview yet, decided from the plan alone.
    decidable = asked_count >= POLICY["min_q"] and not under_measured(state)

    questions = turn_questions(remaining, asking, decidable)
    if not questions:
        return finish(state)
    answers = ask_jev_repeatedly(interview_state(state), questions)

    if answers is None:
        if not remaining:
            return finish(state)
        return remaining[0], question_payload(state, remaining[0])

    next_id = trusted_choice(state, remaining, answers.get("next"))
    heard_enough = answers.get("done", {}).get("noul", 0) > POLICY["done_threshold"]

    if (not next_id
            or asked_count >= POLICY["max_q"]
            or (decidable and heard_enough)):
        return finish(state)
    return next_id, question_payload(state, next_id)


# ── HTML ──────────────────────────────────────────────────────────────────
# The browser is sent markup, not a rendering library. Every widget in the
# vocabulary except `ranking` is an input the browser already has, and the answer
# reaches `do_answer` as text either way, because a turn stores `", ".join(values)`.

def esc(value: object) -> str:
    """Everything interpolated into markup passes through here. No exceptions."""
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def choices(kind: str, options: list[str], label: str) -> str:
    items = "".join(
        f'<label class="pick"><input type="{kind}" name="a" value="{esc(o)}"'
        f'{" required" if kind == "radio" and i == 0 else ""}>'
        f'<i class="{kind}"></i><span>{esc(o)}</span></label>'
        for i, o in enumerate(options))
    return f'<fieldset class="picks" aria-label="{esc(label)}">{items}</fieldset>'


def line_html(kind: str, label: str) -> str:
    return f'<input class="line" type="{kind}" name="a" required aria-label="{esc(label)}">'


def rating_html(label: str) -> str:
    cells = "".join(
        f'<label class="cell"><input type="radio" name="a" value="{n}"'
        f'{" required" if n == 1 else ""}><span>{n}</span></label>' for n in range(1, 6))
    return f'<fieldset class="rating" aria-label="{esc(label)}">{cells}</fieldset>'


def ranking_html(options: list[str], label: str) -> str:
    # ponytail: a number per option instead of drag and drop. No plan has asked for
    # ranking yet; swap in a reorderable list if one ever does.
    rows = "".join(
        f'<label class="rank"><input type="number" name="rank" min="1" max="{len(options)}"'
        f' required value=""><span>{esc(option)}</span>'
        f'<input type="hidden" name="opt" value="{esc(option)}"></label>' for option in options)
    return (f'<fieldset class="ranks" aria-label="{esc(label)}">'
            f'<p class="hint">{esc(TEXT["order"])}</p>{rows}</fieldset>')


def select_html(options: list[str], label: str) -> str:
    picks = "".join(f'<option value="{esc(option)}">{esc(option)}</option>'
                    for option in options)
    # The empty first option is required and disabled, so the browser enforces a choice
    # rather than submitting whatever happened to be at the top of the list.
    return (f'<select class="line" name="a" required aria-label="{esc(label)}">'
            f'<option value="" selected disabled>{esc(TEXT["choose"])}</option>'
            f'{picks}</select>')


# One renderer per name in the vocabulary, so a name is what selects the function and
# nothing else does. The table is the dispatch: an unnamed widget is a KeyError here,
# where it used to fall through to a text box that looked deliberate.
WIDGET_HTML: dict[str, Any] = {
    "text": lambda options, label: line_html("text", label),
    "date": lambda options, label: line_html("date", label),
    "number": lambda options, label: line_html("number", label),
    "textarea": lambda options, label:
        f'<textarea name="a" rows="7" required aria-label="{esc(label)}"></textarea>',
    "boolean": lambda options, label: choices("radio", [TEXT["yes"], TEXT["no"]], label),
    "rating": lambda options, label: rating_html(label),
    "radio": lambda options, label: choices("radio", options or [], label),
    "select": lambda options, label: select_html(options or [], label),
    "checkbox": lambda options, label: choices("checkbox", options or [], label),
    "ranking": lambda options, label: ranking_html(options or [], label),
}


def widget_html(ui: str, options: list[str] | None, label: str) -> str:
    """The renderer the chosen name selects — nothing here decides which."""
    return WIDGET_HTML[ui](options, label)


def answer_values(fields: dict[str, list[str]], ui: str) -> list[str]:
    """What the form said, in the order the answer is meant to carry."""
    if ui != "ranking":
        return fields.get("a", [])
    places, options = fields.get("rank", []), fields.get("opt", [])
    ordered = sorted(zip(places, options),
                     key=lambda pair: int(pair[0]) if pair[0].isdigit() else len(options) + 1)
    return [option for place, option in ordered if place.strip()]


def turn_html(payload: Json) -> str:
    """One question, its form, and the signed state that carries the interview."""
    question, dim = payload["question"]["title"], payload["dim"]
    size = "l" if len(question) > 110 else "m" if len(question) > 55 else "s"
    done = payload["num"] / payload["max"] * 100
    return f"""<div id="turn" class="turn" data-len="{size}">
<header class="stage">
  <div class="prog" aria-hidden="true"><i style="--p:{done:.1f}%"></i></div>
  <div class="wrap">
    <p class="status"><b>{esc(TEXT["meta"])}</b><i aria-hidden="true"></i>
      <span class="dim">{esc(dim)}</span>
      <span class="count">{payload["num"]:02d}&nbsp;/&nbsp;{payload["max"]:02d}</span></p>
    <h1 class="q" tabindex="-1">{esc(question)}</h1>
  </div>
</header>
<main class="desk"><div class="wrap">
  <form hx-post="/turn" hx-target="#turn" hx-swap="outerHTML"
        hx-disabled-elt="find button" hx-sync="this:drop">
    <input type="hidden" name="state" value="{esc(payload["state"])}">
    <input type="hidden" name="id" value="{esc(payload["id"])}">
    <input type="hidden" name="ui" value="{esc(payload["ui"])}">
    <div class="field">
      <div class="field-body">
        <p class="field-head"><span class="req">{esc(TEXT["required"])}</span></p>
        {widget_html(payload["ui"], payload["question"].get("choices"), dim)}
      </div>
    </div>
    <button class="send" type="submit">{esc(TEXT["answer"])}{ARROW}</button>
  </form>
  <p class="err" role="alert">{esc(payload.get("problem", ""))}</p>
</div></main>
</div>"""


def apply_html(token: str, problem: str = "") -> str:
    """The application: background once, no questions, no link yet."""
    return f"""<div id="turn" class="turn" data-len="s" data-form="resume">
<header class="stage"><div class="wrap">
  <p class="status"><b>{esc(TEXT["meta"])}</b><i aria-hidden="true"></i><span class="dim"></span></p>
  <h1 class="q" tabindex="-1">{esc(TEXT["apply"])}</h1>
</div></header>
<main class="desk"><div class="wrap">
  <form hx-post="/apply" hx-target="#turn" hx-swap="outerHTML"
        hx-disabled-elt="find button" hx-sync="this:drop">
    <input type="hidden" name="token" value="{esc(token)}">
    <div class="field"><div class="field-body">
      <p class="field-head"><span class="req">{esc(TEXT["required"])}</span></p>
      <p class="hint">{esc(TEXT["applyHint"])}</p>
      <textarea name="resume" rows="12" required
                aria-label="{esc(TEXT["apply"])}"></textarea>
    </div></div>
    <button class="send" type="submit">{esc(TEXT["start"])}{ARROW}</button>
  </form>
  <p class="err" role="alert">{esc(problem)}</p>
</div></main>
</div>"""


def closing_html(message: str) -> str:
    """The end of it. No mark, no total, no hint of one (Q61)."""
    return (f'<div id="turn" class="turn" data-done="1"><header class="stage">'
            f'<div class="wrap"><h1 class="q" tabindex="-1">{esc(message)}</h1>'
            f"</div></header></div>")


ARROW = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
         ' stroke-width="2" stroke-linecap="square" aria-hidden="true">'
         '<path d="M3.5 12h16M13 5.5l6.5 6.5L13 18.5"/></svg>')


# ── the Interviewer's desk, as markup ─────────────────────────────────────
def stamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def when(value: object) -> str:
    return str(value or "").replace("T", " ").replace("+00:00", " UTC")


def band(score: float | None, top: float) -> str:
    """Where a mark sits on its rubric. A null gets none: that is an outage, not a nil."""
    if score is None:
        return ""
    part = score / (top or 1)
    return f' data-band="{"strong" if part >= 2 / 3 else "mid" if part >= 1 / 3 else "weak"}"'


def bar(label: str, score: float | None, top: float, note: str, sum_row: bool = False) -> str:
    fill = ""
    if score is not None:
        width = max(0.0, min(100.0, score / (top or 1) * 100))
        fill = f'<span class="fill"{band(score, top)} style="width:{width:.1f}%"></span>'
    klass = "bar sum" if sum_row else "bar"
    return (f'<div class="{klass}"><span>{label}</span>'
            f'<span class="track">{fill}</span><span class="num">{note}</span></div>')


def chip(klass: str, text: str) -> str:
    return f'<span class="{klass}">{esc(text)}</span>'


def sign_in_html(problem: str = "") -> str:
    return f"""<form class="signin" id="desk" hx-post="/admin/login" hx-target="#desk"
      hx-swap="outerHTML" hx-disabled-elt="find button">
  <p class="who">{esc(ADMIN_TEXT["title"])}</p>
  <input type="password" name="password" autocomplete="current-password"
         placeholder="{esc(ADMIN_TEXT["password"])}" required autofocus>
  <button class="btn key" type="submit">{esc(ADMIN_TEXT["enter"])}</button>
  <span class="err">{esc(problem)}</span>
</form>"""


def link_html(what: str, path: str) -> str:
    """The one thing this page is for. It ends in a copy, so it ships with one."""
    return (f'<div class="link"><span class="what">{esc(what)}</span>'
            f'<code>{esc(path)}</code>'
            f'<button class="btn copy" type="button" data-said="{esc(ADMIN_TEXT["copied"])}">'
            f'{esc(ADMIN_TEXT["copy"])}</button></div>')


def row_html(target: str, opened: bool, left: str, right: str, meta: str) -> str:
    """One line in the queue. A radio holds the selection, so CSS can show it."""
    return (f'<label class="row"><input type="radio" name="open" hx-get="{target}"'
            f' hx-target="#stack" hx-swap="innerHTML"{" checked" if opened else ""}>'
            f'<span class="when">{left}</span>{right}'
            f'<span class="meta">{meta}</span></label>')


def waiting_records() -> list[Json]:
    records = []
    for path in sorted(CANDIDATES_DIR.glob("*.json"), reverse=True)[:100]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not record.get("approved"):
            records.append(record)
    return records


def finished_reports() -> list[tuple[str, Json]]:
    reports = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:100]:
        try:
            reports.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return reports


def queue_html(session: Json, opened: str = "") -> str:
    """Everything waiting for the Interviewer, in one scannable column."""
    pending = []
    for record in waiting_records():
        count = (ADMIN_TEXT["unwritable"] if record.get("failed")
                 else f'{len(record["questions"])} {ADMIN_TEXT["questions"]}')
        pending.append(row_html(
            f'/admin/applicant?id={quote(record["id"])}', record["id"] == opened,
            f'<b class="id">{esc(record["id"])}</b>',
            chip("score none", count),
            f'{esc(ADMIN_TEXT["waiting"])} {esc(when(record.get("at")))}'))

    rows, unread = [], 0
    for name, report in finished_reports():
        # New means new to this session. The cookie remembers when it began, which is
        # the one piece of "what changed" that a reload used to lose.
        new = bool(session.get("at")) and str(report.get("finished_at", "")) > session["at"]
        unread += new
        top, total, fit = report.get("max") or 1, report.get("total"), report.get("fit")
        flags = [f for f in report.get("flags", []) if f.get("kind") != "human_review"]

        left = ('<i class="dot"></i>' if new else "") + esc(when(report.get("finished_at")))
        if new:
            left += chip("fresh", ADMIN_TEXT["fresh"])
        if total is None:
            score = chip("score none", ADMIN_TEXT["unscored"])
        else:
            score = (f'<span class="score"{band(total, top)}>{total}'
                     f"<small>/{top}</small></span>")
        meta = f'{len(report.get("answers", []))} {esc(ADMIN_TEXT["questions"])}'
        if fit:
            meta += f' · {esc(ADMIN_TEXT["fit"])} {fit["score"]}/{fit["max"]}'
        if flags:
            meta += " " + chip("flag", f'{ADMIN_TEXT["flags"]} {len(flags)}')
        rows.append(row_html(f"/admin/report?file={quote(name)}",
                             name == opened, left, score, meta))

    def heading(label: str, n: int, empty: str, body: str) -> str:
        tally = f'<span class="count">{n}</span>' if n else ""
        note = "" if n else f'<p class="note">{esc(empty)}</p>'
        return f"<h2>{esc(label)}{tally}</h2>{note}{body}"

    # The tab is where they will be when one finishes, so the count goes there too.
    seen = f"({unread}) " if unread else ""
    return (f'<title id="tab" hx-swap-oob="true">{seen}{esc(ADMIN_TEXT["title"])}</title>'
            + heading(ADMIN_TEXT["pending"], len(pending), ADMIN_TEXT["nobody"], "".join(pending))
            + heading(ADMIN_TEXT["results"], len(rows), ADMIN_TEXT["empty"], "".join(rows)))


def applicant_html(record: Json) -> str:
    """The questions a model wrote for one applicant, and the decision they need."""
    # Approving is offered even when nothing could be written. The applicant has already
    # been thanked and is waiting; the Plan's own questions are a whole interview without
    # the extras, and withholding the button left the only person who applied with no way
    # to be interviewed at all.
    # Said on the one screen where it is actionable, and above the button rather than
    # below it, because it is a thing to do first. True in both branches and true in
    # earnest: no version of this program has the account to show.
    approve = (f'<p class="standing">{esc(ADMIN_TEXT["resumeElsewhere"])}</p>'
               f'<button class="btn key" hx-post="/admin/approve"'
               f' hx-vals=\'{{"id": "{esc(record["id"])}"}}\''
               f' hx-target="#stack" hx-swap="innerHTML">'
               f'{esc(ADMIN_TEXT["approve"])}</button>')
    if record.get("failed"):
        body = (f'<p class="note">{chip("flag", ADMIN_TEXT["unwritable"])}</p>'
                f'<p class="note">{esc(ADMIN_TEXT["plainOnly"])}</p>{approve}')
    else:
        # What in the applicant's own account this came out of. The account itself is read
        # once and dropped, so this line is the whole of what survives it — and
        # it is the only thing on the page that answers "is this question fair to ask?".
        # Already written, already in the record, already read on every turn; it was just
        # never drawn. `.get` because a record written before it was required has none.
        def item(q: Json) -> str:
            why = (f'<p class="why"><b>{esc(ADMIN_TEXT["why"])}</b> {esc(q["context"])}</p>'
                   if q.get("context") else "")
            return (f'<li><span class="tag">{esc(q["dimension"])}</span>'
                    f'<q>{esc(q["text"])}</q>{why}'
                    f'<p class="says">{esc(q["rubric"][-1])}</p></li>')

        items = "".join(item(q) for q in record["questions"])
        body = (f'<p class="note"><b>{esc(ADMIN_TEXT["forThem"])}</b> · '
                f'{esc(ADMIN_TEXT["extraNote"])}</p><ol>{items}</ol>{approve}')
    return (f'<div class="rep-head"><b>{esc(record["id"])}</b><span class="when">'
            f'{esc(ADMIN_TEXT["waiting"])} {esc(when(record.get("at")))}</span></div>'
            f'<div class="cand">{body}</div>')


def outcome_html(name: str, report: Json) -> str:
    """The one judgment this program will not make, asked of the person who will.

    It sits after the marks and before the evidence, and stays quiet, because it is
    answered by reading rather than by being shouted at. Fitting a bar wants this filed
    weeks later, once the company has hired or not; asked here it is the interviewer's
    own call at the moment they read the report, which is the shorter loop a demonstration
    wants and a weaker thing to fit a bar against.
    """
    held = report.get("outcome") or {}

    def choice(decision: str, label: str) -> str:
        chosen = " on" if held.get("decision") == decision else ""
        vals = esc(json.dumps({"file": name, "decision": decision}))
        return (f'<button class="btn out{chosen}" hx-post="/admin/outcome"'
                f' hx-vals="{vals}" hx-target="#stack" hx-swap="innerHTML">'
                f'{esc(label)}</button>')

    filed = (f'<span class="when">{esc(ADMIN_TEXT["outcomeAt"])}'
             f' {esc(when(held.get("at")))}</span>' if held.get("at") else "")
    return (f'<div class="outcome"><span class="ask">{esc(ADMIN_TEXT["outcomeAsk"])}</span>'
            f'{choice("hired", ADMIN_TEXT["hired"])}'
            f'{choice("not_hired", ADMIN_TEXT["notHired"])}{filed}</div>')


def report_html(report: Json, name: str = "") -> str:
    """Every answer, the level it landed on, and the numbers — never a verdict."""
    top = report.get("max") or 1
    bars = []
    for dimension, value in (report.get("dimensions") or {}).items():
        note = (f'<strong{band(value["score"], top)}>{value["score"]}</strong>/{top}'
                f' · {esc(value["how"])} · n={value["n"]}'
                f' · {esc(ADMIN_TEXT["conf"])} {value["confidence"]}')
        bars.append(bar(esc(dimension), value["score"], top, note))
    bars.append('<div class="rule"></div>')

    total = report.get("total")
    note = (esc(ADMIN_TEXT["unscored"]) if total is None
            else f'<strong{band(total, top)}>{total}</strong>/{top}')
    bars.append(bar(f'<b>{esc(ADMIN_TEXT["total"])}</b>', total, top, note, True))

    fit = report.get("fit")
    if fit:
        note = (f'<strong{band(fit["score"], fit["max"])}>{fit["score"]}</strong>'
                f'/{fit["max"]} · {esc(ADMIN_TEXT["conf"])} {fit["confidence"]}')
        bars.append(bar(f'<b>{esc(ADMIN_TEXT["fit"])}</b>', fit["score"], fit["max"], note, True))

    flags = report.get("flags", [])
    said = [flag_text(f) for f in flags if f.get("kind") != "human_review"]
    standing = [flag_text(f) for f in flags if f.get("kind") == "human_review"]

    answers = []
    for a in report.get("answers", []):
        head = f'<span>{esc(a["dimension"])}</span>'
        if a.get("extra"):
            head += chip("flag", ADMIN_TEXT["extra"])
        mark = a.get("score")
        head += (f'<span class="mark"{band(mark, top)}>'
                 f'{esc(mark) if mark is not None else "--"}</span>')
        says = f'<p class="says">{esc(a["says"])}</p>' if a.get("says") else ""
        answers.append(f'<div class="qa"><p class="head">{head}</p>'
                       f'<q>{esc(a["text"])}</q>{says}'
                       f'<samp>{esc(a["answer"])}</samp></div>')

    fit_says = f'<p class="says">{esc(fit["says"])}</p>' if fit and fit.get("says") else ""
    said_html = ""
    if said:
        said_html = '<p class="flags">' + "".join(chip("flag", t) for t in said) + "</p>"
    standing_html = ""
    if standing:
        standing_html = f'<p class="standing">{esc(" · ".join(standing))}</p>'

    filed = outcome_html(name, report) if name else ""
    return (f'<div class="rep-head"><b>{esc(report.get("jti"))}</b><span class="when">'
            f'{esc(when(report.get("finished_at")))}</span></div>'
            f'<div class="bars">{"".join(bars)}</div>'
            f"{fit_says}{said_html}{standing_html}{filed}{''.join(answers)}")


def apply_button(klass: str = "btn") -> str:
    """The one way in. It is in the header to be always at hand, and in the middle of
    the empty pane because that is where a person who has never seen this page looks."""
    return (f'<button class="{klass}" hx-post="/admin/apply-link" hx-target="#links"'
            f' hx-swap="innerHTML">{esc(ADMIN_TEXT["applyLink"])}</button>')


def brief_html() -> str:
    """The Brief every question on this page came out of, behind a button.

    Not rendered: it is headings and bullets, and a Markdown parser is a dependency for
    something a <pre> already reads. <dialog> brings the backdrop, Esc and the focus trap.
    """
    return (f'<dialog id="brief"><form method="dialog">'
            f'<b>{esc(ADMIN_TEXT["brief"])}</b>'
            f'<button class="btn">{esc(ADMIN_TEXT["close"])}</button></form>'
            f'<pre>{esc(BRIEF)}</pre></dialog>')


def empty_html() -> str:
    """What the reading pane says before anything is open.

    On a first run the queue is two headings and nothing else, and the old sentence sent
    the reader to a list that had nothing in it. So the pane teaches the order instead —
    which is the question two buttons in a header could not answer — and carries the
    first step as a button. The order is read from the Plan's interface file: how many
    steps there are is a fact about the flow, not about this markup.
    """
    steps = "".join(f'<li><b>{esc(name)}</b><span>{esc(why)}</span></li>'
                    for name, why in ADMIN_TEXT["steps"])
    return (f'<div class="empty"><p class="begin">{esc(ADMIN_TEXT["begin"])}</p>'
            f'<ol class="flow">{steps}</ol>{apply_button("btn key")}'
            f'<p class="or">{esc(ADMIN_TEXT["choose"])}</p></div>')


def desk_html() -> str:
    """The whole interviewer's page, once they are in."""
    return f"""<div id="desk">
<header class="top">
  <h1>{esc(ADMIN_TEXT["title"])}</h1>
  <p class="role">{esc(ADMIN_TEXT["role"])}</p>
  <p class="err" id="trouble" role="alert" data-failed="{esc(ADMIN_TEXT["failed"])}"></p>
  <div class="tools">{apply_button()}<button class="btn" data-brief>{esc(ADMIN_TEXT["brief"])}</button></div>
</header>
<!-- Under the header and across both panes: whichever button issued it, a link turns up
     in the same place, and not inside a column that scrolls away from the button. -->
<div id="links"></div>
<div class="panes" hx-ext="sse" sse-connect="/events">
  <div class="queue">
    <!-- The stream is the push, and `focus` is the admission that a stream can die
         without anyone being told. A laptop sleeps, the connection goes, EventSource
         does not always come back, and this list then sits frozen and confident for as
         long as the tab is open. Coming back to the page re-reads it; throttled, so
         alt-tabbing does not turn into polling by another name. -->
    <div id="queue" hx-get="/admin/queue"
         hx-trigger="load, sse:finished, sse:applied, focus from:window throttle:10s"
         hx-swap="innerHTML"></div>
  </div>
  <div class="stack" id="stack">{empty_html()}</div>
</div>
{brief_html()}
</div>"""


# ── HTTP ──────────────────────────────────────────────────────────────────
def as_text(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value).strip()


class Handler(BaseHTTPRequestHandler):
    def send(self, body: Json | bytes, ctype: str = "application/json; charset=utf-8",
             code: int = 200, cookie: str = "") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes)
                         else json.dumps(body, ensure_ascii=False).encode())

    def body(self) -> Json:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def form(self) -> dict[str, list[str]]:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
        return parse_qs(raw, keep_blank_values=True)

    def html(self, markup: str, code: int = 200) -> None:
        self.send(markup.encode(), "text/html; charset=utf-8", code)

    def start_over(self) -> None:
        """The session is gone. htmx drops a 4xx body, so the page would simply stop
        answering; asking for a reload draws the sign-in form, which is the truth."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("HX-Refresh", "true")
        self.end_headers()

    def query(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.path).query)

    # ── the candidate's turn, as markup ──────────────────────────────────
    def render_turn(self, state: Json, current: Json, problem: str = "") -> None:
        """A payload the JSON routes already build, drawn instead of serialised."""
        self.html(turn_html(client_payload(state, current) | {"problem": problem}))

    def do_turn(self) -> None:
        """One route for the whole interview: answering it, and coming back to it.

        A reload lands here too, carrying the state the browser kept, because the
        server never had it.
        """
        fields = self.form()
        first = lambda key: (fields.get(key) or [""])[0]
        state = unsign(first("state"))
        if not state or not state.get("pending"):
            return self.html(closing_html(TEXT["sendFailed"]), code=400)

        question_id = first("id")
        if not question_id:                       # a reload: show them where they were
            return self.render_turn(state, question_payload(state, state["pending"]))
        if question_id != state["pending"]:
            return self.render_turn(state, question_payload(state, state["pending"]))

        asking = questions_for(state)
        if question_id not in asking:
            return self.html(closing_html(TEXT["sendFailed"]), code=400)

        options = asking[question_id]["options"]
        submitted = answer_values(fields, first("ui"))
        values = [as_text(v) for v in submitted if v is not None and as_text(v)]
        if not values or (options and any(v not in options for v in values)):
            return self.render_turn(state, question_payload(state, question_id),
                                    problem=TEXT["required"])

        state["answers"][question_id] = ", ".join(values)
        state["asked"].append(question_id)
        state["pending"], payload = next_turn(state)
        if state["pending"] is None:
            return self.html(closing_html(TEXT["finished"]))
        self.render_turn(state, payload)

    def do_apply_form(self) -> None:
        fields = self.form()
        first = lambda key: (fields.get(key) or [""])[0]
        link = unsign(first("token"))
        if not link or link.get("apply") is not True or link.get("exp", 0) < now():
            return self.html(closing_html(TEXT["sendFailed"]), code=403)
        resume = as_text(first("resume"))
        if not resume:
            return self.html(apply_html(first("token"), TEXT["applyEmpty"]), code=400)
        if len(resume) > RESUME_MAX_CHARS:
            return self.html(apply_html(first("token"),
                                        TEXT["applyLong"] % RESUME_MAX_CHARS), code=400)
        threading.Thread(target=consider, args=(secrets.token_urlsafe(9), resume),
                         daemon=True).start()
        self.html(closing_html(TEXT["applySent"]))

    def do_GET(self) -> None:
        if self.path.startswith("/events"):
            return self.do_events()
        if self.path.startswith("/admin/queue"):
            return self.do_queue()
        if self.path.startswith("/admin/report"):
            return self.do_report_pane()
        if self.path.startswith("/admin/applicant"):
            return self.do_applicant_pane()
        if self.path.startswith("/admin"):
            return self.do_desk()
        return self.do_shell()

    def do_shell(self) -> None:
        """The page arrives with its first question already drawn.

        A stored state, if the browser kept one, replaces it a moment later — that is
        the one thing markup cannot do for itself, because only the browser has it.
        """
        asked = self.query()
        if asked.get("a"):
            inner = apply_html(asked["a"][0])
        else:
            link = unsign((asked.get("t") or [""])[0])
            if not link or link.get("exp", 0) < now():
                inner = closing_html(TEXT["expired"])
            else:
                state = new_state(link["jti"], link.get("cand", ""))
                current = question_payload(state, FIRST_QUESTION)
                inner = turn_html(client_payload(state, current))
        self.send(SHELL.replace(b"<!--turn-->", inner.encode()),
                  "text/html; charset=utf-8")

    # ── the Interviewer issues a link; the Candidate walks through it ─────
    def session(self) -> Json | None:
        """The signed session the browser carries. A reload keeps it; memory did not.

        It is also what lets /events authorise: EventSource can send no header, and a
        cookie is the one credential it carries without being asked.
        """
        jar = dict(part.strip().partition("=")[::2]
                   for part in (self.headers.get("Cookie") or "").split(";") if "=" in part)
        held = unsign(jar.get(SESSION_COOKIE, ""))
        return held if held and held.get("exp", 0) > now() else None

    def do_admin_login(self) -> None:
        given = as_text((self.form().get("password") or [""])[0])
        if not (ADMIN_PASSWORD and hmac.compare_digest(given, ADMIN_PASSWORD)):
            return self.html(sign_in_html(ADMIN_TEXT["denied"]), code=403)
        token = sign({"admin": True, "at": stamp(), "exp": now() + SESSION_TTL_SECONDS})
        self.send(desk_html().encode(), "text/html; charset=utf-8",
                  cookie=(f"{SESSION_COOKIE}={token}; Path=/; Max-Age={SESSION_TTL_SECONDS}"
                          "; HttpOnly; SameSite=Strict"))

    # ── the desk, and the three things it asks for ───────────────────────
    def do_desk(self) -> None:
        inner = desk_html() if self.session() else sign_in_html()
        self.send(ADMIN_SHELL.replace(b"<!--desk-->", inner.encode()),
                  "text/html; charset=utf-8")

    def do_queue(self) -> None:
        session = self.session()
        if not session:
            return self.start_over()
        self.html(queue_html(session, (self.query().get("open") or [""])[0]))

    def do_report_pane(self) -> None:
        if not self.session():
            return self.start_over()
        name = Path((self.query().get("file") or [""])[0]).name   # nothing escapes RESULTS_DIR
        path = RESULTS_DIR / name
        if not name.endswith(".json") or not path.is_file():
            return self.html(f'<div class="empty"><span>{esc(ADMIN_TEXT["empty"])}</span></div>',
                             code=404)
        self.html(report_html(json.loads(path.read_text(encoding="utf-8")), name))

    def do_outcome_form(self) -> None:
        """The one thing on either surface a person writes into a finished Report."""
        if not self.session():
            return self.start_over()
        form = self.form()
        name = Path(as_text((form.get("file") or [""])[0])).name
        report = record_outcome(name, as_text((form.get("decision") or [""])[0]))
        if report is None:
            return self.html(
                f'<div class="empty"><span>{esc(ADMIN_TEXT["empty"])}</span></div>', code=404)
        self.html(report_html(report, name))

    def do_applicant_pane(self) -> None:
        if not self.session():
            return self.start_over()
        record = load_candidate((self.query().get("id") or [""])[0])
        if not record:
            return self.html(f'<div class="empty"><span>{esc(ADMIN_TEXT["nobody"])}</span></div>',
                             code=404)
        self.html(applicant_html(record))

    def do_make_apply_link(self) -> None:
        if not self.session():
            return self.start_over()
        token = sign({"apply": True, "exp": now() + LINK_TTL_SECONDS})
        self.html(link_html(ADMIN_TEXT["applyWhat"], f"{self.origin()}/?a={token}"))

    def do_approve_form(self) -> None:
        session = self.session()
        if not session:
            return self.start_over()
        record = load_candidate(as_text((self.form().get("id") or [""])[0]))
        if not record:
            return self.html(f'<div class="empty"><span>{esc(ADMIN_TEXT["nobody"])}</span></div>',
                             code=404)
        record["approved"] = True
        save_candidate(record, f"Approved the questions for {record['id']}")
        token = sign({"jti": secrets.token_urlsafe(9), "cand": record["id"],
                      "exp": now() + LINK_TTL_SECONDS})
        # The link belongs beside the other links, and the queue is one shorter now;
        # both ride along out of band so the reading pane can go back to empty.
        self.html(
            f'<div id="links" hx-swap-oob="true">'
            f'{link_html(ADMIN_TEXT["link"], f"{self.origin()}/?t={token}")}</div>'
            f'<div id="queue" hx-swap-oob="true">{queue_html(session)}</div>'
            f'{empty_html()}')

    def origin(self) -> str:
        return f"http://{self.headers.get('Host', 'localhost:8000')}"

    def do_events(self) -> None:
        """One long-lived response per open Interviewer page.

        EventSource cannot send a body or a header, so the page trades its password
        for a signed, expiring watch token first and carries that in the query.
        """
        watch = self.session()
        if not watch:
            return self.send({"error": "denied"}, code=403)

        mine: queue.SimpleQueue[str] = queue.SimpleQueue()
        with WATCHERS_LOCK:
            WATCHERS.add(mine)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(b": open\n\n")
            self.wfile.flush()
            while now() < watch["exp"]:
                try:                              # a comment keeps the pipe honest
                    message = mine.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    message = ": ping\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with WATCHERS_LOCK:
                WATCHERS.discard(mine)

    # ── an applicant applies, then waits to be sent a link ───────────────
    def do_link(self, request: Json) -> None:
        """The one route that still takes the password in a body rather than a
        session; every admin page route goes through the cookie."""
        given = as_text(request.get("password", ""))
        if not (self.session() or (ADMIN_PASSWORD
                                   and hmac.compare_digest(given, ADMIN_PASSWORD))):
            return self.send({"error": "denied"}, code=403)
        # A stateless server cannot remember that a link was used, so it expires
        # instead of burning; the jti rides into the result so repeats are visible.
        token = sign({"jti": secrets.token_urlsafe(9), "exp": now() + LINK_TTL_SECONDS})
        self.send({"token": token, "path": f"/?t={token}"})

    def do_start(self, request: Json) -> None:
        link = unsign(request.get("token"))
        if not link or link.get("exp", 0) < now():
            return self.send({"error": "bad or expired link"}, code=403)
        state = new_state(link["jti"], link.get("cand", ""))
        self.send(client_payload(state, question_payload(state, FIRST_QUESTION)))

    def do_answer(self, request: Json) -> None:
        state = unsign(request.get("state"))
        if not state or "pending" not in state:
            return self.send({"error": "bad state"}, code=400)

        question_id = request.get("id")
        if question_id != state["pending"]:
            return self.send({"error": "stale id"}, code=400)

        asking = questions_for(state)
        if question_id not in asking:
            return self.send({"error": "no such question"}, code=400)
        submitted = request.get("a")
        options = asking[question_id]["options"]
        values = [as_text(v) for v in (submitted if isinstance(submitted, list) else [submitted])
                  if v is not None and as_text(v)]
        if not values or (options and any(v not in options for v in values)):
            return self.send({"error": "bad answer"}, code=400)

        state["answers"][question_id] = ", ".join(values)
        state["asked"].append(question_id)
        state["pending"], payload = next_turn(state)
        self.send(payload if state["pending"] is None else client_payload(state, payload))

    # Markup for the browser; the three JSON routes are what the tests pin a turn
    # through. There was a JSON twin of every admin route once, and nothing called it.
    FORM_ROUTES = {"/turn": do_turn, "/apply": do_apply_form,
                   "/admin/login": do_admin_login,
                   "/admin/apply-link": do_make_apply_link,
                   "/admin/approve": do_approve_form,
                   "/admin/outcome": do_outcome_form}

    ROUTES = {"/link": do_link, "/start": do_start, "/answer": do_answer}

    def do_POST(self) -> None:
        form_route = Handler.FORM_ROUTES.get(self.path)
        if form_route is not None:
            return form_route(self)
        route = Handler.ROUTES.get(self.path)
        if route is None:
            return self.send({"error": "not found"}, code=404)
        route(self, self.body())


if __name__ == "__main__":
    if "--check-repo" in sys.argv:
        raise SystemExit(check_repo())
    if "--rescore" in sys.argv:
        boot()
        index = sys.argv.index("--rescore") + 1
        chosen = ([Path(sys.argv[index])] if index < len(sys.argv)
                  else [p for p in sorted(RESULTS_DIR.glob("*.json"))
                        if not json.loads(p.read_text(encoding="utf-8")).get("scored")])
        if not chosen:
            raise SystemExit(f"nothing to score again in {RESULTS_DIR}/")
        for path in chosen:
            print(f"scoring {path.name} again…", file=sys.stderr)
            if fresh := rescore(path):
                print(f"  total {fresh['total']}/{fresh['max']}"
                      f"   fit {fresh['fit']['score'] if fresh['fit'] else '--'}", file=sys.stderr)
        raise SystemExit(0)
    if "--noise" in sys.argv:
        boot()
        index = sys.argv.index("--noise") + 1
        asked = sys.argv[index] if index < len(sys.argv) else ""
        raise SystemExit(measure_noise(int(asked) if asked.isdigit() else 60))
    if "--report" in sys.argv:
        boot()
        index = sys.argv.index("--report") + 1
        chosen = (Path(sys.argv[index]) if index < len(sys.argv)
                  else max(RESULTS_DIR.glob("*.json"), default=None))
        if chosen is None or not chosen.is_file():
            raise SystemExit(f"no record to read (looked in {RESULTS_DIR}/)")
        print_report(chosen)
        raise SystemExit(0)
    if "--check" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--check") + 1])
        problems = check_plan(json.loads(path.read_text(encoding="utf-8")))
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(f"{path}: {len(problems)} problem(s)" if problems else 0)
    if "--plan" in sys.argv:
        source = Path(sys.argv[sys.argv.index("--plan") + 1])
        # Asked before the two-minute call, and never defaulted to PLAN: that one is
        # DATA_REPO-relative on the server, and this writes a file on this machine.
        if "--out" not in sys.argv:
            raise SystemExit("--plan needs --out <path.json>, somewhere in your own"
                             " repository:\n  python3 app.py --plan"
                             " ../your-records/brief.md"
                             " --out ../your-records/plans/interview.json")
        target = Path(sys.argv[sys.argv.index("--out") + 1])
        model = os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
        print(f"writing a plan from {source} with {model}", file=sys.stderr)
        written = write_plan(source, model)
        if written is None:
            raise SystemExit("no usable plan was written")

        print(f"\n  {len(written['questions'])} questions over"
              f" {len(written['dimensions'])} dimensions", file=sys.stderr)
        for dimension in written["dimensions"]:
            covering = sum(1 for q in written["questions"]
                           if q["dimension"] == dimension["name"])
            print(f"    {dimension['name']:<20} weight {dimension['weight']:.2f}"
                  f"  {dimension['aggregate']:<5} {covering} question(s)", file=sys.stderr)
        print(f"    policy: {written['policy']}", file=sys.stderr)
        # Nothing was booted, so the Replay runs at the candidate's own sample count.
        POLICY = written["policy"]
        regression(written, source.read_text(encoding="utf-8"), RESULTS_DIR)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(written, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {target} — read it, then commit it. The commit is the approval.")
        raise SystemExit(0)
    # Only the server takes them from the operator's repository; a CLI above stands in
    # a clone and reads the files in front of it.
    boot(from_repo=bool(DATA_REPO and GITHUB_TOKEN))
    print(f"  plan   {PLAN_PATH} from {plan_source} — {len(QUESTIONS)} questions,"
          f" {len(WEIGHTS)} dimensions, brief {plan['from']}", file=sys.stderr)
    fill_from_repository()
    ThreadingHTTPServer(("", int(os.environ.get("PORT", 8000))), Handler).serve_forever()
