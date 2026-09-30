# Adaptive Interview on Jev

This program interviews one applicant from a plan of questions. Between answers it asks a judge
which question to ask next, and whether it heard enough to stop. After the interview it asks for
the rubric level of each answer and one `fit` level against the brief, then makes the total by
arithmetic. No generative model runs during an interview. The judge is
[Jev](https://docs.typesafe.ai), a System One model: it writes no text, it reads a JSON state and
gives a probability for each closed answer.

## Result

**In one line.** Jev judged each interview in 4.9 s where Claude Haiku 4.5 needed 40.4 s, and its
scores separated the synthetic hire and reject labels more cleanly.

One test measured the judge and nothing else. 100 invented applicants met the same program, the
same two plans and the same answers. Jev answered each judgment one time. The baseline judge was
Claude Haiku 4.5, also one time, with extended thinking off.

| 100 interviews | Jev, 1 call | Haiku 4.5, 1 call |
|---|---|---|
| judge time, one interview (median) | **4.9 s** | 40.4 s |
| AUC against the labels, of the 90 labelled | **0.993** | 0.969 |
| of the 18 against the principles, `fit` below 1.0 | **18** | 6 |
| failed calls / interviews with no score | 0 / 0 | 0 / 0 |

AUC measures how well a score orders the applicants, and it needs no threshold: it is the chance
that the judge puts a randomly chosen hire above a randomly chosen reject. The same difference,
counted in people: of the 57 applicants the rule says to reject, 4 score above the weakest hire
under Jev and 11 under Haiku, and Jev's 4 are 4 of Haiku's 11. All 11 are one kind of applicant,
strong on the technical questions and in conflict with how the company works. That is the group
`fit` was written for, and it is the row above: Jev put all 18 of them below 1.0, and Haiku put 6.

**These are invented applicants and labels written by a rule.** This test does not show that either
judge screens real people well. It shows what the two judges do with the same 100 interviews.

## How the test ran

- Every arm used the same program, the same approved plan per language, the same applicants,
  answers and order. Only `ask_jev` changed.
- The generative judge got the identical request content, and returned the identical shape of
  answer. Both arms wrap the same call with the same `(state, questions)`.
- A rule fixed each label before any interview ran. Hire needs solid or deep Java, lived money
  experience and aligned principles. Reject needs Java at none or thin, or money at none, or
  principles in conflict. Everybody else is borderline and no accuracy count includes them.
- The applicants are invented: 60 answer in Japanese, 40 in English; 33 hire, 57 reject, 10
  borderline.
- Two generative models ran before any interview: `z-ai/glm-5.3-flash` wrote the plan from the
  brief, and Claude Sonnet 4.5 wrote every answer from its applicant's row.

## What the test does not show

- That either judge screens real people. The applicants and their labels are invented.
- That the result holds against another judge, another route or another brief. There was one
  of each.
- That the answer writer is independent of the judge it is compared against. Both are Claude models.

## Secondary checks

Each of these is in `experiment/logs/analysis.md` with its numbers.

- **Per dimension, Haiku is the better judge.** It has the higher AUC on Java practice, Rust
  readiness and alignment; Jev has the higher AUC on money correctness and on the weighted total.
- **At the shipped boundary of 1.5, Haiku is right about 6 more of the 90.** Both judges separate
  the groups near 2.5, so a boundary at 1.5 measures the boundary.
- **One Jev call is enough.** Three averaged calls changed none of the 100 decisions at 1.5, moved
  the AUC from 0.993 to 0.992, and used 2.3 times the tokens.
- **Neither judge changes its mind.** 12 applicants met a second interview and no decision moved.
- **Tokens do not compare.** Two tokenizers made the two counts, and the program escapes non-ASCII
  while the baseline prompt does not, so Jev read 72.7 MB of request text where Haiku read 48.4 MB
  of the same content. The replies do compare: Jev wrote 41% fewer output tokens.
- **Money does not compare.** The generative run cost $30.86 at list price. Jev reports no price.
- **Extended thinking does not close the gap.** With it on, three interviews took 398 s to 537 s of
  judge time against 29 s to 40 s with it off.

## The data, and how to repeat it

[lab-jev-interview-example](https://github.com/jihnma/lab-jev-interview-example) holds every
record and log. Its `experiment/logs/analysis.md` recomputes every number on this page and much it
leaves out: per applicant type, per language, per half of the order, and the choice set of each
turn.

```sh
git clone https://github.com/jihnma/lab-jev-interview             # this program
git clone https://github.com/jihnma/lab-jev-interview-example     # the data
cd lab-jev-interview-example
export TYPESAFE_API_KEY=...                                       # the Jev arm only
PROGRAM=../lab-jev-interview python3 experiment/run.py jev1       # one Jev call per judgment
PROGRAM=../lab-jev-interview python3 experiment/run.py baseline   # needs the Claude Code CLI
PROGRAM=../lab-jev-interview python3 experiment/run.py jev        # three calls, to price the two
PROGRAM=../lab-jev-interview python3 experiment/analyse.py
```

To run an interview of your own, read `.env.example`: it names what an instance needs.
`interface.json` holds the tuning knobs, and each `_note` in it says what its knob does.
