# Adaptive Interview on Jev

This program interviews one applicant from a plan of questions. Between answers it asks a judge
which question to ask next, and whether it heard enough to stop. After the interview it asks for
the rubric level of each answer and one `fit` level against the brief, then makes the total by
arithmetic. No generative model runs during an interview. The judge is
[Jev](https://docs.typesafe.ai), a System One model: it writes no text, it reads a JSON state and
gives a probability for each closed answer.

## Result

One test measured the judge and nothing else. 100 invented applicants met the same program, the
same two plans and the same answers. Jev answered each judgment one time. The baseline judge was
Claude Haiku 4.5, also one time, with extended thinking off.

| 100 interviews | Jev, 1 call | Haiku 4.5, 1 call |
|---|---|---|
| judge calls | 2,138 | 2,150 |
| tokens, as each service counted them | 14,517,085 | 15,090,228 |
| output tokens, as each service counted them | **180,485** | 308,471 |
| wall time, all interviews, on one clock | **476 s** | 6,663 s |
| judge time, all interviews | **475 s** | 3,893 s |
| judge time, one interview (median) | **4.9 s** | 40.4 s |
| judge time, one turn (mean) | **0.22 s** | 2.03 s |
| AUC of the total, over the 90 labelled | **0.993** | 0.969 |
| of the 57 to reject, how many outrank the weakest hire | **4** | 11 |
| of the 18 against the principles, `fit` below 1.0 | **18** | 6 |
| of the 33 to hire, `fit` at 1.5 or above | 33 | 33 |
| interviews the judge stopped early | 59 | 49 |
| failed calls / interviews with no score | 0 / 0 | 0 / 0 |

Read the resource rows this way. Both judges get the same request, because only `ask_jev`
changed. But two tokenizers made the two token counts, and this test never put one text through
both, so read the 4% between the totals as two services that agree and not as a measurement. What
the counts do compare is the reply, and Jev wrote 41% less of it. Time is the clear result: one
clock measured both arms from end to end, and Haiku took 14 times longer. The time Haiku reported
for its own work is 8.2 times longer than Jev's.

Now the decisions. AUC asks the judge's own question and leaves every bar out of it: take one
applicant to hire and one to reject, and how often does the judge put the hire higher? Jev is right
993 times in 1,000, and Haiku 969. The overlap says the same thing in people: of the 57 applicants
the rule says to reject, 4 outrank the weakest hire under Jev and 11 under Haiku, and Jev's 4 are 4
of Haiku's 11. So Jev makes no mistake here that Haiku does not make too.

All 11 are one kind of applicant: strong on the technical questions, and in conflict with how the
company works. Jev names that conflict directly, because it put all 18 such applicants below 1.0
on `fit` while Haiku put 6 there. Both judges keep all 33 applicants to hire at 1.5 or above. Over
all 100 applicants the two totals correlate at 0.969, so the judges agree nearly everywhere; they
differ on the group the brief cares about most.

Each thing the brief asks for is a dimension of its own, and there Haiku equals Jev or beats it on
three of the four.

| dimension, and its weight in the total | Jev, AUC | Haiku 4.5, AUC |
|---|---|---|
| Money correctness and operational judgment, 0.32 | **0.943** | 0.938 |
| Java backend practice, 0.28 | 0.896 | **0.902** |
| Alignment with how we work, 0.24 | 0.947 | **0.969** |
| Rust readiness, 0.16 | 0.880 | **0.903** |

Jev wins one criterion, the heaviest. It wins the four together, because its scores stay apart on
the applicants where the two groups meet and Haiku's do not. One figure on this page still comes
from a bar: at the shipped drift boundary of 1.5, Haiku is right about 6 more of the 90. Both
judges separate the groups near 2.5, so 1.5 measures the boundary and not the judge.

Three more measurements. The two extra Jev calls buy nothing here: three averaged calls changed
none of the 100 decisions at 1.5, moved the AUC from 0.993 to 0.992, and used 2.3 times the tokens
of one call. Both judges give the same decision twice, because 12
applicants met a second interview and no decision changed (that Jev repeat ran with three calls).
Extended thinking makes the generative judge slower, not equal: with it on, three interviews took
398 s to 537 s of judge time, against 29 s to 40 s with it off.

## How the test ran

- Every arm used the same program, plans, applicants, answers and order. Only `ask_jev` changed.
- The generative judge got the identical request content, and returned the identical shape of
  answer. Both arms wrap the same call with the same `(state, questions)`.
- A rule fixed each label before any interview ran. Hire needs solid or deep Java, lived money
  experience and aligned principles. Reject needs Java at none or thin, or money at none, or
  principles in conflict. Everybody else is borderline.
- The applicants are invented: 60 answer in Japanese, 40 in English; 33 hire, 57 reject, and 10
  borderline that no accuracy count includes.
- Claude Sonnet 4.5 wrote every answer from its applicant's row, before any interview ran.

## What the test does not show

- That either judge selects real people well. The applicants and their labels are invented.
- That Jev needs fewer tokens. Two tokenizers made the two counts. The program also escapes
  every non-ASCII character, so Jev read 72.7 MB of request text where Haiku read 48.4 MB of
  the same content.
- That the numbers hold for another model, route or brief. There was one of each.
- That the judge and the answer writer are independent. Both are Claude models.
- What Jev costs in money. It reports no price per call. The generative run cost $30.86 at
  list price.

## The data, and how to repeat it

[lab-jev-interview-example](https://github.com/jihnma/lab-jev-interview-example) holds every
record and log. Its `experiment/logs/analysis.md` holds each number above and much this page
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
