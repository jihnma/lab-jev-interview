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
| tokens | 14,517,085 | 15,090,228 |
| judge time, all interviews | **475 s** | 3,893 s |
| judge time, one interview (median) | **4.9 s** | 40.4 s |
| judge time, one turn (mean) | **0.22 s** | 2.03 s |
| correct at the shipped bar of 1.5, of 90 | 62 (69%) | **68 (76%)** |
| correct at the fitted bar, leave-one-out, of 90 | **84 (93%)** | 80 (89%) |
| correct when `fit` gates the pass too, of 90 | **80 (89%)** | 73 (81%) |
| of the 18 against the principles, `fit` below 1.0 | **18** | 6 |
| of the 33 to hire, `fit` at 1.5 or above | 33 | 33 |
| interviews the judge stopped early | 59 | 49 |
| failed calls / interviews with no score | 0 / 0 | 0 / 0 |

Read the table this way. The request is the same for both judges, so the tokens are the same to
within 4%. Only the time differs, and Haiku spent 8.2 times more of it. The two judges rank the
applicants almost alike, and agree on 94 of the 100 decisions (r = 0.969). Nobody tuned the
shipped bar of 1.5; it is a drift check. At that bar Haiku was right about 6 more applicants, and
all 6 are rejects near the bar. At the bar the program fits from filed decisions, Jev was right
about 4 more. On `fit`, Jev put all 18 applicants against the company principles below 1.0, and
Haiku put 6 there. The `fit` gate in the third accuracy row came after the results, not before,
so read that one row as an observation.

Three more measurements. One Jev call is enough: three averaged calls changed none of the 100
decisions, and used 2.3 times the tokens of one call. Both judges give the same decision twice: 12
applicants met a second interview, and no decision changed (that Jev repeat ran with three calls).
Extended thinking makes the generative judge slower, not equal: with it on, three interviews took
398 s to 537 s of judge time, against 29 s to 40 s with it off.

## How the test ran

- Every arm used the same program, plans, applicants, answers and order. Only `ask_jev` changed.
- The generative judge got the identical request and returned the identical shape of answer.
- A rule fixed each label before any interview ran. Hire needs solid or deep Java, lived money
  experience and aligned principles; reject needs one of the opposites.
- The applicants are invented: 60 answer in Japanese, 40 in English; 33 hire, 57 reject, and 10
  borderline that no accuracy count includes.
- Claude Sonnet 4.5 wrote every answer from its applicant's row, before any interview ran.

## What the test does not show

- That either judge selects real people well. The applicants and their labels are invented.
- That the token counts are exact. Two different tokenizers read the same text.
- That the numbers hold for another model, route or brief. There was one of each.
- That the judge and the answer writer are independent. Both are Claude models.
- What Jev costs in money. It reports no price per call. The generative run cost $30.86 at list price.

## The data, and how to repeat it

[lab-jev-interview-example](https://github.com/jihnma/lab-jev-interview-example) holds every
record and log. Its `experiment/logs/analysis.md` holds each number above and much this page
leaves out: per applicant type, per language, per half of the order, and the choice set of each
turn. This program must be at commit `8629d80` or later, because `run.py` reads `JEV_USAGE`.

```sh
git clone https://github.com/jihnma/lab-jev-interview             # this program
git clone https://github.com/jihnma/lab-jev-interview-example     # the data
cd lab-jev-interview-example
export TYPESAFE_API_KEY=...                                       # the Jev arm only
PROGRAM=../lab-jev-interview python3 experiment/run.py jev1       # one Jev call per judgment
PROGRAM=../lab-jev-interview python3 experiment/run.py baseline   # needs the Claude Code CLI
PROGRAM=../lab-jev-interview python3 experiment/analyse.py
```

To run an interview of your own, read `.env.example`: it names what an instance needs.
`interface.json` holds the tuning knobs, and each `_note` in it says what its knob does.
