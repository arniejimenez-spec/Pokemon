# The Mirror Test: Why Evaluating an Agent Against Its Own Assumptions Is a Silent Killer

### A Mega Lucario ex deck, three falsified hypotheses, and the anti-mirror methodology that produced our only confirmed ladder win

---

## 1. Deck: Mega Lucario ex, chosen for pilotability, not just power

We built Mega Lucario ex around three deliberate constraints: a **single evolution
step** (Riolu → Mega Lucario ex, not a Stage 2 line), a **single energy color**
(Fighting), and a **dual attack profile** — Aura Jab (cheap, recycles discarded
energy back onto the bench) for tempo, Mega Brave (270 damage) for the finisher.
Bloodmoon Ursaluna ex and Koraidon ex round out the attacker line for redundancy.

This was a deliberate bet: an automatically-piloted agent benefits far more from a
deck a generic policy can execute *correctly* than from a deck that is merely
*powerful* in expert human hands. We tested this bet directly, late in the project,
by porting two real tournament decklists from competitive play data. **Dragapult
ex/Dusknoir** (legal in our card pool, a top-tier human archetype) failed
completely under our agent: its second evolution line never finished developing in
15/15 test games, and 53% of games timed out via deck-out instead of ending in
combat. A **real-world Mega Lucario list** we also tested was fully pilotable
(90% win rate vs. a random-action baseline) but still lost to our own simpler build
head-to-head. Both results affirm the same conclusion: for an automated pilot,
deck *simplicity* — one attacker, one color, no competing win conditions — is
itself a strategic parameter, independent of raw competitive strength.

## 2. Architecture: scoring options, not choosing from a fixed action set

The simulator offers a **different, variable-length set of legal options every
decision** — there is no fixed action space to put a policy head over. Our network
scores each offered option independently, `logit = MLP([state, option])`, and
softmaxes over whatever the engine actually offers that turn. This makes illegal
moves structurally impossible and lets the same architecture handle everything
from "attack" to "discard 2 of 7 cards." Inference runs as a dependency-free numpy
forward pass (no ML runtime required at submission time), completing in under
10ms against a 600-second per-episode budget.

## 3. Hypothesis 1 — search plus opponent modeling will beat a rule-based heuristic

Standard game-AI intuition says lookahead search improves decision quality. We
built a determinized rollout search with an inferred opponent-deck model, and it
won **67–75% locally** against our own rule-based heuristic. We shipped it.

**Ladder result: 456**, against the heuristic's own **601** — an active
regression below the rating's starting point. The cause was not a bug: Kaggle's
agent logs showed zero crashes and normal timing. The cause was **circularity**.
Our search's rollout policy for the opponent's hidden hand was the *same
heuristic* it was being measured against, so its internal model of "what a
reasonable opponent does" was correct by construction — and useless against real
opponents on the ladder. Independent community analysis of the field (based on
per-move timing fingerprints across ~30,000 games) corroborated the mechanism: the
single top-rated player combines a trained value function with light search, but
most of the rest of the top field runs no search at all, because search only pays
off with an accurate value estimate. Ours was hand-tuned, not learned, and the
evaluation that told us it worked was measuring the wrong thing.

## 4. Hypothesis 2 — a cloned-then-fine-tuned policy will generalize better

We pivoted to a learned policy: behavioral cloning (BC) from the heuristic to
bootstrap a competent starting point (avoiding random flailing over ~100
sequential decisions per game), followed by self-play reinforcement learning
(REINFORCE with a value baseline) to push past imitation.

The first version of this policy **beat the heuristic by +105 rating points**,
concurrently confirmed on the ladder — our first real, non-circular win, because
this time we measured against the actual field.

But the very next iteration repeated the same mistake in a new disguise. We
shipped a follow-up model selected because it won **57% of mirror matches**
against its own predecessor. It read as a **statistical tie** on the ladder (a
6.6-point gap, well inside measured noise). Mirror self-play evaluation is exactly
as circular as reusing your own heuristic as the opponent model — it just hides
the assumption one layer deeper. The general principle we extracted: **any
evaluation instrument that shares an assumption with the agent under test will
systematically overstate quality**, regardless of what that shared assumption is.

## 5. The anti-mirror methodology

We redesigned both training and selection around this principle:

- **Diversify training.** The self-play opponent pool now mixes frozen past-
  champion policies, the scripted heuristic, and multiple distinct deck
  archetypes — not primarily copies of the current policy — forcing the agent to
  generalize rather than exploit familiarity with itself.
- **Diversify and hold out evaluation.** Candidates are scored against a 5-opponent
  panel spanning different deck archetypes. One archetype — a Psychic-type deck,
  also our own type weakness — is **deliberately excluded from training entirely**,
  serving as a genuine generalization test. A candidate must also beat the
  reigning champion in a **direct head-to-head**, not merely improve an aggregate
  score.

This caught a real failure during testing: one heavily-trained candidate improved
against every opponent it had trained against while *regressing* on the untouched
holdout deck — precisely the overfitting the holdout exists to catch, invisible
to any metric that only looks at trained-on opponents.

The candidate selected under this methodology shipped to **+58.3 points,
concurrently confirmed on the ladder** — our single largest verified gain, and the
first time a local instrument correctly predicted a ladder outcome in either
direction.

## 6. Testing whether the gain compounds

We then ran four further single-variable experiments against the new champion, to
see whether the anti-mirror recipe itself could be pushed further:

| Experiment | Single variable changed | Outcome |
|---|---|---|
| Cycle 2 | More RL iterations, same opponent pool | No further gain (null) |
| Cycle 3 | One additional deck archetype in the pool | No further gain (null) |
| Cycle 4 | Lower learning rate | Broke a 3-cycle overfitting pattern; promising, not yet conclusive |
| Richer state features | Opponent's revealed hand/discard contents as learned embeddings | Fully built and verified; paused on an uncontrolled training-budget confound before a fair read |

The consistent finding: our one confirmed gain came from a **structural** change
(who the agent trains against), not from spending more compute on an unchanged
recipe. Three of four follow-up experiments testing "more of the same" came back
null or inconclusive — itself useful evidence about where the marginal value in
self-play training actually lives for this problem.

## 7. Results and reflection

| Milestone | Ladder rating | Note |
|---|---|---|
| Rule-based heuristic | 601 | Baseline; starting rating is 600 |
| Determinized search + opponent model | 456 | Regression — circular local evaluation |
| BC + self-play RL (first version) | +105 vs. concurrent heuristic | First confirmed, non-circular win |
| Mirror-selected "improvement" | ≈ tie (6.6 pts) vs. its predecessor | Mirror evaluation, same circularity in disguise |
| Anti-mirror pivot (diverse pool + holdout panel) | **+58.3 vs. concurrent prior best** | Largest confirmed gain; current standing submission |

All comparisons above are **concurrent** (measured against a same-moment baseline,
not a remembered number) — we found early on that absolute ladder ratings drift by
over 100 points across a single day as the field strengthens, making any
non-concurrent comparison meaningless.

Given the competition deadline, we chose to stop active experimentation and
preserve this validated result rather than continue spending ladder submissions
chasing gains our own methodology had just shown were unconfirmed or exhausted —
a decision made from evidence, not from running out of ideas. Given the competition deadline, we chose to stop active experimentation
and preserve this validated result rather than continue spending ladder submissions
chasing gains our own methodology had just shown were unconfirmed or exhausted —
a decision made from evidence, not from running out of ideas.

If given more time, the two clearest next steps are: extending the lower-learning-
rate cycle further, since it is the only untested lead that showed a genuinely new
signal; and completing the richer-feature comparison at a matched training budget,
since it was paused on a confound rather than a negative result.

The single throughline of this project is methodological, more than architectural:
**every real improvement we shipped was confirmed against opponents and
conditions the agent had no assumptions about, and every regression we shipped was
one we had, in hindsight, only ever tested against itself.**
