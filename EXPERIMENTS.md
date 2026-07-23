# Experiment log

Ladder results for every submission. **The ladder is the only ground truth** — local
win rates against our own heuristic have twice failed to predict it (see README).

**Reading the ratings:** μ starts at 600. Two *byte-identical* submissions read 399 and
535 after one hour, so **differences under ~100 points mean nothing** without a day+ of
games. Only the latest 2 submissions are tracked for final scoring.

| # | date | agent | note | git | model | rating | verdict |
|---|------|-------|------|-----|-------|--------|---------|
| 1 | 2026-07-15 | heuristic | first submission, Mega Lucario | `11f31ae` | — | **601.2** | baseline. settled over ~1 day |
| 2 | 2026-07-15 | search | determinized rollout + opponent model, h=3 | `11f31ae` | — | **456.4** | **regression.** below the 600 start = actively losing |
| 3 | 2026-07-16 | heuristic | revert to known-good | `8ff2967` | — | 399 @1h | identical to #4 — see noise note |
| 4 | 2026-07-16 | heuristic | revert, 2nd copy to flush #2 | `8ff2967` | — | **447.8** | settled. identical code read 601 on day 1 — see drift note |
| 5 | 2026-07-16 | policy | RL it12 (BC + 12 self-play iters) | `f8e7e65` | policy_rl_it12.npz | **552.6** | RL beats heuristic by +105 concurrent — first real gain |
| 6 | 2026-07-17 | policy | RL it18: +24 iters from it12; 58.0% vs it12 @300 games (it13 54.3, it24 52.3) | `b4a243b` | policy_rl_it18.npz | 499.8 | settled over weekend; concurrent pair with #7 |
| 7 | 2026-07-17 | policy | RL v2-lineage 30 iters (identity emb); 57.0% vs it18 @300 (it5 56.0, it12 52.0) | `3b13645-dirty` | policy_rl_v2_it30.npz | 477 | re-read @4 days: further drift down from 493.2 (weekend read). Confirms drift, not a real change. |
| 8 | 2026-07-20 | policy | diversified-pool RL it12 (holdout-validated); panel agg 80.9% vs base 78.2%, latias holdout +3.2 | `0d130d5` | policy_rl_pool_it12.npz | 535.3 | +58.3 CONCURRENT vs v2-it30 (477) @21h. Panel/holdout selection's first ladder-confirmed win. |

## What each rating taught us

**#1 vs #2 — search is a dead end here.** The determinized search lost 145 points to the
plain heuristic despite beating it 67–75% locally. Kaggle logs ruled out timeouts/crashes
(zero stderr; ~4.6s used of a 600s/episode budget), so it was pure gameplay. Root cause:
the search uses `HeuristicPolicy` as its opponent-rollout policy and was only ever tested
against that same heuristic — its opponent model was correct *by construction*. Kaggle
discussion 724362 independently confirms the mechanism: an 83rd-place competitor found
search doesn't pay off without an accurate value function, and most of the top uses no
search at all.

**#3 vs #4 — the noise floor.** Byte-identical agents read 399 and 535 after an hour.
This is the single most useful measurement in the log: it means **no ladder comparison is
meaningful until σ shrinks**, and small gaps are meaningless forever.

**#1 vs #4 — ratings DRIFT; only concurrent comparisons count.** The identical heuristic
read **601 on day 1 (#1)** and **447.8 a day later (#4)** — a ~150-point drop with zero code
change, because the ladder strengthened as stronger entries joined and ratings
redistributed. **Absolute ratings are not comparable across days.** Never compare a new
submission to a remembered number; compare it to a baseline running *at the same time*.
This retroactively means #2's 456 was measured against a 601-era ladder and the gap was
real, but its absolute value tells us nothing today.

**#4 vs #5 — RL works (first real gain).** Policy **552.6** vs heuristic **447.8**, both
settled and concurrent: **+105 for the learned policy** — above the practical noise once σ
has shrunk, and pointing the same direction as the local self-play trend (59%→69%). Not a
blowout, and still mid-field (top ~1000+), but the first change in the whole project that
actually beat the baseline on the ladder. Crucially it beat the heuristic against the
*real field*, not just in our own opponent pool — so this is not the circular trap that
sank the search agent.

**#5 — the open question.** Local: RL it12 beats the frozen BC clone 68% and the heuristic
61.5% (200 games each). But `vs_heuristic` is **contaminated** — the heuristic is in the RL
opponent pool. The reliable local signal was self-play win rate climbing 59%→69% over 12
iterations. Whether any of that maps to ladder points is exactly what #5 tests.

**#6 vs #7 — mirror-selection has hit its ceiling.** After a full weekend (settled σ, same
window): it18 **499.8** vs v2-it30 **493.2** — a 6.6-point gap, far below noise. A **57%
Lucario-mirror head-to-head edge produced zero measurable ladder difference.** Both also
sit ~50 points below where the lineage read three days earlier (552.6) — ladder drift
again, the field keeps strengthening. Calibration so far: mirror 57–58% ⇒ ladder ~0.
The local selection instrument (Lucario-vs-Lucario head-to-heads) optimizes something the
ladder doesn't measure; gains that don't generalize beyond the mirror don't move rating.
**Conclusion: stop shipping mirror-selected increments. The next changes must target
generalization against the FIELD** — diverse-opponent training and diverse-panel
evaluation — and/or a representation step-change (id-bag state features), not another
2-3% mirror edge.

**#7 vs #8 — the panel pivot is ladder-confirmed.** it12 (diversified RL pool + frozen
champions as opponents + Latias held out of training) reads **535.3** vs v2-it30's
**477**, both current: **+58.3 concurrent**, well clear of every noise bound seen so far
(6.6-pt tie, 136-pt @1h spread). This is the **second confirmed real gain** in the whole
project (after #4 vs #5's +105), and the **first time the eval-panel instrument correctly
called a ladder outcome** where mirror selection had just failed on the same lineage.
Locally, it12 was picked because it improved EVERY headroom row including the sealed
Latias holdout (+3.2) while a sibling checkpoint (it16) regressed on that same holdout
despite gaining vs trained-on opponents — i.e. the holdout caught overfitting the panel
would have missed by aggregate alone. Confirms: (1) diversify the RL opponent POOL, not
just add more self-play iterations, (2) select with a holdout-containing panel, never
the mirror, (3) `v2-it30`'s repeated re-reads (493.2 weekend -> 477 @4d) reconfirm
absolute-rating drift yet again — always re-read the CONCURRENT baseline, not a
remembered number.

**Cycle 2 (2026-07-21/22) — the recipe saturates after one cycle; it12 holds.** Continued
diversified RL from it12 (16 iters, it12 added to its own opponent pool so the next
generation had to beat the reigning champion, not just its ancestors). Panel vs it12
(250 games/opponent): candidates it10/it14/it16 ALL lost the direct head-to-head to it12
(46.0%/44.8%/47.6%) despite mixed holdout numbers (it14 +2.4 on Latias but the worst
head-to-head of the three; it16, the most-trained checkpoint, repeated the exact
overfitting pattern from cycle 1 — holdout −12.8 as training continued). **No ship. it12
remains champion.** One run in this cycle also hung for ~6 hours (workers alive but <1
CPU-second consumed — a real deadlock, not slowness); resumed cleanly from the last valid
checkpoint (it14) with no code changes, so likely a transient multiprocessing/Windows
issue rather than a reproducible bug — watch for recurrence.
Conclusion: the diversified-pool RECIPE was the source of the cycle-1 gain, not RL
iteration count in general — simply running more iterations against the same fixed
opponent pool does not compound. Getting a further gain likely needs a NEW ingredient
(wider/rotating opponent pool, or the v3 representation rebuild), not more of the same.

**Cycle 3 (2026-07-22/23) — pool WIDENING also fails; it12 holds, recipe plateaued.**
Tested cycle 2's own recommendation directly: added a genuinely new archetype
(Terapagos ex, Colorless/Lightning, validated legal + 91.7% heuristic-vs-random, and
confirmed NOT pathologically long games) to the training pool, 16 more iters from it12.
Panel vs it12 (250 games/opponent, same panel as cycles 1-2): it5 47.6% head-to-head
(holdout +1.6), it8 50.0% (dead tie; holdout -6.4), it16 44.0% (worst; holdout -8.4,
worst). **No candidate beats it12 head-to-head. No ship.** it16 repeats the exact
overfitting signature for the THIRD cycle running (most-trained checkpoint = worst
holdout = worst head-to-head) — a reliable pattern now, not noise.
**Two consecutive null cycles on two different variables (more iterations; a wider
pool) = the diversified-pool-RL recipe has plateaued at the current champion** (it12,
130k params, fv=2 identity-embedding features). The cycle-1 gain came from a one-time
structural change (frozen champions as training opponents); neither more compute nor
incremental pool-widening reproduces it. A further gain most likely needs a genuinely
different lever: the v3 id-bag representation (hand/deck/discard CONTENTS as features,
not just counts — what the official sample notebook and the ~1000-rated field use), not
another RL cycle on the same architecture.

## Local proxies (NOT ladder truth — recorded to check calibration later)

| agent | vs random | vs heuristic | vs frozen BC | notes |
|---|---|---|---|---|
| heuristic | ~100% | — | — | 601 on ladder |
| search (h=3) | — | 75% | — | **456 on ladder.** local number was circular |
| BC clone | 95% | 37–47% | — | 87.2% val agreement; faithful but slightly worse than teacher |
| RL it12 | — | 61.5% | 68% | contaminated vs heuristic; self-play 59→69% |

The `search` row is the cautionary tale: **75% local → 456 ladder.** Any future row
claiming a big local win against our own heuristic should be assumed circular until the
ladder says otherwise.
