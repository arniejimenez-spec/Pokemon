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
| 6 | 2026-07-17 | policy | RL it18: +24 iters from it12; 58.0% vs it12 @300 games (it13 54.3, it24 52.3) | `b4a243b` | policy_rl_it18.npz | pending | challenger vs it12 (concurrent floor) |

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
