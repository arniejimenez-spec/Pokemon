"""Append a submission to EXPERIMENTS.md, or record its ladder rating later.

The point: never again wonder which agent a ladder score belonged to. Reads the
MANIFEST.json out of the built bundle so the row can't drift from what shipped.

Usage:
    python log_submission.py --note "RL it12"                 # after make_submission.py
    python log_submission.py --rating 640 --row 3             # fill in a rating later
    python log_submission.py --list
"""
import argparse
import json
import os
import re
import tarfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "EXPERIMENTS.md")
OUT = os.path.join(ROOT, "submission.tar.gz")

HEADER = """# Experiment log

Ladder results for every submission. **The ladder is the only ground truth** — local
win rates against our own heuristic have twice failed to predict it (see README).

**Reading the ratings:** μ starts at 600. Two *byte-identical* submissions read 399 and
535 after one hour, so **differences under ~100 points mean nothing** without a day+ of
games. Only the latest 2 submissions are tracked for final scoring.

| # | date | agent | note | git | model | rating | verdict |
|---|------|-------|------|-----|-------|--------|---------|
"""


def read_manifest() -> dict:
    if not os.path.exists(OUT):
        raise SystemExit("no submission.tar.gz — run make_submission.py first")
    with tarfile.open(OUT) as t:
        f = t.extractfile("MANIFEST.json")
        return json.load(f)


def ensure_log():
    if not os.path.exists(LOG):
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(HEADER)


def rows() -> list[str]:
    ensure_log()
    with open(LOG, encoding="utf-8") as f:
        return [l for l in f.read().splitlines() if re.match(r"^\|\s*\d+\s*\|", l)]


def append(note: str, rating: str):
    man = read_manifest()
    n = len(rows()) + 1
    row = (f"| {n} | {time.strftime('%Y-%m-%d')} | {man['agent']} | "
           f"{note or man.get('note','')} | `{man['git']}` | "
           f"{man.get('model','—')} | {rating or 'pending'} |  |")
    ensure_log()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(row)
    print(f"\nappended row {n} to EXPERIMENTS.md")


def set_rating(row_id: int, rating: str, verdict: str):
    ensure_log()
    with open(LOG, encoding="utf-8") as f:
        lines = f.read().splitlines()
    hit = False
    for i, l in enumerate(lines):
        m = re.match(r"^\|\s*(\d+)\s*\|", l)
        if m and int(m.group(1)) == row_id:
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            cells[6] = rating
            if verdict:
                cells[7] = verdict
            lines[i] = "| " + " | ".join(cells) + " |"
            hit = True
            print(lines[i])
            break
    if not hit:
        raise SystemExit(f"no row {row_id}")
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    ap.add_argument("--rating", default="")
    ap.add_argument("--row", type=int, help="row to attach --rating to")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        ensure_log()
        print(open(LOG, encoding="utf-8").read())
    elif args.row:
        set_rating(args.row, args.rating, args.verdict)
    else:
        append(args.note, args.rating)


if __name__ == "__main__":
    main()
