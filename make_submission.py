"""Build submission.tar.gz from the submission/ directory.

The bundle must have main.py and deck.csv at the top level (not nested).
Usage: python make_submission.py
"""
import os
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "submission")
OUT = os.path.join(ROOT, "submission.tar.gz")


def main():
    for required in ("main.py", "deck.csv"):
        if not os.path.exists(os.path.join(SRC, required)):
            raise SystemExit(f"missing {required} in submission/")
    if os.path.exists(OUT):
        os.remove(OUT)
    with tarfile.open(OUT, "w:gz") as tar:
        for name in os.listdir(SRC):
            if name == "__pycache__":
                continue
            tar.add(os.path.join(SRC, name), arcname=name,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    size = os.path.getsize(OUT) / (1024 * 1024)
    print(f"wrote submission.tar.gz ({size:.1f} MiB, limit 197.7)")


if __name__ == "__main__":
    main()
