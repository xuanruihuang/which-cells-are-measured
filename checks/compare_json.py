"""Compare two result files value by value (JSON, or NPZ arrays): used to check a rerun from this repository against
the output the paper's numbers were taken from.
Strings are not compared under the key "rules" (the scripts store their docstring there, and the release edited
docstrings where they named server paths). Numbers must agree to a relative tolerance (default 1e-9).
Usage: compare_json.py <original> <rerun> [--rtol R] [--ignore KEY ...]
Prints the number of values compared and every difference; exit status 1 on any difference."""
import argparse
import json
import math
import sys

import numpy as np


def walk(a, b, path, out, rtol, ignore):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in ignore:
                continue
            if k not in a or k not in b:
                out["diff"].append("%s/%s: only in %s" % (path, k, "original" if k in a else "rerun"))
                continue
            walk(a[k], b[k], "%s/%s" % (path, k), out, rtol, ignore)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out["diff"].append("%s: length %d vs %d" % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), out, rtol, ignore)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        out["n"] += 1
        if math.isnan(a) and math.isnan(b):
            return
        if not (a == b or abs(a - b) <= rtol * max(abs(a), abs(b))):
            out["diff"].append("%s: %r vs %r" % (path, a, b))
    else:
        out["n"] += 1
        if a != b:
            out["diff"].append("%s: %r vs %r" % (path, a, b))


def load(p):
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=True)
        return {k: z[k].tolist() for k in z.files}
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original"); ap.add_argument("rerun")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--ignore", nargs="*", default=["rules"])
    a = ap.parse_args()
    out = {"n": 0, "diff": []}
    walk(load(a.original), load(a.rerun), "", out, a.rtol, set(a.ignore))
    for d in out["diff"][:50]:
        print("DIFF", d)
    print("%s: %d values compared, %d differ -> %s" % (a.rerun, out["n"], len(out["diff"]),
                                                        "IDENTICAL" if not out["diff"] else "DIFFERENT"))
    sys.exit(1 if out["diff"] else 0)


if __name__ == "__main__":
    main()
