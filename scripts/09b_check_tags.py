"""Stage 09b: do CB/UB/pa/ts survive STAR realignment of unmapped reads to SIRV Set 4?"""
import argparse
import json
import random
from collections import Counter

import pysam

from pafbench.config import OUT
from pafbench.spikein import module_of, tag_survival

TAGS = ("CB", "UB", "pa", "ts")


def _tags(a):
    return {t: a.get_tag(t) for t in TAGS if a.has_tag(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    base = OUT / args.library / "spikein"
    rng = random.Random(0)
    sample, modules, seen = [], Counter(), 0
    with pysam.AlignmentFile(str(base / "star_Aligned.out.bam")) as fh:
        for a in fh.fetch(until_eof=True):
            if a.is_secondary or a.is_supplementary or a.is_unmapped:
                continue
            modules[module_of(a.reference_name)] += 1
            seen += 1
            item = (a.query_name, _tags(a))
            if len(sample) < args.n:
                sample.append(item)
            else:
                j = rng.randrange(seen)
                if j < args.n:
                    sample[j] = item
    after = dict(sample)
    before = {}
    with pysam.AlignmentFile(str(base / "unmapped.bam"), check_sq=False) as fh:
        for a in fh.fetch(until_eof=True):
            if a.query_name in after:
                before[a.query_name] = _tags(a)
    res = {"aligned_primary": seen, "modules": dict(modules), "survival": tag_survival(before, after),
           "log_final": (base / "star_Log.final.out").read_text()}
    (base / "tag_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["survival"]))


if __name__ == "__main__":
    main()
