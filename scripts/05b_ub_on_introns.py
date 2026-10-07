"""Stage 05b: share of exonic and intronic primary reads carrying UB in 1179 (include-introns false), chr1."""
import json

import pysam

from pafbench.config import LIBRARIES, OUT


def main():
    n = {"E_total": 0, "E_with_UB": 0, "N_total": 0, "N_with_UB": 0}
    with pysam.AlignmentFile(str(LIBRARIES["1179"])) as bam:
        for a in bam.fetch("chr1"):
            if a.is_secondary or a.is_supplementary or not a.has_tag("RE"):
                continue
            region = a.get_tag("RE")
            if region in ("E", "N"):
                n[f"{region}_total"] += 1
                n[f"{region}_with_UB"] += int(a.has_tag("UB"))
    out = OUT / "1179"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ub_on_introns.json").write_text(json.dumps(n, indent=1))
    print(n)


if __name__ == "__main__":
    main()
