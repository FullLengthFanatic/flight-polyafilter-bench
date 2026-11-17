import tomllib

from .config import (LABEL_CUTOFFS, LONG_READ_MIN_MAPQ, LONG_READ_RUN_MIN, LONG_READ_RUN_TOLERANCE, MIN_AFTER, REPO,
                     SHORT_READ_CLIP_MIN, SHORT_READ_CLIP_PURITY, SITE_WINDOW, UNIQUE_MAPQ)


def definition():
    """The internal-priming definition of this benchmark, as docs/definition.json holds it (see docs/definition.md)."""
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    return {
        "source": "flight-polyafilter-bench",
        "version": version,
        "site": {
            "coordinate": "0-based boundary; plus strand: first reference base after the last cDNA base; "
                          "minus strand: the last cDNA base is the base at the site",
            "strand": "read strand; reads are transcript-sense",
            "short_read": {"anchor": "alignment_end", "clip_min": SHORT_READ_CLIP_MIN,
                           "clip_purity_min": SHORT_READ_CLIP_PURITY, "tag": "pa"},
            "long_read": {"anchor": "a_run_start", "run_min": LONG_READ_RUN_MIN, "run_tolerance": LONG_READ_RUN_TOLERANCE,
                          "min_mapq": LONG_READ_MIN_MAPQ, "min_after": dict(MIN_AFTER)},
        },
        "label": {"window": SITE_WINDOW, **LABEL_CUTOFFS},
        "rate": {"unit": "molecule", "unique_mapq": UNIQUE_MAPQ,
                 "share_all": "templated / (templated + nontemplated + ambiguous)",
                 "share_resolved": "templated / (templated + nontemplated)"},
    }
