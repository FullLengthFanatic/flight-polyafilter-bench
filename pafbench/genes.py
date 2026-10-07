from collections import defaultdict

import numpy as np


def load_gene_segments(gtf):
    """{(ref, strd): (bounds, labels)}: genome position -> ';'-joined gene_ids covering it on that strand."""
    spans = defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene":
                continue
            gid = f[8].split('gene_id "', 1)[1].split('"', 1)[0]
            spans[(f[0], f[6] == "+")].append((int(f[3]) - 1, int(f[4]), gid))
    out = {}
    for key, lst in spans.items():
        events = sorted([(s, 1, g) for s, _, g in lst] + [(e, -1, g) for _, e, g in lst])
        bounds, labels, active, i = [], [], set(), 0
        while i < len(events):
            pos = events[i][0]
            while i < len(events) and events[i][0] == pos:
                _, kind, g = events[i]
                if kind == 1:
                    active.add(g)
                else:
                    active.discard(g)
                i += 1
            bounds.append(pos)
            labels.append(";".join(sorted(active)))
        out[key] = (np.asarray(bounds, np.int64), labels)
    return out


def genes_at(segments, ref, strd, pos):
    entry = segments.get((ref, strd))
    if entry is None:
        return ""
    bounds, labels = entry
    i = int(np.searchsorted(bounds, pos, side="right")) - 1
    return labels[i] if i >= 0 else ""
