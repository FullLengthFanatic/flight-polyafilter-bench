"""PolyASite 3.0 proximity of priming sites (spec 4.1). Reported next to the labels, never used to validate them: the atlas
comes from 10x 3' reads primed by the same oligo(dT), the v3.0 documentation describes no filter on A content downstream of a
site, and on chr21 and chr22 the clusters file is not depleted of A-rich downstream sequence (our observation: 9,544 of
465,261 sites, 2.05%, have 14 or more A's in the 20 nt downstream, against 3,282, 0.71%, for the same sites moved 1,000 nt
upstream)."""
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

ATLAS_COLUMNS = ["chrom", "start", "end", "id", "score", "strand", "tissues_pct", "protocols", "stringency", "class", "signals"]
BINS = ("0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand")
NULL_SHIFT = 1000            # chance level: the same sites moved this many nt upstream in transcript sense
NULL_LABEL = "null_shift1kb"


def load_polyasite(path, min_score=None):
    """{(ref, strd): sorted unique int64 ndarray} of the representative cleavage sites of a PolyASite 3.0 atlas BED (the pos
    of the id chrom:pos:strand, 1-based per the atlas documentation), kept when score (column 5, mean RPM) >= min_score.

    Converted to the priming_site boundary (last cDNA base at site - 1 on plus, at site on minus): pos on plus, pos + 1 on
    minus. With the documented reading on both strands (the base [pos - 1, pos) is the last templated base), the mode of the
    signed distance from tail-primed 1417 pa sites (a20 <= 4) to the atlas is 0 on plus but 2 nt downstream on minus, and from
    GENCODE 44 transcript ends 1 nt upstream on plus but 1 nt downstream on minus; pos + 1 on minus makes the strands agree.
    """
    reader = pcsv.open_csv(path, read_options=pcsv.ReadOptions(column_names=ATLAS_COLUMNS, block_size=1 << 26),
                           parse_options=pcsv.ParseOptions(delimiter="\t"),
                           convert_options=pcsv.ConvertOptions(include_columns=["chrom", "id", "score", "strand"],
                                                               column_types={"chrom": pa.string(), "id": pa.string(),
                                                                             "score": pa.float64(), "strand": pa.string()}))
    parts = defaultdict(list)
    for b in reader:
        if min_score is not None:
            b = b.filter(pc.greater_equal(b.column("score"), min_score))
        pos = pc.cast(pc.list_element(pc.split_pattern(b.column("id"), ":"), 1), pa.int64()).to_numpy()
        key = pc.binary_join_element_wise(b.column("chrom"), b.column("strand"), "\t").dictionary_encode()
        code = key.indices.to_numpy(zero_copy_only=False)
        for k, name in enumerate(key.dictionary.to_pylist()):
            parts[name].append(pos[code == k])
    out = {}
    for name, chunks in parts.items():
        ref, strand = name.split("\t")
        out[(ref, strand == "+")] = np.unique(np.concatenate(chunks) + (strand == "-"))
    return out


def nearest_signed(sites, strd, ends):
    """Signed distance from each site to the nearest of ends (sorted atlas sites of the same ref and strand, strd True on
    plus); d > 0 when that atlas site lies downstream in transcript sense (spikein.label_spike_site's rule; equal |d|: the
    positive one). All NaN when ends is None or empty."""
    sites = np.asarray(sites, np.int64)
    if ends is None or len(ends) == 0:
        return np.full(len(sites), np.nan)
    sign = 1 if strd else -1
    i = np.searchsorted(ends, sites)
    far = np.int64(1) << 62
    right = np.where(i < len(ends), ends[np.minimum(i, len(ends) - 1)] - sites, far) * sign
    left = np.where(i > 0, ends[np.maximum(i - 1, 0)] - sites, -far) * sign
    a, b = np.abs(right), np.abs(left)
    return np.where(a < b, right, np.where(b < a, left, np.maximum(right, left))).astype(np.float64)


def site_distances(mol, atlas, upstream=0):
    """nearest_signed for every row of mol (ref, strand '+'/'-', site) against load_polyasite's dict, in row order, after
    moving each site by upstream nt toward the 5' end in transcript sense (lower coordinate on plus, higher on minus)."""
    d = np.full(len(mol), np.nan)
    site = mol.site.to_numpy(np.int64)
    for (ref, strand), idx in mol.groupby(["ref", "strand"], sort=False).indices.items():
        strd = strand == "+"
        d[idx] = nearest_signed(site[idx] - upstream if strd else site[idx] + upstream, strd, atlas.get((ref, strd)))
    return d


def distance_bin(d):
    """BINS name of each distance by |d|; NaN (no atlas site on the ref and strand, e.g. a contig the atlas lacks) is the
    last bin."""
    a = np.abs(np.asarray(d, np.float64))
    return np.select([a <= 10, a <= 50, a <= 200, a > 200], BINS[:4], BINS[4]).astype(object)
