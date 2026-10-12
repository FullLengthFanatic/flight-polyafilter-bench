import hashlib

import pysam

from tests.synth import run_original


def test_original_runs_and_removes_something(synth):
    removed = run_original(synth, 300, 8, 1, True)
    assert len(removed) > 0
    # antisense reads in g1 start at 1200, 1500, ... and are never removed
    assert not any(k[1] & 16 and k[2] == "chrA" and 1200 <= k[3] < 3900 and (k[3] - 1200) % 300 == 0 for k in removed)


def test_removed_keys_detects_subsequence(synth):
    removed = run_original(synth, 100, 5, 0, False)
    assert all(len(k) == 5 for k in removed)


def test_build_keeps_its_genome_and_records(synth):
    # digests of seed 7, taken before the genome and BAM helpers of build() were shared with build_10x
    with pysam.AlignmentFile(str(synth.bam)) as fh:
        records = "\n".join(a.to_string() for a in fh.fetch(until_eof=True))
    digests = [hashlib.sha256(b).hexdigest() for b in (records.encode(), synth.fasta.read_bytes(), synth.gtf.read_bytes())]
    assert digests == ["4f3ac8587685d86ca1691e19b864ede6eb28b65d746ed2f54a0f0456a210dae3",
                       "164823c37d0c156fe86dee45fc7271d2d6acf25de7e3f502833d2c28199346de",
                       "4bfe2c457e38084575ba7d5d5967f5b1c5a9d6eec89702e49a7572b4011dfd8c"]
