import pytest

from pafbench.flags import Setting, flag_reference, removed_from_flags, setting_tag
from pafbench.transcripts import load_transcripts
from tests.synth import run_original

SETTINGS = [
    Setting(100, 5, 0, False),
    Setting(300, 8, 1, False),
    Setting(300, 8, 1, True),
    Setting(150, 3, 2, True),
    Setting(50, 10, 0, True),
    Setting(200, None, 0, True),
    Setting(200, None, 0, False),
]


@pytest.mark.parametrize("st", SETTINGS, ids=setting_tag)
def test_flags_match_original(synth, st):
    want = run_original(synth, st.covlen, st.min_len, st.mism, st.introns)
    tx = load_transcripts(synth.gtf)
    got = set()
    for ref in ("chrA", "chrB"):
        df = flag_reference(synth.bam, synth.fasta, ref, tx, [st])
        got |= removed_from_flags(df, setting_tag(st))
    assert got == want


def test_some_setting_removes_reads(synth):
    assert len(run_original(synth, 300, 8, 1, True)) > 0
