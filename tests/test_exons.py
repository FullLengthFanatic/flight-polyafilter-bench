import numpy as np

from pafbench.exons import exon_distances, exonic_bases, inside_exons, member_distance

EX = (np.array([100, 1000, 5000], np.int64), np.array([200, 1100, 5300], np.int64))


def test_exonic_bases_and_inside():
    assert exonic_bases(EX, 150, 1050) == 50 + 50
    assert exonic_bases(EX, 300, 900) == 0
    assert inside_exons(EX, 1010, 1090) and not inside_exons(EX, 1090, 1110) and not inside_exons(EX, 2000, 2090)


def test_forward_upstream_exon_read():
    # site 5200 (+): last cDNA base 5199 in exon 3; read wholly in exon 2
    assert exon_distances([(1010, 1090)], False, 5200, EX) == (90 + 200, 10 + 200)


def test_forward_spliced_read():
    assert exon_distances([(1050, 1100), (5000, 5040)], False, 5200, EX) == (50 + 200, 160)


def test_reverse_read():
    # site 120 (-): last cDNA base 120 in exon 1; read in exon 2 lies upstream in transcript sense
    assert exon_distances([(1010, 1090)], True, 120, EX) == (80 + 90, 80 + 10)


def test_not_exonic_falls_back_to_genome():
    assert exon_distances([(2000, 2090)], False, 5200, EX) is None      # intronic read
    assert exon_distances([(1010, 1090)], False, 3000, EX) is None      # intronic site
    d5, d3, mode = member_distance([(2000, 2090)], False, 5200, "g1", {"g1": EX}, (3200, 3110))
    assert (d5, d3, mode) == (3200, 3110, "genome")


def test_member_distance_picks_smallest_gene():
    other = (np.array([900], np.int64), np.array([5300], np.int64))      # one long exon
    d5, d3, mode = member_distance([(1010, 1090)], False, 5200, "g1;g2", {"g1": EX, "g2": other}, (4190, 4110))
    assert (d5, mode) == (290, "exon")


def test_read_passing_site_is_negative():
    assert exon_distances([(5150, 5240)], False, 5200, EX) == (50, -40)
