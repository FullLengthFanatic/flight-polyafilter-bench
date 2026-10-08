from pafbench.report import parse_gnu_time, time_row


def test_parse_gnu_time(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n\tMaximum resident set size (kbytes): 2097152\n\tExit status: 0\n")
    r = parse_gnu_time(p)
    assert r == {"elapsed_s": 3723.0, "max_rss_gib": 2.0, "exit_status": 0}
    p.write_text("\tElapsed (wall clock) time (h:mm:ss or m:ss): 2:03.50\n\tMaximum resident set size (kbytes): 1048576\n\tExit status: 137\n")
    assert parse_gnu_time(p) == {"elapsed_s": 123.5, "max_rss_gib": 1.0, "exit_status": 137}


def test_time_row_with_data():
    parsed = {"elapsed_s": 3723.0, "max_rss_gib": 2.0, "exit_status": 0}
    row = time_row("createDB", parsed)
    assert row == "| createDB | 1.03 | 2.0 | 0 |"


def test_time_row_empty_file():
    parsed = {}
    row = time_row("createTRANS_i", parsed)
    assert row == "| createTRANS_i | running | running | running |"
