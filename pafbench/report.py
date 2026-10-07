def parse_gnu_time(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            parts = [float(x) for x in line.rsplit(" ", 1)[1].split(":")]
            secs = 0.0
            for p in parts:
                secs = secs * 60 + p
            out["elapsed_s"] = secs
        elif line.startswith("Maximum resident set size (kbytes)"):
            out["max_rss_gb"] = int(line.rsplit(" ", 1)[1]) / 1024 ** 2
        elif line.startswith("Exit status"):
            out["exit_status"] = int(line.rsplit(" ", 1)[1])
    return out
