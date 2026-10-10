import os
import signal
import time

import pytest

from pafbench.pool import map_forked

STATE = {}


def _init(a, b):
    STATE["v"] = (a, b)


def _square(x):
    return x * x


def _slow_first(x):
    time.sleep(0.3 if x == 0 else 0)
    return x


def _read_state(_):
    return STATE["v"][0], STATE["v"][1]()


def _suicide(x):
    if x == 2:
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(0.1)
    return x


def test_results_in_job_order():
    assert map_forked(_slow_first, list(range(6)), 3) == list(range(6))
    assert map_forked(_square, [3, 1, 2], 2) == [9, 1, 4]


def test_initializer_and_unpicklable_initargs_reach_workers():
    unpicklable = lambda: 1  # noqa: E731  (fork hands it over without pickling)
    res = map_forked(_read_state, [0, 1, 2], 2, initializer=_init, initargs=(7, unpicklable))
    assert res == [(7, 1)] * 3


def test_killed_worker_raises_instead_of_hanging():
    def on_alarm(signum, frame):
        raise AssertionError("map_forked hung after a worker was killed")

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(30)
    t0 = time.monotonic()
    try:
        with pytest.raises(Exception):
            map_forked(_suicide, list(range(6)), 2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert time.monotonic() - t0 < 10
