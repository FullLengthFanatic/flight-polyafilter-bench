"""Process pool for the stages: fork context, fails fast when a worker dies."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def map_forked(fn, jobs, workers, initializer=None, initargs=()):
    """fn over jobs, one job per task, results in job order; BrokenProcessPool if a worker is killed."""
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"), initializer=initializer,
                             initargs=initargs) as pool:
        return list(pool.map(fn, jobs, chunksize=1))
