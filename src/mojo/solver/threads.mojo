"""Worker thread count shared by every threaded phase of a solve.

`STRUT_THREADS=N` caps element sweeps and sparse factorization together, so
`STRUT_THREADS=1` is a true single-threaded run. Unset, the solver uses the
performance cores.
"""

from os import abort, getenv
from sys import num_performance_cores


alias THREADS_ENVIRONMENT_KEY = "STRUT_THREADS"


fn solver_thread_count() -> Int:
    var override = getenv(THREADS_ENVIRONMENT_KEY)
    if override == "":
        return max(1, num_performance_cores())
    try:
        var threads = atol(StringSlice(override))
        if threads > 0:
            return threads
    except:
        pass
    abort(THREADS_ENVIRONMENT_KEY + " must be a positive integer")
    return 1
