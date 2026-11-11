
from time import perf_counter
from collections.abc import Callable
from typing import Any


def benchmark(
    function: Callable,
    *args: Any,
    repeats: int = 3,
    **kwargs: Any
) -> dict:
    """Benchmark the elapsed runtime of a function."""

    if repeats < 1:
        raise ValueError("repeats must be positive")

    runtimes = []

    for _ in range(repeats):
        start = perf_counter()

        function(*args, **kwargs)

        elapsed = perf_counter() - start
        runtimes.append(elapsed)

    return {
        "mean_seconds": sum(runtimes) / repeats,
        "min_seconds": min(runtimes),
        "max_seconds": max(runtimes),
        "runs": repeats
    }
