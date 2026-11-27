"""Generate the missing time series for an OpenSees Tcl ground motion."""

from bisect import bisect_right
import math
from pathlib import Path


def trapezoidal_path(series: dict, dt: float, *, differentiate: bool) -> dict | None:
    """Sample the series using TrapezoidalTimeSeriesIntegrator's endpoint rules."""
    kind = series["type"]
    factor = series.get("factor", 1.0)
    start = 0.0
    duration = 0.0
    if kind == "Trig":
        start = series["t_start"]
        finish = series["t_finish"]
        duration = finish - start
        period = series["period"] or math.tau
        phase = series.get("phase_shift", 0.0)
        zero_shift = series.get("zero_shift", 0.0)
        if factor:
            phase -= period / math.tau * math.asin(zero_shift / factor)

        def value(t):
            if start <= t <= finish:
                return (
                    factor * math.sin(math.tau * (t - start) / period + phase)
                    + zero_shift
                )
            return 0.0

    elif kind == "Path":
        samples = series.get("values")
        if samples is None:
            samples = [
                float(x) for x in Path(series["values_path"]).read_text().split()
            ]
        times = series.get("time")
        use_last = series.get("use_last", False)
        last = len(samples) - 1
        if not samples:
            return None
        if times is None:
            start = series.get("start_time", 0.0)
            step = series["dt"]
            duration = len(samples) * step

            def value(t):
                if t < start:
                    return 0.0
                index = (t - start) / step
                left = math.floor(index)
                if left >= last:
                    return factor * samples[last] if use_last else 0.0
                return factor * (
                    samples[left] + (samples[left + 1] - samples[left]) * (index - left)
                )

        else:
            duration = times[-1]

            def value(t):
                if t < times[0]:
                    return 0.0
                if t >= times[last]:
                    return (
                        factor * samples[last] if t == times[last] or use_last else 0.0
                    )
                left = bisect_right(times, t) - 1
                return factor * (
                    samples[left]
                    + (samples[left + 1] - samples[left])
                    * (t - times[left])
                    / (times[left + 1] - times[left])
                )

    else:
        # Constant and Linear report zero duration and start time.
        def value(t):
            return factor if kind == "Constant" else factor * t

    if dt <= 0.0:
        return None
    values = []
    t = start
    previous = result = 0.0
    for _ in range(int(duration / dt + 1.0)):
        current = value(t)
        if differentiate:
            result = 2.0 * (current - previous) / dt - result
        else:
            result += 0.5 * dt * (previous + current)
        values.append(result)
        previous = current
        t += dt
    if not values:
        return None
    return {
        "type": "Path",
        "dt": dt,
        "values": values,
        "start_time": start,
        "use_last": True,
    }
