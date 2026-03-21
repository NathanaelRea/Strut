from collections import List
from math import asin, floor, sin
from os import abort
from tag_types import TimeSeriesTypeTag


struct TimeSeriesInput(Movable, ImplicitlyCopyable):
    var tag: Int
    var type_tag: Int
    var factor: Float64
    var has_dt: Bool
    var dt: Float64
    var start_time: Float64
    var use_last: Bool
    var values_offset: Int
    var values_count: Int
    var time_offset: Int
    var time_count: Int
    var t_start: Float64
    var t_finish: Float64
    var period: Float64
    var phase_shift: Float64
    var zero_shift: Float64

    fn __init__(out self):
        self.tag = -1
        self.type_tag = TimeSeriesTypeTag.Unknown
        self.factor = 1.0
        self.has_dt = False
        self.dt = 0.0
        self.start_time = 0.0
        self.use_last = False
        self.values_offset = 0
        self.values_count = 0
        self.time_offset = 0
        self.time_count = 0
        self.t_start = 0.0
        self.t_finish = 0.0
        self.period = 0.0
        self.phase_shift = 0.0
        self.zero_shift = 0.0


fn find_time_series_input(ts_list: List[TimeSeriesInput], tag: Int) -> Int:
    for i in range(len(ts_list)):
        if ts_list[i].tag == tag:
            return i
    return -1


fn _path_time_input(
    ts: TimeSeriesInput, idx: Int, path_time: List[Float64]
) raises -> Float64:
    if ts.has_dt:
        return Float64(idx) * ts.dt
    if idx < 0 or idx >= ts.time_count:
        abort("Path time_series time index out of range")
    return path_time[ts.time_offset + idx]


fn eval_time_series_input(
    ts: TimeSeriesInput,
    t: Float64,
    path_values: List[Float64],
    path_time: List[Float64],
) raises -> Float64:
    if ts.type_tag == TimeSeriesTypeTag.Constant:
        return ts.factor
    if ts.type_tag == TimeSeriesTypeTag.Linear:
        return ts.factor * t
    if ts.type_tag == TimeSeriesTypeTag.Path:
        var n = ts.values_count
        if n == 0:
            return 0.0
        if ts.has_dt and ts.time_count == 0:
            if ts.dt <= 0.0:
                abort("Path time_series dt must be > 0")
            if t < ts.start_time:
                return 0.0
            var incr = (t - ts.start_time) / ts.dt
            var incr1 = Int(floor(incr))
            var incr2 = incr1 + 1
            if incr2 >= n:
                if ts.use_last:
                    return ts.factor * path_values[ts.values_offset + n - 1]
                return 0.0
            var v1 = path_values[ts.values_offset + incr1]
            var v2 = path_values[ts.values_offset + incr2]
            return ts.factor * (v1 + (v2 - v1) * (incr - Float64(incr1)))

        if ts.time_count != n:
            abort("Path time_series time/values length mismatch")

        var time0 = _path_time_input(ts, 0, path_time)
        if t < time0:
            return 0.0
        if t == time0:
            return ts.factor * path_values[ts.values_offset]
        var last_idx = n - 1
        var time_last = _path_time_input(ts, last_idx, path_time)
        if t > time_last:
            if ts.use_last:
                return ts.factor * path_values[ts.values_offset + last_idx]
            return 0.0
        if t == time_last:
            return ts.factor * path_values[ts.values_offset + last_idx]
        for i in range(last_idx):
            var t1 = _path_time_input(ts, i, path_time)
            var t2 = _path_time_input(ts, i + 1, path_time)
            if t2 <= t1:
                abort("Path time_series time values must be increasing")
            if t == t2:
                return ts.factor * path_values[ts.values_offset + i + 1]
            if t > t1 and t < t2:
                var v1 = path_values[ts.values_offset + i]
                var v2 = path_values[ts.values_offset + i + 1]
                return ts.factor * (v1 + (v2 - v1) * (t - t1) / (t2 - t1))
        return 0.0
    if ts.type_tag == TimeSeriesTypeTag.Trig:
        if t < ts.t_start or t > ts.t_finish:
            return 0.0
        var period = ts.period
        if period == 0.0:
            period = 2.0 * asin(1.0)
        if ts.factor == 0.0 and ts.zero_shift != 0.0:
            abort("Trig time_series zero_shift requires nonzero factor")
        var twopi = 4.0 * asin(1.0)
        var phi = ts.phase_shift
        if ts.factor != 0.0:
            phi = ts.phase_shift - period / twopi * asin(ts.zero_shift / ts.factor)
        return ts.factor * sin(twopi * (t - ts.t_start) / period + phi) + ts.zero_shift
    abort("unsupported time_series type tag")
    return 0.0
