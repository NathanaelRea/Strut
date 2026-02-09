from math import asin, floor, sin
from os import abort
from python import Python, PythonObject

from strut_io import py_len


fn _is_list(obj: PythonObject) raises -> Bool:
    var builtins = Python.import_module("builtins")
    return Bool(builtins.isinstance(obj, builtins.list))


fn _is_dict(obj: PythonObject) raises -> Bool:
    var builtins = Python.import_module("builtins")
    return Bool(builtins.isinstance(obj, builtins.dict))


fn _as_list(obj: PythonObject) raises -> PythonObject:
    if _is_list(obj):
        return obj
    var builtins = Python.import_module("builtins")
    var lst = builtins.list()
    lst.append(obj)
    return lst


fn _parse_path_values(data: PythonObject, ts: PythonObject) raises -> PythonObject:
    var pathlib = Python.import_module("pathlib")
    var re = Python.import_module("re")
    var builtins = Python.import_module("builtins")

    var values_path = ""
    if ts.__contains__("values_path"):
        values_path = String(ts["values_path"])
    elif ts.__contains__("path"):
        values_path = String(ts["path"])
    else:
        abort("Path time_series missing values_path")

    var path_obj = pathlib.Path(values_path)
    if not Bool(path_obj.is_absolute()) and data.__contains__("__strut_case_dir"):
        var case_dir = pathlib.Path(String(data["__strut_case_dir"]))
        var candidate = case_dir.joinpath(values_path)
        if Bool(candidate.exists()):
            path_obj = candidate
    if not Bool(path_obj.exists()):
        abort("Path time_series values_path not found: " + values_path)

    var text = String(path_obj.read_text())
    var token_pattern = "[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eEdD][-+]?\\d+)?"
    var tokens = re.findall(token_pattern, text)
    if py_len(tokens) == 0:
        abort("Path time_series values_path had no numeric values: " + String(path_obj))

    var values = builtins.list()
    for i in range(py_len(tokens)):
        var token = String(tokens[i]).replace("D", "E").replace("d", "e")
        values.append(builtins.float(token))
    return values


fn _normalize_path_time_series(data: PythonObject, ts_list: PythonObject) raises:
    for i in range(py_len(ts_list)):
        var ts = ts_list[i]
        if not _is_dict(ts):
            abort("time_series entry must be object")
        var typ = String(ts.get("type", ""))
        if typ == "PathFile":
            ts["type"] = "Path"
            typ = "Path"
        if typ != "Path":
            continue
        if ts.__contains__("values") and (
            ts.__contains__("values_path") or ts.__contains__("path")
        ):
            abort("Path time_series cannot specify both values and values_path/path")
        if not ts.__contains__("values"):
            ts["values"] = _parse_path_values(data, ts)


fn parse_time_series(data: PythonObject) raises -> PythonObject:
    if not data.__contains__("time_series"):
        var builtins = Python.import_module("builtins")
        return builtins.list()
    var ts_obj = data["time_series"]
    if _is_list(ts_obj):
        _normalize_path_time_series(data, ts_obj)
        return ts_obj
    if _is_dict(ts_obj):
        var ts_list = _as_list(ts_obj)
        _normalize_path_time_series(data, ts_list)
        return ts_list
    abort("time_series must be list or object")
    var builtins = Python.import_module("builtins")
    return builtins.list()


fn find_time_series(ts_list: PythonObject, tag: Int) raises -> Int:
    var count = py_len(ts_list)
    for i in range(count):
        var entry = ts_list[i]
        if Int(entry["tag"]) == tag:
            return i
    return -1


fn _path_time(ts: PythonObject, idx: Int) raises -> Float64:
    if ts.__contains__("dt"):
        return Float64(idx) * Float64(ts["dt"])
    return Float64(ts["time"][idx])


fn eval_time_series(ts: PythonObject, t: Float64) raises -> Float64:
    var typ = String(ts["type"])
    var factor = Float64(ts.get("factor", 1.0))
    if typ == "Constant":
        return factor
    if typ == "Linear":
        return factor * t
    if typ == "Path":
        if not ts.__contains__("values"):
            abort("Path time_series missing values")
        var values = ts["values"]
        var n = py_len(values)
        if n == 0:
            return 0.0
        if ts.__contains__("dt") and not ts.__contains__("time"):
            var dt = Float64(ts["dt"])
            if dt <= 0.0:
                abort("Path time_series dt must be > 0")
            var start_time = Float64(ts.get("start_time", 0.0))
            if t < start_time:
                return 0.0
            var incr = (t - start_time) / dt
            var incr1 = Int(floor(incr))
            var incr2 = incr1 + 1
            if incr2 >= n:
                if Bool(ts.get("use_last", False)):
                    return factor * Float64(values[n - 1])
                return 0.0
            var v1 = Float64(values[incr1])
            var v2 = Float64(values[incr2])
            return factor * (v1 + (v2 - v1) * (incr - Float64(incr1)))
        if ts.__contains__("time"):
            if py_len(ts["time"]) != n:
                abort("Path time_series time/values length mismatch")
        var time0 = _path_time(ts, 0)
        if t < time0:
            return 0.0
        if t == time0:
            return factor * Float64(values[0])
        var last_idx = n - 1
        var time_last = _path_time(ts, last_idx)
        if t > time_last:
            if Bool(ts.get("use_last", False)):
                return factor * Float64(values[last_idx])
            return 0.0
        if t == time_last:
            return factor * Float64(values[last_idx])
        for i in range(last_idx):
            var t1 = _path_time(ts, i)
            var t2 = _path_time(ts, i + 1)
            if t2 <= t1:
                abort("Path time_series time values must be increasing")
            if t == t2:
                return factor * Float64(values[i + 1])
            if t > t1 and t < t2:
                var v1 = Float64(values[i])
                var v2 = Float64(values[i + 1])
                return factor * (v1 + (v2 - v1) * (t - t1) / (t2 - t1))
        return 0.0
    if typ == "Trig":
        if not ts.__contains__("t_start") or not ts.__contains__("t_finish"):
            abort("Trig time_series missing t_start/t_finish")
        if not ts.__contains__("period"):
            abort("Trig time_series missing period")
        var t_start = Float64(ts["t_start"])
        var t_finish = Float64(ts["t_finish"])
        if t < t_start or t > t_finish:
            return 0.0
        var period = Float64(ts["period"])
        if period == 0.0:
            period = 2.0 * asin(1.0)
        var phase_shift = Float64(ts.get("phase_shift", 0.0))
        var zero_shift = Float64(ts.get("zero_shift", 0.0))
        if factor == 0.0 and zero_shift != 0.0:
            abort("Trig time_series zero_shift requires nonzero factor")
        var twopi = 4.0 * asin(1.0)
        var phi = phase_shift
        if factor != 0.0:
            phi = phase_shift - period / twopi * asin(zero_shift / factor)
        return factor * sin(twopi * (t - t_start) / period + phi) + zero_shift
    abort("unsupported time_series type: " + typ)
    return 0.0
