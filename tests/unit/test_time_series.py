import json
import math
import tempfile
from pathlib import Path
import subprocess
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def _run_mojo_case(case_data, out_dir: Path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(case_data))
        tmp_path = Path(tmp.name)
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_mojo_case.py"),
                "--input",
                str(tmp_path),
                "--output",
                str(out_dir),
            ]
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _expected_cantilever():
    # Expected: cantilever with tip load P
    E = 200000000000.0
    I = 0.0001
    L = 1.0
    P = 1000.0
    expected_v = -P * L**3 / (3 * E * I)
    expected_theta = -P * L**2 / (2 * E * I)
    return expected_v, expected_theta


def _time_series_factor(ts, t):
    ts_type = ts["type"]
    factor = ts.get("factor", 1.0)
    if ts_type == "Constant":
        return factor
    if ts_type == "Linear":
        return factor * t
    if ts_type == "Path":
        values = ts["values"]
        if "dt" in ts and "time" not in ts:
            dt = ts["dt"]
            start_time = ts.get("start_time", 0.0)
            if t < start_time:
                return 0.0
            incr = (t - start_time) / dt
            incr1 = math.floor(incr)
            incr2 = incr1 + 1
            if incr2 >= len(values):
                return factor * values[-1] if ts.get("use_last", False) else 0.0
            v1, v2 = values[incr1], values[incr2]
            return factor * (v1 + (v2 - v1) * (incr - incr1))
        if "dt" in ts:
            times = [i * ts["dt"] for i in range(len(values))]
        else:
            times = ts["time"]
        if t < times[0]:
            return 0.0
        if t >= times[-1]:
            return factor * (values[-1] if ts.get("use_last", False) or t == times[-1] else 0.0)
        for i in range(len(values) - 1):
            t1, t2 = times[i], times[i + 1]
            if t == t2:
                return factor * values[i + 1]
            if t > t1 and t < t2:
                v1, v2 = values[i], values[i + 1]
                return factor * (v1 + (v2 - v1) * (t - t1) / (t2 - t1))
        return 0.0
    if ts_type == "Trig":
        t_start = ts["t_start"]
        t_finish = ts["t_finish"]
        if t < t_start or t > t_finish:
            return 0.0
        period = ts["period"]
        if period == 0.0:
            period = math.pi
        phase_shift = ts.get("phase_shift", 0.0)
        zero_shift = ts.get("zero_shift", 0.0)
        twopi = 2.0 * math.pi
        phi = phase_shift
        if factor != 0.0:
            phi = phase_shift - period / twopi * math.asin(zero_shift / factor)
        return factor * math.sin(twopi * (t - t_start) / period + phi) + zero_shift
    raise ValueError(f"unsupported time series: {ts_type}")


def _run_with_time_series(ts_def):
    base_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "elastic_beam_cantilever"
        / "elastic_beam_cantilever.json"
    )
    case_data = json.loads(base_path.read_text())
    case_data["time_series"] = [ts_def]
    case_data["pattern"] = {"type": "Plain", "tag": 1, "time_series": ts_def["tag"]}
    case_data["analysis"] = {"type": "static_linear", "steps": 1}

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        disp_file = out_dir / "node_disp_node2.out"
        values = [float(v) for v in disp_file.read_text().split()]
    return values


def test_time_series_constant():
    ts = {"type": "Constant", "tag": 1, "factor": 2.5}
    values = _run_with_time_series(ts)
    expected_v, expected_theta = _expected_cantilever()
    scale = _time_series_factor(ts, 1.0)
    assert math.isclose(values[1], expected_v * scale, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], expected_theta * scale, rel_tol=1e-6, abs_tol=1e-12)


def test_time_series_linear():
    ts = {"type": "Linear", "tag": 1, "factor": 1.5}
    values = _run_with_time_series(ts)
    expected_v, expected_theta = _expected_cantilever()
    scale = _time_series_factor(ts, 1.0)
    assert math.isclose(values[1], expected_v * scale, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], expected_theta * scale, rel_tol=1e-6, abs_tol=1e-12)


def test_time_series_path():
    ts = {"type": "Path", "tag": 1, "dt": 0.5, "values": [0.0, 1.0, 2.0]}
    values = _run_with_time_series(ts)
    expected_v, expected_theta = _expected_cantilever()
    scale = _time_series_factor(ts, 1.0)
    assert math.isclose(values[1], expected_v * scale, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], expected_theta * scale, rel_tol=1e-6, abs_tol=1e-12)


def test_time_series_trig():
    ts = {
        "type": "Trig",
        "tag": 1,
        "t_start": 0.0,
        "t_finish": 2.0,
        "period": 4.0,
        "phase_shift": 0.0,
        "factor": 2.0,
        "zero_shift": 0.0,
    }
    values = _run_with_time_series(ts)
    expected_v, expected_theta = _expected_cantilever()
    scale = _time_series_factor(ts, 1.0)
    assert math.isclose(values[1], expected_v * scale, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], expected_theta * scale, rel_tol=1e-6, abs_tol=1e-12)
