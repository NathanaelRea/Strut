import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "synthetic_ground_motion.py"

_spec = importlib.util.spec_from_file_location("strut_synthetic_gm_unit", SCRIPT)
assert _spec is not None and _spec.loader is not None
sgm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sgm
_spec.loader.exec_module(sgm)


def _params(**overrides):
    values = {"seed": 17, "dt": 0.005, "npts": 3001, "pga": 0.35}
    values.update(overrides)
    return sgm.MotionParams(**values)


def _read_smd_dt(text: str):
    """What OpenSees' ReadSMDFile.tcl does: split each header line on single
    spaces, take the word after ``DT=``, and echo every later line."""
    flag = False
    dt_word = None
    data_lines = []
    for line in text.split("\n"):
        if not line.split():
            continue
        if flag:
            data_lines.append(line)
            continue
        words = line.split(" ")
        for idx, word in enumerate(words):
            if word == "DT=":
                flag = True
                dt_word = words[idx + 1]
                break
    return dt_word, data_lines


@pytest.mark.parametrize("suffix", [".AT2", ".DT2", ".acc", ".tcl"])
def test_same_seed_gives_byte_identical_files(tmp_path: Path, suffix: str):
    first = tmp_path / f"first{suffix}"
    second = tmp_path / f"second{suffix}"
    other = tmp_path / f"other{suffix}"
    sgm.write_motion(first, _params())
    sgm.write_motion(second, _params())
    sgm.write_motion(other, _params(seed=18))

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != other.read_bytes()


def test_cli_matches_library_output(tmp_path: Path):
    lib_path = tmp_path / "lib.AT2"
    cli_path = tmp_path / "cli.AT2"
    sgm.write_motion(lib_path, _params())

    assert (
        sgm.main(
            [
                str(cli_path),
                "--seed",
                "17",
                "--dt",
                "0.005",
                "--npts",
                "3001",
                "--pga",
                "0.35",
            ]
        )
        == 0
    )
    assert cli_path.read_bytes() == lib_path.read_bytes()


@pytest.mark.parametrize(
    ("dt", "npts", "dt_text"),
    [(0.005, 7802, ".00500"), (0.01, 4000, ".01000"), (0.02, 1500, ".02000")],
)
def test_at2_header_matches_peer_layout_and_round_trips(
    tmp_path: Path, dt: float, npts: int, dt_text: str
):
    path = tmp_path / "motion.AT2"
    acc = sgm.write_motion(path, _params(dt=dt, npts=npts))
    raw = path.read_bytes()
    text = raw.decode("ascii")
    lines = text.split("\r\n")

    # Same layout as the pinned PEER records: four header lines, CRLF, the
    # NPTS/DT line last, five E15.7 values per line, no trailing newline.
    assert "\n" not in text.replace("\r\n", "")
    assert not raw.endswith(b"\n")
    assert lines[2].startswith("ACCELERATION TIME HISTORY IN UNITS OF G")
    assert lines[3] == f"NPTS={npts:6d}, DT= {dt_text} SEC"
    assert sum("DT=" in line.split(" ") for line in lines) == 1
    data = lines[4:]
    assert all(len(line) == 75 for line in data[:-1])
    assert len(data[-1]) == 15 * (npts - 5 * (len(data) - 1))
    assert data[0][:15].strip().lstrip("-").startswith(".")

    dt_word, data_lines = _read_smd_dt(text)
    assert dt_word == dt_text
    assert float(dt_word) == dt
    tcl_values = [float(tok) for line in data_lines for tok in line.split()]
    assert len(tcl_values) == npts

    record = sgm.read_motion(path)
    assert record.npts == npts
    assert record.dt == dt
    assert record.header == lines[:4]
    assert record.values == tcl_values
    assert record.values == pytest.approx(acc, rel=1e-6, abs=1e-12)


def test_dt2_is_twice_integrated_at2_in_cm(tmp_path: Path):
    params = _params()
    acc = sgm.write_motion(tmp_path / "m.AT2", params)
    sgm.write_motion(tmp_path / "m.DT2", params, acceleration=acc)

    at2 = sgm.read_motion(tmp_path / "m.AT2")
    dt2 = sgm.read_motion(tmp_path / "m.DT2")
    assert dt2.header[2].startswith("DISPLACEMENT TIME HISTORY IN UNITS OF CM")
    assert (dt2.npts, dt2.dt) == (at2.npts, at2.dt)
    _, disp = sgm.integrate(acc, params.dt)
    expected = [d * sgm.G_CM_PER_S2 for d in disp]
    peak = max(abs(v) for v in expected)
    assert dt2.values == pytest.approx(expected, abs=1e-6 * peak)


def test_values_formats_match_real_layouts(tmp_path: Path):
    params = _params(npts=12)
    acc = sgm.write_motion(tmp_path / "BM68elc.acc", params)
    sgm.write_motion(tmp_path / "A10000.tcl", params, acceleration=acc)

    acc_text = (tmp_path / "BM68elc.acc").read_bytes().decode("ascii")
    tcl_text = (tmp_path / "A10000.tcl").read_bytes().decode("ascii")
    for text in (acc_text, tcl_text):
        assert text.endswith("\r\n")
        rows = text.split("\r\n")[:-1]
        assert [len(row) for row in rows] == [75, 75, 30]
    # BM68elc.acc style ".4524259E-02", A10000.tcl style "0.450671E-03".
    assert all(
        tok.lstrip("-").startswith(".") for tok in acc_text.split() if float(tok) != 0
    )
    assert all(tok.lstrip("-").startswith("0.") for tok in tcl_text.split())
    for name, digits in (("BM68elc.acc", 7), ("A10000.tcl", 6)):
        record = sgm.read_motion(tmp_path / name)
        assert record.header == [] and record.dt is None
        assert record.values == pytest.approx(acc, rel=10.0 ** (1 - digits), abs=1e-12)


@pytest.mark.parametrize(
    ("value", "digits", "leading_zero", "text"),
    [
        (-4.524259e-03, 7, False, "  -.4524259E-02"),
        (1.453968e-04, 7, False, "   .1453968E-03"),
        (0.0, 7, False, "   .0000000E+00"),
        (-4.50671e-04, 6, True, "  -0.450671E-03"),
        (12.5, 7, False, "   .1250000E+02"),
        (9.99999996e-03, 7, False, "   .1000000E-01"),
    ],
)
def test_fortran_e_formatting(value, digits, leading_zero, text):
    assert sgm.fortran_e(value, digits, 15, leading_zero) == text


@pytest.mark.parametrize("envelope", sgm.ENVELOPES)
@pytest.mark.parametrize("pga", [0.05, 0.4, 1.2])
def test_pga_scaling_and_baseline_correction(envelope: str, pga: float):
    params = _params(pga=pga, envelope=envelope)
    acc = sgm.generate_acceleration(params)

    assert len(acc) == params.npts
    assert max(abs(v) for v in acc) == pytest.approx(pga, rel=1e-12)
    assert acc[0] == pytest.approx(0.0, abs=1e-12 * pga)
    assert acc[-1] == pytest.approx(0.0, abs=1e-9 * pga)
    vel, disp = sgm.integrate(acc, params.dt)
    peak_vel = max(abs(v) for v in vel)
    peak_disp = max(abs(d) for d in disp)
    assert peak_vel > 0.0 and peak_disp > 0.0
    assert abs(vel[-1]) <= 1e-9 * peak_vel
    assert abs(disp[-1]) <= 1e-9 * peak_disp


def test_written_record_keeps_baseline_correction(tmp_path: Path):
    # Seven significant digits of rounding must not reintroduce drift.
    params = _params(npts=7802)
    sgm.write_motion(tmp_path / "m.AT2", params)
    record = sgm.read_motion(tmp_path / "m.AT2")
    vel, _ = sgm.integrate(record.values, record.dt)
    assert abs(vel[-1]) <= 1e-5 * max(abs(v) for v in vel)


def test_site_frequency_shifts_the_spectrum():
    def zero_crossing_rate(freq):
        params = _params(npts=8001, site_freq_hz=freq, envelope="trapezoidal")
        acc = sgm.generate_acceleration(params)
        crossings = sum(1 for a, b in itertools.pairwise(acc) if a * b < 0.0)
        return crossings / params.duration

    assert zero_crossing_rate(1.0) < zero_crossing_rate(5.0)


def test_envelope_shapes():
    sh = sgm.envelope_values(_params(npts=1001, dt=0.01, end_taper_frac=0.0))
    peak_idx = max(range(len(sh)), key=lambda k: sh[k])
    # Saragoni-Hart peaks (at 1.0) at eps * duration and decays to eta.
    assert sh[peak_idx] == pytest.approx(1.0, rel=1e-3)
    assert peak_idx * 0.01 == pytest.approx(0.2 * 10.0, abs=0.011)
    assert sh[-1] == pytest.approx(0.05, rel=1e-9)
    tapered = sgm.envelope_values(_params(npts=1001, dt=0.01))
    assert tapered[:950] == sh[:950]
    assert tapered[-1] == 0.0
    trap = sgm.envelope_values(_params(npts=1001, dt=0.01, envelope="trapezoidal"))
    assert trap[0] == 0.0 and trap[-1] == 0.0
    assert trap[500] == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"dt": 0.0},
        {"npts": 3},
        {"pga": -1.0},
        {"site_freq_hz": 150.0},
        {"hp_freq_hz": 5.0},
        {"envelope": "box"},
    ],
)
def test_invalid_params_are_rejected(overrides):
    with pytest.raises(ValueError):
        sgm.generate_acceleration(_params(**overrides))


def test_peer_dt_must_fit_the_header_field(tmp_path: Path):
    with pytest.raises(ValueError):
        sgm.write_motion(tmp_path / "m.AT2", _params(dt=0.0012345))
