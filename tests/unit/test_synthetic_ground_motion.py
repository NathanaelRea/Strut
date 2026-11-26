import functools
import importlib.util
import itertools
import math
import random
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


# --------------------------------------------------------------------------- #
# Rezaeian & Der Kiureghian (2008) model
# --------------------------------------------------------------------------- #

# A moderate record: Ia = 0.05 s*g, D5-95 = 8 s, t_mid = 6 s, 5 Hz at t_mid
# falling 0.4 Hz/s, zeta_f = 0.2.
_RDK = sgm.RDKParams(
    arias_intensity=0.05,
    d5_95=8.0,
    t_mid=6.0,
    f_mid_hz=5.0,
    f_rate_hz_per_s=-0.4,
    zeta_f=0.2,
)
_RDK_DT = 0.01


def _rdk_params(seed=3, rdk=_RDK, **overrides):
    values = {
        "seed": seed,
        "dt": _RDK_DT,
        "pga": None,
        "model": "rdk2008",
        "rdk": rdk,
    }
    values.update(overrides)
    if "npts" not in values:
        values["npts"] = math.ceil(sgm.rdk_default_duration(rdk) / _RDK_DT) + 1
    return sgm.MotionParams(**values)


@functools.cache
def _rdk_records(rdk=_RDK, seeds=tuple(range(6))):
    return [sgm.generate_acceleration(_rdk_params(seed, rdk)) for seed in seeds]


def _upcrossing_rate(acc, dt, t0, t1):
    lo, hi = round(t0 / dt), round(t1 / dt)
    window = acc[lo:hi]
    ups = sum(1 for a, b in itertools.pairwise(window) if a < 0.0 <= b)
    return ups / (t1 - t0)


@pytest.mark.parametrize("suffix", [".AT2", ".DT2", ".acc", ".tcl"])
def test_rdk_same_seed_gives_byte_identical_files(tmp_path: Path, suffix: str):
    params = _rdk_params(npts=1500)
    sgm.write_motion(tmp_path / f"a{suffix}", params)
    sgm.write_motion(tmp_path / f"b{suffix}", params)
    sgm.write_motion(tmp_path / f"c{suffix}", _rdk_params(seed=4, npts=1500))
    first = (tmp_path / f"a{suffix}").read_bytes()
    assert first == (tmp_path / f"b{suffix}").read_bytes()
    assert first != (tmp_path / f"c{suffix}").read_bytes()


def test_rdk_at2_header_round_trips(tmp_path: Path):
    params = _rdk_params(npts=1500, pga=0.25)
    acc = sgm.write_motion(tmp_path / "m.AT2", params)
    text = (tmp_path / "m.AT2").read_bytes().decode("ascii")
    lines = text.split("\r\n")
    assert "REZAEIAN-DER KIUREGHIAN" in lines[0]
    assert lines[3] == "NPTS=  1500, DT= .01000 SEC"
    dt_word, _ = _read_smd_dt(text)
    assert dt_word == ".01000"
    record = sgm.read_motion(tmp_path / "m.AT2")
    assert record.values == pytest.approx(acc, rel=1e-6, abs=1e-12)
    assert max(abs(v) for v in acc) == pytest.approx(0.25, rel=1e-12)


def test_rdk_cli_scenario_is_deterministic(tmp_path: Path):
    argv = ["--seed", "9", "--model", "rdk2008", "--dt", "0.01"]
    argv += ["--magnitude", "7", "--rrup", "20", "--vs30", "650", "--fault", "reverse"]
    assert sgm.main([str(tmp_path / "a.AT2"), *argv]) == 0
    assert sgm.main([str(tmp_path / "b.AT2"), *argv]) == 0
    assert (tmp_path / "a.AT2").read_bytes() == (tmp_path / "b.AT2").read_bytes()
    # The library path with the same parameter stream writes the same file.
    scenario = sgm.RDKScenario("reverse", 7.0, 20.0, 650.0)
    rdk = sgm.rdk2010_sample_params(scenario, sgm.rdk2010_params_rng(9))
    npts = math.ceil(sgm.rdk_default_duration(rdk) / 0.01) + 1
    params = sgm.MotionParams(
        seed=9, dt=0.01, npts=npts, pga=None, model="rdk2008", rdk=rdk
    )
    sgm.write_motion(tmp_path / "lib.AT2", params)
    assert (tmp_path / "lib.AT2").read_bytes() == (tmp_path / "a.AT2").read_bytes()


def test_gamma_modulation_solves_the_physical_parameters():
    mod = sgm.gamma_modulation(0.05, 8.0, 6.0)
    assert mod.alpha2 > 1.0 and mod.alpha3 > 0.0
    # Eqs. (4.6), (4.7): the 5/45/95% times of the q^2 gamma distribution.
    t05, t45, t95 = (mod.arias_time(p) for p in (0.05, 0.45, 0.95))
    assert t45 == pytest.approx(6.0, rel=1e-10)
    assert t95 - t05 == pytest.approx(8.0, rel=1e-10)
    # Eq. (4.9): pi/2 * integral(q^2) is the Arias intensity (s*g).
    dt = 0.001
    integral = sum(mod(k * dt) ** 2 for k in range(1, 100_000)) * dt
    assert 0.5 * math.pi * integral == pytest.approx(0.05, rel=1e-6)


def test_gamma_modulation_needs_alpha2_above_one():
    limit = sgm.max_d5_95_over_t_mid()
    assert limit == pytest.approx(
        (math.log(0.95) - math.log(0.05)) / -math.log(0.55), rel=1e-10
    )
    with pytest.raises(ValueError, match="alpha2 > 1"):
        sgm.gamma_modulation(0.05, 1.01 * limit, 1.0)
    assert sgm.gamma_modulation(0.05, 0.99 * limit, 1.0).alpha2 > 1.0


def test_rdk_records_match_requested_arias_duration_and_t_mid():
    """Ia, D5-95 and t_mid are expectations of the process; average 6 seeds."""
    records = _rdk_records()
    ia, d595, tmid = [], [], []
    for acc in records:
        ia.append(sgm.arias_intensity(acc, _RDK_DT))
        t05, t45, t95 = sgm.arias_times(acc, _RDK_DT, (0.05, 0.45, 0.95))
        d595.append(t95 - t05)
        tmid.append(t45)
    # Single records scatter around the targets (sample Arias intensity of one
    # record varies by ~10-15%); the 6-seed means sit within 10% of them.
    assert sum(ia) / len(ia) == pytest.approx(_RDK.arias_intensity, rel=0.10)
    assert sum(d595) / len(d595) == pytest.approx(_RDK.d5_95, rel=0.10)
    assert sum(tmid) / len(tmid) == pytest.approx(_RDK.t_mid, rel=0.10)


def test_rdk_unscaled_record_has_the_process_variance():
    # Before the high-pass filter, pi/2 * sum(x^2) dt matches eq. (4.3) closely
    # because the unit process is normalized at every time step.
    mod = sgm.gamma_modulation(_RDK.arias_intensity, _RDK.d5_95, _RDK.t_mid)
    npts = _rdk_params().npts
    total = 0.0
    seeds = range(6)
    for seed in seeds:
        unit = sgm.rdk_unit_process(_RDK, mod, _RDK_DT, npts, random.Random(seed))
        total += sgm.arias_intensity(
            [mod(k * _RDK_DT) * y for k, y in enumerate(unit)], _RDK_DT
        )
    assert total / len(seeds) == pytest.approx(_RDK.arias_intensity, rel=0.08)


def test_rdk_filter_frequency_drifts():
    """The zero-level up-crossing rate tracks the filter frequency (section
    4.3.2), so it falls over the record when f' < 0 and rises when f' > 0."""
    mod = sgm.gamma_modulation(_RDK.arias_intensity, _RDK.d5_95, _RDK.t_mid)
    t05, t45, t95 = (mod.arias_time(p) for p in (0.05, 0.45, 0.95))
    early_mid = 0.5 * (t05 + t45)
    late_mid = 0.5 * (t45 + t95)

    def rates(rdk):
        early = late = 0.0
        records = _rdk_records(rdk)
        for acc in records:
            early += _upcrossing_rate(acc, _RDK_DT, t05, t45)
            late += _upcrossing_rate(acc, _RDK_DT, t45, t95)
        return early / len(records), late / len(records)

    early, late = rates(_RDK)
    expected_early = _RDK.f_mid_hz + _RDK.f_rate_hz_per_s * (early_mid - _RDK.t_mid)
    expected_late = _RDK.f_mid_hz + _RDK.f_rate_hz_per_s * (late_mid - _RDK.t_mid)
    assert early == pytest.approx(expected_early, rel=0.1)
    assert late == pytest.approx(expected_late, rel=0.1)
    assert early > 1.2 * late

    rising = sgm.RDKParams(**{**_RDK.__dict__, "f_mid_hz": 3.0, "f_rate_hz_per_s": 0.2})
    early_up, late_up = rates(rising)
    assert late_up > 1.2 * early_up


def test_rdk_filter_frequency_is_clamped_and_floored():
    mod = sgm.gamma_modulation(_RDK.arias_intensity, _RDK.d5_95, _RDK.t_mid)
    t_s, t_e = mod.arias_time(0.01), mod.arias_time(0.99)
    two_pi = 2.0 * math.pi
    at = functools.partial(sgm.rdk_filter_frequency, _RDK, mod)
    assert at(_RDK.t_mid) == pytest.approx(two_pi * 5.0)
    assert at(0.0) == at(t_s)
    assert at(100.0) == at(t_e)
    steep = sgm.RDKParams(**{**_RDK.__dict__, "f_rate_hz_per_s": -1.5})
    assert sgm.rdk_filter_frequency(steep, mod, t_e) == pytest.approx(two_pi * 0.3)


def test_rdk_high_pass_leaves_zero_residuals():
    for acc in _rdk_records():
        vel, disp = sgm.integrate(acc, _RDK_DT)
        assert abs(vel[-1]) <= 1e-3 * max(abs(v) for v in vel)
        assert abs(disp[-1]) <= 1e-3 * max(abs(d) for d in disp)
        assert acc[0] == 0.0


def test_rdk_high_pass_is_the_trapezoid_integrated_oscillator():
    # A unit pulse into the eq. (2.28) filter: the trapezoid integrals of the
    # output satisfy the oscillator equation sample by sample, and decay.
    dt, fc = 0.01, 0.15
    pulse = [0.0] * 4001
    pulse[10:20] = [1.0] * 10
    out = sgm.rdk_high_pass(pulse, dt, fc)
    vel, disp = sgm.integrate(out, dt)
    wc = 2.0 * math.pi * fc
    for k in range(0, 4001, 97):
        assert out[k] + 2.0 * wc * vel[k] + wc * wc * disp[k] == pytest.approx(
            pulse[k], abs=1e-12
        )
    assert abs(vel[-1]) < 1e-6 * max(abs(v) for v in vel)
    assert abs(disp[-1]) < 1e-6 * max(abs(d) for d in disp)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rdk": None},
        {"pga": -0.1},
        {"dt": 0.05},  # pi / (2 w_max) is about 0.04 s for 6 Hz
        {"rdk": sgm.RDKParams(0.05, 8.0, 6.0, 5.0, -0.2, 1.0)},
        {"rdk": sgm.RDKParams(0.05, 30.0, 2.0, 5.0, -0.2, 0.2)},
        {"model": "kanai"},
    ],
)
def test_invalid_rdk_params_are_rejected(overrides):
    params = _rdk_params(npts=200, **overrides)
    with pytest.raises(ValueError):
        sgm.generate_acceleration(params)


# --------------------------------------------------------------------------- #
# Rezaeian & Der Kiureghian (2010) predictive model
# --------------------------------------------------------------------------- #

# PEER 2010/02 section 5.3 / Table 5.1: the Tabas, Iran (1978) Dayhook record,
# F = 1 (reverse), M = 7.35, R = 14 km, Vs30 = 660 m/s.
_TABAS = sgm.RDKScenario("reverse", 7.35, 14.0, 660.0)
_TABLE_5_1 = [
    (0.075, 20.1, 7.0, 4.84, -0.012, 0.25),
    (0.288, 21.3, 16.5, 2.48, -0.054, 0.12),
    (0.124, 15.3, 14.9, 3.72, 0.0039, 0.40),
    (0.147, 15.5, 10.0, 6.22, 0.00046, 0.18),
    (0.145, 12.3, 6.8, 5.90, 0.12, 0.26),  # identified from the record
]


def test_predictive_means_follow_table_4_5():
    m, r, v = 7.35 / 7.0, 14.0 / 25.0, 660.0 / 750.0
    expected = [
        -1.844 - 0.071 + 2.944 * m - 1.356 * math.log(r) - 0.265 * math.log(v),
        -6.195 - 0.703 + 6.792 * m + 0.219 * r - 0.523 * v,
        -5.011 - 0.345 + 4.638 * m + 0.348 * r - 0.185 * v,
        2.253 - 0.081 - 1.810 * m - 0.211 * r + 0.012 * v,
        -2.489 + 0.044 + 2.408 * m + 0.065 * r - 0.081 * v,
        -0.258 - 0.477 + 0.905 * m - 0.289 * r + 0.316 * v,
    ]
    assert sgm.rdk2010_mean_normals(_TABAS) == pytest.approx(expected, abs=1e-12)
    strike_slip = sgm.RDKScenario("strike-slip", 7.0, 25.0, 750.0)
    assert sgm.rdk2010_mean_normals(strike_slip) == pytest.approx(
        [
            -1.844 + 2.944,
            -6.195 + 6.792 + 0.219 - 0.523,
            -5.011 + 4.638 + 0.348 - 0.185,
            2.253 - 1.810 - 0.211 + 0.012,
            -2.489 + 2.408 + 0.065 - 0.081,
            -0.258 + 0.905 - 0.289 + 0.316,
        ],
        abs=1e-12,
    )


def test_table_5_1_parameters_are_typical_for_their_scenario():
    """The paper's four simulated and one identified parameter sets for the
    Tabas scenario sit within two standard deviations of the predicted mean,
    which checks the marginal transformations against the paper's own data."""
    mean = sgm.rdk2010_mean_normals(_TABAS)
    cov = sgm.rdk2010_covariance()
    for row in _TABLE_5_1:
        for i, (name, value) in enumerate(zip(sgm.RDK_PARAM_NAMES, row)):
            z = (sgm.rdk2010_to_normal(name, value) - mean[i]) / math.sqrt(cov[i][i])
            assert abs(z) < 2.0, (name, value, z)
    median = sgm.rdk2010_median_params(_TABAS)
    assert 0.1 < median.arias_intensity < 0.5
    assert 10.0 < median.d5_95 < 20.0 and 5.0 < median.t_mid < 12.0
    assert 4.0 < median.f_mid_hz < 8.0 and 0.15 < median.zeta_f < 0.35


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("arias_intensity", 0.0468),
        ("d5_95", 17.25),
        ("t_mid", 0.6),
        ("f_mid_hz", 5.87),
        ("f_rate_hz_per_s", -1.9),
        ("f_rate_hz_per_s", 0.0),
        ("f_rate_hz_per_s", 0.3),
        ("zeta_f", 0.9),
    ],
)
def test_marginal_transformations_round_trip(name, value):
    v = sgm.rdk2010_to_normal(name, value)
    assert sgm.rdk2010_from_normal(name, v) == pytest.approx(value, rel=1e-9)


def test_marginals_use_table_4_3_moments():
    # Lognormal Ia with the sample mean and standard deviation of Table 4.3.
    lam, zeta = sgm._lognormal_shape(0.0468, 0.164)
    assert math.exp(lam + 0.5 * zeta * zeta) == pytest.approx(0.0468)
    # Beta D5-95 on [5, 45] with mean 17.25 s and standard deviation 9.31 s.
    a, b = sgm._beta_shape(17.25, 9.31, 5.0, 45.0)
    assert 5.0 + 40.0 * a / (a + b) == pytest.approx(17.25)
    var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    assert 40.0 * math.sqrt(var) == pytest.approx(9.31)
    # Eq. (4.13): the exponential pieces meet at f' = 0 with density 4.85.
    assert sgm._frate_cdf(-2.0) == 0.0 and sgm._frate_cdf(0.5) == 1.0
    assert sgm._frate_cdf(0.0) == pytest.approx(
        (4.85 / 6.77) * (1 - math.exp(-13.54)) / 0.99993, rel=1e-4
    )


def test_special_functions_match_closed_forms():
    for x in (0.01, 0.7, 3.0, 12.0):
        assert sgm._gamma_p(1.0, x) == pytest.approx(-math.expm1(-x), rel=1e-13)
        poisson = 1.0 - math.exp(-x) * (1.0 + x + x * x / 2.0)
        assert sgm._gamma_p(3.0, x) == pytest.approx(poisson, rel=1e-11, abs=1e-15)
    for p in (0.01, 0.45, 0.99):
        assert sgm._gamma_ppf(p, 1.0) == pytest.approx(-math.log1p(-p), rel=1e-12)
    for x in (0.05, 0.5, 0.95):
        assert sgm._beta_cdf(x, 1.0, 3.5) == pytest.approx(1 - (1 - x) ** 3.5)
        assert sgm._beta_cdf(x, 2.5, 1.0) == pytest.approx(x**2.5)
        assert sgm._beta_cdf(x, 1.3, 4.1) == pytest.approx(
            1.0 - sgm._beta_cdf(1.0 - x, 4.1, 1.3), abs=1e-13
        )
        assert sgm._beta_ppf(sgm._beta_cdf(x, 0.9, 2.0), 0.9, 2.0) == pytest.approx(x)


def test_sampled_parameters_reproduce_table_4_9_correlations():
    rng = random.Random(2024)
    n = 400
    samples = []
    for _ in range(n):
        params = sgm.rdk2010_sample_params(_TABAS, rng)
        samples.append(
            [sgm.rdk2010_to_normal(k, getattr(params, k)) for k in sgm.RDK_PARAM_NAMES]
        )
    mean = sgm.rdk2010_mean_normals(_TABAS)
    cov = sgm.rdk2010_covariance()
    cols = list(zip(*samples))
    avg = [sum(c) / n for c in cols]
    std = [math.sqrt(sum((x - m) ** 2 for x in c) / (n - 1)) for c, m in zip(cols, avg)]
    for i in range(6):
        sd = math.sqrt(cov[i][i])
        assert avg[i] == pytest.approx(mean[i], abs=0.2 * sd)
        assert std[i] == pytest.approx(sd, rel=0.15)
    for i, j in ((0, 1), (1, 2), (3, 5)):
        r = sum((a - avg[i]) * (b - avg[j]) for a, b in zip(cols[i], cols[j])) / (
            (n - 1) * std[i] * std[j]
        )
        assert r == pytest.approx(sgm.RDK2010_CORRELATION[i][j], abs=0.12)


def test_conditional_sampling_holds_fixed_parameters():
    fixed = {"arias_intensity": 0.109}
    free, cond_mean, _, fixed_v = sgm.rdk2010_conditional(_TABAS, fixed)
    assert free == [1, 2, 3, 4, 5]
    mean = sgm.rdk2010_mean_normals(_TABAS)
    cov = sgm.rdk2010_covariance()
    # Eq. (5.4) with one conditioning variable.
    shift = (fixed_v[0] - mean[0]) / cov[0][0]
    for a, i in enumerate(free):
        assert cond_mean[a] == pytest.approx(mean[i] + cov[i][0] * shift)
    rng = random.Random(1)
    for _ in range(5):
        params = sgm.rdk2010_sample_params(_TABAS, rng, fixed)
        assert params.arias_intensity == 0.109


def test_scenario_range_notes():
    assert sgm.RDKScenario("strike-slip", 7.0, 20.0, 650.0).outside_data_range() == []
    notes = sgm.RDKScenario("strike-slip", 7.0, 20.0, 400.0).outside_data_range()
    assert notes == ["Vs30=400.0 m/s < 600 m/s"]
    with pytest.raises(ValueError):
        sgm.rdk2010_mean_normals(sgm.RDKScenario("normal", 7.0, 20.0, 650.0))
