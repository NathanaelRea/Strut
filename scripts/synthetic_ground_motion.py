#!/usr/bin/env python3
"""Seeded synthetic ground motions in the file formats the OpenSees examples read.

The motion is Clough-Penzien filtered Gaussian white noise times a deterministic
envelope, baseline corrected and scaled to a target PGA:

1. White noise ``w`` drives the Kanai-Tajimi soil filter (``site_freq_hz``,
   ``site_damping``), whose absolute acceleration is the classic K-T motion.
2. That acceleration drives the Clough-Penzien high-pass filter
   (``hp_freq_hz``, ``hp_damping``), which removes the K-T spectrum's
   unphysical low-frequency energy.
3. The stationary output is multiplied by a Saragoni-Hart (Boore 2003 form) or
   trapezoidal envelope, with a half-cosine taper to zero over the last
   ``end_taper_frac`` of the record.
4. Two smooth correction shapes, ``sin^2(pi t / T)`` and ``t sin^2(pi t / T)``,
   are subtracted so the trapezoid-integrated final velocity and displacement
   are zero. Both shapes vanish at the ends, so the record starts and ends at
   zero acceleration.
5. The record is scaled so ``max |a| == pga``.

Both filters are integrated with the exact zero-order-hold discretization of
their state-space form, so the spectrum does not depend on ``dt`` beyond
sampling. Everything uses the stdlib; the same parameters give byte-identical
files.

Writers match the layout of the pinned real records that the examples and
Strut's converter parse:

- ``.AT2`` / ``.DT2``: the PEER NGA (old format) layout. Four header lines; the
  last is ``NPTS=  7802, DT= .00500 SEC``, which is where ``ReadSMDFile.tcl``
  finds ``DT=``. Five ``E15.7`` values per line, CRLF, no final newline.
- ``.acc`` (``BM68elc.acc``): no header, five ``E15.7`` values per line, CRLF.
- ``.tcl`` values (``A10000.tcl``): no header, five ``0.dddddd E`` values
  per line (``E15.6`` with a leading zero), CRLF.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

G_CM_PER_S2 = 980.665

FORMATS = ("at2", "dt2", "acc", "tcl")
ENVELOPES = ("saragoni-hart", "trapezoidal")

_SUFFIX_FORMATS = {".at2": "at2", ".dt2": "dt2", ".acc": "acc", ".tcl": "tcl"}
_NEWLINE = "\r\n"
_VALUES_PER_LINE = 5
_NPTS_DT_RE = re.compile(
    r"NPTS=\s*(\d+)\s*,\s*DT=\s*([0-9.Ee+-]+)\s*SEC", re.IGNORECASE
)


@dataclass(frozen=True)
class MotionParams:
    """Inputs of :func:`generate_acceleration`.

    ``pga`` is in the units the record is written in (g for PEER ``.AT2``).
    The defaults are a firm-soil site: Kanai-Tajimi 2.5 Hz / 0.6 and a
    Clough-Penzien high-pass at a tenth of the site frequency.
    """

    seed: int
    dt: float = 0.005
    npts: int = 8000
    pga: float = 0.3
    site_freq_hz: float = 2.5
    site_damping: float = 0.6
    hp_freq_hz: float | None = None
    hp_damping: float | None = None
    envelope: str = "saragoni-hart"
    # Saragoni-Hart: amplitude peaks at eps * t_eta and has decayed to eta at
    # t_eta. t_eta defaults to the record duration.
    sh_eps: float = 0.2
    sh_eta: float = 0.05
    sh_t_eta: float | None = None
    # Trapezoidal: linear rise to 1 by rise_frac * T, flat to decay_start_frac
    # * T, linear decay to 0 at T.
    trap_rise_frac: float = 0.1
    trap_decay_start_frac: float = 0.6
    # Half-cosine taper to zero over this final fraction of the record.
    end_taper_frac: float = 0.05

    @property
    def duration(self) -> float:
        return (self.npts - 1) * self.dt

    def validate(self) -> None:
        if not (self.dt > 0.0 and math.isfinite(self.dt)):
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.npts < 8:
            raise ValueError(f"npts must be at least 8, got {self.npts}")
        if not self.pga > 0.0:
            raise ValueError(f"pga must be positive, got {self.pga}")
        if not (self.site_freq_hz > 0.0 and self.site_damping > 0.0):
            raise ValueError("site_freq_hz and site_damping must be positive")
        if self.site_freq_hz >= 0.5 / self.dt:
            raise ValueError(
                f"site_freq_hz {self.site_freq_hz} is above the Nyquist frequency "
                f"{0.5 / self.dt} for dt={self.dt}"
            )
        hp_freq, hp_damping = self.resolved_high_pass()
        if not (0.0 < hp_freq < self.site_freq_hz and hp_damping > 0.0):
            raise ValueError(
                "hp_freq_hz must be positive and below site_freq_hz, and "
                "hp_damping must be positive"
            )
        if self.envelope not in ENVELOPES:
            raise ValueError(
                f"envelope must be one of {ENVELOPES}, got {self.envelope}"
            )
        if not (0.0 < self.sh_eps < 1.0 and 0.0 < self.sh_eta < 1.0):
            raise ValueError("sh_eps and sh_eta must be in (0, 1)")
        if self.sh_t_eta is not None and not self.sh_t_eta > 0.0:
            raise ValueError("sh_t_eta must be positive")
        if not (0.0 < self.trap_rise_frac <= self.trap_decay_start_frac < 1.0):
            raise ValueError("need 0 < trap_rise_frac <= trap_decay_start_frac < 1")
        if not 0.0 <= self.end_taper_frac < 1.0:
            raise ValueError("end_taper_frac must be in [0, 1)")

    def resolved_high_pass(self) -> tuple[float, float]:
        hp_freq = (
            self.hp_freq_hz if self.hp_freq_hz is not None else 0.1 * self.site_freq_hz
        )
        hp_damping = (
            self.hp_damping if self.hp_damping is not None else self.site_damping
        )
        return hp_freq, hp_damping


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = len(b[0])
    inner = len(b)
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(m)]
        for i in range(n)
    ]


def _expm(a: list[list[float]]) -> list[list[float]]:
    """Matrix exponential by scaling and squaring with a Taylor series."""
    n = len(a)
    norm = max(sum(abs(v) for v in row) for row in a)
    squarings = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    scale = 2.0**-squarings
    scaled = [[v * scale for v in row] for row in a]
    result = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    term = [row[:] for row in result]
    for k in range(1, 30):
        term = [[v / k for v in row] for row in _mat_mul(term, scaled)]
        result = [[result[i][j] + term[i][j] for j in range(n)] for i in range(n)]
        if max(abs(v) for row in term for v in row) < 1.0e-18:
            break
    for _ in range(squarings):
        result = _mat_mul(result, result)
    return result


def _clough_penzien_discrete(
    params: MotionParams,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Exact ZOH discretization (Ad, Bd, C) of the two cascaded filters.

    State ``[u_g, v_g, u_f, v_f]``:
        u_g'' + 2 zg wg u_g' + wg^2 u_g = -w
        u_f'' + 2 zf wf u_f' + wf^2 u_f = -(2 zg wg u_g' + wg^2 u_g)
    and the output is ``u_f''`` (the high-passed Kanai-Tajimi acceleration).
    """
    wg = 2.0 * math.pi * params.site_freq_hz
    zg = params.site_damping
    hp_freq, zf = params.resolved_high_pass()
    wf = 2.0 * math.pi * hp_freq
    c_row = [-(wg * wg), -2.0 * zg * wg, -(wf * wf), -2.0 * zf * wf]
    a = [
        [0.0, 1.0, 0.0, 0.0],
        [-(wg * wg), -2.0 * zg * wg, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        c_row,
    ]
    b = [0.0, -1.0, 0.0, 0.0]
    dt = params.dt
    augmented = [[a[i][j] * dt for j in range(4)] + [b[i] * dt] for i in range(4)]
    augmented.append([0.0] * 5)
    phi = _expm(augmented)
    ad = [row[:4] for row in phi[:4]]
    bd = [phi[i][4] for i in range(4)]
    return ad, bd, c_row


def envelope_values(params: MotionParams) -> list[float]:
    """The deterministic amplitude envelope at each sample time."""
    duration = params.duration
    times = [k * params.dt for k in range(params.npts)]
    if params.envelope == "saragoni-hart":
        eps = params.sh_eps
        eta = params.sh_eta
        t_eta = params.sh_t_eta if params.sh_t_eta is not None else duration
        b = -eps * math.log(eta) / (1.0 + eps * (math.log(eps) - 1.0))
        c = b / eps
        a = (math.e / eps) ** b
        values = [
            0.0 if t <= 0.0 else a * (t / t_eta) ** b * math.exp(-c * t / t_eta)
            for t in times
        ]
    else:
        rise = params.trap_rise_frac * duration
        decay_start = params.trap_decay_start_frac * duration
        values = []
        for t in times:
            if t < rise:
                values.append(t / rise)
            elif t <= decay_start:
                values.append(1.0)
            else:
                values.append(max(0.0, (duration - t) / (duration - decay_start)))
    taper = params.end_taper_frac * duration
    if taper > 0.0:
        taper_start = duration - taper
        for k, t in enumerate(times):
            if t > taper_start:
                values[k] *= 0.5 * (1.0 + math.cos(math.pi * (t - taper_start) / taper))
    return values


def integrate(values: Sequence[float], dt: float) -> tuple[list[float], list[float]]:
    """Trapezoid-rule velocity and displacement from rest."""
    vel = [0.0] * len(values)
    disp = [0.0] * len(values)
    for k in range(1, len(values)):
        vel[k] = vel[k - 1] + 0.5 * dt * (values[k - 1] + values[k])
        disp[k] = disp[k - 1] + 0.5 * dt * (vel[k - 1] + vel[k])
    return vel, disp


def _final_vel_disp(values: Sequence[float], dt: float) -> tuple[float, float]:
    vel, disp = integrate(values, dt)
    return vel[-1], disp[-1]


def baseline_correct(values: Sequence[float], dt: float) -> list[float]:
    """Remove ``p0 h(t) + p1 h(t) t/T`` with ``h = sin^2(pi t/T)`` so the final
    trapezoid-integrated velocity and displacement are zero."""
    n = len(values)
    span = (n - 1) * dt
    shapes = []
    for power in (0, 1):
        shape = []
        for k in range(n):
            t = k * dt
            s = math.sin(math.pi * t / span)
            shape.append(s * s * (t / span) ** power)
        shapes.append(shape)
    v_a, d_a = _final_vel_disp(values, dt)
    v_0, d_0 = _final_vel_disp(shapes[0], dt)
    v_1, d_1 = _final_vel_disp(shapes[1], dt)
    det = v_0 * d_1 - v_1 * d_0
    p0 = (v_a * d_1 - v_1 * d_a) / det
    p1 = (v_0 * d_a - v_a * d_0) / det
    return [values[k] - p0 * shapes[0][k] - p1 * shapes[1][k] for k in range(n)]


def generate_acceleration(params: MotionParams) -> list[float]:
    """Synthetic acceleration history, ``params.npts`` samples at ``params.dt``."""
    params.validate()
    ad, bd, c_row = _clough_penzien_discrete(params)
    rng = random.Random(params.seed)
    # Unit-PSD white noise sampled at dt; the final PGA scaling sets the level.
    noise_scale = 1.0 / math.sqrt(params.dt)
    hp_freq, hp_damping = params.resolved_high_pass()
    slowest_decay = min(
        params.site_damping * params.site_freq_hz, hp_damping * hp_freq
    ) * (2.0 * math.pi)
    # Run the filters long enough before t=0 that the output is stationary.
    warmup = math.ceil(10.0 / (slowest_decay * params.dt))
    state = [0.0, 0.0, 0.0, 0.0]
    stationary = []
    for k in range(warmup + params.npts):
        w = rng.gauss(0.0, 1.0) * noise_scale
        state = [
            sum(ad[i][j] * state[j] for j in range(4)) + bd[i] * w for i in range(4)
        ]
        if k >= warmup:
            stationary.append(sum(c_row[j] * state[j] for j in range(4)))
    shaped = [s * e for s, e in zip(stationary, envelope_values(params))]
    corrected = baseline_correct(shaped, params.dt)
    peak = max(abs(v) for v in corrected)
    scale = params.pga / peak
    return [v * scale for v in corrected]


# --------------------------------------------------------------------------- #
# Formatting and writers
# --------------------------------------------------------------------------- #


def fortran_e(
    value: float, digits: int = 7, width: int = 15, leading_zero: bool = False
) -> str:
    """Fortran ``Ew.d`` text: ``.4524259E-02`` (or ``0.452426E-02``)."""
    if not math.isfinite(value):
        raise ValueError(f"cannot format non-finite value {value}")
    if value == 0.0:
        mantissa, exponent = "0" * digits, 0
        negative = False
    else:
        negative = value < 0.0
        text = f"{abs(value):.{digits - 1}e}"
        coeff, exp_text = text.split("e")
        mantissa = coeff.replace(".", "")
        exponent = int(exp_text) + 1
    sign = "-" if negative else ""
    lead = "0" if leading_zero else ""
    exp_sign = "-" if exponent < 0 else "+"
    body = f"{sign}{lead}.{mantissa}E{exp_sign}{abs(exponent):02d}"
    return body.rjust(width)


def _value_lines(values: Sequence[float], digits: int, leading_zero: bool) -> list[str]:
    lines = []
    for start in range(0, len(values), _VALUES_PER_LINE):
        chunk = values[start : start + _VALUES_PER_LINE]
        lines.append("".join(fortran_e(v, digits, 15, leading_zero) for v in chunk))
    return lines


def _peer_dt_text(dt: float) -> str:
    if not 0.0 < dt < 1.0:
        raise ValueError(f"PEER DT field needs 0 < dt < 1, got {dt}")
    text = f"{dt:.5f}"
    if abs(float(text) - dt) > 1.0e-12:
        raise ValueError(f"dt={dt} does not fit the PEER F6.5 DT field")
    return text[1:]


def peer_header(params: MotionParams, quantity: str) -> list[str]:
    """Four PEER-style header lines. Only the last one carries ``DT=``."""
    hp_freq, hp_damping = params.resolved_high_pass()
    if quantity == "acceleration":
        kind = "ACCELERATION TIME HISTORY IN UNITS OF G"
    elif quantity == "displacement":
        kind = "DISPLACEMENT TIME HISTORY IN UNITS OF CM"
    else:
        raise ValueError(f"unknown PEER quantity: {quantity}")
    return [
        "SYNTHETIC STRONG MOTION RECORD. CLOUGH-PENZIEN FILTERED WHITE NOISE (STRUT).",
        (
            f"SEED {params.seed}, PGA {params.pga:.4f} G, SITE {params.site_freq_hz:.2f} HZ"
            f" / {params.site_damping:.2f}, HIGH-PASS {hp_freq:.3f} HZ / {hp_damping:.2f},"
            f" ENVELOPE {params.envelope.upper()}"
        ),
        f"{kind}. BASELINE CORRECTED TO ZERO FINAL VELOCITY AND DISPLACEMENT",
        f"NPTS={params.npts:6d}, DT= {_peer_dt_text(params.dt)} SEC",
    ]


def format_motion(params: MotionParams, fmt: str, acceleration: Sequence[float]) -> str:
    """File text for one of :data:`FORMATS`."""
    if len(acceleration) != params.npts:
        raise ValueError(f"expected {params.npts} samples, got {len(acceleration)}")
    if fmt == "at2":
        lines = peer_header(params, "acceleration") + _value_lines(
            acceleration, 7, False
        )
        return _NEWLINE.join(lines)
    if fmt == "dt2":
        _, disp = integrate(acceleration, params.dt)
        disp_cm = [d * G_CM_PER_S2 for d in disp]
        lines = peer_header(params, "displacement") + _value_lines(disp_cm, 7, False)
        return _NEWLINE.join(lines)
    if fmt == "acc":
        return _NEWLINE.join(_value_lines(acceleration, 7, False)) + _NEWLINE
    if fmt == "tcl":
        return _NEWLINE.join(_value_lines(acceleration, 6, True)) + _NEWLINE
    raise ValueError(f"unknown ground-motion format {fmt!r}; expected one of {FORMATS}")


def format_for_path(path: Path) -> str:
    fmt = _SUFFIX_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"cannot infer a ground-motion format from {path.name}")
    return fmt


def write_motion(
    path: Path,
    params: MotionParams,
    fmt: str | None = None,
    acceleration: Sequence[float] | None = None,
) -> list[float]:
    """Write a synthetic motion to ``path`` and return its acceleration."""
    resolved_fmt = fmt or format_for_path(path)
    acc = (
        list(acceleration)
        if acceleration is not None
        else generate_acceleration(params)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(format_motion(params, resolved_fmt, acc).encode("ascii"))
    return acc


# --------------------------------------------------------------------------- #
# Readers (for tests and the parity sweep)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MotionRecord:
    values: list[float]
    dt: float | None
    npts: int | None
    header: list[str]


def read_motion(path: Path) -> MotionRecord:
    """Read any of the supported formats. PEER files report ``npts``/``dt``."""
    text = path.read_text(encoding="ascii", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    header: list[str] = []
    npts = None
    dt = None
    if path.suffix.lower() in {".at2", ".dt2"}:
        for idx, line in enumerate(lines):
            match = _NPTS_DT_RE.search(line)
            if match:
                npts = int(match.group(1))
                dt = float(match.group(2))
                header = lines[: idx + 1]
                lines = lines[idx + 1 :]
                break
        else:
            raise ValueError(f"{path}: no 'NPTS=..., DT= ... SEC' header line")
    values = [float(tok) for line in lines for tok in line.split()]
    if npts is not None and len(values) != npts:
        raise ValueError(f"{path}: header NPTS={npts} but {len(values)} values")
    return MotionRecord(values=values, dt=dt, npts=npts, header=header)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, nargs="+", help="files to write")
    parser.add_argument("--format", choices=FORMATS, help="default: from each suffix")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dt", type=float, default=MotionParams.dt)
    parser.add_argument("--npts", type=int, default=MotionParams.npts)
    parser.add_argument("--duration", type=float, help="sets npts = duration/dt + 1")
    parser.add_argument("--pga", type=float, default=MotionParams.pga)
    parser.add_argument("--site-freq-hz", type=float, default=MotionParams.site_freq_hz)
    parser.add_argument("--site-damping", type=float, default=MotionParams.site_damping)
    parser.add_argument("--hp-freq-hz", type=float)
    parser.add_argument("--hp-damping", type=float)
    parser.add_argument("--envelope", choices=ENVELOPES, default=MotionParams.envelope)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    npts = args.npts
    if args.duration is not None:
        npts = round(args.duration / args.dt) + 1
    params = MotionParams(
        seed=args.seed,
        dt=args.dt,
        npts=npts,
        pga=args.pga,
        site_freq_hz=args.site_freq_hz,
        site_damping=args.site_damping,
        hp_freq_hz=args.hp_freq_hz,
        hp_damping=args.hp_damping,
        envelope=args.envelope,
    )
    acc = generate_acceleration(params)
    for path in args.output:
        write_motion(path, params, args.format, acc)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
