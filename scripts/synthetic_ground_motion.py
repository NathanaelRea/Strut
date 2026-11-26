#!/usr/bin/env python3
"""Seeded synthetic ground motions in the file formats the OpenSees examples read.

Two models are available (``--model`` / ``MotionParams.model``):

``clough-penzien`` (the default) is filtered white noise with a fixed spectrum
times an envelope. It stays the default so existing seeds keep producing the
same bytes. It is also faster, and tests and fixtures that only need a valid
record get one cheaply.

``rdk2008`` is the site-based model of Rezaeian & Der Kiureghian (2008),
"A stochastic ground motion model with separable temporal and spectral
nonstationarities", EESD 37:1565-1584. Its dominant frequency drifts over
the record like a real one. The equations follow the authors' PEER Report
2010/02 ("Stochastic Modeling and Simulation of Ground Motions for
Performance-Based Earthquake Engineering"):

1. Unit white-noise pulses ``u_i`` at ``t_i = i dt`` excite a linear filter
   whose impulse response is the pseudo-acceleration of an oscillator with
   frequency ``w_f(t_i)`` and damping ``zeta_f`` (eq. 2.24). Each pulse keeps
   the filter parameters of its own time. The sum of the responses is divided
   by its standard deviation (eqs. 2.12, 2.13), so the result has unit variance.
2. That process is multiplied by the gamma modulating function
   ``q(t) = alpha1 t^(alpha2-1) exp(-alpha3 t)`` (eq. 2.23).
3. A critically damped oscillator high-pass filter at the corner frequency
   ``omega_c`` (eq. 2.28) gives zero residual velocity and displacement.

The physical parameters are the Arias intensity, D5-95, t_mid, the filter
frequency at t_mid and its rate of change, and the damping ratio
(``RDKParams``, section 4.3.1). ``w_f`` is linear between the 1% and 99% Arias
times and constant outside, with a 0.3 Hz floor (eq. 5.6). The parameters can
come directly, or from the Rezaeian & Der Kiureghian (2010) predictive
equations ("Simulation of synthetic ground motions for specified earthquake
and site characteristics", EESD 39:1155-1180; PEER 2010/02 chapters 4-5).
Those equations draw them from fault type, magnitude, R_rup and Vs30, with the
published correlations and marginal distributions (``rdk2010_*``).

The Clough-Penzien model is filtered Gaussian white noise times a deterministic
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

Both Clough-Penzien filters are integrated with the exact zero-order-hold
discretization of their state-space form, so the spectrum does not depend on
``dt`` beyond sampling. Everything uses the stdlib; for either model the same
parameters and seed give byte-identical files.

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
import cmath
import math
import random
import re
import sys
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import accumulate, pairwise, repeat
from operator import add, mul
from pathlib import Path
from statistics import NormalDist

G_CM_PER_S2 = 980.665

FORMATS = ("at2", "dt2", "acc", "tcl")
MODELS = ("clough-penzien", "rdk2008")
DEFAULT_MODEL = "clough-penzien"
DEFAULT_PGA = 0.3
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

    ``model`` selects the generator (:data:`MODELS`):

    - ``"clough-penzien"`` (default): the stationary Clough-Penzien process
      times an envelope, configured by the ``site_*``, ``hp_*``, ``sh_*``,
      ``trap_*`` and ``end_taper_frac`` fields. The defaults are a firm-soil
      site: Kanai-Tajimi 2.5 Hz / 0.6 and a Clough-Penzien high-pass at a
      tenth of the site frequency.
    - ``"rdk2008"``: the Rezaeian & Der Kiureghian (2008) model with the
      physical parameters in ``rdk`` (required); the record is in g.

    ``pga`` is in the units the record is written in (g for PEER ``.AT2``);
    the record is scaled so ``max |a| == pga``. ``rdk2008`` also accepts
    ``pga=None``, which keeps the amplitude implied by the requested Arias
    intensity.
    """

    seed: int
    dt: float = 0.005
    npts: int = 8000
    pga: float | None = DEFAULT_PGA
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
    model: str = DEFAULT_MODEL
    rdk: RDKParams | None = None

    @property
    def duration(self) -> float:
        return (self.npts - 1) * self.dt

    def validate(self) -> None:
        if not (self.dt > 0.0 and math.isfinite(self.dt)):
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.npts < 8:
            raise ValueError(f"npts must be at least 8, got {self.npts}")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if self.model == "rdk2008":
            self.resolved_rdk().validate()
            if self.pga is not None and not self.pga > 0.0:
                raise ValueError(f"pga must be positive or None, got {self.pga}")
            return
        if self.pga is None or not self.pga > 0.0:
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

    def resolved_rdk(self) -> RDKParams:
        if self.rdk is None:
            raise ValueError("model 'rdk2008' needs rdk=RDKParams(...)")
        return self.rdk


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


def _generate_clough_penzien(params: MotionParams) -> list[float]:
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
# Rezaeian & Der Kiureghian (2008) site-based stochastic model
# --------------------------------------------------------------------------- #
#
# Equation numbers refer to Rezaeian, S. and Der Kiureghian, A. (2010),
# "Stochastic Modeling and Simulation of Ground Motions for Performance-Based
# Earthquake Engineering", PEER Report 2010/02, UC Berkeley. Its chapter 2 is
# the model of Rezaeian & Der Kiureghian (2008), EESD 37:1565-1584, and its
# chapters 4-5 are the predictive model of Rezaeian & Der Kiureghian (2010),
# EESD 39:1155-1180.

# Gamma-distribution percentiles that define the physical parameters
# (eqs. 4.6, 4.7) and the time bracket of the linear filter frequency (eq. 5.6).
_RDK_P05, _RDK_P45, _RDK_P95 = 0.05, 0.45, 0.95
_RDK_P01, _RDK_P99 = 0.01, 0.99
# The impulse response of each white-noise pulse is summed until its envelope
# exp(-zeta_f w_f tau) falls below this fraction of its peak (the paper sums
# every earlier pulse; the dropped terms are below double-precision noise of
# the normalized sum).
_RDK_IRF_CUTOFF = 1.0e-10


@dataclass(frozen=True)
class RDKParams:
    """Physical parameters of the Rezaeian & Der Kiureghian (2008) model.

    ``arias_intensity`` is the expected Arias intensity in ``s*g`` (that is,
    ``pi/2 * integral(a^2 dt)`` with ``a`` in g; multiply by 9.81 for m/s).
    ``d5_95`` and ``t_mid`` are the times between the 5% and 95% levels and to
    the 45% level of the expected Arias intensity (s). The filter frequency is
    ``f_mid_hz + f_rate_hz_per_s * (t - t_mid)`` (eq. 4.10, here in Hz rather
    than rad/s), held constant outside the 1%-99% Arias time bracket and
    floored at ``min_filter_freq_hz`` (eq. 5.6). ``zeta_f`` is the constant
    filter damping ratio (eq. 2.26). ``corner_freq_hz`` is ``omega_c / 2 pi`` of
    the critically damped high-pass filter (eq. 2.28); 0.15 Hz is what the
    paper uses for its simulations (section 5.4.1).
    """

    arias_intensity: float
    d5_95: float
    t_mid: float
    f_mid_hz: float
    f_rate_hz_per_s: float
    zeta_f: float
    corner_freq_hz: float = 0.15
    min_filter_freq_hz: float = 0.3

    def validate(self) -> None:
        if not (self.arias_intensity > 0.0 and math.isfinite(self.arias_intensity)):
            raise ValueError("arias_intensity must be positive")
        if not (self.d5_95 > 0.0 and self.t_mid > 0.0):
            raise ValueError("d5_95 and t_mid must be positive")
        if not self.f_mid_hz > 0.0:
            raise ValueError("f_mid_hz must be positive")
        if not math.isfinite(self.f_rate_hz_per_s):
            raise ValueError("f_rate_hz_per_s must be finite")
        if not 0.0 < self.zeta_f < 1.0:
            raise ValueError(f"zeta_f must be in (0, 1), got {self.zeta_f}")
        if not (self.corner_freq_hz > 0.0 and self.min_filter_freq_hz > 0.0):
            raise ValueError("corner_freq_hz and min_filter_freq_hz must be positive")


@dataclass(frozen=True)
class GammaModulation:
    """``q(t) = alpha1 t^(alpha2 - 1) exp(-alpha3 t)`` (eq. 2.23 with T0 = 0).

    ``q^2`` is proportional to a gamma density with shape ``2 alpha2 - 1`` and
    rate ``2 alpha3`` (eq. 4.5); q is in g.
    """

    alpha1: float
    alpha2: float
    alpha3: float

    @property
    def shape(self) -> float:
        return 2.0 * self.alpha2 - 1.0

    @property
    def rate(self) -> float:
        return 2.0 * self.alpha3

    def __call__(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return self.alpha1 * t ** (self.alpha2 - 1.0) * math.exp(-self.alpha3 * t)

    def arias_time(self, fraction: float) -> float:
        """Time at which ``fraction`` of the expected Arias intensity is reached."""
        return _gamma_ppf(fraction, self.shape) / self.rate


# -- special functions (stdlib only) ---------------------------------------- #


def _gamma_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x)."""
    if x <= 0.0:
        return 0.0
    log_prefix = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        # Series: P = x^a e^-x / Gamma(a+1) * sum x^n / ((a+1)...(a+n)).
        term = 1.0 / a
        total = term
        denom = a
        for _ in range(10_000):
            denom += 1.0
            term *= x / denom
            total += term
            if abs(term) < abs(total) * 1.0e-16:
                break
        return min(1.0, total * math.exp(log_prefix))
    # Continued fraction for Q (modified Lentz).
    tiny = 1.0e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1.0e-16:
            break
    return max(0.0, 1.0 - math.exp(log_prefix) * h)


def _gamma_ppf(p: float, a: float) -> float:
    """Quantile of the unit-rate gamma distribution with shape ``a``."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"gamma quantile needs 0 < p < 1, got {p}")
    lo, hi = 0.0, max(1.0, a)
    while _gamma_p(a, hi) < p:
        lo, hi = hi, 2.0 * hi
    x = 0.5 * (lo + hi)
    for _ in range(200):
        err = _gamma_p(a, x) - p
        if err > 0.0:
            hi = x
        else:
            lo = x
        log_pdf = (a - 1.0) * math.log(x) - x - math.lgamma(a)
        pdf = math.exp(log_pdf) if log_pdf > -700.0 else 0.0
        step_x = x - err / pdf if pdf > 0.0 else lo - 1.0
        # Newton step when it stays inside the bracket, bisection otherwise.
        x_new = step_x if lo < step_x < hi else 0.5 * (lo + hi)
        if abs(x_new - x) <= 1.0e-15 * x:
            return x_new
        x = x_new
    return x


def _beta_cf(a: float, b: float, x: float) -> float:
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1.0e-16:
            break
    return h


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1.0e-15:
            break
    return 0.5 * (lo + hi)


# -- modulating function ----------------------------------------------------- #


def _arias_spread_ratio(shape: float) -> float:
    """``(t95 - t5) / t45`` of a gamma distribution; independent of the rate."""
    q05 = _gamma_ppf(_RDK_P05, shape)
    q45 = _gamma_ppf(_RDK_P45, shape)
    q95 = _gamma_ppf(_RDK_P95, shape)
    return (q95 - q05) / q45


def max_d5_95_over_t_mid() -> float:
    """Largest ``D5-95 / t_mid`` a gamma modulating function with ``alpha2 > 1``
    can have (the limit ``alpha2 -> 1``, an exponential ``q^2``)."""
    return _arias_spread_ratio(1.0)


def gamma_modulation(
    arias_intensity: float, d5_95: float, t_mid: float
) -> GammaModulation:
    """Solve eqs. (4.6), (4.7) and (4.9) for ``(alpha1, alpha2, alpha3)``.

    The gamma quantiles scale as ``t_p = g_p(b) / a`` with shape ``b`` and rate
    ``a``, so ``D5-95 / t_mid = (g_95(b) - g_5(b)) / g_45(b)`` fixes ``b``
    alone (it decreases monotonically in ``b``) and ``a = g_45(b) / t_mid``.
    This is the exact solution of the two equations the paper solves by
    nonlinear optimization. The paper requires ``alpha2 > 1`` (``b > 1``).
    """
    if not (arias_intensity > 0.0 and d5_95 > 0.0 and t_mid > 0.0):
        raise ValueError("arias_intensity, d5_95 and t_mid must be positive")
    target = d5_95 / t_mid
    limit = max_d5_95_over_t_mid()
    if not target < limit:
        raise ValueError(
            f"D5-95 / t_mid = {target:.4g} has no gamma modulating function with "
            f"alpha2 > 1 (the ratio must be below {limit:.4g}); increase t_mid or "
            "decrease D5-95"
        )
    lo, hi = 1.0, 2.0
    while _arias_spread_ratio(hi) > target:
        lo, hi = hi, 2.0 * hi
        if hi > 1.0e7:
            raise ValueError(f"D5-95 / t_mid = {target:.4g} is too small")
    # Bisection on log(shape): the ratio is smooth and monotone.
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if _arias_spread_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 <= 1.0e-13:
            break
    shape = math.sqrt(lo * hi)
    rate = _gamma_ppf(_RDK_P45, shape) / t_mid
    alpha2 = 0.5 * (shape + 1.0)
    alpha3 = 0.5 * rate
    # Eq. (4.9) in g units: Ia [s*g] = pi/2 * alpha1^2 Gamma(b) / a^b.
    log_alpha1 = 0.5 * (
        math.log(2.0 / math.pi)
        + math.log(arias_intensity)
        + shape * math.log(rate)
        - math.lgamma(shape)
    )
    return GammaModulation(math.exp(log_alpha1), alpha2, alpha3)


def rdk_filter_frequency(
    rdk: RDKParams, modulation: GammaModulation, t: float
) -> float:
    """Filter frequency in rad/s at pulse time ``t`` (eq. 5.6)."""
    t_s = modulation.arias_time(_RDK_P01)
    t_e = modulation.arias_time(_RDK_P99)
    clamped = min(max(t, t_s), t_e)
    f_hz = rdk.f_mid_hz + rdk.f_rate_hz_per_s * (clamped - rdk.t_mid)
    return 2.0 * math.pi * max(f_hz, rdk.min_filter_freq_hz)


def rdk_high_pass(
    values: Sequence[float], dt: float, corner_freq_hz: float
) -> list[float]:
    """Post-processing high-pass filter (eq. 2.28): the returned ``z''`` of the
    critically damped oscillator ``z'' + 2 wc z' + wc^2 z = x``, driven from rest
    by ``values``.

    The oscillator is integrated with the Newmark average-acceleration rule,
    which is the trapezoid rule :func:`integrate` (and OpenSees' ``Path``
    series) applies to the written record. The record's own velocity and
    displacement are then exactly the oscillator's ``z'`` and ``z``, which the
    critical damping brings back to zero once the input has died out.
    """
    omega_c = 2.0 * math.pi * corner_freq_hz
    two_wc = 2.0 * omega_c
    wc2 = omega_c * omega_c
    lead = 1.0 + omega_c * dt + 0.25 * wc2 * dt * dt
    half_dt = 0.5 * dt
    quarter_dt2 = 0.25 * dt * dt
    out = [0.0] * len(values)
    z = zd = 0.0
    zdd = values[0] if values else 0.0
    if values:
        out[0] = zdd
    for k in range(1, len(values)):
        v_pred = zd + half_dt * zdd
        d_pred = z + dt * zd + quarter_dt2 * zdd
        zdd = (values[k] - two_wc * v_pred - wc2 * d_pred) / lead
        zd = v_pred + half_dt * zdd
        z = d_pred + quarter_dt2 * zdd
        out[k] = zdd
    return out


def rdk_unit_process(
    rdk: RDKParams,
    modulation: GammaModulation,
    dt: float,
    npts: int,
    rng: random.Random,
) -> list[float]:
    """The normalized sum in braces of eq. (2.12)-(2.13) at ``t_k = k dt``.

    ``y(t_k) = sum_i h[t_k - t_i, lambda(t_i)] u_i / sqrt(sum_i h^2[...])`` for
    pulses at ``t_i = i dt``, ``i = 1..k``, with the pseudo-acceleration IRF of
    eq. (2.24). ``h(0) = 0``, so pulse ``i`` first contributes at ``t_{i+1}``.
    Each pulse keeps the filter frequency and damping of its own time ``t_i``,
    which is what makes the frequency content drift.
    """
    zeta = rdk.zeta_f
    root = math.sqrt(1.0 - zeta * zeta)
    num = [0.0] * npts
    den = [0.0] * npts
    cutoff_log = -math.log(_RDK_IRF_CUTOFF)
    for i in range(1, npts):
        u = rng.gauss(0.0, 1.0)
        remaining = npts - 1 - i
        if remaining <= 0:
            continue
        omega = rdk_filter_frequency(rdk, modulation, i * dt)
        decay = zeta * omega
        length = min(remaining, math.ceil(cutoff_log / (decay * dt)) + 1)
        step = cmath.exp(complex(-decay * dt, omega * root * dt))
        scale = omega / root
        # h at t_{i+m}, m = 1..length: scale * Im(step^m).
        h = [
            scale * w.imag
            for w in accumulate(repeat(step, length - 1), mul, initial=step)
        ]
        lo = i + 1
        hi = lo + length
        num[lo:hi] = map(add, num[lo:hi], map(mul, h, repeat(u)))
        den[lo:hi] = map(add, den[lo:hi], map(mul, h, h))
    return [n / math.sqrt(d) if d > 0.0 else 0.0 for n, d in zip(num, den)]


def _validate_rdk_sampling(
    params: MotionParams, rdk: RDKParams, modulation: GammaModulation
) -> None:
    # Section 2.2.4: dt <= pi / (2 w_max) resolves the critical points of a cycle.
    t_s = modulation.arias_time(_RDK_P01)
    t_e = modulation.arias_time(_RDK_P99)
    w_max = max(
        rdk_filter_frequency(rdk, modulation, t_s),
        rdk_filter_frequency(rdk, modulation, t_e),
    )
    if params.dt > math.pi / (2.0 * w_max):
        raise ValueError(
            f"dt={params.dt} is too coarse for a {w_max / (2.0 * math.pi):.3g} Hz filter "
            f"frequency; the model needs dt <= pi / (2 w_max) = "
            f"{math.pi / (2.0 * w_max):.4g} s"
        )


def rdk_modulation_for(params: MotionParams) -> GammaModulation:
    rdk = params.resolved_rdk()
    return gamma_modulation(rdk.arias_intensity, rdk.d5_95, rdk.t_mid)


def _generate_rdk2008(params: MotionParams) -> list[float]:
    rdk = params.resolved_rdk()
    modulation = gamma_modulation(rdk.arias_intensity, rdk.d5_95, rdk.t_mid)
    _validate_rdk_sampling(params, rdk, modulation)
    rng = random.Random(params.seed)
    unit = rdk_unit_process(rdk, modulation, params.dt, params.npts, rng)
    modulated = [modulation(k * params.dt) * y for k, y in enumerate(unit)]
    acc = rdk_high_pass(modulated, params.dt, rdk.corner_freq_hz)
    if params.pga is None:
        return acc
    peak = max(abs(v) for v in acc)
    scale = params.pga / peak
    return [v * scale for v in acc]


def rdk_default_duration(rdk: RDKParams) -> float:
    """Record length for an RDK motion when none is given.

    Section 5.4.3 finds two to three times ``D5-95`` usually long enough for
    the high-pass filter to bring the residuals to zero. This takes ``3 D5-95``
    and, when the motion runs late, extends it to ``20 / omega_c`` past the 99%
    Arias time: about twenty time constants of the critically damped filter,
    which leaves final velocity and displacement near 1e-4 of their peaks.
    """
    modulation = gamma_modulation(rdk.arias_intensity, rdk.d5_95, rdk.t_mid)
    settle = 20.0 / (2.0 * math.pi * rdk.corner_freq_hz)
    return max(3.0 * rdk.d5_95, modulation.arias_time(_RDK_P99) + settle)


# -- record measures ---------------------------------------------------------- #


def arias_intensity(values: Sequence[float], dt: float) -> float:
    """Arias intensity in ``s*g`` of an acceleration record in g (trapezoid rule)."""
    total = 0.0
    for a, b in pairwise(values):
        total += 0.5 * (a * a + b * b)
    return 0.5 * math.pi * total * dt


def arias_times(
    values: Sequence[float], dt: float, fractions: Sequence[float]
) -> list[float]:
    """Times at which the cumulative Arias intensity reaches each fraction."""
    cumulative = [0.0]
    for a, b in pairwise(values):
        cumulative.append(cumulative[-1] + 0.5 * (a * a + b * b))
    total = cumulative[-1]
    times = []
    for fraction in fractions:
        target = fraction * total
        k = bisect_left(cumulative, target)
        if k == 0:
            times.append(0.0)
            continue
        lo, hi = cumulative[k - 1], cumulative[k]
        frac = (target - lo) / (hi - lo) if hi > lo else 0.0
        times.append((k - 1 + frac) * dt)
    return times


# --------------------------------------------------------------------------- #
# Rezaeian & Der Kiureghian (2010) predictive model
# --------------------------------------------------------------------------- #
#
# PEER Report 2010/02, chapter 4 (eqs. 4.1, 4.13, 4.19, 4.20; Tables 4.3, 4.4,
# 4.5, 4.9) and chapter 5 (sections 5.2, 5.3). The data set is NGA records of
# shallow crustal earthquakes in active regions with 6.0 <= M, 10 km <= R_rup
# <= 100 km, and 600 m/s <= Vs30 (section 4.4), strike-slip or reverse faults.

RDK_PARAM_NAMES = (
    "arias_intensity",
    "d5_95",
    "t_mid",
    "f_mid_hz",
    "f_rate_hz_per_s",
    "zeta_f",
)
FAULT_TYPES = {"strike-slip": 0, "reverse": 1}

# Table 4.5: beta_{i,0..4}, tau_i, sigma_i for v_1..v_6.
RDK2010_COEFFICIENTS = (
    (-1.844, -0.071, 2.944, -1.356, -0.265, 0.274, 0.594),
    (-6.195, -0.703, 6.792, 0.219, -0.523, 0.457, 0.569),
    (-5.011, -0.345, 4.638, 0.348, -0.185, 0.511, 0.414),
    (2.253, -0.081, -1.810, -0.211, 0.012, 0.692, 0.723),
    (-2.489, 0.044, 2.408, 0.065, -0.081, 0.129, 0.953),
    (-0.258, -0.477, 0.905, -0.289, 0.316, 0.682, 0.760),
)
# Table 4.9: correlation of the total errors of v_1..v_6.
RDK2010_CORRELATION = (
    (1.00, -0.36, 0.01, -0.15, 0.13, -0.01),
    (-0.36, 1.00, 0.67, -0.13, -0.16, -0.20),
    (0.01, 0.67, 1.00, -0.28, -0.20, -0.22),
    (-0.15, -0.13, -0.28, 1.00, -0.20, 0.28),
    (0.13, -0.16, -0.20, -0.20, 1.00, -0.01),
    (-0.01, -0.20, -0.22, 0.28, -0.01, 1.00),
)
# Table 4.4 distributions with the Table 4.3 sample means and standard
# deviations (Table 4.4, footnote 8): (mean, std, lower, upper).
_RDK2010_LOGNORMAL_IA = (0.0468, 0.164)
_RDK2010_BETA = {
    "d5_95": (17.25, 9.31, 5.0, 45.0),
    "t_mid": (12.38, 7.44, 0.5, 40.0),
    "zeta_f": (0.213, 0.143, 0.02, 1.0),
}
_RDK2010_GAMMA_F_MID = (5.87, 3.11)
# Eq. (4.13): two-sided truncated exponential density of w'/2pi on [-2, 0.5].
_RDK2010_FRATE = (4.85, 6.77, 17.10, -2.0, 0.5)

_NORMAL = NormalDist()


@dataclass(frozen=True)
class RDKScenario:
    """Earthquake and site: fault type, moment magnitude, closest distance to
    the rupture (km), and Vs30 (m/s)."""

    fault: str
    magnitude: float
    r_rup_km: float
    vs30: float

    def validate(self) -> None:
        if self.fault not in FAULT_TYPES:
            raise ValueError(
                f"fault must be one of {tuple(FAULT_TYPES)}, got {self.fault!r}"
            )
        if not (self.magnitude > 0.0 and self.r_rup_km > 0.0 and self.vs30 > 0.0):
            raise ValueError("magnitude, r_rup_km and vs30 must be positive")

    def outside_data_range(self) -> list[str]:
        """Where the scenario lies outside the regression data (section 4.4)."""
        notes = []
        if self.magnitude < 6.0:
            notes.append(f"M={self.magnitude} < 6.0")
        if not 10.0 <= self.r_rup_km <= 100.0:
            notes.append(f"R_rup={self.r_rup_km} km outside [10, 100] km")
        if self.vs30 < 600.0:
            notes.append(f"Vs30={self.vs30} m/s < 600 m/s")
        return notes


def _beta_shape(
    mean: float, std: float, lower: float, upper: float
) -> tuple[float, float]:
    span = upper - lower
    m = (mean - lower) / span
    s = std / span
    common = m * (1.0 - m) / (s * s) - 1.0
    return m * common, (1.0 - m) * common


def _lognormal_shape(mean: float, std: float) -> tuple[float, float]:
    zeta2 = math.log1p((std / mean) ** 2)
    return math.log(mean) - 0.5 * zeta2, math.sqrt(zeta2)


def _frate_mass() -> tuple[float, float]:
    c, k_neg, k_pos, lower, upper = _RDK2010_FRATE
    neg = c / k_neg * (1.0 - math.exp(k_neg * lower))
    pos = c / k_pos * (1.0 - math.exp(-k_pos * upper))
    return neg, pos


def _frate_cdf(x: float) -> float:
    c, k_neg, k_pos, lower, upper = _RDK2010_FRATE
    neg, pos = _frate_mass()
    if x <= lower:
        return 0.0
    if x >= upper:
        return 1.0
    if x <= 0.0:
        mass = c / k_neg * (math.exp(k_neg * x) - math.exp(k_neg * lower))
    else:
        mass = neg + c / k_pos * (1.0 - math.exp(-k_pos * x))
    # Eq. (4.13) integrates to 0.99993; normalize so the CDF reaches 1.
    return mass / (neg + pos)


def _frate_ppf(p: float) -> float:
    c, k_neg, k_pos, lower, _ = _RDK2010_FRATE
    neg, pos = _frate_mass()
    mass = p * (neg + pos)
    if mass <= neg:
        return math.log(mass * k_neg / c + math.exp(k_neg * lower)) / k_neg
    return -math.log(1.0 - (mass - neg) * k_pos / c) / k_pos


def _clamp_unit(p: float) -> float:
    return min(max(p, 1.0e-16), 1.0 - 1.0e-16)


def rdk2010_to_normal(name: str, value: float) -> float:
    """Eq. (4.1): ``v = Phi^-1(F(theta))`` with the Table 4.4 marginals."""
    if name == "arias_intensity":
        lam, zeta = _lognormal_shape(*_RDK2010_LOGNORMAL_IA)
        if not value > 0.0:
            raise ValueError("arias_intensity must be positive")
        return (math.log(value) - lam) / zeta
    if name in _RDK2010_BETA:
        mean, std, lower, upper = _RDK2010_BETA[name]
        a, b = _beta_shape(mean, std, lower, upper)
        if not lower < value < upper:
            raise ValueError(f"{name}={value} is outside its bounds ({lower}, {upper})")
        p = _beta_cdf((value - lower) / (upper - lower), a, b)
    elif name == "f_mid_hz":
        mean, std = _RDK2010_GAMMA_F_MID
        shape, scale = (mean / std) ** 2, std * std / mean
        if not value > 0.0:
            raise ValueError("f_mid_hz must be positive")
        p = _gamma_p(shape, value / scale)
    elif name == "f_rate_hz_per_s":
        _, _, _, lower, upper = _RDK2010_FRATE
        if not lower < value < upper:
            raise ValueError(f"f_rate_hz_per_s={value} is outside ({lower}, {upper})")
        p = _frate_cdf(value)
    else:
        raise ValueError(f"unknown RDK parameter {name!r}")
    return _NORMAL.inv_cdf(_clamp_unit(p))


def rdk2010_from_normal(name: str, v: float) -> float:
    """Inverse of :func:`rdk2010_to_normal`."""
    if name == "arias_intensity":
        lam, zeta = _lognormal_shape(*_RDK2010_LOGNORMAL_IA)
        return math.exp(lam + zeta * v)
    p = _clamp_unit(_NORMAL.cdf(v))
    if name in _RDK2010_BETA:
        mean, std, lower, upper = _RDK2010_BETA[name]
        a, b = _beta_shape(mean, std, lower, upper)
        return lower + (upper - lower) * _beta_ppf(p, a, b)
    if name == "f_mid_hz":
        mean, std = _RDK2010_GAMMA_F_MID
        return _gamma_ppf(p, (mean / std) ** 2) * (std * std / mean)
    if name == "f_rate_hz_per_s":
        return _frate_ppf(p)
    raise ValueError(f"unknown RDK parameter {name!r}")


def rdk2010_mean_normals(scenario: RDKScenario) -> list[float]:
    """Conditional means ``mu_i`` of ``v_1..v_6`` (eqs. 4.19, 4.20)."""
    scenario.validate()
    f = float(FAULT_TYPES[scenario.fault])
    m = scenario.magnitude / 7.0
    r = scenario.r_rup_km / 25.0
    v = scenario.vs30 / 750.0
    means = []
    for idx, (b0, b1, b2, b3, b4, _, _) in enumerate(RDK2010_COEFFICIENTS):
        if idx == 0:
            means.append(b0 + b1 * f + b2 * m + b3 * math.log(r) + b4 * math.log(v))
        else:
            means.append(b0 + b1 * f + b2 * m + b3 * r + b4 * v)
    return means


def rdk2010_covariance() -> list[list[float]]:
    """Covariance of the total errors: Table 4.9 correlations times
    ``sqrt(tau_i^2 + sigma_i^2)`` (section 5.3)."""
    std = [math.hypot(row[5], row[6]) for row in RDK2010_COEFFICIENTS]
    return [
        [RDK2010_CORRELATION[i][j] * std[i] * std[j] for j in range(6)]
        for i in range(6)
    ]


def _cholesky(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("covariance matrix is not positive definite")
                low[i][j] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def _solve_spd(a: list[list[float]], b: list[float]) -> list[float]:
    low = _cholesky(a)
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
    return x


def rdk2010_conditional(
    scenario: RDKScenario, fixed: dict[str, float] | None = None
) -> tuple[list[int], list[float], list[list[float]], dict[int, float]]:
    """Free indices, their conditional mean and covariance in normal space,
    and the fixed normal values (eqs. 5.3-5.5, section 5.4.2)."""
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(RDK_PARAM_NAMES)
    if unknown:
        raise ValueError(f"unknown RDK parameters {sorted(unknown)}")
    mean = rdk2010_mean_normals(scenario)
    cov = rdk2010_covariance()
    fixed_v = {
        RDK_PARAM_NAMES.index(k): rdk2010_to_normal(k, val) for k, val in fixed.items()
    }
    free = [i for i in range(6) if i not in fixed_v]
    known = sorted(fixed_v)
    if not known:
        return (
            free,
            [mean[i] for i in free],
            [[cov[i][j] for j in free] for i in free],
            fixed_v,
        )
    s_bb = [[cov[i][j] for j in known] for i in known]
    delta = [fixed_v[j] - mean[j] for j in known]
    w = _solve_spd(s_bb, delta)
    cond_mean = [
        mean[i] + sum(cov[i][known[k]] * w[k] for k in range(len(known))) for i in free
    ]
    cond_cov = []
    for i in free:
        row = []
        x = _solve_spd(s_bb, [cov[j][i] for j in known])
        for j in free:
            row.append(
                cov[i][j] - sum(cov[j][known[k]] * x[k] for k in range(len(known)))
            )
        cond_cov.append(row)
    return free, cond_mean, cond_cov, fixed_v


def _params_from_normals(v: Sequence[float], **extra: float) -> RDKParams:
    values = {
        name: rdk2010_from_normal(name, v[i]) for i, name in enumerate(RDK_PARAM_NAMES)
    }
    return RDKParams(**values, **extra)


def rdk2010_median_params(scenario: RDKScenario, **extra: float) -> RDKParams:
    """Parameters at the conditional means of ``v`` (zero regression error)."""
    return _params_from_normals(rdk2010_mean_normals(scenario), **extra)


def rdk2010_sample_params(
    scenario: RDKScenario,
    rng: random.Random,
    fixed: dict[str, float] | None = None,
    **extra: float,
) -> RDKParams:
    """One random draw of the six model parameters for ``scenario``.

    ``v`` is jointly normal with the eq. (4.19)/(4.20) means and the section
    5.3 covariance, conditioned on any ``fixed`` physical values (section
    5.4.2), and mapped back through the Table 4.4 marginals. Draws whose
    ``D5-95 / t_mid`` admits no modulating function with ``alpha2 > 1`` (the
    constraint of eq. 2.23) are redrawn from the same stream.
    """
    free, mean, cov, fixed_v = rdk2010_conditional(scenario, fixed)
    low = _cholesky(cov) if free else []
    limit = max_d5_95_over_t_mid()
    for _ in range(1000):
        z = [rng.gauss(0.0, 1.0) for _ in free]
        v = [0.0] * 6
        for idx, val in fixed_v.items():
            v[idx] = val
        for a, i in enumerate(free):
            v[i] = mean[a] + sum(low[a][b] * z[b] for b in range(a + 1))
        params = _params_from_normals(v, **extra)
        if fixed:
            # Keep the user's exact values instead of their round trip.
            params = replace(params, **fixed)
        if params.d5_95 / params.t_mid < limit:
            return params
    raise ValueError(
        "could not draw D5-95 and t_mid with a valid modulating function; "
        "check the fixed parameters"
    )


def generate_acceleration(params: MotionParams) -> list[float]:
    """Synthetic acceleration history, ``params.npts`` samples at ``params.dt``."""
    params.validate()
    if params.model == "rdk2008":
        return _generate_rdk2008(params)
    return _generate_clough_penzien(params)


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
    if quantity == "acceleration":
        kind = "ACCELERATION TIME HISTORY IN UNITS OF G"
    elif quantity == "displacement":
        kind = "DISPLACEMENT TIME HISTORY IN UNITS OF CM"
    else:
        raise ValueError(f"unknown PEER quantity: {quantity}")
    if params.model == "rdk2008":
        rdk = params.resolved_rdk()
        scaling = (
            "UNSCALED" if params.pga is None else f"SCALED TO PGA {params.pga:.4f} G"
        )
        return [
            "SYNTHETIC STRONG MOTION RECORD. REZAEIAN-DER KIUREGHIAN (2008) MODEL (STRUT).",
            (
                f"SEED {params.seed}, IA {rdk.arias_intensity:.4g} S*G, D5-95 {rdk.d5_95:.3f} S,"
                f" TMID {rdk.t_mid:.3f} S, FMID {rdk.f_mid_hz:.3f} HZ,"
                f" F' {rdk.f_rate_hz_per_s:.4f} HZ/S, ZETA {rdk.zeta_f:.3f},"
                f" FC {rdk.corner_freq_hz:.3f} HZ, {scaling}"
            ),
            f"{kind}. HIGH-PASS FILTERED (CRITICALLY DAMPED OSCILLATOR)",
            f"NPTS={params.npts:6d}, DT= {_peer_dt_text(params.dt)} SEC",
        ]
    hp_freq, hp_damping = params.resolved_high_pass()
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


def rdk2010_params_rng(seed: int) -> random.Random:
    """The stream that draws predictive-model parameters for ``seed``. It is
    separate from the white-noise stream (``random.Random(seed)``), so fixing
    the parameters and re-running with the same seed gives the same record."""
    return random.Random(f"rdk2010-params:{seed}")


_RDK_FLAGS = {
    "arias_intensity": "--arias-intensity",
    "d5_95": "--d5-95",
    "t_mid": "--t-mid",
    "f_mid_hz": "--f-mid-hz",
    "f_rate_hz_per_s": "--f-rate",
    "zeta_f": "--zeta-f",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, nargs="+", help="files to write")
    parser.add_argument("--format", choices=FORMATS, help="default: from each suffix")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--dt", type=float, default=MotionParams.dt)
    parser.add_argument(
        "--npts",
        type=int,
        help=f"default {MotionParams.npts} (clough-penzien) or rdk_default_duration (rdk2008)",
    )
    parser.add_argument("--duration", type=float, help="sets npts = duration/dt + 1")
    parser.add_argument(
        "--pga",
        type=float,
        help=f"scale to this PGA; default {DEFAULT_PGA} for clough-penzien, "
        "unscaled (Arias intensity as requested) for rdk2008",
    )
    cp = parser.add_argument_group("clough-penzien")
    cp.add_argument("--site-freq-hz", type=float, default=MotionParams.site_freq_hz)
    cp.add_argument("--site-damping", type=float, default=MotionParams.site_damping)
    cp.add_argument("--hp-freq-hz", type=float)
    cp.add_argument("--hp-damping", type=float)
    cp.add_argument("--envelope", choices=ENVELOPES, default=MotionParams.envelope)
    rdk = parser.add_argument_group(
        "rdk2008",
        "Give all six physical parameters, or a scenario (--magnitude, --rrup, "
        "--vs30, --fault) to draw them from the Rezaeian & Der Kiureghian (2010) "
        "predictive model; any parameters given with a scenario are held fixed "
        "and the rest are drawn conditionally.",
    )
    rdk.add_argument(
        "--arias-intensity", type=float, help="expected Arias intensity, s*g"
    )
    rdk.add_argument("--d5-95", type=float, help="5-95%% Arias duration, s")
    rdk.add_argument("--t-mid", type=float, help="time of 45%% Arias intensity, s")
    rdk.add_argument("--f-mid-hz", type=float, help="filter frequency at t_mid, Hz")
    rdk.add_argument("--f-rate", type=float, help="filter frequency rate, Hz/s")
    rdk.add_argument("--zeta-f", type=float, help="filter damping ratio")
    rdk.add_argument("--corner-freq-hz", type=float, default=RDKParams.corner_freq_hz)
    rdk.add_argument("--fault", choices=tuple(FAULT_TYPES), default="strike-slip")
    rdk.add_argument("--magnitude", type=float, help="moment magnitude")
    rdk.add_argument("--rrup", type=float, help="closest distance to rupture, km")
    rdk.add_argument("--vs30", type=float, help="Vs30, m/s")
    rdk.add_argument(
        "--median",
        action="store_true",
        help="use the median parameters of the scenario instead of a random draw",
    )
    return parser


def _rdk_from_args(args: argparse.Namespace) -> RDKParams:
    given = {
        name: getattr(args, flag[2:].replace("-", "_"))
        for name, flag in _RDK_FLAGS.items()
        if getattr(args, flag[2:].replace("-", "_")) is not None
    }
    extra = {"corner_freq_hz": args.corner_freq_hz}
    scenario_args = (args.magnitude, args.rrup, args.vs30)
    if all(v is None for v in scenario_args):
        missing = [flag for name, flag in _RDK_FLAGS.items() if name not in given]
        if missing:
            raise SystemExit(
                f"--model rdk2008 needs {', '.join(missing)} or a scenario "
                "(--magnitude, --rrup, --vs30)"
            )
        return RDKParams(**given, **extra)
    if any(v is None for v in scenario_args):
        raise SystemExit("a scenario needs all of --magnitude, --rrup and --vs30")
    scenario = RDKScenario(args.fault, args.magnitude, args.rrup, args.vs30)
    for note in scenario.outside_data_range():
        print(
            f"warning: {note}: outside the data behind the Rezaeian & Der Kiureghian "
            "(2010) equations",
            file=sys.stderr,
        )
    if args.median:
        if given:
            raise SystemExit("--median cannot be combined with fixed parameters")
        return rdk2010_median_params(scenario, **extra)
    return rdk2010_sample_params(
        scenario, rdk2010_params_rng(args.seed), given, **extra
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rdk = _rdk_from_args(args) if args.model == "rdk2008" else None
    if args.duration is not None:
        npts = round(args.duration / args.dt) + 1
    elif args.npts is not None:
        npts = args.npts
    elif rdk is not None:
        npts = math.ceil(rdk_default_duration(rdk) / args.dt) + 1
    else:
        npts = MotionParams.npts
    pga = args.pga
    if pga is None and rdk is None:
        pga = DEFAULT_PGA
    params = MotionParams(
        seed=args.seed,
        dt=args.dt,
        npts=npts,
        pga=pga,
        site_freq_hz=args.site_freq_hz,
        site_damping=args.site_damping,
        hp_freq_hz=args.hp_freq_hz,
        hp_damping=args.hp_damping,
        envelope=args.envelope,
        model=args.model,
        rdk=rdk,
    )
    if rdk is not None:
        print(
            "rdk2008: "
            + ", ".join(
                f"{name}={getattr(rdk, name):.5g}"
                for name in (*RDK_PARAM_NAMES, "corner_freq_hz")
            ),
            file=sys.stderr,
        )
    acc = generate_acceleration(params)
    for path in args.output:
        write_motion(path, params, args.format, acc)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
