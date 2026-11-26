#!/usr/bin/env python3
"""Compare a real record with synthetic ones: acceleration traces and spectrograms.

Writes one SVG (stdlib only) with three rows: a real PEER ``.AT2`` record, a
Rezaeian & Der Kiureghian (2008) record and a Clough-Penzien record. Both
synthetics use the real record's dt, length and PGA. Each row shows the
acceleration trace and a short-time Fourier spectrogram (Hann window), so the
falling dominant frequency of the real and RDK records, and the stationary
spectrum of the Clough-Penzien record, are visible side by side.

Example (after ``uv run scripts/setup_reference_data.py``):
    uv run scripts/plot_synthetic_ground_motion.py .tmp/gm_compare.svg \\
        --real tests/fixtures/opensees/OpenSeesExamplesAdvanced/H-E12140.AT2
"""

from __future__ import annotations

import argparse
import cmath
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import synthetic_ground_motion as sgm

DEFAULT_REAL = (
    SCRIPT_DIR.parent
    / "tests"
    / "fixtures"
    / "opensees"
    / "OpenSeesExamplesAdvanced"
    / "H-E12140.AT2"
)

ROW_H = 150
TRACE_W = 520
SPEC_W = 420
PAD = 50
MAX_FREQ_HZ = 15.0


def _fft(values: list[complex]) -> list[complex]:
    """Radix-2 FFT; ``len(values)`` must be a power of two."""
    n = len(values)
    if n == 1:
        return values
    even = _fft(values[0::2])
    odd = _fft(values[1::2])
    out = [0j] * n
    for k in range(n // 2):
        tw = cmath.exp(-2j * math.pi * k / n) * odd[k]
        out[k] = even[k] + tw
        out[k + n // 2] = even[k] - tw
    return out


def spectrogram(
    acc: list[float], dt: float, window_s: float = 2.56, hop_s: float = 0.32
) -> tuple[list[float], list[float], list[list[float]]]:
    """Frame centre times, frequencies, and |STFT| (frames x bins)."""
    n = 1 << max(4, round(math.log2(window_s / dt)))
    hop = max(1, round(hop_s / dt))
    hann = [0.5 - 0.5 * math.cos(2.0 * math.pi * k / (n - 1)) for k in range(n)]
    df = 1.0 / (n * dt)
    bins = min(n // 2, int(MAX_FREQ_HZ / df) + 1)
    times, mags = [], []
    for start in range(0, len(acc) - n + 1, hop):
        frame = [complex(acc[start + k] * hann[k]) for k in range(n)]
        spec = _fft(frame)
        mags.append([abs(spec[b]) for b in range(bins)])
        times.append((start + n / 2) * dt)
    return times, [b * df for b in range(bins)], mags


def _trace_svg(acc: list[float], dt: float, x0: float, y0: float) -> list[str]:
    peak = max(abs(a) for a in acc) or 1.0
    step = max(1, len(acc) // 2000)
    duration = (len(acc) - 1) * dt
    pts = " ".join(
        f"{x0 + TRACE_W * k * dt / duration:.1f},"
        f"{y0 + ROW_H / 2 - (ROW_H / 2 - 8) * acc[k] / peak:.1f}"
        for k in range(0, len(acc), step)
    )
    return [
        (
            f'<rect x="{x0}" y="{y0}" width="{TRACE_W}" height="{ROW_H}" fill="none" '
            'stroke="#bbb"/>'
        ),
        f'<polyline points="{pts}" fill="none" stroke="#1f4e79" stroke-width="0.6"/>',
        f'<text x="{x0}" y="{y0 + ROW_H + 14}" font-size="10">0 s</text>',
        (
            f'<text x="{x0 + TRACE_W - 30}" y="{y0 + ROW_H + 14}" font-size="10">'
            f"{duration:.0f} s</text>"
        ),
    ]


def _spec_svg(acc: list[float], dt: float, x0: float, y0: float) -> list[str]:
    times, freqs, mags = spectrogram(acc, dt)
    duration = (len(acc) - 1) * dt
    top = max(max(row) for row in mags) or 1.0
    cell_w = SPEC_W * (times[1] - times[0]) / duration if len(times) > 1 else SPEC_W
    cell_h = ROW_H / len(freqs)
    out = [
        (
            f'<rect x="{x0}" y="{y0}" width="{SPEC_W}" height="{ROW_H}" fill="#fff" '
            'stroke="#bbb"/>'
        )
    ]
    for t, row in zip(times, mags):
        x = x0 + SPEC_W * t / duration - cell_w / 2
        for b, mag in enumerate(row):
            # 40 dB dynamic range on a white-to-dark scale.
            level = 1.0 + 20.0 * math.log10(max(mag / top, 1e-6)) / 40.0
            if level <= 0.0:
                continue
            shade = round(255 * (1.0 - level))
            y = y0 + ROW_H - (b + 1) * cell_h
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w + 0.4:.1f}" '
                f'height="{cell_h + 0.4:.1f}" fill="rgb({shade},{shade},255)"/>'
            )
    out.append(
        f'<text x="{x0 - 30}" y="{y0 + 8}" font-size="10">{MAX_FREQ_HZ:.0f} Hz</text>'
    )
    out.append(f'<text x="{x0 - 30}" y="{y0 + ROW_H}" font-size="10">0 Hz</text>')
    return out


def build_svg(rows: list[tuple[str, list[float], float]]) -> str:
    width = PAD + TRACE_W + PAD + SPEC_W + PAD
    height = PAD + len(rows) * (ROW_H + PAD)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            'font-family="sans-serif">'
        ),
        (
            f'<text x="{PAD}" y="24" font-size="13">Acceleration (scaled to row peak)'
            "</text>"
        ),
        (
            f'<text x="{2 * PAD + TRACE_W}" y="24" font-size="13">'
            "Spectrogram, |STFT| 2.56 s Hann, 40 dB range</text>"
        ),
    ]
    for idx, (label, acc, dt) in enumerate(rows):
        y0 = PAD + idx * (ROW_H + PAD)
        parts.append(f'<text x="{PAD}" y="{y0 - 6}" font-size="12">{label}</text>')
        parts += _trace_svg(acc, dt, PAD, y0)
        parts += _spec_svg(acc, dt, 2 * PAD + TRACE_W, y0)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="SVG file to write")
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fault", choices=tuple(sgm.FAULT_TYPES), default="reverse")
    parser.add_argument("--magnitude", type=float, default=6.5)
    parser.add_argument("--rrup", type=float, default=15.0)
    parser.add_argument("--vs30", type=float, default=600.0)
    args = parser.parse_args(argv)

    if not args.real.is_file():
        raise SystemExit(
            f"{args.real} not found; run: uv run scripts/setup_reference_data.py"
        )
    real = sgm.read_motion(args.real)
    if real.dt is None:
        raise SystemExit(f"{args.real} has no PEER NPTS/DT header")
    dt, npts = real.dt, len(real.values)
    pga = max(abs(v) for v in real.values)

    scenario = sgm.RDKScenario(args.fault, args.magnitude, args.rrup, args.vs30)
    rng = sgm.rdk2010_params_rng(args.seed)
    for _ in range(1000):
        rdk = sgm.rdk2010_sample_params(scenario, rng)
        mod = sgm.gamma_modulation(rdk.arias_intensity, rdk.d5_95, rdk.t_mid)
        if mod.arias_time(0.99) <= (npts - 1) * dt:
            break
    else:
        raise SystemExit("no RDK draw fits the record length")
    rdk_acc = sgm.generate_acceleration(
        sgm.MotionParams(
            seed=args.seed, dt=dt, npts=npts, pga=pga, model="rdk2008", rdk=rdk
        )
    )
    cp_acc = sgm.generate_acceleration(
        sgm.MotionParams(seed=args.seed, dt=dt, npts=npts, pga=pga)
    )
    rdk_label = (
        f"Rezaeian-Der Kiureghian, seed {args.seed}: {args.fault} M{args.magnitude:g}, "
        f"{args.rrup:g} km, Vs30 {args.vs30:g} (D5-95 {rdk.d5_95:.1f} s, "
        f"f_mid {rdk.f_mid_hz:.2f} Hz, f' {rdk.f_rate_hz_per_s:.3f} Hz/s)"
    )
    rows = [
        (f"Real: {args.real.name}", real.values, dt),
        (rdk_label, rdk_acc, dt),
        (f"Clough-Penzien, seed {args.seed} (stationary spectrum)", cp_acc, dt),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(rows), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
