"""Fiber cell geometry shared by the 2D and 3D fiber sections."""

from collections import InlineArray
from math import acos, cos, sin


@always_inline
fn quad_cell_centroid(
    y0: Float64,
    z0: Float64,
    y1: Float64,
    z1: Float64,
    y2: Float64,
    z2: Float64,
    y3: Float64,
    z3: Float64,
) -> (Float64, Float64):
    """Area centroid of one quadrilateral cell, vertices in order around it.

    OpenSees puts each fiber of a `patch quad` at its cell's area centroid
    (QuadCell::getCentroidPosition), which is not the image of the cell's
    parametric center unless the cell is a parallelogram. The operation order
    follows OpenSees.
    """
    var y = InlineArray[Float64, 4](fill=y0)
    var z = InlineArray[Float64, 4](fill=z0)
    y[1] = y1
    y[2] = y2
    y[3] = y3
    z[1] = z1
    z[2] = z2
    z[3] = z3
    var area = 0.0
    for i in range(4):
        var j = (i + 1) % 4
        area += (z[j] - z[i]) * (y[j] + y[i])
    area /= 2.0
    var cy = 0.0
    var cz = 0.0
    for i in range(4):
        var j = (i + 1) % 4
        var dy = y[j] - y[i]
        var dz = z[j] - z[i]
        var integral = y[i] * z[i] + (y[i] * dz + z[i] * dy) / 2.0 + dy * dz / 3.0
        cy -= dy * integral
        cz += dz * integral
    return (cy / area, cz / area)


@always_inline
fn circ_cell_area_and_centroid(
    r1: Float64,
    r2: Float64,
    alpha: Float64,
    theta_mid: Float64,
    y_center: Float64,
    z_center: Float64,
) -> (Float64, Float64, Float64):
    """Area and centroid (y, z) of one circular sector cell, matching OpenSees CircSectionCell."""
    var a = alpha / 2.0
    var at = a * r2 * r2
    var ct = 2.0 * r2 * sin(a) / (3.0 * a)
    var a1 = a * r1 * r1
    var c1 = 2.0 * r1 * sin(a) / (3.0 * a)
    var area = at - a1
    var c = (at * ct - a1 * c1) / area
    var cy = cos(theta_mid) * c + y_center
    var cz = sin(theta_mid) * c + z_center
    return (area, cy, cz)


@always_inline
fn circ_layer_effective_angles(
    num_bars: Int, start_ang: Float64, end_ang: Float64, has_angles: Bool
) -> (Float64, Float64):
    """Return (init_ang_rad, dtheta_rad) matching OpenSees CircReinfLayer."""
    var init_deg = start_ang
    var final_deg = end_ang
    if not has_angles:
        init_deg = 0.0
        final_deg = 360.0 - 360.0 / Float64(num_bars) if num_bars > 0 else 0.0
    else:
        if final_deg - init_deg >= 360.0 and num_bars > 0:
            final_deg = init_deg + 360.0 - 360.0 / Float64(num_bars)
    var pi = acos(-1.0)
    var init_rad = pi * init_deg / 180.0
    var final_rad = pi * final_deg / 180.0
    var dtheta = (final_rad - init_rad) / Float64(num_bars - 1) if num_bars > 1 else 0.0
    return (init_rad, dtheta)
