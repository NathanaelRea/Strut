"""Fiber cell geometry shared by the 2D and 3D fiber sections."""

from collections import InlineArray


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
