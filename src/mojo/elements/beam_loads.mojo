from collections import List

from solver.run_case.input_types import ElementLoadInput
from tag_types import ElementLoadTypeTag


fn beam2d_section_load_response(
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    elem_index: Int,
    load_scale: Float64,
    x: Float64,
    l: Float64,
) -> (Float64, Float64, Float64):
    var axial = 0.0
    var moment_z = 0.0
    var shear_y = 0.0
    if (
        load_scale == 0.0
        or elem_index < 0
        or elem_index + 1 >= len(elem_load_offsets)
        or len(elem_load_pool) == 0
    ):
        return (axial, moment_z, shear_y)

    for slot in range(elem_load_offsets[elem_index], elem_load_offsets[elem_index + 1]):
        if slot < 0 or slot >= len(elem_load_pool):
            continue
        var load_index = elem_load_pool[slot]
        if load_index < 0 or load_index >= len(element_loads):
            continue
        var load = element_loads[load_index]
        if load.type_tag == ElementLoadTypeTag.BeamUniform:
            var wa = load.wx * load_scale
            var wy = load.wy * load_scale
            axial += wa * (l - x)
            moment_z += wy * 0.5 * x * (x - l)
            shear_y += wy * (x - 0.5 * l)
        elif load.type_tag == ElementLoadTypeTag.BeamPoint:
            var p = load.py * load_scale
            var n = load.px * load_scale
            var a_over_l = load.x
            if a_over_l < 0.0 or a_over_l > 1.0:
                continue
            var a = a_over_l * l
            var v1 = p * (1.0 - a_over_l)
            var v2 = p * a_over_l
            if x <= a:
                axial += n
                moment_z -= x * v1
                shear_y -= v1
            else:
                moment_z -= (l - x) * v2
                shear_y += v2
    return (axial, moment_z, shear_y)


fn beam3d_section_load_response(
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    elem_index: Int,
    load_scale: Float64,
    x: Float64,
    l: Float64,
) -> (Float64, Float64, Float64, Float64, Float64):
    var axial = 0.0
    var moment_y = 0.0
    var moment_z = 0.0
    var shear_y = 0.0
    var shear_z = 0.0
    if (
        load_scale == 0.0
        or elem_index < 0
        or elem_index + 1 >= len(elem_load_offsets)
        or len(elem_load_pool) == 0
    ):
        return (axial, moment_y, moment_z, shear_y, shear_z)

    for slot in range(elem_load_offsets[elem_index], elem_load_offsets[elem_index + 1]):
        if slot < 0 or slot >= len(elem_load_pool):
            continue
        var load_index = elem_load_pool[slot]
        if load_index < 0 or load_index >= len(element_loads):
            continue
        var load = element_loads[load_index]
        if load.type_tag == ElementLoadTypeTag.BeamUniform:
            var wa = load.wx * load_scale
            var wy = load.wy * load_scale
            var wz = load.wz * load_scale
            axial += wa * (l - x)
            moment_z += wy * 0.5 * x * (x - l)
            shear_y += wy * (x - 0.5 * l)
            moment_y += wz * 0.5 * x * (l - x)
            shear_z += wz * (0.5 * l - x)
        elif load.type_tag == ElementLoadTypeTag.BeamPoint:
            var py = load.py * load_scale
            var pz = load.pz * load_scale
            var n = load.px * load_scale
            var a_over_l = load.x
            if a_over_l < 0.0 or a_over_l > 1.0:
                continue
            var a = a_over_l * l
            var vy1 = py * (1.0 - a_over_l)
            var vy2 = py * a_over_l
            var vz1 = pz * (1.0 - a_over_l)
            var vz2 = pz * a_over_l
            if x <= a:
                axial += n
                moment_z -= x * vy1
                shear_y -= vy1
                moment_y += x * vz1
                shear_z += vz1
            else:
                moment_z -= (l - x) * vy2
                shear_y += vy2
                moment_y += (l - x) * vz2
                shear_z -= vz2
    return (axial, moment_y, moment_z, shear_y, shear_z)


fn beam2d_basic_fixed_end_and_reactions(
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    elem_index: Int,
    load_scale: Float64,
    l: Float64,
) -> (Float64, Float64, Float64, Float64, Float64, Float64):
    var q0_0 = 0.0
    var q0_1 = 0.0
    var q0_2 = 0.0
    var p0_0 = 0.0
    var p0_1 = 0.0
    var p0_2 = 0.0
    if (
        load_scale == 0.0
        or elem_index < 0
        or elem_index + 1 >= len(elem_load_offsets)
        or len(elem_load_pool) == 0
    ):
        return (q0_0, q0_1, q0_2, p0_0, p0_1, p0_2)

    for slot in range(elem_load_offsets[elem_index], elem_load_offsets[elem_index + 1]):
        if slot < 0 or slot >= len(elem_load_pool):
            continue
        var load_index = elem_load_pool[slot]
        if load_index < 0 or load_index >= len(element_loads):
            continue
        var load = element_loads[load_index]
        if load.type_tag == ElementLoadTypeTag.BeamUniform:
            var wt = load.wy * load_scale
            var wa = load.wx * load_scale
            var v = 0.5 * wt * l
            var m = v * l / 6.0
            var p = wa * l
            p0_0 -= p
            p0_1 -= v
            p0_2 -= v
            q0_0 -= 0.5 * p
            q0_1 -= m
            q0_2 += m
        elif load.type_tag == ElementLoadTypeTag.BeamPoint:
            var p = load.py * load_scale
            var n = load.px * load_scale
            var a_over_l = load.x
            if a_over_l < 0.0 or a_over_l > 1.0:
                continue
            var a = a_over_l * l
            var b = l - a
            var v1 = p * (1.0 - a_over_l)
            var v2 = p * a_over_l
            p0_0 -= n
            p0_1 -= v1
            p0_2 -= v2
            var inv_l2 = 1.0 / (l * l)
            q0_0 -= n * a_over_l
            q0_1 += -a * b * b * p * inv_l2
            q0_2 += a * a * b * p * inv_l2
    return (q0_0, q0_1, q0_2, p0_0, p0_1, p0_2)


fn beam3d_basic_fixed_end_and_reactions(
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    elem_index: Int,
    load_scale: Float64,
    l: Float64,
) -> (
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
):
    var q0_0 = 0.0
    var q0_1 = 0.0
    var q0_2 = 0.0
    var q0_3 = 0.0
    var q0_4 = 0.0
    var p0_0 = 0.0
    var p0_1 = 0.0
    var p0_2 = 0.0
    var p0_3 = 0.0
    var p0_4 = 0.0
    if (
        load_scale == 0.0
        or elem_index < 0
        or elem_index + 1 >= len(elem_load_offsets)
        or len(elem_load_pool) == 0
    ):
        return (q0_0, q0_1, q0_2, q0_3, q0_4, p0_0, p0_1, p0_2, p0_3, p0_4)

    for slot in range(elem_load_offsets[elem_index], elem_load_offsets[elem_index + 1]):
        if slot < 0 or slot >= len(elem_load_pool):
            continue
        var load_index = elem_load_pool[slot]
        if load_index < 0 or load_index >= len(element_loads):
            continue
        var load = element_loads[load_index]
        if load.type_tag == ElementLoadTypeTag.BeamUniform:
            var wy = load.wy * load_scale
            var wz = load.wz * load_scale
            var wx = load.wx * load_scale
            var vy = 0.5 * wy * l
            var vz = 0.5 * wz * l
            var mz = vy * l / 6.0
            var my = vz * l / 6.0
            var p = wx * l
            p0_0 -= p
            p0_1 -= vy
            p0_2 -= vy
            p0_3 -= vz
            p0_4 -= vz
            q0_0 -= 0.5 * p
            q0_1 -= mz
            q0_2 += mz
            q0_3 += my
            q0_4 -= my
        elif load.type_tag == ElementLoadTypeTag.BeamPoint:
            var py = load.py * load_scale
            var pz = load.pz * load_scale
            var n = load.px * load_scale
            var a_over_l = load.x
            if a_over_l < 0.0 or a_over_l > 1.0:
                continue
            var a = a_over_l * l
            var b = l - a
            p0_0 -= n
            var vy1 = py * (1.0 - a_over_l)
            var vy2 = py * a_over_l
            var vz1 = pz * (1.0 - a_over_l)
            var vz2 = pz * a_over_l
            p0_1 -= vy1
            p0_2 -= vy2
            p0_3 -= vz1
            p0_4 -= vz2
            var inv_l2 = 1.0 / (l * l)
            q0_0 -= n * a_over_l
            var m1 = -a * b * b * py * inv_l2
            var m2 = a * a * b * py * inv_l2
            q0_1 += m1
            q0_2 += m2
            m1 = -a * b * b * pz * inv_l2
            m2 = a * a * b * pz * inv_l2
            q0_3 -= m1
            q0_4 -= m2
    return (q0_0, q0_1, q0_2, q0_3, q0_4, p0_0, p0_1, p0_2, p0_3, p0_4)
