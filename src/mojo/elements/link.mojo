from collections import List
from math import sqrt
from os import abort


fn link_global_stiffness(
    dirs: List[Int],
    ks: List[Float64],
) -> List[List[Float64]]:
    if len(dirs) != len(ks):
        abort("dirs/materials length mismatch")

    var k_global: List[List[Float64]] = []
    for _ in range(4):
        var row: List[Float64] = []
        row.resize(4, 0.0)
        k_global.append(row^)

    for i in range(len(dirs)):
        var dof = dirs[i]
        var k = ks[i]
        var a = -1
        var b = -1
        if dof == 1:
            a = 0
            b = 2
        elif dof == 2:
            a = 1
            b = 3
        else:
            abort("unsupported link dir")

        k_global[a][a] += k
        k_global[b][b] += k
        k_global[a][b] -= k
        k_global[b][a] -= k

    return k_global^


fn zero_length_global_stiffness(
    dirs: List[Int],
    ks: List[Float64],
) -> List[List[Float64]]:
    return link_global_stiffness(dirs, ks)


fn two_node_link_global_stiffness(
    dirs: List[Int],
    ks: List[Float64],
) -> List[List[Float64]]:
    return link_global_stiffness(dirs, ks)


fn link_element_dof_count(ndm: Int, ndf: Int) -> Int:
    if ndm == 2 and ndf == 2:
        return 4
    if ndm == 2 and ndf == 3:
        return 6
    if ndm == 3 and ndf == 3:
        return 6
    if ndm == 3 and ndf == 6:
        return 12
    abort("unsupported link ndm/ndf combination")
    return 0


fn zero_length_internal_dir(dir: Int, ndm: Int, ndf: Int) -> Int:
    if ndm == 2 and ndf == 2:
        if dir == 1 or dir == 2:
            return dir
        abort("zeroLength requires dirs 1..2 for ndm=2 ndf=2")
    if ndm == 2 and ndf == 3:
        if dir == 1 or dir == 2 or dir == 6:
            return dir
        if dir == 3:
            return 6
        abort("zeroLength requires dirs 1, 2, or 6 for ndm=2 ndf=3")
    if ndm == 3 and ndf == 3:
        if dir >= 1 and dir <= 3:
            return dir
        abort("zeroLength requires dirs 1..3 for ndm=3 ndf=3")
    if ndm == 3 and ndf == 6:
        if dir >= 1 and dir <= 6:
            return dir
        abort("zeroLength requires dirs 1..6 for ndm=3 ndf=6")
    abort("unsupported zeroLength ndm/ndf combination")
    return 0


fn two_node_link_internal_dir(dir: Int, ndm: Int, ndf: Int) -> Int:
    if ndm == 2 and ndf == 2:
        if dir == 1 or dir == 2:
            return dir
        abort("twoNodeLink requires dirs 1..2 for ndm=2 ndf=2")
    if ndm == 2 and ndf == 3:
        if dir >= 1 and dir <= 3:
            return dir
        abort("twoNodeLink requires dirs 1..3 for ndm=2 ndf=3")
    if ndm == 3 and ndf == 3:
        if dir >= 1 and dir <= 3:
            return dir
        abort("twoNodeLink requires dirs 1..3 for ndm=3 ndf=3")
    if ndm == 3 and ndf == 6:
        if dir >= 1 and dir <= 6:
            return dir
        abort("twoNodeLink requires dirs 1..6 for ndm=3 ndf=6")
    abort("unsupported twoNodeLink ndm/ndf combination")
    return 0


fn _normalize3(v0: Float64, v1: Float64, v2: Float64) -> List[Float64]:
    var norm = sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    if norm == 0.0:
        abort("invalid zero-length orientation vector")
    return [v0 / norm, v1 / norm, v2 / norm]


fn _reference_y_for_x(x0: Float64, x1: Float64, x2: Float64) -> List[Float64]:
    var dot_y = x1
    if dot_y > -0.9 and dot_y < 0.9:
        return [0.0, 1.0, 0.0]
    var dot_x = x0
    if dot_x > -0.9 and dot_x < 0.9:
        return [1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0]


fn link_orientation_matrix(
    ndm: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    has_orient_x: Bool,
    orient_x_1: Float64,
    orient_x_2: Float64,
    orient_x_3: Float64,
    has_orient_y: Bool,
    orient_y_1: Float64,
    orient_y_2: Float64,
    orient_y_3: Float64,
    use_nodes_as_default_x: Bool,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var x_local = [1.0, 0.0, 0.0]
    if use_nodes_as_default_x:
        if ndm == 2:
            var L2 = sqrt(dx * dx + dy * dy)
            if L2 > 0.0:
                x_local = [dx / L2, dy / L2, 0.0]
        else:
            var L3 = sqrt(dx * dx + dy * dy + dz * dz)
            if L3 > 0.0:
                x_local = [dx / L3, dy / L3, dz / L3]
    if has_orient_x:
        x_local = _normalize3(orient_x_1, orient_x_2, orient_x_3)

    var y_ref: List[Float64]
    if ndm == 2:
        y_ref = [-x_local[1], x_local[0], 0.0]
    elif has_orient_y:
        y_ref = _normalize3(orient_y_1, orient_y_2, orient_y_3)
    else:
        y_ref = _reference_y_for_x(x_local[0], x_local[1], x_local[2])
    if ndm == 2 and has_orient_y:
        y_ref = _normalize3(orient_y_1, orient_y_2, orient_y_3)

    var z0 = x_local[1] * y_ref[2] - x_local[2] * y_ref[1]
    var z1v = x_local[2] * y_ref[0] - x_local[0] * y_ref[2]
    var z2v = x_local[0] * y_ref[1] - x_local[1] * y_ref[0]
    var z_local = _normalize3(z0, z1v, z2v)
    var y0 = z_local[1] * x_local[2] - z_local[2] * x_local[1]
    var y1v = z_local[2] * x_local[0] - z_local[0] * x_local[2]
    var y2v = z_local[0] * x_local[1] - z_local[1] * x_local[0]
    var y_local = _normalize3(y0, y1v, y2v)

    return [x_local^, y_local^, z_local^]
