from collections import List
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
