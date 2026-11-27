from collections import List
from math import sqrt


fn krylov_apply_qr_acceleration(
    mut update: List[Float64],
    history_v: List[List[Float64]],
    mut history_av: List[List[Float64]],
    mut columns: List[List[Float64]],
    mut rhs: List[Float64],
):
    """Solve the Krylov least-squares problem with Householder QR."""
    var dimension = len(history_v)
    if dimension == 0:
        return

    var count = len(update)
    for i in range(count):
        history_av[dimension - 1][i] -= update[i]

    columns = history_av.copy()
    rhs = update.copy()
    for k in range(dimension):
        ref column = columns[k]
        var tail_sq = 0.0
        for i in range(k + 1, count):
            tail_sq += column[i] * column[i]
        if tail_sq == 0.0:
            if column[k] == 0.0:
                return
            continue

        var diagonal = column[k]
        var reflected = sqrt(diagonal * diagonal + tail_sq)
        if diagonal >= 0.0:
            reflected = -reflected
        var tau = (reflected - diagonal) / reflected
        var inverse_scale = 1.0 / (diagonal - reflected)
        for i in range(k + 1, count):
            column[i] *= inverse_scale
        column[k] = reflected

        for j in range(k + 1, dimension):
            ref target = columns[j]
            var projection = target[k]
            for i in range(k + 1, count):
                projection += column[i] * target[i]
            projection *= tau
            target[k] -= projection
            for i in range(k + 1, count):
                target[i] -= column[i] * projection

        var projection = rhs[k]
        for i in range(k + 1, count):
            projection += column[i] * rhs[i]
        projection *= tau
        rhs[k] -= projection
        for i in range(k + 1, count):
            rhs[i] -= column[i] * projection

    for j in range(dimension - 1, -1, -1):
        rhs[j] /= columns[j][j]
        for i in range(j):
            rhs[i] -= columns[j][i] * rhs[j]
    for j in range(dimension):
        for i in range(count):
            update[i] += rhs[j] * history_v[j][i]
        for i in range(count):
            update[i] -= rhs[j] * history_av[j][i]
