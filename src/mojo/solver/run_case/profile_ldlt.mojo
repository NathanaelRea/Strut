from collections import List
from os import abort


fn factor_profile_ldlt(
    first_row: List[Int],
    col_start: List[Int],
    mut values: List[Float64],
    mut inverse_diagonal: List[Float64],
):
    var columns = len(first_row)
    inverse_diagonal.resize(columns, 0.0)
    for col in range(columns):
        var first = first_row[col]
        var offset = col_start[col] - first
        for row in range(first, col):
            var row_offset = col_start[row] - first_row[row]
            var value = values[offset + row]
            for k in range(max(first, first_row[row]), row):
                value -= values[offset + k] * values[row_offset + k]
            values[offset + row] = value
        var diagonal = values[offset + col]
        for row in range(first, col):
            var reduced = values[offset + row]
            var multiplier = reduced * inverse_diagonal[row]
            values[offset + row] = multiplier
            diagonal -= multiplier * reduced
        if diagonal <= 0.0 if col == 0 else abs(diagonal) <= 1.0e-12:
            abort("ProfileSPD has a zero or inadmissible pivot in the coupled system")
        inverse_diagonal[col] = 1.0 / diagonal


fn solve_profile_ldlt(
    first_row: List[Int],
    col_start: List[Int],
    values: List[Float64],
    inverse_diagonal: List[Float64],
    rhs: List[Float64],
    mut solution: List[Float64],
):
    var n = len(rhs)
    solution.resize(n, 0.0)
    for col in range(n):
        var correction = 0.0
        var first = first_row[col]
        var offset = col_start[col] - first
        for row in range(first, col):
            correction -= values[offset + row] * solution[row]
        solution[col] = rhs[col] + correction
    for col in range(n):
        solution[col] *= inverse_diagonal[col]
    for col in range(n - 1, 0, -1):
        var first = first_row[col]
        var offset = col_start[col] - first
        for row in range(first, col):
            solution[row] -= values[offset + row] * solution[col]
