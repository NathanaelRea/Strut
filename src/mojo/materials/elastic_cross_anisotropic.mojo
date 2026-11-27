from collections import InlineArray
from linalg import lu_factorize_into, lu_solve_into


fn cross_anisotropic_tangent(
    eh: Float64, ev: Float64, nuhv: Float64, nuhh: Float64, ghv: Float64
) -> InlineArray[Float64, 36]:
    # The horizontal plane is x-y. Components use engineering shear strains.
    var compliance = [
        [1.0 / eh, -nuhh / eh, -nuhv / ev],
        [-nuhh / eh, 1.0 / eh, -nuhv / ev],
        [-nuhv / ev, -nuhv / ev, 1.0 / ev],
    ]
    var pivots: List[Int] = []
    lu_factorize_into(compliance, pivots)
    var tangent = InlineArray[Float64, 36](fill=0.0)
    var work: List[Float64] = []
    var result: List[Float64] = []
    for column in range(3):
        var rhs: List[Float64] = [0.0, 0.0, 0.0]
        rhs[column] = 1.0
        lu_solve_into(compliance, pivots, rhs, work, result)
        for row in range(3):
            tangent[6 * row + column] = result[row]
    tangent[21] = eh / (2.0 * (1.0 + nuhh))
    tangent[28] = ghv
    tangent[35] = ghv
    return tangent
