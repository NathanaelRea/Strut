from elements.beam_integration import BeamIntegrationCache, beam_integration_rule
from elements.utils import _ensure_zero_matrix
from linalg import gaussian_elimination, lu_factorize_into, lu_solve_into


@fieldwise_init
struct CbdiSettings(ImplicitlyCopyable, Movable):
    var max_iters: Int
    var tolerance: Float64

    fn enabled(self) -> Bool:
        return self.max_iters > 0


fn _curvature_influence(xis: List[Float64], length: Float64) -> List[List[Float64]]:
    # Twice integrate the interpolating curvature polynomial and impose
    # zero transverse displacement at both element ends.
    var n = len(xis)
    var vandermonde: List[List[Float64]] = []
    var influence: List[List[Float64]] = []
    _ensure_zero_matrix(vandermonde, n, n)
    _ensure_zero_matrix(influence, n, n)
    for i in range(n):
        var power = 1.0
        for k in range(n):
            vandermonde[k][i] = power
            power *= xis[i]
    var pivots: List[Int] = []
    var integrals: List[Float64] = []
    var work: List[Float64] = []
    integrals.resize(n, 0.0)
    lu_factorize_into(vandermonde, pivots)
    for i in range(n):
        var x = xis[i]
        var power = x * x
        for k in range(n):
            integrals[k] = length * length * (power - x) / Float64((k + 1) * (k + 2))
            power *= x
        lu_solve_into(vandermonde, pivots, integrals, work, influence[i])
    return influence^


fn cbdi_elastic_curvature(
    length: Float64,
    ei: Float64,
    integration: String,
    num_int_pts: Int,
    point: Int,
    axial: Float64,
    moment_i: Float64,
    moment_j: Float64,
    particular_moments: List[Float64] = List[Float64](),
) -> Float64:
    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var matrix = _curvature_influence(xis, length)
    var moments: List[Float64] = []
    for i in range(num_int_pts):
        for j in range(num_int_pts):
            matrix[i][j] *= -axial
        matrix[i][i] += ei
        var particular = 0.0 if len(particular_moments) == 0 else particular_moments[i]
        moments.append((xis[i] - 1.0) * moment_i + xis[i] * moment_j + particular)
    return gaussian_elimination(matrix, moments)[point]


fn cbdi_elastic_basic_state(
    length: Float64,
    axial_rigidity: Float64,
    bending_rigidities: List[Float64],
    quadrature: BeamIntegrationCache,
    section_loads: List[List[Float64]],
    target: List[Float64],
    settings: CbdiSettings,
    mut q: List[Float64],
    mut tangent: List[List[Float64]],
):
    var n = quadrature.num_int_pts
    var basic_size = len(target)
    var influence = _curvature_influence(quadrature.xis, length)
    var section_matrix: List[List[Float64]] = []
    var flexibility: List[List[Float64]] = []
    var section_derivatives: List[List[Float64]] = []
    _ensure_zero_matrix(section_matrix, n, n)
    _ensure_zero_matrix(section_derivatives, 3, n)
    _ensure_zero_matrix(flexibility, basic_size, basic_size)
    _ensure_zero_matrix(tangent, basic_size, basic_size)
    q.resize(basic_size, 0.0)
    var section_pivots: List[Int] = []
    var basic_pivots: List[Int] = []
    var curvature: List[Float64] = []
    var transverse: List[Float64] = []
    var section_rhs: List[Float64] = []
    var section_work: List[Float64] = []
    var basic_rhs: List[Float64] = []
    var basic_work: List[Float64] = []
    var increment: List[Float64] = []
    section_rhs.resize(n, 0.0)
    transverse.resize(n, 0.0)
    basic_rhs.resize(basic_size, 0.0)

    # For an elastic section and a fixed axial force the section equilibrium
    # equations are linear in curvature. Differentiate that system to obtain
    # the basic flexibility used in the compatibility iteration.
    var axial_flexibility = length / axial_rigidity
    q[0] = target[0] / axial_flexibility
    for iteration in range(settings.max_iters):
        _ensure_zero_matrix(flexibility, basic_size, basic_size)
        for i in range(basic_size):
            basic_rhs[i] = target[i]
        basic_rhs[0] -= q[0] * axial_flexibility
        flexibility[0][0] = axial_flexibility
        for ip in range(n):
            basic_rhs[0] -= quadrature.weights[ip] * axial_flexibility * section_loads[0][ip]
        for plane in range(len(bending_rigidities)):
            var i_end = 1 + 2 * plane
            var j_end = i_end + 1
            for i in range(n):
                var x = quadrature.xis[i]
                for j in range(n):
                    section_matrix[i][j] = -q[0] * influence[i][j]
                section_matrix[i][i] += bending_rigidities[plane]
                section_rhs[i] = (x - 1.0) * q[i_end] + x * q[j_end] + section_loads[plane + 1][i]
            lu_factorize_into(section_matrix, section_pivots)
            lu_solve_into(section_matrix, section_pivots, section_rhs, section_work, curvature)
            for i in range(n):
                transverse[i] = 0.0
                for j in range(n):
                    transverse[i] += influence[i][j] * curvature[j]
                section_rhs[i] = transverse[i]
            lu_solve_into(
                section_matrix, section_pivots, section_rhs, section_work, section_derivatives[0]
            )
            for i in range(n):
                section_rhs[i] = quadrature.xis[i] - 1.0
            lu_solve_into(
                section_matrix, section_pivots, section_rhs, section_work, section_derivatives[1]
            )
            for i in range(n):
                section_rhs[i] = quadrature.xis[i]
            lu_solve_into(
                section_matrix, section_pivots, section_rhs, section_work, section_derivatives[2]
            )

            for i in range(n):
                var x = quadrature.xis[i]
                var weight = quadrature.weights[i] * length
                var weighted_curvature = weight * curvature[i]
                basic_rhs[0] -= 0.5 * transverse[i] * weighted_curvature
                basic_rhs[i_end] -= (x - 1.0) * weighted_curvature
                basic_rhs[j_end] -= x * weighted_curvature
                for local_column in range(3):
                    var column = 0 if local_column == 0 else i_end + local_column - 1
                    var d_curvature = section_derivatives[local_column][i]
                    var d_transverse = 0.0
                    for j in range(n):
                        d_transverse += influence[i][j] * section_derivatives[local_column][j]
                    flexibility[0][column] += (
                        0.5 * weight * (transverse[i] * d_curvature + d_transverse * curvature[i])
                    )
                    flexibility[i_end][column] += weight * (x - 1.0) * d_curvature
                    flexibility[j_end][column] += weight * x * d_curvature

        lu_factorize_into(flexibility, basic_pivots)
        lu_solve_into(flexibility, basic_pivots, basic_rhs, basic_work, increment)
        var residual_work = 0.0
        var relative_correction = 0.0
        for i in range(basic_size):
            residual_work += basic_rhs[i] * increment[i]
            relative_correction = max(relative_correction, abs(increment[i]) / max(1.0, abs(q[i])))
        if (
            abs(residual_work) < settings.tolerance and relative_correction < 8.0e-15
        ) or iteration + 1 == settings.max_iters:
            break
        for i in range(basic_size):
            q[i] += increment[i]

    for column in range(basic_size):
        for i in range(basic_size):
            basic_rhs[i] = 1.0 if i == column else 0.0
        lu_solve_into(flexibility, basic_pivots, basic_rhs, basic_work, increment)
        for i in range(basic_size):
            tangent[i][column] = increment[i]
