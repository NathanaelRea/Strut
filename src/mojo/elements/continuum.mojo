from collections import InlineArray, List
from linalg import lu_factorize_into, lu_solve_into
from math import sqrt
from materials.drucker_prager import (
    DruckerPragerDefinition,
    DruckerPragerHistory,
    DruckerPragerTrial,
    DruckerPragerVector,
    drucker_prager_trial,
)
from os import abort

alias ContinuumVector = SIMD[DType.float64, 4]
alias ContinuumTangent = InlineArray[Float64, 36]
alias Jacobian = InlineArray[Float64, 9]

alias QUAD_R = InlineArray[Int, 9](-1, 1, 1, -1, 0, 1, 0, -1, 0)
alias QUAD_S = InlineArray[Int, 9](-1, -1, 1, 1, -1, 0, 1, 0, 0)
alias BRICK_R = InlineArray[Int, 27](
    -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 0, 0
)
alias BRICK_S = InlineArray[Int, 27](
    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, 0, 0, 0
)
alias BRICK_T = InlineArray[Int, 27](
    -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, -1, 1, 0
)
alias GAUSS_2 = 0.5773502691896258
alias GAUSS_3 = 0.7745966692414834


fn continuum_nodes(kind: String) -> Int:
    if (
        kind == "quad8n"
        or kind == "stdBrick"
        or kind == "bbarBrick"
        or kind == "SSPbrick"
        or kind == "Brick8N"
    ):
        return 8
    if kind == "quad9n":
        return 9
    if kind == "Brick27N":
        return 27
    if kind == "quad3d" or kind == "enhancedQuad" or kind == "bbarQuad" or kind == "SSPquad":
        return 4
    return 0


fn continuum_dimension(kind: String) -> Int:
    if (
        kind == "stdBrick"
        or kind == "bbarBrick"
        or kind == "SSPbrick"
        or kind == "Brick8N"
        or kind == "Brick27N"
        or kind == "quad3d"
    ):
        return 3
    return 2


fn isotropic_continuum_tangent(
    E: Float64, nu: Float64, components: Int, plane_strain: Bool
) -> ContinuumTangent:
    var tangent = ContinuumTangent(fill=0.0)
    var shear = E / (2.0 * (1.0 + nu))
    if components == 3 and not plane_strain:
        tangent[0] = E / (1.0 - nu * nu)
        tangent[7] = tangent[0]
        tangent[1] = nu * tangent[0]
        tangent[6] = tangent[1]
        tangent[14] = shear
        return tangent
    var lame = 2.0 * shear * nu / (1.0 - 2.0 * nu)
    var normals = 3 if components == 6 else 2
    for i in range(normals):
        for j in range(normals):
            tangent[6 * i + j] = lame
        tangent[7 * i] += 2.0 * shear
    for i in range(normals, components):
        tangent[7 * i] = shear
    return tangent


fn _basis(x: Float64, sign: Int, quadratic: Bool) -> Tuple[Float64, Float64]:
    var s = Float64(sign)
    if not quadratic:
        return (0.5 * (1.0 + s * x), 0.5 * s)
    if sign == 0:
        return (1.0 - x * x, -2.0 * x)
    return (0.5 * x * (x + s), x + 0.5 * s)


fn _quadrilateral_shape(r: Float64, s: Float64, a: Int, count: Int) -> ContinuumVector:
    var i = QUAD_R[a]
    var j = QUAD_S[a]
    if count == 8 and a < 4:
        var x = Float64(i)
        var y = Float64(j)
        var rx = 1.0 + x * r
        var sy = 1.0 + y * s
        var edge = x * r + y * s - 1.0
        return ContinuumVector(
            0.25 * rx * sy * edge, 0.25 * x * sy * (edge + rx), 0.25 * y * rx * (edge + sy), 0.0
        )
    var rx = _basis(r, i, count == 9 or i == 0)
    var sy = _basis(s, j, count == 9 or j == 0)
    return ContinuumVector(rx[0] * sy[0], rx[1] * sy[0], rx[0] * sy[1], 0.0)


fn _brick_shape(r: Float64, s: Float64, t: Float64, a: Int, count: Int) -> ContinuumVector:
    var quadratic = count == 27
    var rx = _basis(r, BRICK_R[a], quadratic)
    var sy = _basis(s, BRICK_S[a], quadratic)
    var tz = _basis(t, BRICK_T[a], quadratic)
    return ContinuumVector(
        rx[0] * sy[0] * tz[0], rx[1] * sy[0] * tz[0], rx[0] * sy[1] * tz[0], rx[0] * sy[0] * tz[1]
    )


@fieldwise_init
struct ContinuumPoint(Copyable, Movable):
    var weight: Float64
    var shape: List[ContinuumVector]
    var jacobian: Jacobian
    var inverse: Jacobian
    var determinant: Float64
    var natural: ContinuumVector


fn _point(
    coords: List[ContinuumVector],
    axes: InlineArray[Int, 3],
    dim: Int,
    r: Float64,
    s: Float64,
    t: Float64,
    weight: Float64,
    membrane: Bool = False,
) -> ContinuumPoint:
    var shape: List[ContinuumVector] = []
    var jacobian = Jacobian(fill=0.0)
    for a in range(len(coords)):
        var value = _brick_shape(r, s, t, a, len(coords)) if dim == 3 else _quadrilateral_shape(
            r, s, a, len(coords)
        )
        shape.append(value)
        for i in range(dim):
            for j in range(dim):
                jacobian[3 * i + j] += (coords[a][axes[i]] - coords[0][axes[i]]) * value[j + 1]
    if membrane:
        # FourNodeQuad3d uses the second node's global x coordinate in J[0,1].
        jacobian[1] += 0.25 * (1.0 + r) * (coords[1][axes[0]] - coords[1][0])
    if dim == 2:
        jacobian[8] = 1.0
    var inverse = Jacobian(fill=0.0)
    for i in range(3):
        var j = (i + 1) % 3
        var k = (i + 2) % 3
        for d in range(3):
            var e = (d + 1) % 3
            var f = (d + 2) % 3
            inverse[3 * d + i] = (
                jacobian[3 * j + e] * jacobian[3 * k + f]
                - jacobian[3 * j + f] * jacobian[3 * k + e]
            )
    var det = jacobian[0] * inverse[0] + jacobian[1] * inverse[3] + jacobian[2] * inverse[6]
    # Legacy brick connectivity can traverse the natural thickness in reverse.
    if det == 0.0:
        abort("singular continuum element Jacobian")
    for i in range(9):
        inverse[i] /= det
    for a in range(len(shape)):
        var value = shape[a]
        for i in range(dim):
            shape[a][i + 1] = 0.0
            for j in range(dim):
                shape[a][i + 1] += value[j + 1] * inverse[3 * j + i]
    return ContinuumPoint(
        weight * abs(det), shape^, jacobian, inverse, det, ContinuumVector(r, s, t, 0.0)
    )


fn _points(
    kind: String,
    coords: List[ContinuumVector],
    axes: InlineArray[Int, 3],
    dim: Int,
    thickness: Float64,
) -> List[ContinuumPoint]:
    var points: List[ContinuumPoint] = []
    if dim == 2:
        var order = 9 if len(coords) > 4 else 4
        var g = GAUSS_3 if order == 9 else GAUSS_2
        for i in range(order):
            var r = g * Float64(QUAD_R[i])
            var s = g * Float64(QUAD_S[i])
            var weight = thickness
            if order == 9:
                weight *= (8.0 / 9.0 if r == 0.0 else 5.0 / 9.0) * (
                    8.0 / 9.0 if s == 0.0 else 5.0 / 9.0
                )
            points.append(_point(coords, axes, dim, r, s, 0.0, weight, kind == "quad3d"))
    else:
        var order = 3 if kind == "Brick8N" or kind == "Brick27N" or kind == "SSPbrick" else 2
        var gauss = [-GAUSS_2, GAUSS_2] if order == 2 else [-GAUSS_3, 0.0, GAUSS_3]
        var weights = [1.0, 1.0] if order == 2 else [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        for i in range(order):
            for j in range(order):
                for k in range(order):
                    points.append(
                        _point(
                            coords,
                            axes,
                            dim,
                            gauss[i],
                            gauss[j],
                            gauss[k],
                            weights[i] * weights[j] * weights[k],
                        )
                    )
    return points^


fn _strain_column(
    mut B: List[Float64],
    width: Int,
    column: Int,
    direction: Int,
    gradient: ContinuumVector,
    components: Int,
):
    B[direction * width + column] = gradient[direction + 1]
    if direction < 2:
        var shear_xy = 3 if components == 6 else 2
        B[shear_xy * width + column] = gradient[2 - direction]
    if components == 6:
        if direction > 0:
            B[4 * width + column] = gradient[4 - direction]
        if direction != 1:
            B[5 * width + column] = gradient[3 - direction]
    return


fn _zeros(size: Int) -> List[Float64]:
    var values: List[Float64] = []
    values.resize(size, 0.0)
    return values^


fn _add_energy(
    mut K: List[Float64],
    B: List[Float64],
    tangent: ContinuumTangent,
    width: Int,
    components: Int,
    weight: Float64,
):
    var cb = _zeros(width * components)
    for i in range(components):
        for j in range(components):
            var value = weight * tangent[6 * i + j]
            if value != 0.0:
                for a in range(width):
                    cb[i * width + a] += value * B[j * width + a]
    for i in range(components):
        for a in range(width):
            var value = B[i * width + a]
            if value != 0.0:
                for b in range(width):
                    K[a * width + b] += value * cb[i * width + b]


fn _condensation(K: List[Float64], retained: Int, internal: Int) -> List[Float64]:
    var width = retained + internal
    var kee: List[List[Float64]] = []
    for i in range(retained, width):
        var row = _zeros(internal)
        for j in range(internal):
            row[j] = K[i * width + retained + j]
        kee.append(row^)
    var pivots: List[Int] = []
    lu_factorize_into(kee, pivots)
    var modes = _zeros(internal * retained)
    var work: List[Float64] = []
    var result: List[Float64] = []
    var rhs = _zeros(internal)
    for a in range(retained):
        for i in range(internal):
            rhs[i] = -K[(retained + i) * width + a]
        lu_solve_into(kee, pivots, rhs, work, result)
        for i in range(internal):
            modes[i * retained + a] = result[i]
    return modes^


@fieldwise_init
struct ContinuumElement(Copyable, Movable):
    var dofs: Int
    var components: Int
    var stiffness: List[Float64]
    var mass: List[Float64]
    var lumped_mass: List[Float64]
    var body_force: List[Float64]
    var strains: List[List[Float64]]
    var tangent: ContinuumTangent
    var weights: List[Float64]
    var drucker_prager: Optional[DruckerPragerDefinition]
    var committed: List[DruckerPragerHistory]
    var trial: List[DruckerPragerTrial]
    var trial_force: List[Float64]

    fn set_drucker_prager(mut self, definition: DruckerPragerDefinition):
        self.drucker_prager = definition
        self.committed.resize(len(self.strains), DruckerPragerHistory())

    fn commit(mut self):
        for p in range(len(self.trial)):
            self.committed[p] = self.trial[p].history

    fn evaluate(mut self, u: List[Float64], tangent: Bool) -> List[Float64]:
        if not self.drucker_prager:
            return self.resisting_force(u)
        var force = self.body_force.copy()
        for i in range(self.dofs):
            force[i] = -force[i]
        if tangent:
            for i in range(len(self.stiffness)):
                self.stiffness[i] = 0.0
        self.trial.clear()
        var definition = self.drucker_prager.value()
        for p in range(len(self.strains)):
            ref B = self.strains[p]
            var strain = DruckerPragerVector(fill=0.0)
            for i in range(6):
                for j in range(self.dofs):
                    strain[i] += B[i * self.dofs + j] * u[j]
            var trial = drucker_prager_trial(definition, self.committed[p], strain)
            var weight = self.weights[p]
            for i in range(6):
                for j in range(self.dofs):
                    force[j] += weight * B[i * self.dofs + j] * trial.stress[i]
            if tangent:
                _add_energy(self.stiffness, B, trial.tangent, self.dofs, 6, weight)
            self.trial.append(trial)
        self.trial_force = force.copy()
        return force^

    fn resisting_force(self, u: List[Float64]) -> List[Float64]:
        var result = self.body_force.copy()
        for i in range(self.dofs):
            result[i] = -result[i]
            for j in range(self.dofs):
                result[i] += self.stiffness[i * self.dofs + j] * u[j]
        return result^

    fn response(
        self, u: List[Float64], point: Int, stress: Bool, state: Bool = False
    ) -> List[Float64]:
        if point < 0 or point >= len(self.strains):
            abort("continuum material recorder integration point out of range")
        if self.drucker_prager:
            if point >= len(self.trial):
                abort("DruckerPrager recorder has no trial response")
            ref trial = self.trial[point]
            if state:
                return trial.state()
            var result: List[Float64] = []
            for i in range(6):
                result.append(trial.stress[i] if stress else trial.strain[i])
            return result^
        if state:
            abort("material state recorder requires DruckerPrager")
        var strain = _zeros(self.components)
        for i in range(self.components):
            for j in range(self.dofs):
                strain[i] += self.strains[point][i * self.dofs + j] * u[j]
        if not stress:
            return strain^
        var result = _zeros(self.components)
        for i in range(self.components):
            for j in range(self.components):
                result[i] += self.tangent[i * 6 + j] * strain[j]
        return result^


fn _add_quad_stabilization(
    mut stiffness: List[Float64],
    coords: List[ContinuumVector],
    center: ContinuumPoint,
    tangent: ContinuumTangent,
    thickness: Float64,
):
    var hourglass = coords[0] - coords[1] + coords[2] - coords[3]
    var gamma = InlineArray[Float64, 4](fill=0.0)
    for a in range(4):
        gamma[a] = 0.25 * (
            Float64(QUAD_R[a] * QUAD_S[a])
            - hourglass[0] * center.shape[a][1]
            - hourglass[1] * center.shape[a][2]
        )
    var j = center.jacobian
    var g1 = ContinuumVector(j[0], j[3], 0.0, 0.0) / sqrt(j[0] * j[0] + j[3] * j[3])
    var g2 = ContinuumVector(j[1], j[4], 0.0, 0.0) / sqrt(j[1] * j[1] + j[4] * j[4])
    var scale = thickness * center.determinant / 3.0
    var ixx = scale * (g1[0] * g1[0] + g2[0] * g2[0])
    var ixy = scale * (g1[0] * g1[1] + g2[0] * g2[1])
    var iyy = scale * (g1[1] * g1[1] + g2[1] * g2[1])
    var inv = center.inverse
    var hss = ixx * inv[3] * inv[3] + ixy * inv[0] * inv[3] + iyy * inv[0] * inv[0]
    var htt = ixx * inv[4] * inv[4] + ixy * inv[1] * inv[4] + iyy * inv[1] * inv[1]
    var hst = (
        ixx * inv[4] * inv[3] + ixy * (inv[3] * inv[1] + inv[4] * inv[0]) + iyy * inv[1] * inv[0]
    )
    var modulus = tangent[0] + tangent[7] - tangent[1] - tangent[6]
    var fcf = InlineArray[Float64, 4](modulus * hss, -modulus * hst, -modulus * hst, modulus * htt)
    for a in range(4):
        for b in range(4):
            for i in range(2):
                for k in range(2):
                    stiffness[(2 * a + i) * 8 + 2 * b + k] += gamma[a] * fcf[2 * i + k] * gamma[b]


fn _det(a: ContinuumVector, b: ContinuumVector, c: ContinuumVector) -> Float64:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


fn _ssp_brick_linear_jacobian(coords: List[ContinuumVector]) -> ContinuumVector:
    var map = InlineArray[ContinuumVector, 7](fill=ContinuumVector(0.0))
    for a in range(8):
        var r = Float64(BRICK_R[a])
        var s = Float64(BRICK_S[a])
        var t = Float64(BRICK_T[a])
        var x = 0.125 * (coords[a] - coords[0])
        map[0] += r * x
        map[1] += s * x
        map[2] += t * x
        map[3] += s * t * x
        map[4] += r * t * x
        map[5] += r * s * x
    return ContinuumVector(
        _det(map[0], map[5], map[2]) + _det(map[0], map[1], map[4]),
        _det(map[5], map[1], map[2]) + _det(map[0], map[1], map[3]),
        _det(map[4], map[1], map[2]) + _det(map[0], map[3], map[2]),
        0.0,
    )


fn _add_brick_stabilization(
    mut stiffness: List[Float64],
    coords: List[ContinuumVector],
    points: List[ContinuumPoint],
    center: ContinuumPoint,
    average: List[ContinuumVector],
    tangent: ContinuumTangent,
):
    # The four hourglass modes are rs, st, rt, rst. The last mode enters
    # stabilization through its shear components. Condense the nine enhanced
    # strain modes from the first three hourglass modes.
    var projection = _zeros(4 * 8)
    for mode in range(4):
        var projected_coords = ContinuumVector(0.0)
        for a in range(8):
            var r = BRICK_R[a]
            var s = BRICK_S[a]
            var t = BRICK_T[a]
            var signs = InlineArray[Int, 4](r * s, s * t, r * t, r * s * t)
            projection[mode * 8 + a] = 0.125 * Float64(signs[mode])
            projected_coords += projection[mode * 8 + a] * (coords[a] - coords[0])
        for a in range(8):
            for d in range(3):
                projection[mode * 8 + a] -= projected_coords[d] * average[a][d + 1]
    var energy = _zeros(21 * 21)
    var removed_linear = _ssp_brick_linear_jacobian(coords)
    for i in range(len(points)):
        ref point = points[i]
        var r = point.natural[0]
        var s = point.natural[1]
        var t = point.natural[2]
        var derivatives = InlineArray[ContinuumVector, 7](
            ContinuumVector(s, r, 0.0, 0.0),
            ContinuumVector(0.0, t, s, 0.0),
            ContinuumVector(t, 0.0, r, 0.0),
            ContinuumVector(s * t, r * t, r * s, 0.0),
            ContinuumVector(r, 0.0, 0.0, 0.0),
            ContinuumVector(0.0, s, 0.0, 0.0),
            ContinuumVector(0.0, 0.0, t, 0.0),
        )
        var B = _zeros(6 * 21)
        for mode in range(7):
            var gradient = ContinuumVector(0.0)
            for d in range(3):
                for j in range(3):
                    gradient[d + 1] += derivatives[mode][j] * center.inverse[3 * j + d]
            for d in range(3):
                var col = 3 * mode + d
                _strain_column(B, 21, col, d, gradient, 6)
                if mode == 3:
                    B[d * 21 + col] = 0.0
        var weight = point.weight * (
            1.0
            - (r * removed_linear[0] + s * removed_linear[1] + t * removed_linear[2])
            / point.determinant
        )
        _add_energy(energy, B, tangent, 21, 6, weight)
    # OpenSees leaves the trilinear hourglass column uncoupled from the
    # enhanced modes in this condensation.
    for i in range(12, 21):
        for j in range(9, 12):
            energy[i * 21 + j] = 0.0
            energy[j * 21 + i] = 0.0
    var modes = _condensation(energy, 12, 9)
    for i in range(12):
        for j in range(12):
            var kij = energy[i * 21 + j]
            for k in range(9):
                kij += energy[i * 21 + 12 + k] * modes[k * 12 + j]
            for a in range(8):
                for b in range(8):
                    stiffness[(3 * a + i % 3) * 24 + 3 * b + j % 3] += (
                        kij * projection[(i // 3) * 8 + a] * projection[(j // 3) * 8 + b]
                    )


fn _apply_quad_pressure(
    mut body_force: List[Float64],
    coords: List[ContinuumVector],
    axes: InlineArray[Int, 3],
    ndf: Int,
    pressure: Float64,
):
    for a in range(4):
        var b = (a + 1) % 4
        var dx = coords[b][axes[0]] - coords[a][axes[0]]
        var dy = coords[b][axes[1]] - coords[a][axes[1]]
        if len(coords) == 4:
            for node in [a, b]:
                body_force[node * ndf + axes[0]] += 0.5 * pressure * dy
                body_force[node * ndf + axes[1]] -= 0.5 * pressure * dx
        else:
            var mid = a + 4
            var dx_am = coords[mid][axes[0]] - coords[a][axes[0]]
            var dy_am = coords[mid][axes[1]] - coords[a][axes[1]]
            body_force[a * ndf + axes[0]] += pressure * dy_am / 3.0
            body_force[a * ndf + axes[1]] -= pressure * dx_am / 3.0
            body_force[mid * ndf + axes[0]] += 2.0 * pressure * dy / 3.0
            body_force[mid * ndf + axes[1]] -= 2.0 * pressure * dx / 3.0
            body_force[b * ndf + axes[0]] += pressure * (dy - dy_am) / 3.0
            body_force[b * ndf + axes[1]] -= pressure * (dx - dx_am) / 3.0


fn prepare_continuum(
    kind: String,
    coords: List[ContinuumVector],
    thickness: Float64,
    density: Float64,
    body: ContinuumVector,
    pressure: Float64,
    tangent: ContinuumTangent,
) -> ContinuumElement:
    var ndf = continuum_dimension(kind)
    var axes = InlineArray[Int, 3](0, 1, 2)
    var dim = ndf
    if kind == "quad3d":
        dim = 2
        var count = 0
        for d in range(3):
            var varies = False
            for a in range(1, len(coords)):
                varies = varies or coords[a][d] != coords[0][d]
            if varies:
                axes[count] = d
                count += 1
        if count != 2:
            abort("quad3d requires its nodes to lie in a coordinate plane")

    var count = len(coords)
    var dofs = ndf * count
    var mixed = kind == "bbarBrick" or kind == "bbarQuad"
    var components = 6 if dim == 3 or mixed else 3
    var enhanced = kind == "enhancedQuad"
    var ssp_quad = kind == "SSPquad"
    var ssp_brick = kind == "SSPbrick"
    var ssp = ssp_quad or ssp_brick
    var consistent = (dim == 3 and not ssp_brick) or enhanced or kind == "bbarQuad"
    var dimensions = thickness if dim == 2 else 1.0
    var width = dofs + 4 if enhanced else dofs
    var K = _zeros(width * width)
    var mass: List[Float64] = []
    if consistent and density != 0.0:
        mass = _zeros(dofs * dofs)
    var lumped = _zeros(dofs)
    var force = _zeros(dofs)
    var strains: List[List[Float64]] = []
    var weights: List[Float64] = []
    var points = _points(kind, coords, axes, dim, dimensions)
    var average: List[ContinuumVector] = []
    average.resize(count, ContinuumVector(0.0))
    var volume = 0.0
    for i in range(len(points)):
        ref point = points[i]
        volume += point.weight
        for a in range(count):
            average[a] += point.weight * point.shape[a]
    for a in range(count):
        average[a] /= volume
    var center = _point(coords, axes, dim, 0.0, 0.0, 0.0, 1.0)
    if ssp_quad:
        volume = 4.0 * thickness * center.determinant
    var ssp_linear = _ssp_brick_linear_jacobian(coords) if ssp_brick else ContinuumVector(0.0)
    for i in range(0 if ssp_brick else len(points)):
        ref point = points[i]
        for a in range(count):
            var nodal_weight = point.shape[a][0] * point.weight
            if ssp_quad:
                # The OpenSees SSPquad diagonal uses the Jacobian sampled at the corner.
                nodal_weight = 0.0
                if i == a:
                    var corner = _point(
                        coords, axes, dim, Float64(QUAD_R[a]), Float64(QUAD_S[a]), 0.0, thickness
                    )
                    nodal_weight = corner.weight
            for d in range(dim):
                var row = a * ndf + axes[d]
                force[row] += nodal_weight * body[d]
                if len(mass) > 0:
                    for b in range(count):
                        var col = b * ndf + axes[d]
                        mass[row * dofs + col] += nodal_weight * point.shape[b][0] * density
                elif not consistent:
                    lumped[row] += nodal_weight * density
    if ssp_brick:
        # OpenSees scales the odd Jacobian moments by the signed hourglass
        # coefficients (1/8) and multiplies the result by the center Jacobian.
        var average_det = volume / 8.0
        var nodal_moments = _zeros(count)
        for i in range(len(points)):
            ref point = points[i]
            var q = point.natural
            var linear = q[0] * ssp_linear[0] + q[1] * ssp_linear[1] + q[2] * ssp_linear[2]
            var weight = point.weight * (point.determinant - linear) / point.determinant
            for a in range(count):
                nodal_moments[a] += weight * point.shape[a][0]
        for a in range(count):
            var weight = center.determinant * (
                1.0 + average_det - center.determinant + (nodal_moments[a] - average_det) / 8.0
            )
            for d in range(dim):
                var row = a * ndf + axes[d]
                force[row] = weight * body[d]
                lumped[row] = weight * density
    if pressure != 0.0:
        _apply_quad_pressure(force, coords, axes, ndf, pressure)
    for i in range(1 if ssp else len(points)):
        var point = center.copy() if ssp else points[i].copy()
        var B = _zeros(components * width)
        for a in range(count):
            var grad = average[a] if ssp_brick else point.shape[a]
            for d in range(dim):
                var col = a * ndf + axes[d]
                _strain_column(B, width, col, d, grad, components)
                if mixed:
                    var correction = (average[a][d + 1] - grad[d + 1]) / 3.0
                    for normal in range(3):
                        B[normal * width + col] += correction
        if enhanced:
            for mode in range(2):
                var scale = point.natural[mode] / point.determinant
                var gradient = ContinuumVector(
                    0.0, scale * center.inverse[3 * mode], scale * center.inverse[3 * mode + 1], 0.0
                )
                for d in range(2):
                    _strain_column(B, width, dofs + 2 * mode + d, d, gradient, components)
        var weight = volume if ssp else point.weight
        _add_energy(K, B, tangent, width, components, weight)
        strains.append(B^)
        weights.append(weight)

    if enhanced:
        var modes = _condensation(K, dofs, 4)
        var condensed = _zeros(dofs * dofs)
        for a in range(dofs):
            for b in range(dofs):
                condensed[a * dofs + b] = K[a * width + b]
                for i in range(4):
                    condensed[a * dofs + b] += K[a * width + dofs + i] * modes[i * dofs + b]
        K = condensed^
        for point in range(len(strains)):
            var B = _zeros(components * dofs)
            for i in range(components):
                for a in range(dofs):
                    B[i * dofs + a] = strains[point][i * width + a]
                    for m in range(4):
                        B[i * dofs + a] += (
                            strains[point][i * width + dofs + m] * modes[m * dofs + a]
                        )
            strains[point] = B^

    if ssp_quad:
        _add_quad_stabilization(K, coords, center, tangent, thickness)
    elif ssp_brick:
        _add_brick_stabilization(K, coords, points, center, average, tangent)
    return ContinuumElement(
        dofs, components, K^, mass^, lumped^, force^, strains^, tangent, weights^, None, [], [], []
    )
