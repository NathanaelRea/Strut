from collections import List
from math import sqrt
from os import abort

from elements.beam_loads import beam3d_basic_fixed_end_and_reactions, beam3d_section_load_response
from elements.beam_integration import BeamIntegrationCache, beam_integration_cache_ensure
from elements.utils import (
    _cross,
    _dot,
    _ensure_zero_matrix,
    _ensure_zero_vector,
    _normalize,
)
from materials import UniMaterialDef, UniMaterialState
from solver.run_case.input_types import ElementLoadInput
from sections import (
    FiberCell,
    FiberSection3dDef,
    FiberSection3dResponse,
    fiber_section3d_set_trial_from_offset,
)
from utils import StaticTuple


alias Beam3dMat3 = StaticTuple[Float64, 9]
alias Beam3dCoeff72 = StaticTuple[Float64, 72]
alias Beam3dVec10 = StaticTuple[Float64, 10]
alias Beam3dMat5 = StaticTuple[Float64, 25]
alias Beam3dVec5 = StaticTuple[Float64, 5]
alias Beam3dMat12 = StaticTuple[Float64, 144]
alias Beam3dVec12 = StaticTuple[Float64, 12]


struct ForceBeamColumn3dScratch(Movable):
    var integration_cache: BeamIntegrationCache
    var R: Beam3dMat3
    var u_local: Beam3dVec12
    var section_load_axial: List[Float64]
    var section_load_my: List[Float64]
    var section_load_mz: List[Float64]
    var k_basic_flat: Beam3dMat5
    var f_basic_flat: Beam3dMat5
    var f_basic_copy_flat: Beam3dMat5
    var v_from: Beam3dVec5
    var residual: Beam3dVec5
    var dq: Beam3dVec5
    var k_local_flat: Beam3dMat12
    var f_local: Beam3dVec12
    var k_work_flat: Beam3dMat12
    var geometry_valid: List[Bool]
    var load_valid: List[Bool]
    var cached_length: List[Float64]
    var cached_inv_length: List[Float64]
    var basic_coeff_cache: List[Beam3dCoeff72]
    var rotation_cache: List[Beam3dMat3]
    var section_load_axial_cache: List[List[Float64]]
    var section_load_my_cache: List[List[Float64]]
    var section_load_mz_cache: List[List[Float64]]
    var fixed_end_cache: List[Beam3dVec10]
    var section_vs_eps0: List[Float64]
    var section_vs_ky: List[Float64]
    var section_vs_kz: List[Float64]
    var section_ssr_axial: List[Float64]
    var section_ssr_my: List[Float64]
    var section_ssr_mz: List[Float64]
    var section_fs11: List[Float64]
    var section_fs12: List[Float64]
    var section_fs13: List[Float64]
    var section_fs22: List[Float64]
    var section_fs23: List[Float64]
    var section_fs33: List[Float64]
    var section_vs_subdivide_eps0: List[Float64]
    var section_vs_subdivide_ky: List[Float64]
    var section_vs_subdivide_kz: List[Float64]
    var section_ssr_subdivide_axial: List[Float64]
    var section_ssr_subdivide_my: List[Float64]
    var section_ssr_subdivide_mz: List[Float64]
    var section_fs_subdivide11: List[Float64]
    var section_fs_subdivide12: List[Float64]
    var section_fs_subdivide13: List[Float64]
    var section_fs_subdivide22: List[Float64]
    var section_fs_subdivide23: List[Float64]
    var section_fs_subdivide33: List[Float64]

    fn __init__(out self):
        self.integration_cache = BeamIntegrationCache()
        self.R = Beam3dMat3(fill=0.0)
        self.u_local = Beam3dVec12(fill=0.0)
        self.section_load_axial = []
        self.section_load_my = []
        self.section_load_mz = []
        self.k_basic_flat = Beam3dMat5(fill=0.0)
        self.f_basic_flat = Beam3dMat5(fill=0.0)
        self.f_basic_copy_flat = Beam3dMat5(fill=0.0)
        self.v_from = Beam3dVec5(fill=0.0)
        self.residual = Beam3dVec5(fill=0.0)
        self.dq = Beam3dVec5(fill=0.0)
        self.k_local_flat = Beam3dMat12(fill=0.0)
        self.f_local = Beam3dVec12(fill=0.0)
        self.k_work_flat = Beam3dMat12(fill=0.0)
        self.geometry_valid = []
        self.load_valid = []
        self.cached_length = []
        self.cached_inv_length = []
        self.basic_coeff_cache = []
        self.rotation_cache = []
        self.section_load_axial_cache = []
        self.section_load_my_cache = []
        self.section_load_mz_cache = []
        self.fixed_end_cache = []
        self.section_vs_eps0 = []
        self.section_vs_ky = []
        self.section_vs_kz = []
        self.section_ssr_axial = []
        self.section_ssr_my = []
        self.section_ssr_mz = []
        self.section_fs11 = []
        self.section_fs12 = []
        self.section_fs13 = []
        self.section_fs22 = []
        self.section_fs23 = []
        self.section_fs33 = []
        self.section_vs_subdivide_eps0 = []
        self.section_vs_subdivide_ky = []
        self.section_vs_subdivide_kz = []
        self.section_ssr_subdivide_axial = []
        self.section_ssr_subdivide_my = []
        self.section_ssr_subdivide_mz = []
        self.section_fs_subdivide11 = []
        self.section_fs_subdivide12 = []
        self.section_fs_subdivide13 = []
        self.section_fs_subdivide22 = []
        self.section_fs_subdivide23 = []
        self.section_fs_subdivide33 = []


fn reset_force_beam_column3d_scratch(mut scratch: ForceBeamColumn3dScratch):
    scratch.integration_cache.is_valid = False
    scratch.geometry_valid = []
    scratch.load_valid = []
    scratch.cached_length = []
    scratch.cached_inv_length = []
    scratch.basic_coeff_cache = []
    scratch.rotation_cache = []
    scratch.section_load_axial_cache = []
    scratch.section_load_my_cache = []
    scratch.section_load_mz_cache = []
    scratch.fixed_end_cache = []


fn invalidate_force_beam_column3d_load_cache(mut scratch: ForceBeamColumn3dScratch):
    for i in range(len(scratch.load_valid)):
        scratch.load_valid[i] = False


@always_inline
fn _mat3_index(row: Int, col: Int) -> Int:
    return row * 3 + col


fn _beam3d_rotation_into(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    mut R: List[List[Float64]],
):
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var lx: Float64
    var ly: Float64
    var lz: Float64
    (lx, ly, lz) = _normalize(dx, dy, dz)

    var vx = 1.0
    var vy = 0.0
    var vz = 0.0
    if abs(_dot(lx, ly, lz, vx, vy, vz)) >= 0.9:
        vx = 0.0
        vy = 1.0
        vz = 0.0
        if abs(_dot(lx, ly, lz, vx, vy, vz)) >= 0.9:
            vx = 0.0
            vy = 0.0
            vz = 1.0

    var yx: Float64
    var yy: Float64
    var yz: Float64
    # Match OpenSees Linear/PDelta/Corotational vecxz orientation:
    # local y = vecxz x local x, local z = local x x local y.
    (yx, yy, yz) = _cross(vx, vy, vz, lx, ly, lz)
    (yx, yy, yz) = _normalize(yx, yy, yz)

    var zx: Float64
    var zy: Float64
    var zz: Float64
    (zx, zy, zz) = _cross(lx, ly, lz, yx, yy, yz)
    _ensure_zero_matrix(R, 3, 3)
    R[0][0] = lx
    R[0][1] = ly
    R[0][2] = lz
    R[1][0] = yx
    R[1][1] = yy
    R[1][2] = yz
    R[2][0] = zx
    R[2][1] = zy
    R[2][2] = zz


fn _beam3d_rotation_into_static(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    mut R: Beam3dMat3,
):
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var lx: Float64
    var ly: Float64
    var lz: Float64
    (lx, ly, lz) = _normalize(dx, dy, dz)

    var vx = 1.0
    var vy = 0.0
    var vz = 0.0
    if abs(_dot(lx, ly, lz, vx, vy, vz)) >= 0.9:
        vx = 0.0
        vy = 1.0
        vz = 0.0
        if abs(_dot(lx, ly, lz, vx, vy, vz)) >= 0.9:
            vx = 0.0
            vy = 0.0
            vz = 1.0

    var yx: Float64
    var yy: Float64
    var yz: Float64
    (yx, yy, yz) = _cross(vx, vy, vz, lx, ly, lz)
    (yx, yy, yz) = _normalize(yx, yy, yz)

    var zx: Float64
    var zy: Float64
    var zz: Float64
    (zx, zy, zz) = _cross(lx, ly, lz, yx, yy, yz)
    R[_mat3_index(0, 0)] = lx
    R[_mat3_index(0, 1)] = ly
    R[_mat3_index(0, 2)] = lz
    R[_mat3_index(1, 0)] = yx
    R[_mat3_index(1, 1)] = yy
    R[_mat3_index(1, 2)] = yz
    R[_mat3_index(2, 0)] = zx
    R[_mat3_index(2, 1)] = zy
    R[_mat3_index(2, 2)] = zz


fn _beam3d_rotation(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
) -> List[List[Float64]]:
    var R: List[List[Float64]] = []
    _beam3d_rotation_into(x1, y1, z1, x2, y2, z2, R)
    return R^


fn _beam3d_transform_u_global_to_local(
    R: List[List[Float64]], u_global: List[Float64], mut u_local: List[Float64]
):
    _ensure_zero_vector(u_local, 12)
    for block in range(4):
        var offset = 3 * block
        var g0 = u_global[offset]
        var g1 = u_global[offset + 1]
        var g2 = u_global[offset + 2]
        for i in range(3):
            u_local[offset + i] = R[i][0] * g0 + R[i][1] * g1 + R[i][2] * g2


fn _beam3d_transform_u_global_to_local_static(
    R: Beam3dMat3, u_global: List[Float64], mut u_local: Beam3dVec12
):
    for i in range(12):
        u_local[i] = 0.0
    for block in range(4):
        var offset = 3 * block
        var g0 = u_global[offset]
        var g1 = u_global[offset + 1]
        var g2 = u_global[offset + 2]
        for i in range(3):
            u_local[offset + i] = (
                R[_mat3_index(i, 0)] * g0
                + R[_mat3_index(i, 1)] * g1
                + R[_mat3_index(i, 2)] * g2
            )


fn _beam3d_transform_f_local_to_global(
    R: List[List[Float64]], f_local: List[Float64], mut f_global: List[Float64]
):
    _ensure_zero_vector(f_global, 12)
    for block in range(4):
        var offset = 3 * block
        var l0 = f_local[offset]
        var l1 = f_local[offset + 1]
        var l2 = f_local[offset + 2]
        for i in range(3):
            f_global[offset + i] = R[0][i] * l0 + R[1][i] * l1 + R[2][i] * l2


fn _beam3d_transform_f_local_to_global_static(
    R: Beam3dMat3, f_local: Beam3dVec12, mut f_global: List[Float64]
):
    _ensure_zero_vector(f_global, 12)
    for block in range(4):
        var offset = 3 * block
        var l0 = f_local[offset]
        var l1 = f_local[offset + 1]
        var l2 = f_local[offset + 2]
        for i in range(3):
            f_global[offset + i] = (
                R[_mat3_index(0, i)] * l0
                + R[_mat3_index(1, i)] * l1
                + R[_mat3_index(2, i)] * l2
            )


fn _beam3d_transform_stiffness_local_to_global(
    R: List[List[Float64]],
    k_local: List[List[Float64]],
    mut work: List[List[Float64]],
    mut k_global: List[List[Float64]],
):
    _ensure_zero_matrix(work, 12, 12)
    _ensure_zero_matrix(k_global, 12, 12)
    for row_block in range(4):
        var row_offset = 3 * row_block
        for col_block in range(4):
            var col_offset = 3 * col_block
            for i in range(3):
                var src0 = k_local[row_offset + i][col_offset]
                var src1 = k_local[row_offset + i][col_offset + 1]
                var src2 = k_local[row_offset + i][col_offset + 2]
                for j in range(3):
                    work[row_offset + i][col_offset + j] = (
                        src0 * R[0][j] + src1 * R[1][j] + src2 * R[2][j]
                    )
    for row_block in range(4):
        var row_offset = 3 * row_block
        for col in range(12):
            for i in range(3):
                k_global[row_offset + i][col] = (
                    R[0][i] * work[row_offset][col]
                    + R[1][i] * work[row_offset + 1][col]
                    + R[2][i] * work[row_offset + 2][col]
                )


@always_inline
fn _beam3d_local_basic_coeff(row: Int, col: Int, inv_L: Float64) -> Float64:
    if row == 0:
        if col == 0:
            return -1.0
        if col == 6:
            return 1.0
        return 0.0
    if row == 1:
        if col == 1:
            return inv_L
        if col == 5:
            return 1.0
        if col == 7:
            return -inv_L
        return 0.0
    if row == 2:
        if col == 1:
            return inv_L
        if col == 7:
            return -inv_L
        if col == 11:
            return 1.0
        return 0.0
    if row == 3:
        if col == 2:
            return -inv_L
        if col == 4:
            return 1.0
        if col == 8:
            return inv_L
        return 0.0
    if row == 4:
        if col == 2:
            return -inv_L
        if col == 8:
            return inv_L
        if col == 10:
            return 1.0
        return 0.0
    if col == 3:
        return -1.0
    if col == 9:
        return 1.0
    return 0.0


fn _beam3d_add_axial_geometric_stiffness(
    mut k_local: List[List[Float64]], axial_force: Float64, L: Float64
):
    var N_over_L = axial_force / L

    k_local[1][1] += N_over_L
    k_local[1][7] -= N_over_L
    k_local[7][1] -= N_over_L
    k_local[7][7] += N_over_L

    k_local[2][2] += N_over_L
    k_local[2][8] -= N_over_L
    k_local[8][2] -= N_over_L
    k_local[8][8] += N_over_L


fn _element_rotation_into(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    mut R: List[List[Float64]],
):
    var dx0 = x2 - x1
    var dy0 = y2 - y1
    var dz0 = z2 - z1
    var L0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    if L0 == 0.0:
        abort("zero-length element")

    if geom_transf == "Corotational":
        _beam3d_rotation_into(
            x1 + u_elem_global[0],
            y1 + u_elem_global[1],
            z1 + u_elem_global[2],
            x2 + u_elem_global[6],
            y2 + u_elem_global[7],
            z2 + u_elem_global[8],
            R,
        )
        return
    _beam3d_rotation_into(x1, y1, z1, x2, y2, z2, R)


fn _element_rotation_into_static(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    mut R: Beam3dMat3,
):
    var dx0 = x2 - x1
    var dy0 = y2 - y1
    var dz0 = z2 - z1
    var L0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    if L0 == 0.0:
        abort("zero-length element")

    if geom_transf == "Corotational":
        _beam3d_rotation_into_static(
            x1 + u_elem_global[0],
            y1 + u_elem_global[1],
            z1 + u_elem_global[2],
            x2 + u_elem_global[6],
            y2 + u_elem_global[7],
            z2 + u_elem_global[8],
            R,
        )
        return
    _beam3d_rotation_into_static(x1, y1, z1, x2, y2, z2, R)


fn _element_rotation(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
) -> List[List[Float64]]:
    var R: List[List[Float64]] = []
    _element_rotation_into(x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf, R)
    return R^


fn _invert_3x3_values(
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
    e: Float64,
    f: Float64,
    g: Float64,
    h: Float64,
    i: Float64,
) -> (
    Bool,
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
    var co00 = e * i - f * h
    var co01 = -(d * i - f * g)
    var co02 = d * h - e * g
    var co10 = -(b * i - c * h)
    var co11 = a * i - c * g
    var co12 = -(a * h - b * g)
    var co20 = b * f - c * e
    var co21 = -(a * f - c * d)
    var co22 = a * e - b * d

    var det = a * co00 + b * co01 + c * co02
    if abs(det) <= 1.0e-40:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var inv_det = 1.0 / det
    return (
        True,
        co00 * inv_det,
        co10 * inv_det,
        co20 * inv_det,
        co01 * inv_det,
        co11 * inv_det,
        co21 * inv_det,
        co02 * inv_det,
        co12 * inv_det,
        co22 * inv_det,
    )


fn _invert_5x5_into(mut a: List[List[Float64]], mut inv_out: List[List[Float64]]) -> Bool:
    var n = 5
    _ensure_zero_matrix(inv_out, n, n)
    for i in range(n):
        inv_out[i][i] = 1.0

    for i in range(n):
        var pivot = i
        var max_val = abs(a[i][i])
        for r in range(i + 1, n):
            var abs_val = abs(a[r][i])
            if abs_val > max_val:
                max_val = abs_val
                pivot = r
        if max_val <= 1.0e-40:
            return False

        if pivot != i:
            for c in range(n):
                var tmp = a[i][c]
                a[i][c] = a[pivot][c]
                a[pivot][c] = tmp
                var inv_tmp = inv_out[i][c]
                inv_out[i][c] = inv_out[pivot][c]
                inv_out[pivot][c] = inv_tmp

        var piv = a[i][i]
        for c in range(n):
            a[i][c] /= piv
            inv_out[i][c] /= piv

        for r in range(n):
            if r == i:
                continue
            var factor = a[r][i]
            if factor == 0.0:
                continue
            for c in range(n):
                a[r][c] -= factor * a[i][c]
                inv_out[r][c] -= factor * inv_out[i][c]
    return True


@always_inline
fn _mat5_index(row: Int, col: Int) -> Int:
    return row * 5 + col


@always_inline
fn _mat12_index(row: Int, col: Int) -> Int:
    return row * 12 + col


fn _zero_static_tuple[size: Int](mut values: StaticTuple[Float64, size]):
    for i in range(size):
        values[i] = 0.0


fn _invert_5x5_flat_into(mut a: Beam3dMat5, mut inv_out: Beam3dMat5) -> Bool:
    var n = 5
    _zero_static_tuple(inv_out)
    for i in range(n):
        inv_out[_mat5_index(i, i)] = 1.0

    for i in range(n):
        var pivot = i
        var max_val = abs(a[_mat5_index(i, i)])
        for r in range(i + 1, n):
            var abs_val = abs(a[_mat5_index(r, i)])
            if abs_val > max_val:
                max_val = abs_val
                pivot = r
        if max_val <= 1.0e-40:
            return False

        if pivot != i:
            for c in range(n):
                var idx_i = _mat5_index(i, c)
                var idx_p = _mat5_index(pivot, c)
                var tmp = a[idx_i]
                a[idx_i] = a[idx_p]
                a[idx_p] = tmp
                var inv_tmp = inv_out[idx_i]
                inv_out[idx_i] = inv_out[idx_p]
                inv_out[idx_p] = inv_tmp

        var piv = a[_mat5_index(i, i)]
        for c in range(n):
            var idx = _mat5_index(i, c)
            a[idx] /= piv
            inv_out[idx] /= piv

        for r in range(n):
            if r == i:
                continue
            var factor = a[_mat5_index(r, i)]
            if factor == 0.0:
                continue
            for c in range(n):
                var idx_rc = _mat5_index(r, c)
                var idx_ic = _mat5_index(i, c)
                a[idx_rc] -= factor * a[idx_ic]
                inv_out[idx_rc] -= factor * inv_out[idx_ic]
    return True


fn _beam3d_transform_stiffness_local_to_global_flat(
    R: Beam3dMat3,
    k_local: Beam3dMat12,
    mut work: Beam3dMat12,
    mut k_global: List[List[Float64]],
):
    _zero_static_tuple(work)
    _ensure_zero_matrix(k_global, 12, 12)
    for row_block in range(4):
        var row_offset = 3 * row_block
        for col_block in range(4):
            var col_offset = 3 * col_block
            for i in range(3):
                var base = _mat12_index(row_offset + i, col_offset)
                var src0 = k_local[base]
                var src1 = k_local[base + 1]
                var src2 = k_local[base + 2]
                for j in range(3):
                    work[base + j] = (
                        src0 * R[_mat3_index(0, j)]
                        + src1 * R[_mat3_index(1, j)]
                        + src2 * R[_mat3_index(2, j)]
                    )
    for row_block in range(4):
        var row_offset = 3 * row_block
        for col in range(12):
            for i in range(3):
                k_global[row_offset + i][col] = (
                    R[_mat3_index(0, i)] * work[_mat12_index(row_offset, col)]
                    + R[_mat3_index(1, i)] * work[_mat12_index(row_offset + 1, col)]
                    + R[_mat3_index(2, i)] * work[_mat12_index(row_offset + 2, col)]
                )


fn _ensure_force_beam_column3d_cache_slot(
    mut scratch: ForceBeamColumn3dScratch, elem_index: Int
):
    if elem_index < 0:
        return
    var needed = elem_index + 1
    if len(scratch.geometry_valid) < needed:
        scratch.geometry_valid.resize(needed, False)
        scratch.load_valid.resize(needed, False)
        scratch.cached_length.resize(needed, 0.0)
        scratch.cached_inv_length.resize(needed, 0.0)
        while len(scratch.basic_coeff_cache) < needed:
            scratch.basic_coeff_cache.append(Beam3dCoeff72(fill=0.0))
        while len(scratch.rotation_cache) < needed:
            scratch.rotation_cache.append(Beam3dMat3(fill=0.0))
        while len(scratch.section_load_axial_cache) < needed:
            scratch.section_load_axial_cache.append(List[Float64]())
        while len(scratch.section_load_my_cache) < needed:
            scratch.section_load_my_cache.append(List[Float64]())
        while len(scratch.section_load_mz_cache) < needed:
            scratch.section_load_mz_cache.append(List[Float64]())
        while len(scratch.fixed_end_cache) < needed:
            scratch.fixed_end_cache.append(Beam3dVec10(fill=0.0))


fn _ensure_force_beam_column3d_geometry_cache(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    mut scratch: ForceBeamColumn3dScratch,
):
    if elem_index < 0:
        return
    _ensure_force_beam_column3d_cache_slot(scratch, elem_index)
    if scratch.geometry_valid[elem_index]:
        return
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")
    var inv_L = 1.0 / L
    scratch.cached_length[elem_index] = L
    scratch.cached_inv_length[elem_index] = inv_L

    var basic_coeff = scratch.basic_coeff_cache[elem_index]
    for row in range(6):
        for col in range(12):
            basic_coeff[row * 12 + col] = _beam3d_local_basic_coeff(row, col, inv_L)
    scratch.basic_coeff_cache[elem_index] = basic_coeff

    var rotation = scratch.rotation_cache[elem_index]
    _beam3d_rotation_into_static(x1, y1, z1, x2, y2, z2, scratch.R)
    for i in range(3):
        for j in range(3):
            rotation[_mat3_index(i, j)] = scratch.R[_mat3_index(i, j)]
    scratch.rotation_cache[elem_index] = rotation
    scratch.geometry_valid[elem_index] = True


fn _ensure_force_beam_column3d_load_cache(
    elem_index: Int,
    integration: String,
    num_int_pts: Int,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut scratch: ForceBeamColumn3dScratch,
):
    if elem_index < 0:
        return
    _ensure_force_beam_column3d_cache_slot(scratch, elem_index)
    if scratch.load_valid[elem_index]:
        return
    beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
    var L = scratch.cached_length[elem_index]
    var axial = scratch.section_load_axial_cache[elem_index].copy()
    var my = scratch.section_load_my_cache[elem_index].copy()
    var mz = scratch.section_load_mz_cache[elem_index].copy()
    axial.resize(num_int_pts, 0.0)
    my.resize(num_int_pts, 0.0)
    mz.resize(num_int_pts, 0.0)
    for ip in range(num_int_pts):
        var loads = beam3d_section_load_response(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            scratch.integration_cache.xis[ip] * L,
            L,
        )
        axial[ip] = loads[0]
        my[ip] = loads[1]
        mz[ip] = loads[2]
    scratch.section_load_axial_cache[elem_index] = axial^
    scratch.section_load_my_cache[elem_index] = my^
    scratch.section_load_mz_cache[elem_index] = mz^

    var fixed_end = scratch.fixed_end_cache[elem_index]
    var fixed = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    fixed_end[0] = fixed[0]
    fixed_end[1] = fixed[1]
    fixed_end[2] = fixed[2]
    fixed_end[3] = fixed[3]
    fixed_end[4] = fixed[4]
    fixed_end[5] = fixed[5]
    fixed_end[6] = fixed[6]
    fixed_end[7] = fixed[7]
    fixed_end[8] = fixed[8]
    fixed_end[9] = fixed[9]
    scratch.fixed_end_cache[elem_index] = fixed_end
    scratch.load_valid[elem_index] = True


fn _fiber_section3d_set_trial_from_offset(
    sec_def: FiberSection3dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
) -> FiberSection3dResponse:
    return fiber_section3d_set_trial_from_offset(
        sec_def,
        uniaxial_states,
        section_state_offset,
        section_state_count,
        eps0,
        kappa_y,
        kappa_z,
    )


fn _fiber_section3d_solve_for_force(
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    axial_force_target: Float64,
    moment_y_target: Float64,
    moment_z_target: Float64,
    mut eps0_guess: Float64,
    mut kappa_y_guess: Float64,
    mut kappa_z_guess: Float64,
    max_iters: Int,
    tol: Float64,
) -> (FiberSection3dResponse, Float64, Float64, Float64):
    var iter = 0
    var resp = FiberSection3dResponse()
    while iter < max_iters:
        resp = _fiber_section3d_set_trial_from_offset(
            sec_def,
            uniaxial_states,
            section_state_offset,
            section_state_count,
            eps0_guess,
            kappa_y_guess,
            kappa_z_guess,
        )

        var r_axial = axial_force_target - resp.axial_force
        var r_my = moment_y_target - resp.moment_y
        var r_mz = moment_z_target - resp.moment_z
        if abs(r_axial) <= tol and abs(r_my) <= tol and abs(r_mz) <= tol:
            return (resp, eps0_guess, kappa_y_guess, kappa_z_guess)

        var inv_sec = _invert_3x3_values(
            resp.k11,
            resp.k12,
            resp.k13,
            resp.k12,
            resp.k22,
            resp.k23,
            resp.k13,
            resp.k23,
            resp.k33,
        )
        if not inv_sec[0]:
            return (resp, eps0_guess, kappa_y_guess, kappa_z_guess)

        var de0 = inv_sec[1] * r_axial + inv_sec[4] * r_my + inv_sec[7] * r_mz
        var dky = inv_sec[2] * r_axial + inv_sec[5] * r_my + inv_sec[8] * r_mz
        var dkz = inv_sec[3] * r_axial + inv_sec[6] * r_my + inv_sec[9] * r_mz
        eps0_guess += de0
        kappa_y_guess += dky
        kappa_z_guess += dkz
        iter += 1
    return (resp, eps0_guess, kappa_y_guess, kappa_z_guess)


fn _fiber_section3d_response_flexibility(
    resp: FiberSection3dResponse
) -> (Bool, Float64, Float64, Float64, Float64, Float64, Float64):
    var inv_sec = _invert_3x3_values(
        resp.k11,
        resp.k12,
        resp.k13,
        resp.k12,
        resp.k22,
        resp.k23,
        resp.k13,
        resp.k23,
        resp.k33,
    )
    if not inv_sec[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        True,
        inv_sec[1],
        inv_sec[2],
        inv_sec[3],
        inv_sec[5],
        inv_sec[6],
        inv_sec[9],
    )


fn _ensure_force_beam_column3d_section_history_capacity(
    mut scratch: ForceBeamColumn3dScratch, num_int_pts: Int
):
    scratch.section_vs_eps0.resize(num_int_pts, 0.0)
    scratch.section_vs_ky.resize(num_int_pts, 0.0)
    scratch.section_vs_kz.resize(num_int_pts, 0.0)
    scratch.section_ssr_axial.resize(num_int_pts, 0.0)
    scratch.section_ssr_my.resize(num_int_pts, 0.0)
    scratch.section_ssr_mz.resize(num_int_pts, 0.0)
    scratch.section_fs11.resize(num_int_pts, 0.0)
    scratch.section_fs12.resize(num_int_pts, 0.0)
    scratch.section_fs13.resize(num_int_pts, 0.0)
    scratch.section_fs22.resize(num_int_pts, 0.0)
    scratch.section_fs23.resize(num_int_pts, 0.0)
    scratch.section_fs33.resize(num_int_pts, 0.0)
    scratch.section_vs_subdivide_eps0.resize(num_int_pts, 0.0)
    scratch.section_vs_subdivide_ky.resize(num_int_pts, 0.0)
    scratch.section_vs_subdivide_kz.resize(num_int_pts, 0.0)
    scratch.section_ssr_subdivide_axial.resize(num_int_pts, 0.0)
    scratch.section_ssr_subdivide_my.resize(num_int_pts, 0.0)
    scratch.section_ssr_subdivide_mz.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide11.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide12.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide13.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide22.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide23.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide33.resize(num_int_pts, 0.0)


fn _restore_force_beam_column3d_predictor_state(
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    num_int_pts: Int,
    q0: Float64,
    q1: Float64,
    q2: Float64,
    q3: Float64,
    q4: Float64,
    section_eps0: List[Float64],
    section_ky: List[Float64],
    section_kz: List[Float64],
):
    force_basic_q_state[force_basic_q_offset] = q0
    force_basic_q_state[force_basic_q_offset + 1] = q1
    force_basic_q_state[force_basic_q_offset + 2] = q2
    force_basic_q_state[force_basic_q_offset + 3] = q3
    force_basic_q_state[force_basic_q_offset + 4] = q4
    var eps0_offset = force_basic_q_offset + 5
    var ky_offset = eps0_offset + num_int_pts
    var kz_offset = ky_offset + num_int_pts
    for ip in range(num_int_pts):
        force_basic_q_state[eps0_offset + ip] = section_eps0[ip]
        force_basic_q_state[ky_offset + ip] = section_ky[ip]
        force_basic_q_state[kz_offset + ip] = section_kz[ip]


fn _force_beam_column3d_section_basic_deformation(
    L: Float64,
    num_int_pts: Int,
    scratch: ForceBeamColumn3dScratch,
    force_basic_q_state: List[Float64],
    eps0_offset: Int,
    ky_offset: Int,
    kz_offset: Int,
) -> (Float64, Float64, Float64, Float64, Float64):
    var v0 = 0.0
    var v1 = 0.0
    var v2 = 0.0
    var v3 = 0.0
    var v4 = 0.0
    for ip in range(num_int_pts):
        var xi = scratch.integration_cache.xis[ip]
        var wL = scratch.integration_cache.weights[ip] * L
        var b_mi = xi - 1.0
        var b_mj = xi
        v0 += wL * force_basic_q_state[eps0_offset + ip]
        v1 += wL * b_mi * force_basic_q_state[kz_offset + ip]
        v2 += wL * b_mj * force_basic_q_state[kz_offset + ip]
        v3 += wL * b_mi * force_basic_q_state[ky_offset + ip]
        v4 += wL * b_mj * force_basic_q_state[ky_offset + ip]
    return (v0, v1, v2, v3, v4)


fn _force_beam_column3d_try_increment(
    L: Float64,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    fibers_per_section: Int,
    num_int_pts: Int,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    mut scratch: ForceBeamColumn3dScratch,
    base_v0: Float64,
    base_v1: Float64,
    base_v2: Float64,
    base_v3: Float64,
    base_v4: Float64,
    dv_trial0: Float64,
    dv_trial1: Float64,
    dv_trial2: Float64,
    dv_trial3: Float64,
    dv_trial4: Float64,
    use_initial_section_flexibility: Int,
) -> (Bool, Float64, Float64, Float64, Float64, Float64):
    var q0 = force_basic_q_state[force_basic_q_offset]
    var q1 = force_basic_q_state[force_basic_q_offset + 1]
    var q2 = force_basic_q_state[force_basic_q_offset + 2]
    var q3 = force_basic_q_state[force_basic_q_offset + 3]
    var q4 = force_basic_q_state[force_basic_q_offset + 4]
    var eps0_offset = force_basic_q_offset + 5
    var ky_offset = eps0_offset + num_int_pts
    var kz_offset = ky_offset + num_int_pts

    var elem_tol = 1.0e-12
    var max_elem_iters = 10

    var target_v0 = base_v0 + dv_trial0
    var target_v1 = base_v1 + dv_trial1
    var target_v2 = base_v2 + dv_trial2
    var target_v3 = base_v3 + dv_trial3
    var target_v4 = base_v4 + dv_trial4

    _ensure_force_beam_column3d_section_history_capacity(scratch, num_int_pts)

    var init_fs11: List[Float64] = []
    var init_fs12: List[Float64] = []
    var init_fs13: List[Float64] = []
    var init_fs22: List[Float64] = []
    var init_fs23: List[Float64] = []
    var init_fs33: List[Float64] = []
    init_fs11.resize(num_int_pts, 0.0)
    init_fs12.resize(num_int_pts, 0.0)
    init_fs13.resize(num_int_pts, 0.0)
    init_fs22.resize(num_int_pts, 0.0)
    init_fs23.resize(num_int_pts, 0.0)
    init_fs33.resize(num_int_pts, 0.0)

    for ip in range(num_int_pts):
        var eps0 = force_basic_q_state[eps0_offset + ip]
        var kappa_y = force_basic_q_state[ky_offset + ip]
        var kappa_z = force_basic_q_state[kz_offset + ip]
        scratch.section_vs_eps0[ip] = eps0
        scratch.section_vs_ky[ip] = kappa_y
        scratch.section_vs_kz[ip] = kappa_z

        var ip_state_offset = elem_state_offset + ip * fibers_per_section
        var resp_trial = _fiber_section3d_set_trial_from_offset(
            sec_def,
            uniaxial_states,
            ip_state_offset,
            fibers_per_section,
            eps0,
            kappa_y,
            kappa_z,
        )
        scratch.section_ssr_axial[ip] = resp_trial.axial_force
        scratch.section_ssr_my[ip] = resp_trial.moment_y
        scratch.section_ssr_mz[ip] = resp_trial.moment_z
        var sec_flex = _fiber_section3d_response_flexibility(resp_trial)
        if not sec_flex[0]:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
        scratch.section_fs11[ip] = sec_flex[1]
        scratch.section_fs12[ip] = sec_flex[2]
        scratch.section_fs13[ip] = sec_flex[3]
        scratch.section_fs22[ip] = sec_flex[4]
        scratch.section_fs23[ip] = sec_flex[5]
        scratch.section_fs33[ip] = sec_flex[6]

        scratch.section_vs_subdivide_eps0[ip] = eps0
        scratch.section_vs_subdivide_ky[ip] = kappa_y
        scratch.section_vs_subdivide_kz[ip] = kappa_z
        scratch.section_ssr_subdivide_axial[ip] = resp_trial.axial_force
        scratch.section_ssr_subdivide_my[ip] = resp_trial.moment_y
        scratch.section_ssr_subdivide_mz[ip] = resp_trial.moment_z
        scratch.section_fs_subdivide11[ip] = sec_flex[1]
        scratch.section_fs_subdivide12[ip] = sec_flex[2]
        scratch.section_fs_subdivide13[ip] = sec_flex[3]
        scratch.section_fs_subdivide22[ip] = sec_flex[4]
        scratch.section_fs_subdivide23[ip] = sec_flex[5]
        scratch.section_fs_subdivide33[ip] = sec_flex[6]

        var init_resp = _fiber_section3d_set_trial_from_offset(
            sec_def,
            uniaxial_states,
            ip_state_offset,
            fibers_per_section,
            0.0,
            0.0,
            0.0,
        )
        var init_flex = _fiber_section3d_response_flexibility(init_resp)
        if not init_flex[0]:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_fs11[ip] = init_flex[1]
        init_fs12[ip] = init_flex[2]
        init_fs13[ip] = init_flex[3]
        init_fs22[ip] = init_flex[4]
        init_fs23[ip] = init_flex[5]
        init_fs33[ip] = init_flex[6]
    var predictor_f = Beam3dMat5(fill=0.0)
    for ip in range(num_int_pts):
        var xi = scratch.integration_cache.xis[ip]
        var wL = scratch.integration_cache.weights[ip] * L
        var b_mi = xi - 1.0
        var b_mj = xi
        var sec_f11 = scratch.section_fs11[ip]
        var sec_f12 = scratch.section_fs12[ip]
        var sec_f13 = scratch.section_fs13[ip]
        var sec_f22 = scratch.section_fs22[ip]
        var sec_f23 = scratch.section_fs23[ip]
        var sec_f33 = scratch.section_fs33[ip]
        predictor_f[_mat5_index(0, 0)] += wL * sec_f11
        predictor_f[_mat5_index(0, 1)] += wL * sec_f13 * b_mi
        predictor_f[_mat5_index(0, 2)] += wL * sec_f13 * b_mj
        predictor_f[_mat5_index(0, 3)] += wL * sec_f12 * b_mi
        predictor_f[_mat5_index(0, 4)] += wL * sec_f12 * b_mj
        predictor_f[_mat5_index(1, 0)] += wL * b_mi * sec_f13
        predictor_f[_mat5_index(1, 1)] += wL * b_mi * sec_f33 * b_mi
        predictor_f[_mat5_index(1, 2)] += wL * b_mi * sec_f33 * b_mj
        predictor_f[_mat5_index(1, 3)] += wL * b_mi * sec_f23 * b_mi
        predictor_f[_mat5_index(1, 4)] += wL * b_mi * sec_f23 * b_mj
        predictor_f[_mat5_index(2, 0)] += wL * b_mj * sec_f13
        predictor_f[_mat5_index(2, 1)] += wL * b_mj * sec_f33 * b_mi
        predictor_f[_mat5_index(2, 2)] += wL * b_mj * sec_f33 * b_mj
        predictor_f[_mat5_index(2, 3)] += wL * b_mj * sec_f23 * b_mi
        predictor_f[_mat5_index(2, 4)] += wL * b_mj * sec_f23 * b_mj
        predictor_f[_mat5_index(3, 0)] += wL * b_mi * sec_f12
        predictor_f[_mat5_index(3, 1)] += wL * b_mi * sec_f23 * b_mi
        predictor_f[_mat5_index(3, 2)] += wL * b_mi * sec_f23 * b_mj
        predictor_f[_mat5_index(3, 3)] += wL * b_mi * sec_f22 * b_mi
        predictor_f[_mat5_index(3, 4)] += wL * b_mi * sec_f22 * b_mj
        predictor_f[_mat5_index(4, 0)] += wL * b_mj * sec_f12
        predictor_f[_mat5_index(4, 1)] += wL * b_mj * sec_f23 * b_mi
        predictor_f[_mat5_index(4, 2)] += wL * b_mj * sec_f23 * b_mj
        predictor_f[_mat5_index(4, 3)] += wL * b_mj * sec_f22 * b_mi
        predictor_f[_mat5_index(4, 4)] += wL * b_mj * sec_f22 * b_mj
    var predictor_f_copy = predictor_f
    if _invert_5x5_flat_into(predictor_f_copy, scratch.k_basic_flat):
        q0 += (
            scratch.k_basic_flat[_mat5_index(0, 0)] * dv_trial0
            + scratch.k_basic_flat[_mat5_index(0, 1)] * dv_trial1
            + scratch.k_basic_flat[_mat5_index(0, 2)] * dv_trial2
            + scratch.k_basic_flat[_mat5_index(0, 3)] * dv_trial3
            + scratch.k_basic_flat[_mat5_index(0, 4)] * dv_trial4
        )
        q1 += (
            scratch.k_basic_flat[_mat5_index(1, 0)] * dv_trial0
            + scratch.k_basic_flat[_mat5_index(1, 1)] * dv_trial1
            + scratch.k_basic_flat[_mat5_index(1, 2)] * dv_trial2
            + scratch.k_basic_flat[_mat5_index(1, 3)] * dv_trial3
            + scratch.k_basic_flat[_mat5_index(1, 4)] * dv_trial4
        )
        q2 += (
            scratch.k_basic_flat[_mat5_index(2, 0)] * dv_trial0
            + scratch.k_basic_flat[_mat5_index(2, 1)] * dv_trial1
            + scratch.k_basic_flat[_mat5_index(2, 2)] * dv_trial2
            + scratch.k_basic_flat[_mat5_index(2, 3)] * dv_trial3
            + scratch.k_basic_flat[_mat5_index(2, 4)] * dv_trial4
        )
        q3 += (
            scratch.k_basic_flat[_mat5_index(3, 0)] * dv_trial0
            + scratch.k_basic_flat[_mat5_index(3, 1)] * dv_trial1
            + scratch.k_basic_flat[_mat5_index(3, 2)] * dv_trial2
            + scratch.k_basic_flat[_mat5_index(3, 3)] * dv_trial3
            + scratch.k_basic_flat[_mat5_index(3, 4)] * dv_trial4
        )
        q4 += (
            scratch.k_basic_flat[_mat5_index(4, 0)] * dv_trial0
            + scratch.k_basic_flat[_mat5_index(4, 1)] * dv_trial1
            + scratch.k_basic_flat[_mat5_index(4, 2)] * dv_trial2
            + scratch.k_basic_flat[_mat5_index(4, 3)] * dv_trial3
            + scratch.k_basic_flat[_mat5_index(4, 4)] * dv_trial4
        )

    var num_elem_iters = max_elem_iters
    if use_initial_section_flexibility == 1:
        num_elem_iters = 10 * max_elem_iters

    for elem_iter in range(num_elem_iters):
        _zero_static_tuple(scratch.f_basic_flat)
        _zero_static_tuple(scratch.v_from)
        var use_initial_flex = (
            use_initial_section_flexibility == 1
            or (
                use_initial_section_flexibility == 2
                and elem_iter == 0
            )
        )
        for ip in range(num_int_pts):
            var xi = scratch.integration_cache.xis[ip]
            var wL = scratch.integration_cache.weights[ip] * L
            var b_mi = xi - 1.0
            var b_mj = xi
            var axial_target = q0 + scratch.section_load_axial[ip]
            var moment_z_target = b_mi * q1 + b_mj * q2 + scratch.section_load_mz[ip]
            var moment_y_target = b_mi * q3 + b_mj * q4 + scratch.section_load_my[ip]
            var dss0 = axial_target - scratch.section_ssr_subdivide_axial[ip]
            var dss1 = moment_y_target - scratch.section_ssr_subdivide_my[ip]
            var dss2 = moment_z_target - scratch.section_ssr_subdivide_mz[ip]

            var solve_f11: Float64
            var solve_f12: Float64
            var solve_f13: Float64
            var solve_f22: Float64
            var solve_f23: Float64
            var solve_f33: Float64
            if use_initial_flex:
                solve_f11 = init_fs11[ip]
                solve_f12 = init_fs12[ip]
                solve_f13 = init_fs13[ip]
                solve_f22 = init_fs22[ip]
                solve_f23 = init_fs23[ip]
                solve_f33 = init_fs33[ip]
            else:
                solve_f11 = scratch.section_fs_subdivide11[ip]
                solve_f12 = scratch.section_fs_subdivide12[ip]
                solve_f13 = scratch.section_fs_subdivide13[ip]
                solve_f22 = scratch.section_fs_subdivide22[ip]
                solve_f23 = scratch.section_fs_subdivide23[ip]
                solve_f33 = scratch.section_fs_subdivide33[ip]
            var dvs0 = solve_f11 * dss0 + solve_f12 * dss1 + solve_f13 * dss2
            var dvs1 = solve_f12 * dss0 + solve_f22 * dss1 + solve_f23 * dss2
            var dvs2 = solve_f13 * dss0 + solve_f23 * dss1 + solve_f33 * dss2

            var sec_vs0 = scratch.section_vs_subdivide_eps0[ip] + dvs0
            var sec_vs1 = scratch.section_vs_subdivide_ky[ip] + dvs1
            var sec_vs2 = scratch.section_vs_subdivide_kz[ip] + dvs2
            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            var resp_trial = _fiber_section3d_set_trial_from_offset(
                sec_def,
                uniaxial_states,
                ip_state_offset,
                fibers_per_section,
                sec_vs0,
                sec_vs1,
                sec_vs2,
            )
            scratch.section_vs_subdivide_eps0[ip] = sec_vs0
            scratch.section_vs_subdivide_ky[ip] = sec_vs1
            scratch.section_vs_subdivide_kz[ip] = sec_vs2
            scratch.section_ssr_subdivide_axial[ip] = resp_trial.axial_force
            scratch.section_ssr_subdivide_my[ip] = resp_trial.moment_y
            scratch.section_ssr_subdivide_mz[ip] = resp_trial.moment_z
            var sec_flex = _fiber_section3d_response_flexibility(resp_trial)
            if not sec_flex[0]:
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
            scratch.section_fs_subdivide11[ip] = sec_flex[1]
            scratch.section_fs_subdivide12[ip] = sec_flex[2]
            scratch.section_fs_subdivide13[ip] = sec_flex[3]
            scratch.section_fs_subdivide22[ip] = sec_flex[4]
            scratch.section_fs_subdivide23[ip] = sec_flex[5]
            scratch.section_fs_subdivide33[ip] = sec_flex[6]

            var dss_res0 = axial_target - scratch.section_ssr_subdivide_axial[ip]
            var dss_res1 = moment_y_target - scratch.section_ssr_subdivide_my[ip]
            var dss_res2 = moment_z_target - scratch.section_ssr_subdivide_mz[ip]
            var dvs_res0 = sec_flex[1] * dss_res0 + sec_flex[2] * dss_res1 + sec_flex[3] * dss_res2
            var dvs_res1 = sec_flex[2] * dss_res0 + sec_flex[4] * dss_res1 + sec_flex[5] * dss_res2
            var dvs_res2 = sec_flex[3] * dss_res0 + sec_flex[5] * dss_res1 + sec_flex[6] * dss_res2

            scratch.v_from[0] += wL * (sec_vs0 + dvs_res0)
            scratch.v_from[1] += wL * b_mi * (sec_vs2 + dvs_res2)
            scratch.v_from[2] += wL * b_mj * (sec_vs2 + dvs_res2)
            scratch.v_from[3] += wL * b_mi * (sec_vs1 + dvs_res1)
            scratch.v_from[4] += wL * b_mj * (sec_vs1 + dvs_res1)

            for a in range(5):
                var Ba_n = 0.0
                var Ba_my = 0.0
                var Ba_mz = 0.0
                if a == 0:
                    Ba_n = 1.0
                elif a == 1:
                    Ba_mz = b_mi
                elif a == 2:
                    Ba_mz = b_mj
                elif a == 3:
                    Ba_my = b_mi
                else:
                    Ba_my = b_mj
                for b in range(5):
                    var Bb_n = 0.0
                    var Bb_my = 0.0
                    var Bb_mz = 0.0
                    if b == 0:
                        Bb_n = 1.0
                    elif b == 1:
                        Bb_mz = b_mi
                    elif b == 2:
                        Bb_mz = b_mj
                    elif b == 3:
                        Bb_my = b_mi
                    else:
                        Bb_my = b_mj
                    var dEps = sec_flex[1] * Bb_n + sec_flex[2] * Bb_my + sec_flex[3] * Bb_mz
                    var dKy = sec_flex[2] * Bb_n + sec_flex[4] * Bb_my + sec_flex[5] * Bb_mz
                    var dKz = sec_flex[3] * Bb_n + sec_flex[5] * Bb_my + sec_flex[6] * Bb_mz
                    scratch.f_basic_flat[_mat5_index(a, b)] += (
                        wL * (Ba_n * dEps + Ba_my * dKy + Ba_mz * dKz)
                    )
        _zero_static_tuple(scratch.f_basic_copy_flat)
        for i in range(25):
            scratch.f_basic_copy_flat[i] = scratch.f_basic_flat[i]
        if not _invert_5x5_flat_into(
            scratch.f_basic_copy_flat, scratch.k_basic_flat
        ):
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0)

        var residual_0 = target_v0 - scratch.v_from[0]
        var residual_1 = target_v1 - scratch.v_from[1]
        var residual_2 = target_v2 - scratch.v_from[2]
        var residual_3 = target_v3 - scratch.v_from[3]
        var residual_4 = target_v4 - scratch.v_from[4]
        var dq0 = (
            scratch.k_basic_flat[_mat5_index(0, 0)] * residual_0
            + scratch.k_basic_flat[_mat5_index(0, 1)] * residual_1
            + scratch.k_basic_flat[_mat5_index(0, 2)] * residual_2
            + scratch.k_basic_flat[_mat5_index(0, 3)] * residual_3
            + scratch.k_basic_flat[_mat5_index(0, 4)] * residual_4
        )
        var dq1 = (
            scratch.k_basic_flat[_mat5_index(1, 0)] * residual_0
            + scratch.k_basic_flat[_mat5_index(1, 1)] * residual_1
            + scratch.k_basic_flat[_mat5_index(1, 2)] * residual_2
            + scratch.k_basic_flat[_mat5_index(1, 3)] * residual_3
            + scratch.k_basic_flat[_mat5_index(1, 4)] * residual_4
        )
        var dq2 = (
            scratch.k_basic_flat[_mat5_index(2, 0)] * residual_0
            + scratch.k_basic_flat[_mat5_index(2, 1)] * residual_1
            + scratch.k_basic_flat[_mat5_index(2, 2)] * residual_2
            + scratch.k_basic_flat[_mat5_index(2, 3)] * residual_3
            + scratch.k_basic_flat[_mat5_index(2, 4)] * residual_4
        )
        var dq3 = (
            scratch.k_basic_flat[_mat5_index(3, 0)] * residual_0
            + scratch.k_basic_flat[_mat5_index(3, 1)] * residual_1
            + scratch.k_basic_flat[_mat5_index(3, 2)] * residual_2
            + scratch.k_basic_flat[_mat5_index(3, 3)] * residual_3
            + scratch.k_basic_flat[_mat5_index(3, 4)] * residual_4
        )
        var dq4 = (
            scratch.k_basic_flat[_mat5_index(4, 0)] * residual_0
            + scratch.k_basic_flat[_mat5_index(4, 1)] * residual_1
            + scratch.k_basic_flat[_mat5_index(4, 2)] * residual_2
            + scratch.k_basic_flat[_mat5_index(4, 3)] * residual_3
            + scratch.k_basic_flat[_mat5_index(4, 4)] * residual_4
        )
        var work_norm = abs(
            residual_0 * dq0
            + residual_1 * dq1
            + residual_2 * dq2
            + residual_3 * dq3
            + residual_4 * dq4
        )
        q0 += dq0
        q1 += dq1
        q2 += dq2
        q3 += dq3
        q4 += dq4
        if work_norm < elem_tol:
            force_basic_q_state[force_basic_q_offset] = q0
            force_basic_q_state[force_basic_q_offset + 1] = q1
            force_basic_q_state[force_basic_q_offset + 2] = q2
            force_basic_q_state[force_basic_q_offset + 3] = q3
            force_basic_q_state[force_basic_q_offset + 4] = q4
            for ip in range(num_int_pts):
                force_basic_q_state[eps0_offset + ip] = scratch.section_vs_subdivide_eps0[ip]
                force_basic_q_state[ky_offset + ip] = scratch.section_vs_subdivide_ky[ip]
                force_basic_q_state[kz_offset + ip] = scratch.section_vs_subdivide_kz[ip]
                scratch.section_vs_eps0[ip] = scratch.section_vs_subdivide_eps0[ip]
                scratch.section_vs_ky[ip] = scratch.section_vs_subdivide_ky[ip]
                scratch.section_vs_kz[ip] = scratch.section_vs_subdivide_kz[ip]
                scratch.section_ssr_axial[ip] = scratch.section_ssr_subdivide_axial[ip]
                scratch.section_ssr_my[ip] = scratch.section_ssr_subdivide_my[ip]
                scratch.section_ssr_mz[ip] = scratch.section_ssr_subdivide_mz[ip]
                scratch.section_fs11[ip] = scratch.section_fs_subdivide11[ip]
                scratch.section_fs12[ip] = scratch.section_fs_subdivide12[ip]
                scratch.section_fs13[ip] = scratch.section_fs_subdivide13[ip]
                scratch.section_fs22[ip] = scratch.section_fs_subdivide22[ip]
                scratch.section_fs23[ip] = scratch.section_fs_subdivide23[ip]
                scratch.section_fs33[ip] = scratch.section_fs_subdivide33[ip]
            return (True, q0, q1, q2, q3, q4)
    return (False, 0.0, 0.0, 0.0, 0.0, 0.0)


fn _force_beam_column3d_basic_state(
    u_local: List[Float64], L: Float64
) -> (Float64, Float64, Float64, Float64, Float64, Float64):
    var chord_z = (u_local[7] - u_local[1]) / L
    var chord_y = (u_local[8] - u_local[2]) / L
    return (
        u_local[6] - u_local[0],
        u_local[5] - chord_z,
        u_local[11] - chord_z,
        u_local[4] + chord_y,
        u_local[10] + chord_y,
        u_local[9] - u_local[3],
    )


fn _force_beam_column3d_basic_state(
    u_local: Beam3dVec12, L: Float64
) -> (Float64, Float64, Float64, Float64, Float64, Float64):
    var chord_z = (u_local[7] - u_local[1]) / L
    var chord_y = (u_local[8] - u_local[2]) / L
    return (
        u_local[6] - u_local[0],
        u_local[5] - chord_z,
        u_local[11] - chord_z,
        u_local[4] + chord_y,
        u_local[10] + chord_y,
        u_local[9] - u_local[3],
    )


fn force_beam_column3d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut scratch: ForceBeamColumn3dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    force_beam_column3d_global_tangent_and_internal(
        0,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var scratch = ForceBeamColumn3dScratch()
    force_beam_column3d_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var scratch = ForceBeamColumn3dScratch()
    force_beam_column3d_global_tangent_and_internal(
        elem_index,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut scratch: ForceBeamColumn3dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    _ensure_force_beam_column3d_geometry_cache(
        elem_index, x1, y1, z1, x2, y2, z2, scratch
    )
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if elem_index >= 0 and elem_index < len(scratch.cached_length):
        L = scratch.cached_length[elem_index]
    if L == 0.0:
        abort("zero-length element")

    if geom_transf == "Corotational":
        _element_rotation_into_static(
            x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf, scratch.R
        )
    elif elem_index >= 0 and elem_index < len(scratch.rotation_cache):
        scratch.R = scratch.rotation_cache[elem_index]
    else:
        _element_rotation_into_static(
            x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf, scratch.R
        )
    _beam3d_transform_u_global_to_local_static(scratch.R, u_elem_global, scratch.u_local)
    var fixed_end = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    if elem_index >= 0 and elem_index < len(scratch.fixed_end_cache):
        fixed_end = (
            scratch.fixed_end_cache[elem_index][0],
            scratch.fixed_end_cache[elem_index][1],
            scratch.fixed_end_cache[elem_index][2],
            scratch.fixed_end_cache[elem_index][3],
            scratch.fixed_end_cache[elem_index][4],
            scratch.fixed_end_cache[elem_index][5],
            scratch.fixed_end_cache[elem_index][6],
            scratch.fixed_end_cache[elem_index][7],
            scratch.fixed_end_cache[elem_index][8],
            scratch.fixed_end_cache[elem_index][9],
        )

    var v_basic_0: Float64
    var v_basic_1: Float64
    var v_basic_2: Float64
    var v_basic_3: Float64
    var v_basic_4: Float64
    var torsion_basic: Float64
    (v_basic_0, v_basic_1, v_basic_2, v_basic_3, v_basic_4, torsion_basic) = (
        _force_beam_column3d_basic_state(scratch.u_local, L)
    )
    var q_basic_0 = (E * A / L) * v_basic_0 + fixed_end[0]
    var q_basic_1 = (E * Iz / L) * (4.0 * v_basic_1 + 2.0 * v_basic_2) + fixed_end[1]
    var q_basic_2 = (E * Iz / L) * (2.0 * v_basic_1 + 4.0 * v_basic_2) + fixed_end[2]
    var q_basic_3 = (E * Iy / L) * (4.0 * v_basic_3 + 2.0 * v_basic_4) + fixed_end[3]
    var q_basic_4 = (E * Iy / L) * (2.0 * v_basic_3 + 4.0 * v_basic_4) + fixed_end[4]
    var q_basic_5 = 0.0
    if G > 0.0 and J > 0.0:
        q_basic_5 = (G * J / L) * torsion_basic

    var inv_L = 1.0 / L
    if elem_index >= 0 and elem_index < len(scratch.cached_inv_length):
        inv_L = scratch.cached_inv_length[elem_index]
    _zero_static_tuple(scratch.k_local_flat)
    var axial_k = E * A / L
    var flex_z = E * Iz / L
    var flex_y = E * Iy / L
    var torsion_k = 0.0
    if G > 0.0 and J > 0.0:
        torsion_k = G * J / L
    var use_cached_basic_coeff = elem_index >= 0 and elem_index < len(scratch.basic_coeff_cache)
    if use_cached_basic_coeff:
        var coeff = scratch.basic_coeff_cache[elem_index]
        for a in range(12):
            for b in range(12):
                var a0 = coeff[0 * 12 + a]
                var a1 = coeff[1 * 12 + a]
                var a2 = coeff[2 * 12 + a]
                var a3 = coeff[3 * 12 + a]
                var a4 = coeff[4 * 12 + a]
                var a5 = coeff[5 * 12 + a]
                var b0 = coeff[0 * 12 + b]
                var b1 = coeff[1 * 12 + b]
                var b2 = coeff[2 * 12 + b]
                var b3 = coeff[3 * 12 + b]
                var b4 = coeff[4 * 12 + b]
                var b5 = coeff[5 * 12 + b]
                var idx = _mat12_index(a, b)
                scratch.k_local_flat[idx] += a0 * axial_k * b0
                scratch.k_local_flat[idx] += flex_z * (
                    a1 * (4.0 * b1 + 2.0 * b2) + a2 * (2.0 * b1 + 4.0 * b2)
                )
                scratch.k_local_flat[idx] += flex_y * (
                    a3 * (4.0 * b3 + 2.0 * b4) + a4 * (2.0 * b3 + 4.0 * b4)
                )
                if torsion_k != 0.0:
                    scratch.k_local_flat[idx] += a5 * torsion_k * b5
    else:
        for a in range(12):
            for b in range(12):
                var a0 = _beam3d_local_basic_coeff(0, a, inv_L)
                var a1 = _beam3d_local_basic_coeff(1, a, inv_L)
                var a2 = _beam3d_local_basic_coeff(2, a, inv_L)
                var a3 = _beam3d_local_basic_coeff(3, a, inv_L)
                var a4 = _beam3d_local_basic_coeff(4, a, inv_L)
                var a5 = _beam3d_local_basic_coeff(5, a, inv_L)
                var b0 = _beam3d_local_basic_coeff(0, b, inv_L)
                var b1 = _beam3d_local_basic_coeff(1, b, inv_L)
                var b2 = _beam3d_local_basic_coeff(2, b, inv_L)
                var b3 = _beam3d_local_basic_coeff(3, b, inv_L)
                var b4 = _beam3d_local_basic_coeff(4, b, inv_L)
                var b5 = _beam3d_local_basic_coeff(5, b, inv_L)
                var idx = _mat12_index(a, b)
                scratch.k_local_flat[idx] += a0 * axial_k * b0
                scratch.k_local_flat[idx] += flex_z * (
                    a1 * (4.0 * b1 + 2.0 * b2) + a2 * (2.0 * b1 + 4.0 * b2)
                )
                scratch.k_local_flat[idx] += flex_y * (
                    a3 * (4.0 * b3 + 2.0 * b4) + a4 * (2.0 * b3 + 4.0 * b4)
                )
                if torsion_k != 0.0:
                    scratch.k_local_flat[idx] += a5 * torsion_k * b5

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        scratch.k_local_flat[_mat12_index(1, 1)] += q_basic_0 / L
        scratch.k_local_flat[_mat12_index(1, 7)] -= q_basic_0 / L
        scratch.k_local_flat[_mat12_index(7, 1)] -= q_basic_0 / L
        scratch.k_local_flat[_mat12_index(7, 7)] += q_basic_0 / L
        scratch.k_local_flat[_mat12_index(2, 2)] += q_basic_0 / L
        scratch.k_local_flat[_mat12_index(2, 8)] -= q_basic_0 / L
        scratch.k_local_flat[_mat12_index(8, 2)] -= q_basic_0 / L
        scratch.k_local_flat[_mat12_index(8, 8)] += q_basic_0 / L

    _zero_static_tuple(scratch.f_local)
    if use_cached_basic_coeff:
        var coeff = scratch.basic_coeff_cache[elem_index]
        for a in range(12):
            scratch.f_local[a] += coeff[0 * 12 + a] * q_basic_0
            scratch.f_local[a] += coeff[1 * 12 + a] * q_basic_1
            scratch.f_local[a] += coeff[2 * 12 + a] * q_basic_2
            scratch.f_local[a] += coeff[3 * 12 + a] * q_basic_3
            scratch.f_local[a] += coeff[4 * 12 + a] * q_basic_4
            scratch.f_local[a] += coeff[5 * 12 + a] * q_basic_5
    else:
        for a in range(12):
            scratch.f_local[a] += _beam3d_local_basic_coeff(0, a, inv_L) * q_basic_0
            scratch.f_local[a] += _beam3d_local_basic_coeff(1, a, inv_L) * q_basic_1
            scratch.f_local[a] += _beam3d_local_basic_coeff(2, a, inv_L) * q_basic_2
            scratch.f_local[a] += _beam3d_local_basic_coeff(3, a, inv_L) * q_basic_3
            scratch.f_local[a] += _beam3d_local_basic_coeff(4, a, inv_L) * q_basic_4
            scratch.f_local[a] += _beam3d_local_basic_coeff(5, a, inv_L) * q_basic_5
    scratch.f_local[0] += fixed_end[5]
    scratch.f_local[1] += fixed_end[6]
    scratch.f_local[2] += fixed_end[8]
    scratch.f_local[7] += fixed_end[7]
    scratch.f_local[8] += fixed_end[9]

    _beam3d_transform_stiffness_local_to_global_flat(
        scratch.R, scratch.k_local_flat, scratch.k_work_flat, k_global_out
    )
    _beam3d_transform_f_local_to_global_static(scratch.R, scratch.f_local, f_global_out)


fn force_beam_column3d_fiber_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut scratch: ForceBeamColumn3dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    force_beam_column3d_fiber_global_tangent_and_internal(
        0,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        elem_state_count,
        integration,
        num_int_pts,
        G,
        J,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_fiber_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var scratch = ForceBeamColumn3dScratch()
    force_beam_column3d_fiber_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        elem_state_count,
        integration,
        num_int_pts,
        G,
        J,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_fiber_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var scratch = ForceBeamColumn3dScratch()
    force_beam_column3d_fiber_global_tangent_and_internal(
        elem_index,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        elem_state_count,
        integration,
        num_int_pts,
        G,
        J,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_fiber_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut scratch: ForceBeamColumn3dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var fibers_per_section = sec_def.fiber_count
    if elem_state_count != num_int_pts * fibers_per_section:
        abort("forceBeamColumn3d element state count mismatch")

    var predictor_state_count = 5 + 3 * num_int_pts
    if force_basic_q_count < predictor_state_count:
        abort("forceBeamColumn3d basic force state count mismatch")
    if (
        force_basic_q_offset < 0
        or force_basic_q_offset + predictor_state_count > len(force_basic_q_state)
    ):
        abort("forceBeamColumn3d basic force state out of range")

    _ensure_force_beam_column3d_geometry_cache(
        elem_index, x1, y1, z1, x2, y2, z2, scratch
    )
    _ensure_force_beam_column3d_load_cache(
        elem_index,
        integration,
        num_int_pts,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        scratch,
    )
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if elem_index >= 0 and elem_index < len(scratch.cached_length):
        L = scratch.cached_length[elem_index]
    if L == 0.0:
        abort("zero-length element")

    if geom_transf == "Corotational":
        _element_rotation_into_static(
            x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf, scratch.R
        )
    elif elem_index >= 0 and elem_index < len(scratch.rotation_cache):
        scratch.R = scratch.rotation_cache[elem_index]
    else:
        _element_rotation_into_static(
            x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf, scratch.R
        )
    _beam3d_transform_u_global_to_local_static(scratch.R, u_elem_global, scratch.u_local)

    beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
    scratch.section_load_axial.resize(num_int_pts, 0.0)
    scratch.section_load_my.resize(num_int_pts, 0.0)
    scratch.section_load_mz.resize(num_int_pts, 0.0)
    for ip in range(num_int_pts):
        scratch.section_load_axial[ip] = scratch.section_load_axial_cache[elem_index][ip]
        scratch.section_load_my[ip] = scratch.section_load_my_cache[elem_index][ip]
        scratch.section_load_mz[ip] = scratch.section_load_mz_cache[elem_index][ip]
    var fixed_end = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    if elem_index >= 0 and elem_index < len(scratch.fixed_end_cache):
        fixed_end = (
            scratch.fixed_end_cache[elem_index][0],
            scratch.fixed_end_cache[elem_index][1],
            scratch.fixed_end_cache[elem_index][2],
            scratch.fixed_end_cache[elem_index][3],
            scratch.fixed_end_cache[elem_index][4],
            scratch.fixed_end_cache[elem_index][5],
            scratch.fixed_end_cache[elem_index][6],
            scratch.fixed_end_cache[elem_index][7],
            scratch.fixed_end_cache[elem_index][8],
            scratch.fixed_end_cache[elem_index][9],
        )

    var v_basic_0: Float64
    var v_basic_1: Float64
    var v_basic_2: Float64
    var v_basic_3: Float64
    var v_basic_4: Float64
    var torsion_basic: Float64
    (v_basic_0, v_basic_1, v_basic_2, v_basic_3, v_basic_4, torsion_basic) = (
        _force_beam_column3d_basic_state(scratch.u_local, L)
    )

    var eps0_offset = force_basic_q_offset + 5
    var ky_offset = eps0_offset + num_int_pts
    var kz_offset = ky_offset + num_int_pts

    _zero_static_tuple(scratch.k_basic_flat)
    _ensure_force_beam_column3d_section_history_capacity(scratch, num_int_pts)
    var current_basic = _force_beam_column3d_section_basic_deformation(
        L,
        num_int_pts,
        scratch,
        force_basic_q_state,
        eps0_offset,
        ky_offset,
        kz_offset,
    )
    var accepted_basic_0 = current_basic[0]
    var accepted_basic_1 = current_basic[1]
    var accepted_basic_2 = current_basic[2]
    var accepted_basic_3 = current_basic[3]
    var accepted_basic_4 = current_basic[4]
    var accepted_q0 = force_basic_q_state[force_basic_q_offset]
    var accepted_q1 = force_basic_q_state[force_basic_q_offset + 1]
    var accepted_q2 = force_basic_q_state[force_basic_q_offset + 2]
    var accepted_q3 = force_basic_q_state[force_basic_q_offset + 3]
    var accepted_q4 = force_basic_q_state[force_basic_q_offset + 4]
    for ip in range(num_int_pts):
        scratch.section_vs_eps0[ip] = force_basic_q_state[eps0_offset + ip]
        scratch.section_vs_ky[ip] = force_basic_q_state[ky_offset + ip]
        scratch.section_vs_kz[ip] = force_basic_q_state[kz_offset + ip]

    var tolerance = 1.0e-12
    var cutback_factor = 10.0
    var max_subdivisions = 4
    var converged = False
    var remaining_0 = v_basic_0 - accepted_basic_0
    var remaining_1 = v_basic_1 - accepted_basic_1
    var remaining_2 = v_basic_2 - accepted_basic_2
    var remaining_3 = v_basic_3 - accepted_basic_3
    var remaining_4 = v_basic_4 - accepted_basic_4
    var attempt_0 = remaining_0
    var attempt_1 = remaining_1
    var attempt_2 = remaining_2
    var attempt_3 = remaining_3
    var attempt_4 = remaining_4
    var num_subdivide = 1

    while True:
        if num_subdivide > max_subdivisions:
            break
        var target_v0 = accepted_basic_0 + attempt_0
        var target_v1 = accepted_basic_1 + attempt_1
        var target_v2 = accepted_basic_2 + attempt_2
        var target_v3 = accepted_basic_3 + attempt_3
        var target_v4 = accepted_basic_4 + attempt_4
        var scheme_success = False
        for use_initial in range(3):
            _restore_force_beam_column3d_predictor_state(
                force_basic_q_state,
                force_basic_q_offset,
                num_int_pts,
                accepted_q0,
                accepted_q1,
                accepted_q2,
                accepted_q3,
                accepted_q4,
                scratch.section_vs_eps0,
                scratch.section_vs_ky,
                scratch.section_vs_kz,
            )
            var solved = _force_beam_column3d_try_increment(
                L,
                sec_def,
                fibers,
                uniaxial_defs,
                uniaxial_states,
                elem_state_ids,
                elem_state_offset,
                fibers_per_section,
                num_int_pts,
                force_basic_q_state,
                force_basic_q_offset,
                scratch,
                accepted_basic_0,
                accepted_basic_1,
                accepted_basic_2,
                accepted_basic_3,
                accepted_basic_4,
                attempt_0,
                attempt_1,
                attempt_2,
                attempt_3,
                attempt_4,
                use_initial,
            )
            if not solved[0]:
                continue
            accepted_basic_0 = target_v0
            accepted_basic_1 = target_v1
            accepted_basic_2 = target_v2
            accepted_basic_3 = target_v3
            accepted_basic_4 = target_v4
            accepted_q0 = solved[1]
            accepted_q1 = solved[2]
            accepted_q2 = solved[3]
            accepted_q3 = solved[4]
            accepted_q4 = solved[5]
            for ip in range(num_int_pts):
                scratch.section_vs_eps0[ip] = force_basic_q_state[eps0_offset + ip]
                scratch.section_vs_ky[ip] = force_basic_q_state[ky_offset + ip]
                scratch.section_vs_kz[ip] = force_basic_q_state[kz_offset + ip]
            remaining_0 = v_basic_0 - accepted_basic_0
            remaining_1 = v_basic_1 - accepted_basic_1
            remaining_2 = v_basic_2 - accepted_basic_2
            remaining_3 = v_basic_3 - accepted_basic_3
            remaining_4 = v_basic_4 - accepted_basic_4
            var remaining_norm = max(
                max(abs(remaining_0), abs(remaining_1)),
                max(max(abs(remaining_2), abs(remaining_3)), abs(remaining_4)),
            )
            if remaining_norm <= tolerance:
                converged = True
            else:
                attempt_0 = remaining_0
                attempt_1 = remaining_1
                attempt_2 = remaining_2
                attempt_3 = remaining_3
                attempt_4 = remaining_4
                num_subdivide = 1
            scheme_success = True
            break
        if converged:
            break
        if scheme_success:
            continue
        attempt_0 /= cutback_factor
        attempt_1 /= cutback_factor
        attempt_2 /= cutback_factor
        attempt_3 /= cutback_factor
        attempt_4 /= cutback_factor
        num_subdivide += 1

    if not converged:
        _restore_force_beam_column3d_predictor_state(
            force_basic_q_state,
            force_basic_q_offset,
            num_int_pts,
            accepted_q0,
            accepted_q1,
            accepted_q2,
            accepted_q3,
            accepted_q4,
            scratch.section_vs_eps0,
            scratch.section_vs_ky,
            scratch.section_vs_kz,
        )
        abort("forceBeamColumn3d element compatibility did not converge")
    var q0 = accepted_q0
    var q1 = accepted_q1
    var q2 = accepted_q2
    var q3 = accepted_q3
    var q4 = accepted_q4

    var inv_L = 1.0 / L
    if elem_index >= 0 and elem_index < len(scratch.cached_inv_length):
        inv_L = scratch.cached_inv_length[elem_index]
    _zero_static_tuple(scratch.k_local_flat)
    var use_cached_basic_coeff = elem_index >= 0 and elem_index < len(scratch.basic_coeff_cache)

    if use_cached_basic_coeff:
        var coeff = scratch.basic_coeff_cache[elem_index]
        for a in range(12):
            for b in range(12):
                var a0 = coeff[0 * 12 + a]
                var a1 = coeff[1 * 12 + a]
                var a2 = coeff[2 * 12 + a]
                var a3 = coeff[3 * 12 + a]
                var a4 = coeff[4 * 12 + a]
                var b0 = coeff[0 * 12 + b]
                var b1 = coeff[1 * 12 + b]
                var b2 = coeff[2 * 12 + b]
                var b3 = coeff[3 * 12 + b]
                var b4 = coeff[4 * 12 + b]
                for i in range(5):
                    for j in range(5):
                        var ai = a4
                        if i == 0:
                            ai = a0
                        elif i == 1:
                            ai = a1
                        elif i == 2:
                            ai = a2
                        elif i == 3:
                            ai = a3
                        var bj = b4
                        if j == 0:
                            bj = b0
                        elif j == 1:
                            bj = b1
                        elif j == 2:
                            bj = b2
                        elif j == 3:
                            bj = b3
                        scratch.k_local_flat[_mat12_index(a, b)] += (
                            ai * scratch.k_basic_flat[_mat5_index(i, j)] * bj
                        )
    else:
        for a in range(12):
            for b in range(12):
                var a0 = _beam3d_local_basic_coeff(0, a, inv_L)
                var a1 = _beam3d_local_basic_coeff(1, a, inv_L)
                var a2 = _beam3d_local_basic_coeff(2, a, inv_L)
                var a3 = _beam3d_local_basic_coeff(3, a, inv_L)
                var a4 = _beam3d_local_basic_coeff(4, a, inv_L)
                var b0 = _beam3d_local_basic_coeff(0, b, inv_L)
                var b1 = _beam3d_local_basic_coeff(1, b, inv_L)
                var b2 = _beam3d_local_basic_coeff(2, b, inv_L)
                var b3 = _beam3d_local_basic_coeff(3, b, inv_L)
                var b4 = _beam3d_local_basic_coeff(4, b, inv_L)
                for i in range(5):
                    for j in range(5):
                        var ai = a4
                        if i == 0:
                            ai = a0
                        elif i == 1:
                            ai = a1
                        elif i == 2:
                            ai = a2
                        elif i == 3:
                            ai = a3
                        var bj = b4
                        if j == 0:
                            bj = b0
                        elif j == 1:
                            bj = b1
                        elif j == 2:
                            bj = b2
                        elif j == 3:
                            bj = b3
                        scratch.k_local_flat[_mat12_index(a, b)] += (
                            ai * scratch.k_basic_flat[_mat5_index(i, j)] * bj
                        )
    if G > 0.0 and J > 0.0:
        var torsion_k = G * J / L
        if use_cached_basic_coeff:
            var coeff = scratch.basic_coeff_cache[elem_index]
            for a in range(12):
                for b in range(12):
                    scratch.k_local_flat[_mat12_index(a, b)] += (
                        coeff[5 * 12 + a] * torsion_k * coeff[5 * 12 + b]
                    )
        else:
            for a in range(12):
                for b in range(12):
                    scratch.k_local_flat[_mat12_index(a, b)] += (
                        _beam3d_local_basic_coeff(5, a, inv_L)
                        * torsion_k
                        * _beam3d_local_basic_coeff(5, b, inv_L)
                    )

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        scratch.k_local_flat[_mat12_index(1, 1)] += q0 / L
        scratch.k_local_flat[_mat12_index(1, 7)] -= q0 / L
        scratch.k_local_flat[_mat12_index(7, 1)] -= q0 / L
        scratch.k_local_flat[_mat12_index(7, 7)] += q0 / L
        scratch.k_local_flat[_mat12_index(2, 2)] += q0 / L
        scratch.k_local_flat[_mat12_index(2, 8)] -= q0 / L
        scratch.k_local_flat[_mat12_index(8, 2)] -= q0 / L
        scratch.k_local_flat[_mat12_index(8, 8)] += q0 / L

    _zero_static_tuple(scratch.f_local)
    var q_basic_5 = 0.0
    if G > 0.0 and J > 0.0:
        q_basic_5 = (G * J / L) * torsion_basic
    if use_cached_basic_coeff:
        var coeff = scratch.basic_coeff_cache[elem_index]
        for a in range(12):
            scratch.f_local[a] += coeff[0 * 12 + a] * q0
            scratch.f_local[a] += coeff[1 * 12 + a] * q1
            scratch.f_local[a] += coeff[2 * 12 + a] * q2
            scratch.f_local[a] += coeff[3 * 12 + a] * q3
            scratch.f_local[a] += coeff[4 * 12 + a] * q4
            scratch.f_local[a] += coeff[5 * 12 + a] * q_basic_5
    else:
        for a in range(12):
            scratch.f_local[a] += _beam3d_local_basic_coeff(0, a, inv_L) * q0
            scratch.f_local[a] += _beam3d_local_basic_coeff(1, a, inv_L) * q1
            scratch.f_local[a] += _beam3d_local_basic_coeff(2, a, inv_L) * q2
            scratch.f_local[a] += _beam3d_local_basic_coeff(3, a, inv_L) * q3
            scratch.f_local[a] += _beam3d_local_basic_coeff(4, a, inv_L) * q4
            scratch.f_local[a] += _beam3d_local_basic_coeff(5, a, inv_L) * q_basic_5
    scratch.f_local[0] += fixed_end[5]
    scratch.f_local[1] += fixed_end[6]
    scratch.f_local[2] += fixed_end[8]
    scratch.f_local[7] += fixed_end[7]
    scratch.f_local[8] += fixed_end[9]

    _beam3d_transform_stiffness_local_to_global_flat(
        scratch.R, scratch.k_local_flat, scratch.k_work_flat, k_global_out
    )
    _beam3d_transform_f_local_to_global_static(scratch.R, scratch.f_local, f_global_out)


fn force_beam_column3d_fiber_section_response(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    section_no: Int,
    want_deformation: Bool,
) -> List[Float64]:
    if section_no < 1 or section_no > num_int_pts:
        abort("forceBeamColumn3d section index out of range")
    var k_dummy: List[List[Float64]] = []
    var f_dummy: List[Float64] = []
    force_beam_column3d_fiber_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        elem_state_count,
        integration,
        num_int_pts,
        G,
        J,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        k_dummy,
        f_dummy,
    )

    var ip = section_no - 1
    var q0 = force_basic_q_state[force_basic_q_offset]
    var q1 = force_basic_q_state[force_basic_q_offset + 1]
    var q2 = force_basic_q_state[force_basic_q_offset + 2]
    var q3 = force_basic_q_state[force_basic_q_offset + 3]
    var q4 = force_basic_q_state[force_basic_q_offset + 4]
    var eps0_offset = force_basic_q_offset + 5
    var ky_offset = eps0_offset + num_int_pts
    var kz_offset = ky_offset + num_int_pts
    var u_local: List[Float64] = []
    _beam3d_transform_u_global_to_local(
        _element_rotation(x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf),
        u_elem_global,
        u_local,
    )
    var twist = (u_local[9] - u_local[3]) / sqrt(
        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)
    )

    if want_deformation:
        return [
            force_basic_q_state[eps0_offset + ip],
            force_basic_q_state[kz_offset + ip],
            force_basic_q_state[ky_offset + ip],
            twist,
        ]

    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var xi = xis[ip]
    var b_mi = xi - 1.0
    var b_mj = xi
    return [
        q0,
        b_mi * q1 + b_mj * q2,
        b_mi * q3 + b_mj * q4,
        G * J * twist,
    ]
