from collections import List
from math import sqrt
from os import abort

from elements.beam_loads import beam3d_basic_fixed_end_and_reactions, beam3d_section_load_response
from elements.beam_integration import (
    BeamIntegrationCache,
    beam_integration_cache_ensure,
    beam_integration_rule,
)
from elements.utils import (
    _beam3d_local_axes_from_vecxz,
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
    fiber_section3d_set_trial_instance,
)
from utils import StaticTuple


alias Beam3dMat3 = StaticTuple[Float64, 9]
alias Beam3dVec10 = StaticTuple[Float64, 10]
alias Beam3dMat5 = StaticTuple[Float64, 25]
alias Beam3dVec5 = StaticTuple[Float64, 5]
alias Beam3dMat12 = StaticTuple[Float64, 144]
alias Beam3dVec12 = StaticTuple[Float64, 12]


struct ForceBeamColumn3dScratch(Copyable, Movable):
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
    var load_has_nonzero: List[Bool]
    var cached_length: List[Float64]
    var cached_inv_length: List[Float64]
    var rotation_cache: List[Beam3dMat3]
    var section_load_offsets: List[Int]
    var section_load_counts: List[Int]
    var section_load_axial_pool: List[Float64]
    var section_load_my_pool: List[Float64]
    var section_load_mz_pool: List[Float64]
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
    var section_init_fs11: List[Float64]
    var section_init_fs12: List[Float64]
    var section_init_fs13: List[Float64]
    var section_init_fs22: List[Float64]
    var section_init_fs23: List[Float64]
    var section_init_fs33: List[Float64]

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
        self.load_has_nonzero = []
        self.cached_length = []
        self.cached_inv_length = []
        self.rotation_cache = []
        self.section_load_offsets = []
        self.section_load_counts = []
        self.section_load_axial_pool = []
        self.section_load_my_pool = []
        self.section_load_mz_pool = []
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
        self.section_init_fs11 = []
        self.section_init_fs12 = []
        self.section_init_fs13 = []
        self.section_init_fs22 = []
        self.section_init_fs23 = []
        self.section_init_fs33 = []


fn reset_force_beam_column3d_scratch(mut scratch: ForceBeamColumn3dScratch):
    scratch.integration_cache.is_valid = False
    scratch.geometry_valid.clear()
    scratch.load_valid.clear()
    scratch.load_has_nonzero.clear()
    scratch.cached_length.clear()
    scratch.cached_inv_length.clear()
    scratch.rotation_cache.clear()
    scratch.section_load_offsets.clear()
    scratch.section_load_counts.clear()
    scratch.section_load_axial_pool.clear()
    scratch.section_load_my_pool.clear()
    scratch.section_load_mz_pool.clear()
    scratch.fixed_end_cache.clear()


fn invalidate_force_beam_column3d_load_cache(mut scratch: ForceBeamColumn3dScratch):
    var ptr = scratch.load_valid.unsafe_ptr()
    for i in range(len(scratch.load_valid)):
        ptr[i] = False


@always_inline
fn _mat3_index(row: Int, col: Int) -> Int:
    return row * 3 + col


@always_inline
fn _mat5_index(row: Int, col: Int) -> Int:
    return row * 5 + col


@always_inline
fn _mat12_index(row: Int, col: Int) -> Int:
    return row * 12 + col


@always_inline
fn _zero_static_tuple[size: Int](mut values: StaticTuple[Float64, size]):
    for i in range(size):
        values[i] = 0.0


fn _beam3d_rotation_into_static(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    mut R: Beam3dMat3,
):
    _beam3d_rotation_into_static(x1, y1, z1, x2, y2, z2, False, 0.0, 0.0, 0.0, R)


fn _beam3d_rotation_into_static(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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

    var yx: Float64
    var yy: Float64
    var yz: Float64
    var zx: Float64
    var zy: Float64
    var zz: Float64
    (yx, yy, yz, zx, zy, zz) = _beam3d_local_axes_from_vecxz(
        lx,
        ly,
        lz,
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
    )
    R[0] = lx
    R[1] = ly
    R[2] = lz
    R[3] = yx
    R[4] = yy
    R[5] = yz
    R[6] = zx
    R[7] = zy
    R[8] = zz


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
    _element_rotation_into_static(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
        0.0,
        R,
    )


fn _element_rotation_into_static(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            R,
        )
        return
    _beam3d_rotation_into_static(x1, y1, z1, x2, y2, z2, has_vecxz, vecxz_x, vecxz_y, vecxz_z, R)


@always_inline
fn _beam3d_transform_u_global_to_local_static(
    R: Beam3dMat3, u_global: List[Float64], mut u_local: Beam3dVec12
):
    var r00 = R[0]
    var r01 = R[1]
    var r02 = R[2]
    var r10 = R[3]
    var r11 = R[4]
    var r12 = R[5]
    var r20 = R[6]
    var r21 = R[7]
    var r22 = R[8]
    var ug = u_global.unsafe_ptr()
    for block in range(4):
        var offset = 3 * block
        var g0 = ug[offset]
        var g1 = ug[offset + 1]
        var g2 = ug[offset + 2]
        u_local[offset] = r00 * g0 + r01 * g1 + r02 * g2
        u_local[offset + 1] = r10 * g0 + r11 * g1 + r12 * g2
        u_local[offset + 2] = r20 * g0 + r21 * g1 + r22 * g2


@always_inline
fn _beam3d_transform_f_local_to_global_static(
    R: Beam3dMat3, f_local: Beam3dVec12, mut f_global: List[Float64]
):
    if len(f_global) != 12:
        _ensure_zero_vector(f_global, 12)
    var r00 = R[0]
    var r01 = R[1]
    var r02 = R[2]
    var r10 = R[3]
    var r11 = R[4]
    var r12 = R[5]
    var r20 = R[6]
    var r21 = R[7]
    var r22 = R[8]
    var out = f_global.unsafe_ptr()
    for block in range(4):
        var offset = 3 * block
        var l0 = f_local[offset]
        var l1 = f_local[offset + 1]
        var l2 = f_local[offset + 2]
        out[offset] = r00 * l0 + r10 * l1 + r20 * l2
        out[offset + 1] = r01 * l0 + r11 * l1 + r21 * l2
        out[offset + 2] = r02 * l0 + r12 * l1 + r22 * l2


@always_inline
fn _beam3d_transform_stiffness_local_to_global_flat(
    R: Beam3dMat3,
    k_local: Beam3dMat12,
    mut work: Beam3dMat12,
    mut k_global: List[List[Float64]],
):
    if len(k_global) != 12 or len(k_global[0]) != 12:
        _ensure_zero_matrix(k_global, 12, 12)
    var r00 = R[0]
    var r01 = R[1]
    var r02 = R[2]
    var r10 = R[3]
    var r11 = R[4]
    var r12 = R[5]
    var r20 = R[6]
    var r21 = R[7]
    var r22 = R[8]
    var r0_v4 = SIMD[DType.float64, 4](r00, r01, r02, 0.0)
    var r1_v4 = SIMD[DType.float64, 4](r10, r11, r12, 0.0)
    var r2_v4 = SIMD[DType.float64, 4](r20, r21, r22, 0.0)
    for row in range(12):
        var row_base = row * 12

        @parameter
        for col_block in range(4):
            var base = row_base + 3 * col_block
            var out = r0_v4 * k_local[base] + r1_v4 * k_local[base + 1] + r2_v4 * k_local[base + 2]
            work[base] = out[0]
            work[base + 1] = out[1]
            work[base + 2] = out[2]

    var r00_8 = SIMD[DType.float64, 8](r00)
    var r10_8 = SIMD[DType.float64, 8](r10)
    var r20_8 = SIMD[DType.float64, 8](r20)
    var r01_8 = SIMD[DType.float64, 8](r01)
    var r11_8 = SIMD[DType.float64, 8](r11)
    var r21_8 = SIMD[DType.float64, 8](r21)
    var r02_8 = SIMD[DType.float64, 8](r02)
    var r12_8 = SIMD[DType.float64, 8](r12)
    var r22_8 = SIMD[DType.float64, 8](r22)

    var r00_4 = SIMD[DType.float64, 4](r00)
    var r10_4 = SIMD[DType.float64, 4](r10)
    var r20_4 = SIMD[DType.float64, 4](r20)
    var r01_4 = SIMD[DType.float64, 4](r01)
    var r11_4 = SIMD[DType.float64, 4](r11)
    var r21_4 = SIMD[DType.float64, 4](r21)
    var r02_4 = SIMD[DType.float64, 4](r02)
    var r12_4 = SIMD[DType.float64, 4](r12)
    var r22_4 = SIMD[DType.float64, 4](r22)

    @parameter
    for row_block in range(4):
        var row_offset = 3 * row_block
        var w0_base = row_offset * 12
        var w1_base = (row_offset + 1) * 12
        var w2_base = (row_offset + 2) * 12
        var dst0 = k_global[row_offset].unsafe_ptr()
        var dst1 = k_global[row_offset + 1].unsafe_ptr()
        var dst2 = k_global[row_offset + 2].unsafe_ptr()

        var w0_lo = SIMD[DType.float64, 8](
            work[w0_base],
            work[w0_base + 1],
            work[w0_base + 2],
            work[w0_base + 3],
            work[w0_base + 4],
            work[w0_base + 5],
            work[w0_base + 6],
            work[w0_base + 7],
        )
        var w1_lo = SIMD[DType.float64, 8](
            work[w1_base],
            work[w1_base + 1],
            work[w1_base + 2],
            work[w1_base + 3],
            work[w1_base + 4],
            work[w1_base + 5],
            work[w1_base + 6],
            work[w1_base + 7],
        )
        var w2_lo = SIMD[DType.float64, 8](
            work[w2_base],
            work[w2_base + 1],
            work[w2_base + 2],
            work[w2_base + 3],
            work[w2_base + 4],
            work[w2_base + 5],
            work[w2_base + 6],
            work[w2_base + 7],
        )
        dst0.store(0, r00_8 * w0_lo + r10_8 * w1_lo + r20_8 * w2_lo)
        dst1.store(0, r01_8 * w0_lo + r11_8 * w1_lo + r21_8 * w2_lo)
        dst2.store(0, r02_8 * w0_lo + r12_8 * w1_lo + r22_8 * w2_lo)

        var w0_hi = SIMD[DType.float64, 4](
            work[w0_base + 8],
            work[w0_base + 9],
            work[w0_base + 10],
            work[w0_base + 11],
        )
        var w1_hi = SIMD[DType.float64, 4](
            work[w1_base + 8],
            work[w1_base + 9],
            work[w1_base + 10],
            work[w1_base + 11],
        )
        var w2_hi = SIMD[DType.float64, 4](
            work[w2_base + 8],
            work[w2_base + 9],
            work[w2_base + 10],
            work[w2_base + 11],
        )
        dst0.store(8, r00_4 * w0_hi + r10_4 * w1_hi + r20_4 * w2_hi)
        dst1.store(8, r01_4 * w0_hi + r11_4 * w1_hi + r21_4 * w2_hi)
        dst2.store(8, r02_4 * w0_hi + r12_4 * w1_hi + r22_4 * w2_hi)


@always_inline
fn _beam3d_write_local_stiffness_col_pair(
    col: Int,
    neg_col: Int,
    m0: Float64,
    m1: Float64,
    m2: Float64,
    m3: Float64,
    m4: Float64,
    inv_L: Float64,
    mut k_local: Beam3dMat12,
):
    var r0 = -m0
    var r1 = (m1 + m2) * inv_L
    var r2 = -(m3 + m4) * inv_L
    k_local[0 * 12 + col] = r0
    k_local[1 * 12 + col] = r1
    k_local[2 * 12 + col] = r2
    k_local[3 * 12 + col] = 0.0
    k_local[4 * 12 + col] = m3
    k_local[5 * 12 + col] = m1
    k_local[6 * 12 + col] = -r0
    k_local[7 * 12 + col] = -r1
    k_local[8 * 12 + col] = -r2
    k_local[9 * 12 + col] = 0.0
    k_local[10 * 12 + col] = m4
    k_local[11 * 12 + col] = m2

    k_local[0 * 12 + neg_col] = -r0
    k_local[1 * 12 + neg_col] = -r1
    k_local[2 * 12 + neg_col] = -r2
    k_local[3 * 12 + neg_col] = 0.0
    k_local[4 * 12 + neg_col] = -m3
    k_local[5 * 12 + neg_col] = -m1
    k_local[6 * 12 + neg_col] = r0
    k_local[7 * 12 + neg_col] = r1
    k_local[8 * 12 + neg_col] = r2
    k_local[9 * 12 + neg_col] = 0.0
    k_local[10 * 12 + neg_col] = -m4
    k_local[11 * 12 + neg_col] = -m2


@always_inline
fn _beam3d_write_local_stiffness_col_single(
    col: Int,
    m0: Float64,
    m1: Float64,
    m2: Float64,
    m3: Float64,
    m4: Float64,
    inv_L: Float64,
    mut k_local: Beam3dMat12,
):
    var r0 = -m0
    var r1 = (m1 + m2) * inv_L
    var r2 = -(m3 + m4) * inv_L
    k_local[0 * 12 + col] = r0
    k_local[1 * 12 + col] = r1
    k_local[2 * 12 + col] = r2
    k_local[3 * 12 + col] = 0.0
    k_local[4 * 12 + col] = m3
    k_local[5 * 12 + col] = m1
    k_local[6 * 12 + col] = -r0
    k_local[7 * 12 + col] = -r1
    k_local[8 * 12 + col] = -r2
    k_local[9 * 12 + col] = 0.0
    k_local[10 * 12 + col] = m4
    k_local[11 * 12 + col] = m2


@always_inline
fn _beam3d_assemble_local_stiffness_from_basic(
    k_basic: Beam3dMat5,
    inv_L: Float64,
    torsion_k: Float64,
    mut k_local: Beam3dMat12,
):
    var k00 = k_basic[0]
    var k01 = k_basic[1]
    var k02 = k_basic[2]
    var k03 = k_basic[3]
    var k04 = k_basic[4]
    var k10 = k_basic[5]
    var k11 = k_basic[6]
    var k12 = k_basic[7]
    var k13 = k_basic[8]
    var k14 = k_basic[9]
    var k20 = k_basic[10]
    var k21 = k_basic[11]
    var k22 = k_basic[12]
    var k23 = k_basic[13]
    var k24 = k_basic[14]
    var k30 = k_basic[15]
    var k31 = k_basic[16]
    var k32 = k_basic[17]
    var k33 = k_basic[18]
    var k34 = k_basic[19]
    var k40 = k_basic[20]
    var k41 = k_basic[21]
    var k42 = k_basic[22]
    var k43 = k_basic[23]
    var k44 = k_basic[24]

    # Columns 0 and 6 (col 6 = -col 0)
    _beam3d_write_local_stiffness_col_pair(0, 6, -k00, -k10, -k20, -k30, -k40, inv_L, k_local)
    # Columns 1 and 7 (col 7 = -col 1)
    _beam3d_write_local_stiffness_col_pair(
        1,
        7,
        (k01 + k02) * inv_L,
        (k11 + k12) * inv_L,
        (k21 + k22) * inv_L,
        (k31 + k32) * inv_L,
        (k41 + k42) * inv_L,
        inv_L,
        k_local,
    )
    # Columns 2 and 8 (col 8 = -col 2)
    _beam3d_write_local_stiffness_col_pair(
        2,
        8,
        -(k03 + k04) * inv_L,
        -(k13 + k14) * inv_L,
        -(k23 + k24) * inv_L,
        -(k33 + k34) * inv_L,
        -(k43 + k44) * inv_L,
        inv_L,
        k_local,
    )
    # Rotational columns 4, 5, 10, 11
    _beam3d_write_local_stiffness_col_single(4, k03, k13, k23, k33, k43, inv_L, k_local)
    _beam3d_write_local_stiffness_col_single(5, k01, k11, k21, k31, k41, inv_L, k_local)
    _beam3d_write_local_stiffness_col_single(10, k04, k14, k24, k34, k44, inv_L, k_local)
    _beam3d_write_local_stiffness_col_single(11, k02, k12, k22, k32, k42, inv_L, k_local)
    # Torsion columns 3 and 9
    for r in range(12):
        k_local[r * 12 + 3] = 0.0
        k_local[r * 12 + 9] = 0.0
    if torsion_k != 0.0:
        k_local[3 * 12 + 3] = torsion_k
        k_local[3 * 12 + 9] = -torsion_k
        k_local[9 * 12 + 3] = -torsion_k
        k_local[9 * 12 + 9] = torsion_k


@always_inline
fn _beam3d_assemble_local_force_from_basic(
    q0: Float64,
    q1: Float64,
    q2: Float64,
    q3: Float64,
    q4: Float64,
    q5: Float64,
    inv_L: Float64,
    mut f_local: Beam3dVec12,
):
    var vz = (q1 + q2) * inv_L
    var vy = -(q3 + q4) * inv_L
    f_local[0] = -q0
    f_local[1] = vz
    f_local[2] = vy
    f_local[3] = -q5
    f_local[4] = q3
    f_local[5] = q1
    f_local[6] = q0
    f_local[7] = -vz
    f_local[8] = -vy
    f_local[9] = q5
    f_local[10] = q4
    f_local[11] = q2


@always_inline
fn _beam3d_add_axial_geometric_stiffness_flat(
    mut k_local: Beam3dMat12, axial_force: Float64, inv_L: Float64
):
    var N_over_L = axial_force * inv_L
    k_local[1 * 12 + 1] += N_over_L
    k_local[1 * 12 + 7] -= N_over_L
    k_local[7 * 12 + 1] -= N_over_L
    k_local[7 * 12 + 7] += N_over_L
    k_local[2 * 12 + 2] += N_over_L
    k_local[2 * 12 + 8] -= N_over_L
    k_local[8 * 12 + 2] -= N_over_L
    k_local[8 * 12 + 8] += N_over_L


@always_inline
fn _beam3d_add_axial_geometric_force(
    mut f_local: Beam3dVec12, u_local: Beam3dVec12, axial_force: Float64, inv_L: Float64
):
    """Add the P-Delta shears of an axial force acting through the chord drift.

    This is the force of `_beam3d_add_axial_geometric_stiffness_flat` at `u_local`
    (OpenSees PDeltaCrdTransf3d::getGlobalResistingForce). Without it the tangent
    carries P-Delta but the converged state does not.
    """
    var N_over_L = axial_force * inv_L
    var shear_y = (u_local[1] - u_local[7]) * N_over_L
    var shear_z = (u_local[2] - u_local[8]) * N_over_L
    f_local[1] += shear_y
    f_local[7] -= shear_y
    f_local[2] += shear_z
    f_local[8] -= shear_z


@always_inline
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
) -> (Bool, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,):
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


@always_inline
fn _invert_5x5_flat_into(mut a: Beam3dMat5, mut inv_out: Beam3dMat5) -> Bool:
    var a_rows = InlineArray[SIMD[DType.float64, 8], 5](
        SIMD[DType.float64, 8](a[0], a[1], a[2], a[3], a[4], 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](a[5], a[6], a[7], a[8], a[9], 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](a[10], a[11], a[12], a[13], a[14], 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](a[15], a[16], a[17], a[18], a[19], 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](a[20], a[21], a[22], a[23], a[24], 0.0, 0.0, 0.0),
    )
    var inv_rows = InlineArray[SIMD[DType.float64, 8], 5](
        SIMD[DType.float64, 8](1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        SIMD[DType.float64, 8](0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )

    @parameter
    for i in range(5):
        var pivot = i
        var max_val = abs(a_rows[i][i])

        @parameter
        for r in range(i + 1, 5):
            var abs_val = abs(a_rows[r][i])
            if abs_val > max_val:
                max_val = abs_val
                pivot = r
        if max_val <= 1.0e-40:
            return False

        if pivot != i:
            var tmp_a = a_rows[i]
            a_rows[i] = a_rows[pivot]
            a_rows[pivot] = tmp_a
            var tmp_inv = inv_rows[i]
            inv_rows[i] = inv_rows[pivot]
            inv_rows[pivot] = tmp_inv

        var inv_piv = 1.0 / a_rows[i][i]
        var ai = a_rows[i] * inv_piv
        var invi = inv_rows[i] * inv_piv
        a_rows[i] = ai
        inv_rows[i] = invi

        @parameter
        for r in range(5):

            @parameter
            if r != i:
                var factor = a_rows[r][i]
                a_rows[r] -= factor * ai
                inv_rows[r] -= factor * invi

    @parameter
    for r in range(5):
        var row = inv_rows[r]
        var base = r * 5
        inv_out[base] = row[0]
        inv_out[base + 1] = row[1]
        inv_out[base + 2] = row[2]
        inv_out[base + 3] = row[3]
        inv_out[base + 4] = row[4]
    return True


fn _ensure_force_beam_column3d_cache_slot(mut scratch: ForceBeamColumn3dScratch, elem_index: Int):
    if elem_index < 0:
        return
    var needed = elem_index + 1
    if len(scratch.geometry_valid) < needed:
        # Elements arrive in index order, so grow capacity geometrically; the
        # lengths below stay exact because callers test them for cache hits.
        if scratch.geometry_valid.capacity < needed:
            var capacity = max(needed, 2 * scratch.geometry_valid.capacity)
            scratch.geometry_valid.reserve(capacity)
            scratch.load_valid.reserve(capacity)
            scratch.load_has_nonzero.reserve(capacity)
            scratch.cached_length.reserve(capacity)
            scratch.cached_inv_length.reserve(capacity)
            scratch.section_load_offsets.reserve(capacity)
            scratch.section_load_counts.reserve(capacity)
        scratch.geometry_valid.resize(needed, False)
        scratch.load_valid.resize(needed, False)
        scratch.load_has_nonzero.resize(needed, False)
        scratch.cached_length.resize(needed, 0.0)
        scratch.cached_inv_length.resize(needed, 0.0)
        scratch.section_load_offsets.resize(needed, -1)
        scratch.section_load_counts.resize(needed, 0)
        while len(scratch.rotation_cache) < needed:
            scratch.rotation_cache.append(Beam3dMat3(fill=0.0))
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
    _ensure_force_beam_column3d_geometry_cache(
        elem_index, x1, y1, z1, x2, y2, z2, False, 0.0, 0.0, 0.0, scratch
    )


fn _ensure_force_beam_column3d_geometry_cache(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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

    _beam3d_rotation_into_static(
        x1, y1, z1, x2, y2, z2, has_vecxz, vecxz_x, vecxz_y, vecxz_z, scratch.R
    )
    scratch.rotation_cache[elem_index] = scratch.R
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

    var has_elem_loads = (
        load_scale != 0.0
        and elem_index + 1 < len(elem_load_offsets)
        and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
    )
    scratch.load_has_nonzero[elem_index] = has_elem_loads
    if not has_elem_loads:
        scratch.fixed_end_cache[elem_index] = Beam3dVec10(fill=0.0)
        scratch.load_valid[elem_index] = True
        return

    beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
    var L = scratch.cached_length[elem_index]
    var offset = scratch.section_load_offsets[elem_index]
    if offset < 0 or scratch.section_load_counts[elem_index] < num_int_pts:
        offset = len(scratch.section_load_axial_pool)
        scratch.section_load_offsets[elem_index] = offset
        scratch.section_load_counts[elem_index] = num_int_pts
        scratch.section_load_axial_pool.resize(offset + num_int_pts, 0.0)
        scratch.section_load_my_pool.resize(offset + num_int_pts, 0.0)
        scratch.section_load_mz_pool.resize(offset + num_int_pts, 0.0)

    var axial_ptr = scratch.section_load_axial_pool.unsafe_ptr()
    var my_ptr = scratch.section_load_my_pool.unsafe_ptr()
    var mz_ptr = scratch.section_load_mz_pool.unsafe_ptr()
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
        axial_ptr[offset + ip] = loads[0]
        my_ptr[offset + ip] = loads[1]
        mz_ptr[offset + ip] = loads[2]

    var fixed = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    var fixed_end = Beam3dVec10(fill=0.0)
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


@always_inline
fn _fiber_section3d_response_flexibility(
    resp: FiberSection3dResponse,
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


@always_inline
fn _accum_basic_flexibility_ip(
    mut f_basic: Beam3dMat5,
    wL: Float64,
    b_mi: Float64,
    b_mj: Float64,
    sec_f11: Float64,
    sec_f12: Float64,
    sec_f13: Float64,
    sec_f22: Float64,
    sec_f23: Float64,
    sec_f33: Float64,
):
    var wf11 = wL * sec_f11
    var wf12 = wL * sec_f12
    var wf13 = wL * sec_f13
    var wf22 = wL * sec_f22
    var wf23 = wL * sec_f23
    var wf33 = wL * sec_f33
    var b_mi2 = b_mi * b_mi
    var b_mimj = b_mi * b_mj
    var b_mj2 = b_mj * b_mj

    var wf13_mi = wf13 * b_mi
    var wf13_mj = wf13 * b_mj
    var wf12_mi = wf12 * b_mi
    var wf12_mj = wf12 * b_mj
    var wf23_mi2 = wf23 * b_mi2
    var wf23_mimj = wf23 * b_mimj
    var wf23_mj2 = wf23 * b_mj2
    var wf33_mimj = wf33 * b_mimj
    var wf22_mimj = wf22 * b_mimj

    f_basic[0] += wf11
    f_basic[1] += wf13_mi
    f_basic[2] += wf13_mj
    f_basic[3] += wf12_mi
    f_basic[4] += wf12_mj

    f_basic[5] += wf13_mi
    f_basic[6] += wf33 * b_mi2
    f_basic[7] += wf33_mimj
    f_basic[8] += wf23_mi2
    f_basic[9] += wf23_mimj

    f_basic[10] += wf13_mj
    f_basic[11] += wf33_mimj
    f_basic[12] += wf33 * b_mj2
    f_basic[13] += wf23_mimj
    f_basic[14] += wf23_mj2

    f_basic[15] += wf12_mi
    f_basic[16] += wf23_mi2
    f_basic[17] += wf23_mimj
    f_basic[18] += wf22 * b_mi2
    f_basic[19] += wf22_mimj

    f_basic[20] += wf12_mj
    f_basic[21] += wf23_mimj
    f_basic[22] += wf23_mj2
    f_basic[23] += wf22_mimj
    f_basic[24] += wf22 * b_mj2


fn _ensure_force_beam_column3d_section_history_capacity(
    mut scratch: ForceBeamColumn3dScratch, num_int_pts: Int
):
    if len(scratch.section_vs_eps0) >= num_int_pts:
        return
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
    scratch.section_init_fs11.resize(num_int_pts, 0.0)
    scratch.section_init_fs12.resize(num_int_pts, 0.0)
    scratch.section_init_fs13.resize(num_int_pts, 0.0)
    scratch.section_init_fs22.resize(num_int_pts, 0.0)
    scratch.section_init_fs23.resize(num_int_pts, 0.0)
    scratch.section_init_fs33.resize(num_int_pts, 0.0)


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
    mut sec_def: FiberSection3dDef,
    elem_inst_base: Int,
    num_int_pts: Int,
    has_elem_loads: Bool,
    section_load_offset: Int,
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
    var need_init_flex = use_initial_section_flexibility != 0

    var predictor_f = Beam3dMat5(fill=0.0)
    for ip in range(num_int_pts):
        var eps0 = force_basic_q_state[eps0_offset + ip]
        var kappa_y = force_basic_q_state[ky_offset + ip]
        var kappa_z = force_basic_q_state[kz_offset + ip]
        scratch.section_vs_eps0[ip] = eps0
        scratch.section_vs_ky[ip] = kappa_y
        scratch.section_vs_kz[ip] = kappa_z

        var resp_trial = fiber_section3d_set_trial_instance(
            sec_def, elem_inst_base + ip, eps0, kappa_y, kappa_z
        )
        scratch.section_ssr_axial[ip] = resp_trial.axial_force
        scratch.section_ssr_my[ip] = resp_trial.moment_y
        scratch.section_ssr_mz[ip] = resp_trial.moment_z
        var sec_flex = _fiber_section3d_response_flexibility(resp_trial)
        if not sec_flex[0]:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
        var sf11 = sec_flex[1]
        var sf12 = sec_flex[2]
        var sf13 = sec_flex[3]
        var sf22 = sec_flex[4]
        var sf23 = sec_flex[5]
        var sf33 = sec_flex[6]
        scratch.section_fs11[ip] = sf11
        scratch.section_fs12[ip] = sf12
        scratch.section_fs13[ip] = sf13
        scratch.section_fs22[ip] = sf22
        scratch.section_fs23[ip] = sf23
        scratch.section_fs33[ip] = sf33

        scratch.section_vs_subdivide_eps0[ip] = eps0
        scratch.section_vs_subdivide_ky[ip] = kappa_y
        scratch.section_vs_subdivide_kz[ip] = kappa_z
        scratch.section_ssr_subdivide_axial[ip] = resp_trial.axial_force
        scratch.section_ssr_subdivide_my[ip] = resp_trial.moment_y
        scratch.section_ssr_subdivide_mz[ip] = resp_trial.moment_z
        scratch.section_fs_subdivide11[ip] = sf11
        scratch.section_fs_subdivide12[ip] = sf12
        scratch.section_fs_subdivide13[ip] = sf13
        scratch.section_fs_subdivide22[ip] = sf22
        scratch.section_fs_subdivide23[ip] = sf23
        scratch.section_fs_subdivide33[ip] = sf33

        if need_init_flex:
            var init_resp = fiber_section3d_set_trial_instance(
                sec_def, elem_inst_base + ip, 0.0, 0.0, 0.0
            )
            var init_flex = _fiber_section3d_response_flexibility(init_resp)
            if not init_flex[0]:
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
            scratch.section_init_fs11[ip] = init_flex[1]
            scratch.section_init_fs12[ip] = init_flex[2]
            scratch.section_init_fs13[ip] = init_flex[3]
            scratch.section_init_fs22[ip] = init_flex[4]
            scratch.section_init_fs23[ip] = init_flex[5]
            scratch.section_init_fs33[ip] = init_flex[6]

        var xi = scratch.integration_cache.xis[ip]
        var wL = scratch.integration_cache.weights[ip] * L
        _accum_basic_flexibility_ip(
            predictor_f, wL, xi - 1.0, xi, sf11, sf12, sf13, sf22, sf23, sf33
        )

    var predictor_f_copy = predictor_f
    if _invert_5x5_flat_into(predictor_f_copy, scratch.k_basic_flat):
        q0 += (
            scratch.k_basic_flat[0] * dv_trial0
            + scratch.k_basic_flat[1] * dv_trial1
            + scratch.k_basic_flat[2] * dv_trial2
            + scratch.k_basic_flat[3] * dv_trial3
            + scratch.k_basic_flat[4] * dv_trial4
        )
        q1 += (
            scratch.k_basic_flat[5] * dv_trial0
            + scratch.k_basic_flat[6] * dv_trial1
            + scratch.k_basic_flat[7] * dv_trial2
            + scratch.k_basic_flat[8] * dv_trial3
            + scratch.k_basic_flat[9] * dv_trial4
        )
        q2 += (
            scratch.k_basic_flat[10] * dv_trial0
            + scratch.k_basic_flat[11] * dv_trial1
            + scratch.k_basic_flat[12] * dv_trial2
            + scratch.k_basic_flat[13] * dv_trial3
            + scratch.k_basic_flat[14] * dv_trial4
        )
        q3 += (
            scratch.k_basic_flat[15] * dv_trial0
            + scratch.k_basic_flat[16] * dv_trial1
            + scratch.k_basic_flat[17] * dv_trial2
            + scratch.k_basic_flat[18] * dv_trial3
            + scratch.k_basic_flat[19] * dv_trial4
        )
        q4 += (
            scratch.k_basic_flat[20] * dv_trial0
            + scratch.k_basic_flat[21] * dv_trial1
            + scratch.k_basic_flat[22] * dv_trial2
            + scratch.k_basic_flat[23] * dv_trial3
            + scratch.k_basic_flat[24] * dv_trial4
        )

    var num_elem_iters = max_elem_iters
    if use_initial_section_flexibility == 1:
        num_elem_iters = 10 * max_elem_iters

    for elem_iter in range(num_elem_iters):
        _zero_static_tuple(scratch.f_basic_flat)
        var v_from_0 = 0.0
        var v_from_1 = 0.0
        var v_from_2 = 0.0
        var v_from_3 = 0.0
        var v_from_4 = 0.0
        var use_initial_flex = use_initial_section_flexibility == 1 or (
            use_initial_section_flexibility == 2 and elem_iter == 0
        )
        for ip in range(num_int_pts):
            var xi = scratch.integration_cache.xis[ip]
            var wL = scratch.integration_cache.weights[ip] * L
            var b_mi = xi - 1.0
            var b_mj = xi
            var axial_target = q0
            var moment_z_target = b_mi * q1 + b_mj * q2
            var moment_y_target = b_mi * q3 + b_mj * q4
            if has_elem_loads:
                if section_load_offset >= 0:
                    axial_target += scratch.section_load_axial_pool[section_load_offset + ip]
                    moment_y_target += scratch.section_load_my_pool[section_load_offset + ip]
                    moment_z_target += scratch.section_load_mz_pool[section_load_offset + ip]
                else:
                    axial_target += scratch.section_load_axial[ip]
                    moment_y_target += scratch.section_load_my[ip]
                    moment_z_target += scratch.section_load_mz[ip]

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
                solve_f11 = scratch.section_init_fs11[ip]
                solve_f12 = scratch.section_init_fs12[ip]
                solve_f13 = scratch.section_init_fs13[ip]
                solve_f22 = scratch.section_init_fs22[ip]
                solve_f23 = scratch.section_init_fs23[ip]
                solve_f33 = scratch.section_init_fs33[ip]
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
            var resp_trial = fiber_section3d_set_trial_instance(
                sec_def, elem_inst_base + ip, sec_vs0, sec_vs1, sec_vs2
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
            var sf11 = sec_flex[1]
            var sf12 = sec_flex[2]
            var sf13 = sec_flex[3]
            var sf22 = sec_flex[4]
            var sf23 = sec_flex[5]
            var sf33 = sec_flex[6]
            scratch.section_fs_subdivide11[ip] = sf11
            scratch.section_fs_subdivide12[ip] = sf12
            scratch.section_fs_subdivide13[ip] = sf13
            scratch.section_fs_subdivide22[ip] = sf22
            scratch.section_fs_subdivide23[ip] = sf23
            scratch.section_fs_subdivide33[ip] = sf33

            var dss_res0 = axial_target - resp_trial.axial_force
            var dss_res1 = moment_y_target - resp_trial.moment_y
            var dss_res2 = moment_z_target - resp_trial.moment_z
            var dvs_res0 = sf11 * dss_res0 + sf12 * dss_res1 + sf13 * dss_res2
            var dvs_res1 = sf12 * dss_res0 + sf22 * dss_res1 + sf23 * dss_res2
            var dvs_res2 = sf13 * dss_res0 + sf23 * dss_res1 + sf33 * dss_res2

            var tot_vs0 = sec_vs0 + dvs_res0
            var tot_vs1 = sec_vs1 + dvs_res1
            var tot_vs2 = sec_vs2 + dvs_res2
            v_from_0 += wL * tot_vs0
            v_from_1 += wL * b_mi * tot_vs2
            v_from_2 += wL * b_mj * tot_vs2
            v_from_3 += wL * b_mi * tot_vs1
            v_from_4 += wL * b_mj * tot_vs1

            _accum_basic_flexibility_ip(
                scratch.f_basic_flat, wL, b_mi, b_mj, sf11, sf12, sf13, sf22, sf23, sf33
            )

        scratch.f_basic_copy_flat = scratch.f_basic_flat
        if not _invert_5x5_flat_into(scratch.f_basic_copy_flat, scratch.k_basic_flat):
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0)

        var residual_0 = target_v0 - v_from_0
        var residual_1 = target_v1 - v_from_1
        var residual_2 = target_v2 - v_from_2
        var residual_3 = target_v3 - v_from_3
        var residual_4 = target_v4 - v_from_4
        var dq0 = (
            scratch.k_basic_flat[0] * residual_0
            + scratch.k_basic_flat[1] * residual_1
            + scratch.k_basic_flat[2] * residual_2
            + scratch.k_basic_flat[3] * residual_3
            + scratch.k_basic_flat[4] * residual_4
        )
        var dq1 = (
            scratch.k_basic_flat[5] * residual_0
            + scratch.k_basic_flat[6] * residual_1
            + scratch.k_basic_flat[7] * residual_2
            + scratch.k_basic_flat[8] * residual_3
            + scratch.k_basic_flat[9] * residual_4
        )
        var dq2 = (
            scratch.k_basic_flat[10] * residual_0
            + scratch.k_basic_flat[11] * residual_1
            + scratch.k_basic_flat[12] * residual_2
            + scratch.k_basic_flat[13] * residual_3
            + scratch.k_basic_flat[14] * residual_4
        )
        var dq3 = (
            scratch.k_basic_flat[15] * residual_0
            + scratch.k_basic_flat[16] * residual_1
            + scratch.k_basic_flat[17] * residual_2
            + scratch.k_basic_flat[18] * residual_3
            + scratch.k_basic_flat[19] * residual_4
        )
        var dq4 = (
            scratch.k_basic_flat[20] * residual_0
            + scratch.k_basic_flat[21] * residual_1
            + scratch.k_basic_flat[22] * residual_2
            + scratch.k_basic_flat[23] * residual_3
            + scratch.k_basic_flat[24] * residual_4
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


@always_inline
fn _force_beam_column3d_basic_state(
    u_local: Beam3dVec12, inv_L: Float64
) -> (Float64, Float64, Float64, Float64, Float64, Float64):
    var chord_z = (u_local[7] - u_local[1]) * inv_L
    var chord_y = (u_local[8] - u_local[2]) * inv_L
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
    force_beam_column3d_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
    force_beam_column3d_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
        0.0,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
        False,
        0.0,
        0.0,
        0.0,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
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
        elem_index, x1, y1, z1, x2, y2, z2, has_vecxz, vecxz_x, vecxz_y, vecxz_z, scratch
    )
    var L: Float64
    var inv_L: Float64
    if elem_index >= 0 and elem_index < len(scratch.cached_length):
        L = scratch.cached_length[elem_index]
        inv_L = scratch.cached_inv_length[elem_index]
    else:
        var dx = x2 - x1
        var dy = y2 - y1
        var dz = z2 - z1
        L = sqrt(dx * dx + dy * dy + dz * dz)
        if L == 0.0:
            abort("zero-length element")
        inv_L = 1.0 / L

    if geom_transf == "Corotational":
        _element_rotation_into_static(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            geom_transf,
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            scratch.R,
        )
    elif elem_index >= 0 and elem_index < len(scratch.rotation_cache):
        scratch.R = scratch.rotation_cache[elem_index]
    else:
        _element_rotation_into_static(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            geom_transf,
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            scratch.R,
        )
    _beam3d_transform_u_global_to_local_static(scratch.R, u_elem_global, scratch.u_local)

    var fe0 = 0.0
    var fe1 = 0.0
    var fe2 = 0.0
    var fe3 = 0.0
    var fe4 = 0.0
    var fe5 = 0.0
    var fe6 = 0.0
    var fe7 = 0.0
    var fe8 = 0.0
    var fe9 = 0.0
    var has_elem_loads = (
        elem_index >= 0
        and load_scale != 0.0
        and elem_index + 1 < len(elem_load_offsets)
        and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
    )
    if has_elem_loads:
        if elem_index < len(scratch.load_valid) and scratch.load_valid[elem_index]:
            ref fe = scratch.fixed_end_cache[elem_index]
            fe0 = fe[0]
            fe1 = fe[1]
            fe2 = fe[2]
            fe3 = fe[3]
            fe4 = fe[4]
            fe5 = fe[5]
            fe6 = fe[6]
            fe7 = fe[7]
            fe8 = fe[8]
            fe9 = fe[9]
        else:
            var fixed = beam3d_basic_fixed_end_and_reactions(
                element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
            )
            fe0 = fixed[0]
            fe1 = fixed[1]
            fe2 = fixed[2]
            fe3 = fixed[3]
            fe4 = fixed[4]
            fe5 = fixed[5]
            fe6 = fixed[6]
            fe7 = fixed[7]
            fe8 = fixed[8]
            fe9 = fixed[9]
            if elem_index < len(scratch.fixed_end_cache):
                var fe_cache = Beam3dVec10(fill=0.0)
                fe_cache[0] = fe0
                fe_cache[1] = fe1
                fe_cache[2] = fe2
                fe_cache[3] = fe3
                fe_cache[4] = fe4
                fe_cache[5] = fe5
                fe_cache[6] = fe6
                fe_cache[7] = fe7
                fe_cache[8] = fe8
                fe_cache[9] = fe9
                scratch.fixed_end_cache[elem_index] = fe_cache
                scratch.load_has_nonzero[elem_index] = True
                scratch.load_valid[elem_index] = True

    var v_basic_0: Float64
    var v_basic_1: Float64
    var v_basic_2: Float64
    var v_basic_3: Float64
    var v_basic_4: Float64
    var torsion_basic: Float64
    (
        v_basic_0,
        v_basic_1,
        v_basic_2,
        v_basic_3,
        v_basic_4,
        torsion_basic,
    ) = _force_beam_column3d_basic_state(scratch.u_local, inv_L)
    var axial_k = E * A * inv_L
    var flex_z = E * Iz * inv_L
    var flex_y = E * Iy * inv_L
    var torsion_k = 0.0
    if G > 0.0 and J > 0.0:
        torsion_k = G * J * inv_L

    var q_basic_0 = axial_k * v_basic_0 + fe0
    var q_basic_1 = flex_z * (4.0 * v_basic_1 + 2.0 * v_basic_2) + fe1
    var q_basic_2 = flex_z * (2.0 * v_basic_1 + 4.0 * v_basic_2) + fe2
    var q_basic_3 = flex_y * (4.0 * v_basic_3 + 2.0 * v_basic_4) + fe3
    var q_basic_4 = flex_y * (2.0 * v_basic_3 + 4.0 * v_basic_4) + fe4
    var q_basic_5 = torsion_k * torsion_basic

    _zero_static_tuple(scratch.k_basic_flat)
    scratch.k_basic_flat[0] = axial_k
    scratch.k_basic_flat[6] = 4.0 * flex_z
    scratch.k_basic_flat[7] = 2.0 * flex_z
    scratch.k_basic_flat[11] = 2.0 * flex_z
    scratch.k_basic_flat[12] = 4.0 * flex_z
    scratch.k_basic_flat[18] = 4.0 * flex_y
    scratch.k_basic_flat[19] = 2.0 * flex_y
    scratch.k_basic_flat[23] = 2.0 * flex_y
    scratch.k_basic_flat[24] = 4.0 * flex_y
    _beam3d_assemble_local_stiffness_from_basic(
        scratch.k_basic_flat, inv_L, torsion_k, scratch.k_local_flat
    )

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_stiffness_flat(scratch.k_local_flat, q_basic_0, inv_L)

    _beam3d_assemble_local_force_from_basic(
        q_basic_0, q_basic_1, q_basic_2, q_basic_3, q_basic_4, q_basic_5, inv_L, scratch.f_local
    )
    if has_elem_loads:
        scratch.f_local[0] += fe5
        scratch.f_local[1] += fe6
        scratch.f_local[2] += fe8
        scratch.f_local[7] += fe7
        scratch.f_local[8] += fe9
    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_force(scratch.f_local, scratch.u_local, q_basic_0, inv_L)

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
    mut sec_def: FiberSection3dDef,
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
    force_beam_column3d_fiber_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    mut sec_def: FiberSection3dDef,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
    mut sec_def: FiberSection3dDef,
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
    force_beam_column3d_fiber_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    mut sec_def: FiberSection3dDef,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
    mut sec_def: FiberSection3dDef,
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
        False,
        0.0,
        0.0,
        0.0,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut sec_def: FiberSection3dDef,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut sec_def: FiberSection3dDef,
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

    var local_state_offset = elem_state_offset - sec_def.runtime_state_offset_base
    if (
        local_state_offset < 0
        or local_state_offset + elem_state_count > sec_def.runtime_state_count
    ):
        abort("forceBeamColumn3d section states out of range")
    var elem_inst_base = local_state_offset // fibers_per_section

    var predictor_state_count = 5 + 3 * num_int_pts
    if force_basic_q_count < predictor_state_count:
        abort("forceBeamColumn3d basic force state count mismatch")
    if force_basic_q_offset < 0 or force_basic_q_offset + predictor_state_count > len(
        force_basic_q_state
    ):
        abort("forceBeamColumn3d basic force state out of range")

    _ensure_force_beam_column3d_geometry_cache(
        elem_index, x1, y1, z1, x2, y2, z2, has_vecxz, vecxz_x, vecxz_y, vecxz_z, scratch
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
    var L: Float64
    var inv_L: Float64
    if elem_index >= 0 and elem_index < len(scratch.cached_length):
        L = scratch.cached_length[elem_index]
        inv_L = scratch.cached_inv_length[elem_index]
    else:
        var dx = x2 - x1
        var dy = y2 - y1
        var dz = z2 - z1
        L = sqrt(dx * dx + dy * dy + dz * dz)
        if L == 0.0:
            abort("zero-length element")
        inv_L = 1.0 / L

    if geom_transf == "Corotational":
        _element_rotation_into_static(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            geom_transf,
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            scratch.R,
        )
    elif elem_index >= 0 and elem_index < len(scratch.rotation_cache):
        scratch.R = scratch.rotation_cache[elem_index]
    else:
        _element_rotation_into_static(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            geom_transf,
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            scratch.R,
        )
    _beam3d_transform_u_global_to_local_static(scratch.R, u_elem_global, scratch.u_local)

    beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
    var has_elem_loads: Bool
    var section_load_offset = -1
    var fe5 = 0.0
    var fe6 = 0.0
    var fe7 = 0.0
    var fe8 = 0.0
    var fe9 = 0.0
    if elem_index >= 0 and elem_index < len(scratch.load_has_nonzero):
        has_elem_loads = scratch.load_has_nonzero[elem_index]
        if has_elem_loads:
            section_load_offset = scratch.section_load_offsets[elem_index]
            ref fe = scratch.fixed_end_cache[elem_index]
            fe5 = fe[5]
            fe6 = fe[6]
            fe7 = fe[7]
            fe8 = fe[8]
            fe9 = fe[9]
    else:
        has_elem_loads = (
            load_scale != 0.0
            and elem_index >= 0
            and elem_index + 1 < len(elem_load_offsets)
            and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
        )
        if has_elem_loads:
            scratch.section_load_axial.resize(num_int_pts, 0.0)
            scratch.section_load_my.resize(num_int_pts, 0.0)
            scratch.section_load_mz.resize(num_int_pts, 0.0)
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
                scratch.section_load_axial[ip] = loads[0]
                scratch.section_load_my[ip] = loads[1]
                scratch.section_load_mz[ip] = loads[2]
            var fixed = beam3d_basic_fixed_end_and_reactions(
                element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
            )
            fe5 = fixed[5]
            fe6 = fixed[6]
            fe7 = fixed[7]
            fe8 = fixed[8]
            fe9 = fixed[9]

    var v_basic_0: Float64
    var v_basic_1: Float64
    var v_basic_2: Float64
    var v_basic_3: Float64
    var v_basic_4: Float64
    var torsion_basic: Float64
    (
        v_basic_0,
        v_basic_1,
        v_basic_2,
        v_basic_3,
        v_basic_4,
        torsion_basic,
    ) = _force_beam_column3d_basic_state(scratch.u_local, inv_L)

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
                elem_inst_base,
                num_int_pts,
                has_elem_loads,
                section_load_offset,
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

    var torsion_k = 0.0
    if G > 0.0 and J > 0.0:
        torsion_k = G * J * inv_L
    _beam3d_assemble_local_stiffness_from_basic(
        scratch.k_basic_flat, inv_L, torsion_k, scratch.k_local_flat
    )

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_stiffness_flat(scratch.k_local_flat, q0, inv_L)

    var q_basic_5 = torsion_k * torsion_basic
    _beam3d_assemble_local_force_from_basic(q0, q1, q2, q3, q4, q_basic_5, inv_L, scratch.f_local)
    if has_elem_loads:
        scratch.f_local[0] += fe5
        scratch.f_local[1] += fe6
        scratch.f_local[2] += fe8
        scratch.f_local[7] += fe7
        scratch.f_local[8] += fe9
    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_force(scratch.f_local, scratch.u_local, q0, inv_L)

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
    mut sec_def: FiberSection3dDef,
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
    return force_beam_column3d_fiber_section_response(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
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
        section_no,
        want_deformation,
    )


fn force_beam_column3d_fiber_section_response(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    mut sec_def: FiberSection3dDef,
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
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
    var R = Beam3dMat3(fill=0.0)
    var u_local = Beam3dVec12(fill=0.0)
    _element_rotation_into_static(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
        R,
    )
    _beam3d_transform_u_global_to_local_static(R, u_elem_global, u_local)
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
