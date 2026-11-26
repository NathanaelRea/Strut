from algorithm import vectorize
from collections import List
from sys import align_of, size_of

from materials import UniMaterialDef
from materials.uniaxial.concrete01 import _concrete01_reload
from materials.uniaxial.core import _abs, _sign
from simd_math import simd_pow_nonneg
from solver.simd_contiguous import (
    FLOAT64_SIMD_WIDTH,
    copy_float64_contiguous_simd,
    load_float64_contiguous_simd,
    store_float64_contiguous_simd,
)


@always_inline
fn _fiber_padded_family_count(count: Int) -> Int:
    if count <= 0:
        return 0
    return (
        (count + FLOAT64_SIMD_WIDTH - 1) // FLOAT64_SIMD_WIDTH
    ) * FLOAT64_SIMD_WIDTH


@always_inline
fn _fiber_copy_float64_range_simd[width: Int](
    mut dst: List[Float64],
    dst_start: Int,
    src: List[Float64],
    src_start: Int,
    count: Int,
):
    if count <= 0:
        return
    if dst_start == 0 and src_start == 0:
        copy_float64_contiguous_simd[width](dst, src, count)
        return

    @parameter
    fn copy_chunk[chunk: Int](i: Int):
        var value_vec = load_float64_contiguous_simd[chunk](src, src_start + i)
        store_float64_contiguous_simd[chunk](dst, dst_start + i, value_vec)

    vectorize[copy_chunk, width](count)


@always_inline
fn _fiber_copy_int_range(
    mut dst: List[Int], dst_start: Int, src: List[Int], src_start: Int, count: Int
):
    for i in range(count):
        dst[dst_start + i] = src[src_start + i]


@always_inline
fn load_int32_ptr_contiguous_simd[width: Int](
    ptr: UnsafePointer[Int32], start: Int
) -> SIMD[DType.int32, width]:
    alias alignment = align_of[SIMD[DType.int32, width]]()
    var byte_offset = start * size_of[Int32]()
    if (Int(ptr) + byte_offset) % alignment == 0:
        return ptr.load[width=width, alignment=alignment](start)
    return ptr.load[width=width, alignment=1](start)


@always_inline
fn store_int32_ptr_contiguous_simd[width: Int](
    ptr: UnsafePointer[Int32], start: Int, vec: SIMD[DType.int32, width]
):
    alias alignment = align_of[SIMD[DType.int32, width]]()
    var byte_offset = start * size_of[Int32]()
    if (Int(ptr) + byte_offset) % alignment == 0:
        ptr.store[width=width, alignment=alignment](start, vec)
        return
    ptr.store[width=width, alignment=1](start, vec)


@always_inline
fn _fiber_copy_int32_range_simd[width: Int](
    mut dst: List[Int32],
    dst_start: Int,
    src: List[Int32],
    src_start: Int,
    count: Int,
):
    if count <= 0:
        return
    var dst_ptr = dst.unsafe_ptr()
    var src_ptr = src.unsafe_ptr()
    var dst_offset = dst_start
    var src_offset = src_start

    @parameter
    fn copy_chunk[chunk: Int](i: Int):
        var value_vec = load_int32_ptr_contiguous_simd[chunk](
            src_ptr, src_offset + i
        )
        store_int32_ptr_contiguous_simd[chunk](dst_ptr, dst_offset + i, value_vec)

    vectorize[copy_chunk, width](count)


@always_inline
fn _simd_abs_float64[width: Int](
    x: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    return x.lt(0.0).select(-x, x)


@always_inline
fn _concrete02_compr_envlp_simd[width: Int](
    fc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    fcu: SIMD[DType.float64, width],
    epscu: SIMD[DType.float64, width],
    strain: SIMD[DType.float64, width],
) -> (SIMD[DType.float64, width], SIMD[DType.float64, width]):
    var ec0 = (SIMD[DType.float64, width](2.0) * fc) / epsc0
    return _concrete02_compr_envlp_precomputed_ec0_simd[width](
        fc, epsc0, fcu, epscu, ec0, strain
    )


@always_inline
fn _concrete02_compr_envlp_precomputed_ec0_simd[width: Int](
    fc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    fcu: SIMD[DType.float64, width],
    epscu: SIMD[DType.float64, width],
    ec0: SIMD[DType.float64, width],
    strain: SIMD[DType.float64, width],
) -> (SIMD[DType.float64, width], SIMD[DType.float64, width]):
    var rat = strain / epsc0
    var sig = fc * rat * (SIMD[DType.float64, width](2.0) - rat)
    var tangent = ec0 * (SIMD[DType.float64, width](1.0) - rat)

    var softening_sig = (fcu - fc) * (strain - epsc0) / (epscu - epsc0) + fc
    var softening_tangent = (fcu - fc) / (epscu - epsc0)
    var peak_mask = strain.ge(epsc0)
    sig = peak_mask.select(sig, softening_sig)
    tangent = peak_mask.select(tangent, softening_tangent)

    var crush_mask = strain.le(epscu)
    sig = crush_mask.select(fcu, sig)
    tangent = crush_mask.select(SIMD[DType.float64, width](1.0e-10), tangent)
    return (sig, tangent)


@always_inline
fn _concrete02_tens_envlp_simd[width: Int](
    fc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    ft: SIMD[DType.float64, width],
    ets: SIMD[DType.float64, width],
    strain: SIMD[DType.float64, width],
) -> (SIMD[DType.float64, width], SIMD[DType.float64, width]):
    var ec0 = (SIMD[DType.float64, width](2.0) * fc) / epsc0
    var eps0 = ft / ec0
    var epsu = ft * (
        SIMD[DType.float64, width](1.0) / ets + SIMD[DType.float64, width](1.0) / ec0
    )
    return _concrete02_tens_envlp_precomputed_simd[width](
        ft, ets, ec0, eps0, epsu, strain
    )


@always_inline
fn _concrete02_tens_envlp_precomputed_simd[width: Int](
    ft: SIMD[DType.float64, width],
    ets: SIMD[DType.float64, width],
    ec0: SIMD[DType.float64, width],
    eps0: SIMD[DType.float64, width],
    epsu: SIMD[DType.float64, width],
    strain: SIMD[DType.float64, width],
) -> (SIMD[DType.float64, width], SIMD[DType.float64, width]):
    var sig = strain * ec0
    var tangent = ec0

    var softening_mask = strain.gt(eps0)
    var softening_sig = ft - ets * (strain - eps0)
    sig = softening_mask.select(softening_sig, sig)
    tangent = softening_mask.select(-ets, tangent)

    var zero_mask = strain.gt(epsu)
    sig = zero_mask.select(SIMD[DType.float64, width](0.0), sig)
    tangent = zero_mask.select(SIMD[DType.float64, width](1.0e-10), tangent)
    return (sig, tangent)


@always_inline
fn _concrete02_eval_simd_step[width: Int](
    strain: SIMD[DType.float64, width],
    eps_c: SIMD[DType.float64, width],
    sig_c: SIMD[DType.float64, width],
    tangent_c: SIMD[DType.float64, width],
    ecmin_c: SIMD[DType.float64, width],
    dept_c: SIMD[DType.float64, width],
    fc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    fcu: SIMD[DType.float64, width],
    epscu: SIMD[DType.float64, width],
    ft: SIMD[DType.float64, width],
    ets: SIMD[DType.float64, width],
    ec0: SIMD[DType.float64, width],
    eps0_tension: SIMD[DType.float64, width],
    epsu_tension: SIMD[DType.float64, width],
    epsr: SIMD[DType.float64, width],
    sigmr: SIMD[DType.float64, width],
) -> (
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
):
    var deps = strain - eps_c
    var tol = SIMD[DType.float64, width](2.220446049250313e-16)
    var sig_t = sig_c
    var tangent_t = tangent_c
    var ecmin_t = ecmin_c
    var dept_t = dept_c

    var done_mask = _simd_abs_float64[width](deps).lt(tol)
    if all(done_mask):
        return (sig_t, tangent_t, ecmin_t, dept_t)

    var compression_mask = strain.lt(ecmin_t) & ~done_mask
    if not all(~compression_mask):
        var compr_env = _concrete02_compr_envlp_precomputed_ec0_simd[width](
            fc, epsc0, fcu, epscu, ec0, strain
        )
        sig_t = compression_mask.select(compr_env[0], sig_t)
        tangent_t = compression_mask.select(compr_env[1], tangent_t)
        ecmin_t = compression_mask.select(strain, ecmin_t)
    done_mask = done_mask | compression_mask
    if all(done_mask):
        return (sig_t, tangent_t, ecmin_t, dept_t)

    var remaining_mask = ~done_mask
    var half = SIMD[DType.float64, width](0.5)
    var sigmm_env = _concrete02_compr_envlp_precomputed_ec0_simd[width](
        fc, epsc0, fcu, epscu, ec0, ecmin_t
    )
    var sigmm = sigmm_env[0]
    var er = (sigmm - sigmr) / (ecmin_t - epsr)
    var ept = ecmin_t - sigmm / er

    var reload_mask = remaining_mask & strain.le(ept)
    if not all(~reload_mask):
        var sig_trial = sig_c + ec0 * deps
        var tangent_trial = ec0
        var sigmin = sigmm + er * (strain - ecmin_t)
        var sigmax = half * er * (strain - ept)
        var reload_low_mask = reload_mask & sig_trial.le(sigmin)
        var reload_high_mask = reload_mask & sig_trial.ge(sigmax)
        sig_trial = reload_low_mask.select(sigmin, sig_trial)
        tangent_trial = reload_low_mask.select(er, tangent_trial)
        sig_trial = reload_high_mask.select(sigmax, sig_trial)
        tangent_trial = reload_high_mask.select(half * er, tangent_trial)
        sig_t = reload_mask.select(sig_trial, sig_t)
        tangent_t = reload_mask.select(tangent_trial, tangent_t)
    done_mask = done_mask | reload_mask
    if all(done_mask):
        return (sig_t, tangent_t, ecmin_t, dept_t)

    var tension_mask = ~done_mask
    var epn = ept + dept_t
    var pre_tension_mask = tension_mask & strain.le(epn)
    if not all(~pre_tension_mask):
        var sicn_env = _concrete02_tens_envlp_precomputed_simd[width](
            ft, ets, ec0, eps0_tension, epsu_tension, dept_t
        )
        var dept_nonzero_mask = dept_t.ne(0.0)
        var e = dept_nonzero_mask.select(sicn_env[0] / dept_t, ec0)
        var sig_pre_tension = e * (strain - ept)
        sig_t = pre_tension_mask.select(sig_pre_tension, sig_t)
        tangent_t = pre_tension_mask.select(e, tangent_t)
    var tension_env_mask = tension_mask & ~pre_tension_mask
    if not all(~tension_env_mask):
        var tens_env = _concrete02_tens_envlp_precomputed_simd[width](
            ft, ets, ec0, eps0_tension, epsu_tension, strain - ept
        )
        sig_t = tension_env_mask.select(tens_env[0], sig_t)
        tangent_t = tension_env_mask.select(tens_env[1], tangent_t)
        dept_t = tension_env_mask.select(strain - ept, dept_t)
    return (sig_t, tangent_t, ecmin_t, dept_t)


@always_inline
fn _steel02_eval_simd_step[width: Int](
    trial_eps: SIMD[DType.float64, width],
    eps_c: SIMD[DType.float64, width],
    sig_c: SIMD[DType.float64, width],
    tangent_c: SIMD[DType.float64, width],
    epsmin_c: SIMD[DType.float64, width],
    epsmax_c: SIMD[DType.float64, width],
    epspl_c: SIMD[DType.float64, width],
    epss0_c: SIMD[DType.float64, width],
    sigs0_c: SIMD[DType.float64, width],
    epsr_c: SIMD[DType.float64, width],
    sigr_c: SIMD[DType.float64, width],
    kon_c: SIMD[DType.int32, width],
    fy: SIMD[DType.float64, width],
    e0: SIMD[DType.float64, width],
    b: SIMD[DType.float64, width],
    r0: SIMD[DType.float64, width],
    cr1: SIMD[DType.float64, width],
    cr2: SIMD[DType.float64, width],
    a1: SIMD[DType.float64, width],
    a3: SIMD[DType.float64, width],
    sigini: SIMD[DType.float64, width],
    esh: SIMD[DType.float64, width],
    epsy: SIMD[DType.float64, width],
    sigini_over_e0: SIMD[DType.float64, width],
    pos_inv_denom: SIMD[DType.float64, width],
    neg_inv_denom: SIMD[DType.float64, width],
) -> (
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.int32, width],
):
    var eps = sigini.ne(0.0).select(trial_eps + sigini_over_e0, trial_eps)
    var deps = eps - eps_c

    var epsmin_t = epsmin_c
    var epsmax_t = epsmax_c
    var epspl_t = epspl_c
    var epss0_t = epss0_c
    var sigs0_t = sigs0_c
    var epsr_t = epsr_c
    var sigr_t = sigr_c
    var kon_t = kon_c
    var sig_t = sig_c
    var tangent_t = tangent_c

    var tol = SIMD[DType.float64, width](2.220446049250313e-15)
    var no_change_mask = (kon_c.eq(0) | kon_c.eq(3)) & _simd_abs_float64[width](
        deps
    ).lt(tol)
    sig_t = no_change_mask.select(sigini, sig_t)
    tangent_t = no_change_mask.select(e0, tangent_t)
    kon_t = no_change_mask.select(SIMD[DType.int32, width](3), kon_t)
    if all(no_change_mask):
        return (
            eps,
            sig_t,
            tangent_t,
            epsmin_t,
            epsmax_t,
            epspl_t,
            epss0_t,
            sigs0_t,
            epsr_t,
            sigr_t,
            kon_t,
        )

    var active_mask = ~no_change_mask
    var initial_mask = active_mask & (kon_c.eq(0) | kon_c.eq(3))
    epsmax_t = initial_mask.select(epsy, epsmax_t)
    epsmin_t = initial_mask.select(-epsy, epsmin_t)

    var initial_neg_mask = initial_mask & deps.lt(0.0)
    var initial_pos_mask = initial_mask & ~deps.lt(0.0)
    kon_t = initial_neg_mask.select(SIMD[DType.int32, width](2), kon_t)
    epss0_t = initial_neg_mask.select(epsmin_t, epss0_t)
    sigs0_t = initial_neg_mask.select(-fy, sigs0_t)
    epspl_t = initial_neg_mask.select(epsmin_t, epspl_t)

    kon_t = initial_pos_mask.select(SIMD[DType.int32, width](1), kon_t)
    epss0_t = initial_pos_mask.select(epsmax_t, epss0_t)
    sigs0_t = initial_pos_mask.select(fy, sigs0_t)
    epspl_t = initial_pos_mask.select(epsmax_t, epspl_t)

    var pos_reversal_mask = active_mask & kon_t.eq(2) & deps.gt(0.0)
    if not all(~pos_reversal_mask):
        var epsmin_after_pos = eps_c.lt(epsmin_t).select(eps_c, epsmin_t)
        var d1_pos = (epsmax_t - epsmin_after_pos) * pos_inv_denom
        var shft_pos = SIMD[DType.float64, width](1.0) + a3 * (
            simd_pow_nonneg[width](d1_pos, SIMD[DType.float64, width](0.8))
        )
        var epss0_pos = (
            fy * shft_pos - esh * epsy * shft_pos - sig_c + e0 * eps_c
        ) / (e0 - esh)
        var sigs0_pos = fy * shft_pos + esh * (epss0_pos - epsy * shft_pos)
        kon_t = pos_reversal_mask.select(SIMD[DType.int32, width](1), kon_t)
        epsr_t = pos_reversal_mask.select(eps_c, epsr_t)
        sigr_t = pos_reversal_mask.select(sig_c, sigr_t)
        epsmin_t = pos_reversal_mask.select(epsmin_after_pos, epsmin_t)
        epss0_t = pos_reversal_mask.select(epss0_pos, epss0_t)
        sigs0_t = pos_reversal_mask.select(sigs0_pos, sigs0_t)
        epspl_t = pos_reversal_mask.select(epsmax_t, epspl_t)

    var neg_reversal_mask = active_mask & kon_t.eq(1) & deps.lt(0.0)
    if not all(~neg_reversal_mask):
        var epsmax_after_neg = eps_c.gt(epsmax_t).select(eps_c, epsmax_t)
        var d1_neg = (epsmax_after_neg - epsmin_t) * neg_inv_denom
        var shft_neg = SIMD[DType.float64, width](1.0) + a1 * (
            simd_pow_nonneg[width](d1_neg, SIMD[DType.float64, width](0.8))
        )
        var epss0_neg = (
            -fy * shft_neg + esh * epsy * shft_neg - sig_c + e0 * eps_c
        ) / (e0 - esh)
        var sigs0_neg = -fy * shft_neg + esh * (epss0_neg + epsy * shft_neg)
        kon_t = neg_reversal_mask.select(SIMD[DType.int32, width](2), kon_t)
        epsr_t = neg_reversal_mask.select(eps_c, epsr_t)
        sigr_t = neg_reversal_mask.select(sig_c, sigr_t)
        epsmax_t = neg_reversal_mask.select(epsmax_after_neg, epsmax_t)
        epss0_t = neg_reversal_mask.select(epss0_neg, epss0_t)
        sigs0_t = neg_reversal_mask.select(sigs0_neg, sigs0_t)
        epspl_t = neg_reversal_mask.select(epsmin_t, epspl_t)

    var xi = _simd_abs_float64[width]((epspl_t - epss0_t) / epsy)
    var r = r0 * (SIMD[DType.float64, width](1.0) - (cr1 * xi) / (cr2 + xi))
    var epsrat = (eps - epsr_t) / (epss0_t - epsr_t)
    var dum1 = SIMD[DType.float64, width](1.0) + (
        simd_pow_nonneg[width](_simd_abs_float64[width](epsrat), r)
    )
    var dum2 = simd_pow_nonneg[width](
        dum1, SIMD[DType.float64, width](1.0) / r
    )

    var sig_active = (
        b * epsrat + (SIMD[DType.float64, width](1.0) - b) * epsrat / dum2
    )
    sig_active = sig_active * (sigs0_t - sigr_t) + sigr_t
    var tangent_active = b + (SIMD[DType.float64, width](1.0) - b) / (dum1 * dum2)
    tangent_active = tangent_active * (sigs0_t - sigr_t) / (epss0_t - epsr_t)
    sig_t = active_mask.select(sig_active, sig_t)
    tangent_t = active_mask.select(tangent_active, tangent_t)
    return (
        eps,
        sig_t,
        tangent_t,
        epsmin_t,
        epsmax_t,
        epspl_t,
        epss0_t,
        sigs0_t,
        epsr_t,
        sigr_t,
        kon_t,
    )


@always_inline
fn _steel01_eval_scalar(
    mat_def: UniMaterialDef,
    eps_p_c: Float64,
    alpha_c: Float64,
    eps: Float64,
) -> (Float64, Float64, Float64, Float64):
    var Fy = mat_def.p0
    var E0 = mat_def.p1
    var b = mat_def.p2
    var H = (b * E0) / (1.0 - b)
    var sigma_trial = E0 * (eps - eps_p_c)
    var xi = sigma_trial - alpha_c
    var f = _abs(xi) - Fy
    if f <= 0.0:
        return (sigma_trial, E0, eps_p_c, alpha_c)
    var dg = f / (E0 + H)
    var sgn = _sign(xi)
    return (
        sigma_trial - E0 * dg * sgn,
        (E0 * H) / (E0 + H),
        eps_p_c + dg * sgn,
        alpha_c + H * dg * sgn,
    )


@always_inline
fn _concrete01_eval_scalar(
    mat_def: UniMaterialDef,
    eps_c: Float64,
    sig_c: Float64,
    tangent_c: Float64,
    min_strain_c: Float64,
    end_strain_c: Float64,
    unload_slope_c: Float64,
    eps: Float64,
) -> (Float64, Float64, Float64, Float64, Float64):
    var fpc = mat_def.p0
    var epsc0 = mat_def.p1
    var fpcu = mat_def.p2
    var epscu = mat_def.p3
    var d_strain = eps - eps_c
    if _abs(d_strain) < 1.0e-14:
        return (sig_c, tangent_c, min_strain_c, end_strain_c, unload_slope_c)
    var min_strain_t = min_strain_c
    var end_strain_t = end_strain_c
    var unload_slope_t = unload_slope_c
    if eps > 0.0:
        return (0.0, 0.0, min_strain_t, end_strain_t, unload_slope_t)
    var temp_stress = sig_c + unload_slope_t * eps - unload_slope_t * eps_c
    if eps <= eps_c:
        var reload = _concrete01_reload(
            fpc,
            epsc0,
            fpcu,
            epscu,
            eps,
            min_strain_t,
            end_strain_t,
            unload_slope_t,
        )
        var sig_t = reload.stress
        var tangent_t = reload.tangent
        if temp_stress > sig_t:
            sig_t = temp_stress
            tangent_t = unload_slope_t
        return (sig_t, tangent_t, min_strain_t, end_strain_t, unload_slope_t)
    if temp_stress <= 0.0:
        return (temp_stress, unload_slope_t, min_strain_t, end_strain_t, unload_slope_t)
    return (0.0, 0.0, min_strain_t, end_strain_t, unload_slope_t)


@always_inline
fn _concrete01_eval_simd_step[width: Int](
    eps: SIMD[DType.float64, width],
    eps_c: SIMD[DType.float64, width],
    sig_c: SIMD[DType.float64, width],
    tangent_c: SIMD[DType.float64, width],
    min_strain_c: SIMD[DType.float64, width],
    end_strain_c: SIMD[DType.float64, width],
    unload_slope_c: SIMD[DType.float64, width],
    fpc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    fpcu: SIMD[DType.float64, width],
    epscu: SIMD[DType.float64, width],
    ec0: SIMD[DType.float64, width],
    softening_tangent: SIMD[DType.float64, width],
) -> (
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
):
    var zero = SIMD[DType.float64, width](0.0)
    var d_strain = eps - eps_c
    var sig_t = sig_c
    var tangent_t = tangent_c
    var min_strain_t = min_strain_c
    var end_strain_t = end_strain_c
    var unload_slope_t = unload_slope_c

    var done_mask = _simd_abs_float64[width](d_strain).lt(1.0e-14)
    if all(done_mask):
        return (sig_t, tangent_t, min_strain_t, end_strain_t, unload_slope_t)

    var tens_mask = (~done_mask) & eps.gt(0.0)
    sig_t = tens_mask.select(zero, sig_t)
    tangent_t = tens_mask.select(zero, tangent_t)
    done_mask = done_mask | tens_mask
    if all(done_mask):
        return (sig_t, tangent_t, min_strain_t, end_strain_t, unload_slope_t)

    var rem_mask = ~done_mask
    var temp_stress = sig_c + unload_slope_c * eps - unload_slope_c * eps_c
    var load_mask = rem_mask & eps.le(eps_c)
    if not all(~load_mask):
        var new_min_mask = load_mask & eps.le(min_strain_c)
        var reload_sig = zero
        var reload_tan = zero
        if not all(~new_min_mask):
            var eta_env = eps / epsc0
            var par_sig = fpc * (
                SIMD[DType.float64, width](2.0) * eta_env - eta_env * eta_env
            )
            var par_tan = ec0 * (SIMD[DType.float64, width](1.0) - eta_env)
            var lin_sig = fpc + softening_tangent * (eps - epsc0)
            var above_c0 = eps.gt(epsc0)
            var above_cu = eps.gt(epscu)
            var env_sig = above_c0.select(par_sig, above_cu.select(lin_sig, fpcu))
            var env_tan = above_c0.select(
                par_tan, above_cu.select(softening_tangent, zero)
            )

            var temp_strain = eps.lt(epscu).select(epscu, eps)
            var eta = temp_strain / epsc0
            var ratio_lin = SIMD[DType.float64, width](0.707) * (
                eta - SIMD[DType.float64, width](2.0)
            ) + SIMD[DType.float64, width](0.834)
            var ratio_quad = (
                SIMD[DType.float64, width](0.145) * eta * eta
                + SIMD[DType.float64, width](0.13) * eta
            )
            var ratio = eta.lt(2.0).select(ratio_quad, ratio_lin)
            var end_s = ratio * epsc0
            var temp1 = eps - end_s
            var temp2 = env_sig / ec0
            var t1_gt_tol = temp1.gt(-1.0e-14)
            var t1_le_t2 = temp1.le(temp2)
            var safe_temp1 = (t1_le_t2 & ~t1_gt_tol).select(
                temp1, SIMD[DType.float64, width](-1.0)
            )
            var new_end_s = t1_gt_tol.select(
                end_s,
                t1_le_t2.select(eps - temp1, eps - temp2),
            )
            var new_slope = t1_gt_tol.select(
                ec0,
                t1_le_t2.select(env_sig / safe_temp1, ec0),
            )
            min_strain_t = new_min_mask.select(eps, min_strain_t)
            end_strain_t = new_min_mask.select(new_end_s, end_strain_t)
            unload_slope_t = new_min_mask.select(new_slope, unload_slope_t)
            reload_sig = new_min_mask.select(env_sig, reload_sig)
            reload_tan = new_min_mask.select(env_tan, reload_tan)

        var mid_mask = load_mask & (~new_min_mask) & eps.le(end_strain_c)
        reload_sig = mid_mask.select(
            unload_slope_c * (eps - end_strain_c), reload_sig
        )
        reload_tan = mid_mask.select(unload_slope_c, reload_tan)

        var use_temp = temp_stress.gt(reload_sig)
        var final_load_sig = use_temp.select(temp_stress, reload_sig)
        var final_load_tan = use_temp.select(unload_slope_t, reload_tan)
        sig_t = load_mask.select(final_load_sig, sig_t)
        tangent_t = load_mask.select(final_load_tan, tangent_t)

    var unload_mask = rem_mask & (~load_mask)
    if not all(~unload_mask):
        var neg_temp = temp_stress.le(0.0)
        sig_t = unload_mask.select(neg_temp.select(temp_stress, zero), sig_t)
        tangent_t = unload_mask.select(
            neg_temp.select(unload_slope_c, zero), tangent_t
        )

    return (sig_t, tangent_t, min_strain_t, end_strain_t, unload_slope_t)


@always_inline
fn _steel01_eval_simd_step[width: Int](
    eps: SIMD[DType.float64, width],
    eps_p_c: SIMD[DType.float64, width],
    alpha_c: SIMD[DType.float64, width],
    fy: SIMD[DType.float64, width],
    e0: SIMD[DType.float64, width],
    h: SIMD[DType.float64, width],
    inv_e0_plus_h: SIMD[DType.float64, width],
    e_ep: SIMD[DType.float64, width],
) -> (
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
    SIMD[DType.float64, width],
):
    var sigma_trial = e0 * (eps - eps_p_c)
    var xi = sigma_trial - alpha_c
    var f = _simd_abs_float64[width](xi) - fy
    var yield_mask = f.gt(0.0)
    if all(~yield_mask):
        return (sigma_trial, e0, eps_p_c, alpha_c)
    var dg = f * inv_e0_plus_h
    var sgn = xi.lt(0.0).select(
        SIMD[DType.float64, width](-1.0), SIMD[DType.float64, width](1.0)
    )
    var dgs = dg * sgn
    var sig_t = yield_mask.select(sigma_trial - e0 * dgs, sigma_trial)
    var tan_t = yield_mask.select(e_ep, e0)
    var eps_p_t = yield_mask.select(eps_p_c + dgs, eps_p_c)
    var alpha_t = yield_mask.select(alpha_c + h * dgs, alpha_c)
    return (sig_t, tan_t, eps_p_t, alpha_t)
