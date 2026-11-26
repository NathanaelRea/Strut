"""SIMD elementary functions for float64 material kernels.

The standard library evaluates `SIMD ** SIMD` one lane at a time through a
scalar software pow, which makes Menegotto-Pinto steel (two pows per fiber
evaluation) the most expensive thing in a fiber section. These versions stay in
vector registers end to end.

Accuracy: `simd_log` and `simd_exp` follow the fdlibm reductions and are good
to about 1 ulp. `simd_pow_nonneg` is `exp(y * log(x))`, so its relative error is
about `|y * log(x)|` ulps: a few 1e-15 for the exponents and ratios the material
models use. That is far below any solver tolerance, but it is not correctly
rounded, so results can differ from a scalar pow in the last digits.
"""

from memory import bitcast


alias _LN2_HI: Float64 = 6.93147180369123816490e-01
alias _LN2_LO: Float64 = 1.90821492927058770002e-10
alias _INV_LN2: Float64 = 1.44269504088896338700e+00
alias _SQRT2: Float64 = 1.41421356237309504880e+00
alias _TWO54: Float64 = 18014398509481984.0
alias _FLOAT64_MIN_NORMAL: Float64 = 2.2250738585072014e-308
alias _EXP_OVERFLOW: Float64 = 7.09782712893383973096e+02
alias _EXP_UNDERFLOW: Float64 = -7.45133219101941108420e+02

alias _LG1: Float64 = 6.666666666666735130e-01
alias _LG2: Float64 = 3.999999999940941908e-01
alias _LG3: Float64 = 2.857142874366239149e-01
alias _LG4: Float64 = 2.222219843214978396e-01
alias _LG5: Float64 = 1.818357216161805012e-01
alias _LG6: Float64 = 1.531383769920937332e-01
alias _LG7: Float64 = 1.479819860511658591e-01

alias _P1: Float64 = 1.66666666666666019037e-01
alias _P2: Float64 = -2.77777777770155933842e-03
alias _P3: Float64 = 6.61375632143793436117e-05
alias _P4: Float64 = -1.65339022054652515390e-06
alias _P5: Float64 = 4.13813679705723846039e-08


@always_inline
fn simd_log[width: Int](x: SIMD[DType.float64, width]) -> SIMD[DType.float64, width]:
    """Natural log of each lane. Lanes must be positive and finite."""
    alias F = SIMD[DType.float64, width]
    alias I = SIMD[DType.int64, width]

    # Lift subnormals into the normal range so the exponent field is meaningful.
    var subnormal = x.lt(F(_FLOAT64_MIN_NORMAL))
    var scaled = subnormal.select(x * F(_TWO54), x)
    var exponent_bias = subnormal.select(I(-54), I(0))

    var bits = bitcast[DType.int64, width](scaled)
    var exponent = ((bits >> I(52)) & I(0x7FF)) - I(1023) + exponent_bias
    var mantissa_bits = (bits & I(0x000FFFFFFFFFFFFF)) | I(0x3FF0000000000000)
    var mantissa = bitcast[DType.float64, width](mantissa_bits)

    # Keep the mantissa in [sqrt(2)/2, sqrt(2)) so f = m - 1 stays small.
    var upper = mantissa.gt(F(_SQRT2))
    mantissa = upper.select(mantissa * F(0.5), mantissa)
    exponent = upper.select(exponent + I(1), exponent)
    var k = exponent.cast[DType.float64]()

    var f = mantissa - F(1.0)
    var s = f / (F(2.0) + f)
    var z = s * s
    var w = z * z
    var t1 = w * (F(_LG2) + w * (F(_LG4) + w * F(_LG6)))
    var t2 = z * (F(_LG1) + w * (F(_LG3) + w * (F(_LG5) + w * F(_LG7))))
    var r = t1 + t2
    var hfsq = F(0.5) * f * f
    return k * F(_LN2_HI) - ((hfsq - (s * (hfsq + r) + k * F(_LN2_LO))) - f)


@always_inline
fn simd_exp[width: Int](x: SIMD[DType.float64, width]) -> SIMD[DType.float64, width]:
    """e**x for each lane, saturating to 0 and +inf outside the float64 range."""
    alias F = SIMD[DType.float64, width]
    alias I = SIMD[DType.int64, width]

    var overflow = x.gt(F(_EXP_OVERFLOW))
    var underflow = x.lt(F(_EXP_UNDERFLOW))
    # Clamp so the reduction below stays finite; the masks restore the limits.
    var clamped = overflow.select(F(_EXP_OVERFLOW), x)
    clamped = underflow.select(F(_EXP_UNDERFLOW), clamped)

    var scaled = clamped * F(_INV_LN2)
    var k_float = (scaled + scaled.lt(F(0.0)).select(F(-0.5), F(0.5))).cast[
        DType.int64
    ]().cast[DType.float64]()
    var hi = clamped - k_float * F(_LN2_HI)
    var lo = k_float * F(_LN2_LO)
    var r = hi - lo
    var rr = r * r
    var c = r - rr * (
        F(_P1) + rr * (F(_P2) + rr * (F(_P3) + rr * (F(_P4) + rr * F(_P5))))
    )
    var y = F(1.0) - ((lo - (r * c) / (F(2.0) - c)) - hi)

    # Scale by 2**k in two steps so results in the subnormal range stay exact.
    var k = k_float.cast[DType.int64]()
    var k_half = k >> I(1)
    var k_rest = k - k_half
    var scale_half = bitcast[DType.float64, width]((k_half + I(1023)) << I(52))
    var scale_rest = bitcast[DType.float64, width]((k_rest + I(1023)) << I(52))
    var result = (y * scale_half) * scale_rest
    result = underflow.select(F(0.0), result)
    return overflow.select(F(Float64.MAX_FINITE) * F(Float64.MAX_FINITE), result)


@always_inline
fn simd_pow_nonneg[width: Int](
    x: SIMD[DType.float64, width], y: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    """x**y for each lane with x >= 0 and finite, and y finite.

    Zero bases follow the usual convention: 0**y is 0 for y > 0, 1 for y == 0,
    and +inf for y < 0.
    """
    alias F = SIMD[DType.float64, width]

    var zero_base = x.le(F(0.0))
    # A zero base would send log to -inf times y; substitute 1 and patch after.
    var safe_x = zero_base.select(F(1.0), x)
    var result = simd_exp[width](y * simd_log[width](safe_x))
    var zero_result = y.gt(F(0.0)).select(
        F(0.0),
        y.lt(F(0.0)).select(F(Float64.MAX_FINITE) * F(Float64.MAX_FINITE), F(1.0)),
    )
    return zero_base.select(zero_result, result)


@always_inline
fn pow_nonneg(x: Float64, y: Float64) -> Float64:
    """Scalar `simd_pow_nonneg`, so scalar and SIMD material paths agree bit for bit."""
    return simd_pow_nonneg[1](
        SIMD[DType.float64, 1](x), SIMD[DType.float64, 1](y)
    )[0]
