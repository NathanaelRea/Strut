from collections import List
from sys import align_of, simd_width_of, size_of


alias FLOAT64_SIMD_WIDTH = simd_width_of[DType.float64]()


@always_inline
fn load_float64_contiguous_simd[width: Int](
    values: List[Float64], start: Int
) -> SIMD[DType.float64, width]:
    var ptr = values.unsafe_ptr()
    alias alignment = align_of[SIMD[DType.float64, width]]()
    var byte_offset = start * size_of[DType.float64]()
    if (Int(ptr) + byte_offset) % alignment == 0:
        return ptr.load[width=width, alignment=alignment](start)
    return ptr.load[width=width, alignment=1](start)


@always_inline
fn store_float64_contiguous_simd[width: Int](
    mut values: List[Float64], start: Int, vec: SIMD[DType.float64, width]
):
    var ptr = values.unsafe_ptr()
    alias alignment = align_of[SIMD[DType.float64, width]]()
    var byte_offset = start * size_of[DType.float64]()
    if (Int(ptr) + byte_offset) % alignment == 0:
        ptr.store[width=width, alignment=alignment](start, vec)
        return
    ptr.store[width=width, alignment=1](start, vec)


@always_inline
fn load_float64_contiguous(
    values: List[Float64], start: Int
) -> SIMD[DType.float64, FLOAT64_SIMD_WIDTH]:
    return load_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](values, start)


@always_inline
fn store_float64_contiguous(
    mut values: List[Float64], start: Int, vec: SIMD[DType.float64, FLOAT64_SIMD_WIDTH]
):
    store_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](values, start, vec)


@always_inline
fn copy_float64_contiguous_simd[width: Int](
    mut dst: List[Float64], src: List[Float64], count: Int
):
    if count <= 0:
        return
    var i = 0
    while i + width <= count:
        store_float64_contiguous_simd[width](
            dst, i, load_float64_contiguous_simd[width](src, i)
        )
        i += width
    while i < count:
        dst[i] = src[i]
        i += 1


@always_inline
fn copy_float64_contiguous(mut dst: List[Float64], src: List[Float64], count: Int):
    copy_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](dst, src, count)


@always_inline
fn dot_float64_contiguous_simd[width: Int](
    lhs: List[Float64], rhs: List[Float64], count: Int
) -> Float64:
    if count <= 0:
        return 0.0
    var sum = 0.0
    var i = 0
    while i + width <= count:
        var lhs_vec = load_float64_contiguous_simd[width](lhs, i)
        var rhs_vec = load_float64_contiguous_simd[width](rhs, i)
        sum += (lhs_vec * rhs_vec).reduce_add()
        i += width
    while i < count:
        sum += lhs[i] * rhs[i]
        i += 1
    return sum


@always_inline
fn dot_float64_contiguous(
    lhs: List[Float64], rhs: List[Float64], count: Int
) -> Float64:
    return dot_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](lhs, rhs, count)


@always_inline
fn sum_sq_float64_contiguous_simd[width: Int](
    values: List[Float64], count: Int
) -> Float64:
    if count <= 0:
        return 0.0
    var sum = 0.0
    var i = 0
    while i + width <= count:
        var vec = load_float64_contiguous_simd[width](values, i)
        sum += (vec * vec).reduce_add()
        i += width
    while i < count:
        sum += values[i] * values[i]
        i += 1
    return sum


@always_inline
fn sum_sq_float64_contiguous(values: List[Float64], count: Int) -> Float64:
    return sum_sq_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](values, count)


@always_inline
fn scale_float64_segment_simd[width: Int](
    mut values: List[Float64], start: Int, count: Int, scale: Float64
):
    if count <= 0:
        return
    var i = 0
    var scale_vec = SIMD[DType.float64, width](scale)
    while i + width <= count:
        var vec = load_float64_contiguous_simd[width](values, start + i)
        store_float64_contiguous_simd[width](values, start + i, vec * scale_vec)
        i += width
    while i < count:
        values[start + i] *= scale
        i += 1


@always_inline
fn scale_float64_segment(
    mut values: List[Float64], start: Int, count: Int, scale: Float64
):
    scale_float64_segment_simd[FLOAT64_SIMD_WIDTH](values, start, count, scale)


@always_inline
fn axpy_float64_segments_simd[width: Int](
    mut dst: List[Float64],
    dst_start: Int,
    src: List[Float64],
    src_start: Int,
    alpha: Float64,
    count: Int,
):
    if count <= 0:
        return
    var i = 0
    var alpha_vec = SIMD[DType.float64, width](alpha)
    while i + width <= count:
        var dst_vec = load_float64_contiguous_simd[width](dst, dst_start + i)
        var src_vec = load_float64_contiguous_simd[width](src, src_start + i)
        store_float64_contiguous_simd[width](
            dst, dst_start + i, dst_vec + alpha_vec * src_vec
        )
        i += width
    while i < count:
        dst[dst_start + i] += alpha * src[src_start + i]
        i += 1


@always_inline
fn axpy_float64_segments(
    mut dst: List[Float64],
    dst_start: Int,
    src: List[Float64],
    src_start: Int,
    alpha: Float64,
    count: Int,
):
    axpy_float64_segments_simd[FLOAT64_SIMD_WIDTH](
        dst, dst_start, src, src_start, alpha, count
    )


@always_inline
fn dot_float64_segments_simd[width: Int](
    lhs: List[Float64],
    lhs_start: Int,
    rhs: List[Float64],
    rhs_start: Int,
    count: Int,
) -> Float64:
    if count <= 0:
        return 0.0
    var sum = 0.0
    var i = 0
    while i + width <= count:
        var lhs_vec = load_float64_contiguous_simd[width](lhs, lhs_start + i)
        var rhs_vec = load_float64_contiguous_simd[width](rhs, rhs_start + i)
        sum += (lhs_vec * rhs_vec).reduce_add()
        i += width
    while i < count:
        sum += lhs[lhs_start + i] * rhs[rhs_start + i]
        i += 1
    return sum


@always_inline
fn dot_float64_segments(
    lhs: List[Float64],
    lhs_start: Int,
    rhs: List[Float64],
    rhs_start: Int,
    count: Int,
) -> Float64:
    return dot_float64_segments_simd[FLOAT64_SIMD_WIDTH](
        lhs, lhs_start, rhs, rhs_start, count
    )
