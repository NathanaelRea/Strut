from algorithm import vectorize
from collections import List
from sys import align_of, simd_width_of, size_of


alias FLOAT64_SIMD_WIDTH = simd_width_of[DType.float64]()


@always_inline
fn load_float64_ptr_contiguous_simd[width: Int](
    ptr: UnsafePointer[Float64], start: Int
) -> SIMD[DType.float64, width]:
    alias alignment = align_of[SIMD[DType.float64, width]]()
    var byte_offset = start * size_of[DType.float64]()
    if (Int(ptr) + byte_offset) % alignment == 0:
        return ptr.load[width=width, alignment=alignment](start)
    return ptr.load[width=width, alignment=1](start)


@always_inline
fn store_float64_ptr_contiguous_simd[width: Int](
    ptr: UnsafePointer[Float64], start: Int, vec: SIMD[DType.float64, width]
):
    alias alignment = align_of[SIMD[DType.float64, width]]()
    var byte_offset = start * size_of[DType.float64]()
    if (Int(ptr) + byte_offset) % alignment == 0:
        ptr.store[width=width, alignment=alignment](start, vec)
        return
    ptr.store[width=width, alignment=1](start, vec)


@always_inline
fn load_float64_contiguous_simd[width: Int](
    values: List[Float64], start: Int
) -> SIMD[DType.float64, width]:
    return load_float64_ptr_contiguous_simd[width](values.unsafe_ptr(), start)


@always_inline
fn store_float64_contiguous_simd[width: Int](
    mut values: List[Float64], start: Int, vec: SIMD[DType.float64, width]
):
    store_float64_ptr_contiguous_simd[width](values.unsafe_ptr(), start, vec)


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
fn vectorize_contiguous_unary_map_float64_simd[
    simd_width: Int
](
    mut dst: List[Float64],
    dst_start: Int,
    src: List[Float64],
    src_start: Int,
    count: Int,
    scale: Float64,
    bias: Float64,
):
    if count <= 0:
        return
    var dst_ptr = dst.unsafe_ptr()
    var src_ptr = src.unsafe_ptr()
    var dst_offset = dst_start
    var src_offset = src_start
    var scale_value = scale
    var bias_value = bias

    @parameter
    fn map_values[width: Int](i: Int):
        var src_vec = load_float64_ptr_contiguous_simd[width](src_ptr, src_offset + i)
        var mapped_vec = src_vec * SIMD[DType.float64, width](scale_value) + SIMD[
            DType.float64, width
        ](bias_value)
        store_float64_ptr_contiguous_simd[width](dst_ptr, dst_offset + i, mapped_vec)

    vectorize[map_values, simd_width](count)


@always_inline
fn vectorize_contiguous_unary_map_inplace_float64_simd[
    simd_width: Int
](
    mut values: List[Float64],
    start: Int,
    count: Int,
    scale: Float64,
    bias: Float64,
):
    if count <= 0:
        return
    var value_ptr = values.unsafe_ptr()
    var value_start = start
    var scale_value = scale
    var bias_value = bias

    @parameter
    fn map_values[width: Int](i: Int):
        var value_vec = load_float64_ptr_contiguous_simd[width](value_ptr, value_start + i)
        var mapped_vec = value_vec * SIMD[DType.float64, width](scale_value) + SIMD[
            DType.float64, width
        ](bias_value)
        store_float64_ptr_contiguous_simd[width](value_ptr, value_start + i, mapped_vec)

    vectorize[map_values, simd_width](count)


@always_inline
fn vectorize_contiguous_binary_map_float64_simd[
    simd_width: Int
](
    mut dst: List[Float64],
    dst_start: Int,
    lhs: List[Float64],
    lhs_start: Int,
    rhs: List[Float64],
    rhs_start: Int,
    count: Int,
    lhs_scale: Float64,
    rhs_scale: Float64,
    bias: Float64,
):
    if count <= 0:
        return
    var dst_ptr = dst.unsafe_ptr()
    var lhs_ptr = lhs.unsafe_ptr()
    var rhs_ptr = rhs.unsafe_ptr()
    var dst_offset = dst_start
    var lhs_offset = lhs_start
    var rhs_offset = rhs_start
    var lhs_scale_value = lhs_scale
    var rhs_scale_value = rhs_scale
    var bias_value = bias

    @parameter
    fn map_values[width: Int](i: Int):
        var lhs_vec = load_float64_ptr_contiguous_simd[width](lhs_ptr, lhs_offset + i)
        var rhs_vec = load_float64_ptr_contiguous_simd[width](rhs_ptr, rhs_offset + i)
        var mapped_vec = (
            lhs_vec * SIMD[DType.float64, width](lhs_scale_value)
            + rhs_vec * SIMD[DType.float64, width](rhs_scale_value)
            + SIMD[DType.float64, width](bias_value)
        )
        store_float64_ptr_contiguous_simd[width](dst_ptr, dst_offset + i, mapped_vec)

    vectorize[map_values, simd_width](count)


@always_inline
fn vectorize_contiguous_binary_map_accumulate_float64_simd[
    simd_width: Int
](
    mut dst: List[Float64],
    dst_start: Int,
    src: List[Float64],
    src_start: Int,
    count: Int,
    src_scale: Float64,
    bias: Float64,
):
    if count <= 0:
        return
    var dst_ptr = dst.unsafe_ptr()
    var src_ptr = src.unsafe_ptr()
    var dst_offset = dst_start
    var src_offset = src_start
    var src_scale_value = src_scale
    var bias_value = bias

    @parameter
    fn map_values[width: Int](i: Int):
        var dst_vec = load_float64_ptr_contiguous_simd[width](dst_ptr, dst_offset + i)
        var src_vec = load_float64_ptr_contiguous_simd[width](src_ptr, src_offset + i)
        var mapped_vec = (
            dst_vec
            + src_vec * SIMD[DType.float64, width](src_scale_value)
            + SIMD[DType.float64, width](bias_value)
        )
        store_float64_ptr_contiguous_simd[width](dst_ptr, dst_offset + i, mapped_vec)

    vectorize[map_values, simd_width](count)


@always_inline
fn vectorize_contiguous_binary_map_reduce_float64_simd[
    simd_width: Int
](
    lhs: List[Float64],
    lhs_start: Int,
    rhs: List[Float64],
    rhs_start: Int,
    count: Int,
    factor: Float64 = 1.0,
) -> Float64:
    if count <= 0:
        return 0.0
    var lhs_ptr = lhs.unsafe_ptr()
    var rhs_ptr = rhs.unsafe_ptr()
    var sum = 0.0
    var lhs_offset = lhs_start
    var rhs_offset = rhs_start
    var factor_value = factor

    @parameter
    fn reduce_values[width: Int](i: Int):
        var lhs_vec = load_float64_ptr_contiguous_simd[width](lhs_ptr, lhs_offset + i)
        var rhs_vec = load_float64_ptr_contiguous_simd[width](rhs_ptr, rhs_offset + i)
        sum += (
            (lhs_vec * rhs_vec) * SIMD[DType.float64, width](factor_value)
        ).reduce_add()

    vectorize[reduce_values, simd_width](count)
    return sum


@always_inline
fn copy_float64_contiguous_simd[width: Int](
    mut dst: List[Float64], src: List[Float64], count: Int
):
    vectorize_contiguous_unary_map_float64_simd[width](
        dst, 0, src, 0, count, 1.0, 0.0
    )


@always_inline
fn copy_float64_contiguous(mut dst: List[Float64], src: List[Float64], count: Int):
    copy_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](dst, src, count)


@always_inline
fn dot_float64_contiguous_simd[width: Int](
    lhs: List[Float64], rhs: List[Float64], count: Int
) -> Float64:
    return vectorize_contiguous_binary_map_reduce_float64_simd[width](
        lhs, 0, rhs, 0, count
    )


@always_inline
fn dot_float64_contiguous(
    lhs: List[Float64], rhs: List[Float64], count: Int
) -> Float64:
    return dot_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](lhs, rhs, count)


@always_inline
fn sum_sq_float64_contiguous_simd[width: Int](
    values: List[Float64], count: Int
) -> Float64:
    return vectorize_contiguous_binary_map_reduce_float64_simd[width](
        values, 0, values, 0, count
    )


@always_inline
fn sum_sq_float64_contiguous(values: List[Float64], count: Int) -> Float64:
    return sum_sq_float64_contiguous_simd[FLOAT64_SIMD_WIDTH](values, count)


@always_inline
fn scale_float64_segment_simd[width: Int](
    mut values: List[Float64], start: Int, count: Int, scale: Float64
):
    vectorize_contiguous_unary_map_inplace_float64_simd[width](
        values, start, count, scale, 0.0
    )


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
    vectorize_contiguous_binary_map_accumulate_float64_simd[width](
        dst, dst_start, src, src_start, count, alpha, 0.0
    )


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
    return vectorize_contiguous_binary_map_reduce_float64_simd[width](
        lhs, lhs_start, rhs, rhs_start, count
    )


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
