from collections import List
from sys import simd_width_of

from solver.simd_contiguous import (
    load_float64_contiguous_simd,
    store_float64_contiguous_simd,
)


alias FLOAT64_SIMD_WIDTH = simd_width_of[DType.float64]()


@always_inline
fn gather_float64_by_index_simd[width: Int](
    indices: List[Int], start: Int, src: List[Float64]
) -> SIMD[DType.float64, width]:
    var vec = SIMD[DType.float64, width](0.0)
    @parameter
    for lane in range(width):
        vec[lane] = src[indices[start + lane]]
    return vec


@always_inline
fn gather_float64_by_index(
    indices: List[Int], start: Int, src: List[Float64]
) -> SIMD[DType.float64, FLOAT64_SIMD_WIDTH]:
    return gather_float64_by_index_simd[FLOAT64_SIMD_WIDTH](indices, start, src)


@always_inline
fn scatter_float64_by_index_simd[width: Int](
    indices: List[Int],
    start: Int,
    mut dst: List[Float64],
    vec: SIMD[DType.float64, width],
):
    @parameter
    for lane in range(width):
        dst[indices[start + lane]] = vec[lane]


@always_inline
fn scatter_float64_by_index(
    indices: List[Int],
    start: Int,
    mut dst: List[Float64],
    vec: SIMD[DType.float64, FLOAT64_SIMD_WIDTH],
):
    scatter_float64_by_index_simd[FLOAT64_SIMD_WIDTH](indices, start, dst, vec)


@always_inline
fn scatter_add_float64_by_index_simd[width: Int](
    indices: List[Int],
    start: Int,
    mut dst: List[Float64],
    vec: SIMD[DType.float64, width],
):
    @parameter
    for lane in range(width):
        dst[indices[start + lane]] += vec[lane]


@always_inline
fn scatter_add_float64_by_index(
    indices: List[Int],
    start: Int,
    mut dst: List[Float64],
    vec: SIMD[DType.float64, FLOAT64_SIMD_WIDTH],
):
    scatter_add_float64_by_index_simd[FLOAT64_SIMD_WIDTH](indices, start, dst, vec)


@always_inline
fn vectorize_indexed_gather_unary_map_to_contiguous_float64_simd[
    simd_width: Int
](
    indices: List[Int],
    indices_start: Int,
    src: List[Float64],
    mut dst: List[Float64],
    dst_start: Int,
    count: Int,
    scale: Float64,
    bias: Float64,
):
    if count <= 0:
        return
    var i = 0
    var scale_vec = SIMD[DType.float64, simd_width](scale)
    var bias_vec = SIMD[DType.float64, simd_width](bias)
    while i + simd_width <= count:
        var gathered_vec = gather_float64_by_index_simd[simd_width](
            indices, indices_start + i, src
        )
        var mapped_vec = gathered_vec * scale_vec + bias_vec
        store_float64_contiguous_simd[simd_width](dst, dst_start + i, mapped_vec)
        i += simd_width
    while i < count:
        dst[dst_start + i] = src[indices[indices_start + i]] * scale + bias
        i += 1


@always_inline
fn vectorize_contiguous_unary_map_to_indexed_scatter_float64_simd[
    simd_width: Int
](
    src: List[Float64],
    src_start: Int,
    indices: List[Int],
    indices_start: Int,
    mut dst: List[Float64],
    count: Int,
    scale: Float64,
    bias: Float64,
):
    if count <= 0:
        return
    var i = 0
    var scale_vec = SIMD[DType.float64, simd_width](scale)
    var bias_vec = SIMD[DType.float64, simd_width](bias)
    while i + simd_width <= count:
        var src_vec = load_float64_contiguous_simd[simd_width](src, src_start + i)
        var mapped_vec = src_vec * scale_vec + bias_vec
        scatter_float64_by_index_simd[simd_width](indices, indices_start + i, dst, mapped_vec)
        i += simd_width
    while i < count:
        dst[indices[indices_start + i]] = src[src_start + i] * scale + bias
        i += 1
