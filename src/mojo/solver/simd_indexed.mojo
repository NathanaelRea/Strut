from collections import List
from sys import simd_width_of


alias FLOAT64_SIMD_WIDTH = simd_width_of[DType.float64]()


@always_inline
fn gather_float64_by_index_simd[width: Int](
    indices: List[Int], start: Int, src: List[Float64]
) -> SIMD[DType.float64, width]:
    var vec = SIMD[DType.float64, width](0.0)
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
