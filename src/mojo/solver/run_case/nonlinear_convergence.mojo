from math import sqrt
from os import abort

from solver.simd_contiguous import dot_float64_contiguous, sum_sq_float64_contiguous
from tag_types import NonlinearTestTypeTag


fn nonlinear_test_metric(
    test_type_tag: Int, label: String, solution: List[Float64], rhs: List[Float64], count: Int
) -> Float64:
    if test_type_tag == NonlinearTestTypeTag.NormDispIncr:
        return sqrt(sum_sq_float64_contiguous(solution, count))
    if test_type_tag == NonlinearTestTypeTag.NormUnbalance:
        return sqrt(sum_sq_float64_contiguous(rhs, count))
    if test_type_tag == NonlinearTestTypeTag.EnergyIncr:
        return 0.5 * abs(dot_float64_contiguous(solution, rhs, count))
    if test_type_tag == NonlinearTestTypeTag.MaxDispIncr:
        abort(label + " test_type MaxDispIncr is not supported in OpenSees-parity mode")
    abort(label + " has unsupported test_type")
    return 0.0

