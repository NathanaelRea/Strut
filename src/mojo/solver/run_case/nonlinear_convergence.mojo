from math import sqrt
from os import abort
from sys.ffi import external_call

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


@fieldwise_init
struct RelativeNormDispIncrOptions(ImplicitlyCopyable, Movable):
    var norm_type: Int
    var print_flag: Int

    fn norm(self, solution: List[Float64], count: Int) -> Float64:
        # OpenSees Vector::pNorm accumulates in equation order.
        var value = 0.0
        if self.norm_type == 2:
            for i in range(count):
                value += solution[i] * solution[i]
            return sqrt(value)
        if self.norm_type <= 0:
            for i in range(count):
                var entry = abs(solution[i])
                if entry > value:
                    value = entry
            return value
        var power = Float64(self.norm_type)
        for i in range(count):
            value += external_call["pow", Float64](abs(solution[i]), power)
        return external_call["pow", Float64](value, 1.0 / power)

    fn accepts(
        self,
        norm: Float64,
        mut first_norm: Float64,
        iteration: Int,
        max_iters: Int,
        tol: Float64,
    ) -> Bool:
        if iteration == 1:
            first_norm = norm
        var ratio = norm / first_norm if first_norm != 0.0 else norm
        return ratio <= tol or (
            iteration == max_iters and (self.print_flag == 5 or self.print_flag == 6)
        )
