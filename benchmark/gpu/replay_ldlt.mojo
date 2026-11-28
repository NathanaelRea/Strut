"""Replay captured stiffness matrices and actual Newton right-hand sides."""

from collections import List
from math import fma, inf, isfinite, sqrt
from os import abort, getenv
from sys import argv
from solver.sparse_ldlt import SparseLDLT
from trace_io import TraceFile, read_ints, read_vector
from time import perf_counter_ns


fn norm_inf(values: List[Float64]) -> Float64:
    var result = 0.0
    for value in values:
        if not isfinite(value):
            return inf[DType.float64]()
        result = max(result, abs(value))
    return result


fn subtract_matrix_product(
    columns: List[Int],
    rows: List[Int],
    values: List[Float64],
    x: List[Float64],
    mut output: List[Float64],
):
    var dst = output.unsafe_ptr()
    var src = x.unsafe_ptr()
    var starts = columns.unsafe_ptr()
    var indices = rows.unsafe_ptr()
    var entries = values.unsafe_ptr()
    for col in range(len(x)):
        var begin = starts[col]
        var acc = fma(-entries[begin], src[col], dst[col])
        for p in range(begin + 1, starts[col + 1]):
            var row = indices[p]
            var value = entries[p]
            dst[row] = fma(-value, src[col], dst[row])
            acc = fma(-value, src[row], acc)
        dst[col] = acc


fn dot(x: List[Float64], y: List[Float64]) -> Float64:
    var lhs = x.unsafe_ptr()
    var rhs = y.unsafe_ptr()
    var sum = 0.0
    for i in range(len(x)):
        sum = fma(lhs[i], rhs[i], sum)
    return sum


fn axpy(alpha: Float64, x: List[Float64], mut y: List[Float64]):
    var src = x.unsafe_ptr()
    var dst = y.unsafe_ptr()
    for i in range(len(x)):
        dst[i] = fma(alpha, src[i], dst[i])


struct Replay(Movable):
    var solver: SparseLDLT
    var columns: List[Int]
    var rows: List[Int]
    var values: List[Float64]
    var scaled_values: List[Float64]
    var weights: List[Float64]
    var rhs: List[Float64]
    var expected: List[Float64]
    var scaled_rhs: List[Float64]
    var residual: List[Float64]
    var solution: List[Float64]
    var correction: List[Float64]
    var valid: Bool
    var equilibrate: Bool
    var gmres: Bool
    var max_refine: Int
    var factors: Int
    var solves: Int
    var rejected: Int
    var inaccurate: Int
    var symbolic_ns: UInt64
    var factor_ns: UInt64
    var solve_ns: UInt64
    var refine_ns: UInt64
    var refinements: Int
    var worst_initial_error: Float64
    var worst_error: Float64
    var worst_residual: Float64

    fn __init__(out self, equilibrate: Bool, max_refine: Int, gmres: Bool):
        self.solver = SparseLDLT(8)
        self.columns = []
        self.rows = []
        self.values = []
        self.scaled_values = []
        self.weights = []
        self.rhs = []
        self.expected = []
        self.scaled_rhs = []
        self.residual = []
        self.solution = []
        self.correction = []
        self.valid = False
        self.equilibrate = equilibrate
        self.gmres = gmres
        self.max_refine = max_refine
        self.factors = 0
        self.solves = 0
        self.rejected = 0
        self.inaccurate = 0
        self.symbolic_ns = 0
        self.factor_ns = 0
        self.solve_ns = 0
        self.refine_ns = 0
        self.refinements = 0
        self.worst_initial_error = 0.0
        self.worst_error = 0.0
        self.worst_residual = 0.0

    fn pattern(mut self, file: UnsafePointer[UInt8]) raises:
        var dimensions = read_ints(file, 2)
        var n = dimensions[0]
        var entries = dimensions[1]
        if n < 1 or entries < n:
            raise Error("invalid dimensions in sparse linear trace")
        self.columns = read_ints(file, n + 1)
        self.rows = read_ints(file, entries)
        self.values.resize(entries, 0.0)
        self.scaled_values.resize(entries if self.equilibrate else 0, 0.0)
        self.weights.resize(n, 1.0)
        self.rhs.resize(n, 0.0)
        self.expected.resize(n, 0.0)
        self.scaled_rhs.resize(n, 0.0)
        self.residual.resize(n, 0.0)
        self.solution.resize(n, 0.0)
        self.correction.resize(n, 0.0)
        self.valid = False

        # New analysis stages start with a fresh solver, as in the full model.
        self.solver = SparseLDLT(8)
        self.solver.gpu_requested = getenv("STRUT_REPLAY_GPU") == "1"
        var start = perf_counter_ns()
        self.solver.analyze(n, self.columns, self.rows)
        self.symbolic_ns += perf_counter_ns() - start

    fn factor(mut self, file: UnsafePointer[UInt8]) raises:
        if len(self.values) == 0:
            abort("missing sparse trace pattern before tangent")
        read_vector(file, self.values)
        var start = perf_counter_ns()
        if self.equilibrate:
            var n = len(self.weights)
            for c in range(n):
                var diagonal = abs(self.values[self.columns[c]])
                self.weights[c] = 1.0 / sqrt(diagonal) if diagonal > 0.0 else 1.0
            for c in range(n):
                var weight = self.weights[c]
                for p in range(self.columns[c], self.columns[c + 1]):
                    self.scaled_values[p] = (self.values[p] * weight) * self.weights[self.rows[p]]
            self.valid = self.solver.factorize(self.scaled_values)
        else:
            self.valid = self.solver.factorize(self.values)
        self.factor_ns += perf_counter_ns() - start
        self.factors += 1
        self.rejected += Int(not self.valid)

    fn form_residual(mut self) -> Float64:
        var residual = self.residual.unsafe_ptr()
        var b = self.rhs.unsafe_ptr()
        var weights = self.weights.unsafe_ptr()
        var n = len(self.rhs)
        for i in range(n):
            residual[i] = b[i]
        subtract_matrix_product(self.columns, self.rows, self.values, self.solution, self.residual)
        var r_norm = 0.0
        var b_norm = 0.0
        for i in range(n):
            if not isfinite(residual[i]):
                return inf[DType.float64]()
            r_norm = max(r_norm, abs(residual[i] * weights[i]))
            b_norm = max(b_norm, abs(b[i] * weights[i]))
        return r_norm / max(1e-300, b_norm)

    fn solution_error(self) -> Float64:
        var difference = 0.0
        for i in range(len(self.solution)):
            if not isfinite(self.solution[i]):
                return inf[DType.float64]()
            difference = max(difference, abs(self.solution[i] - self.expected[i]))
        return difference / max(1e-300, norm_inf(self.expected))

    fn refine_with_gmres(mut self) -> Float64:
        # Arnoldi in the equilibrated coordinates, with the rounded factor
        # as a right preconditioner. Matrix products and updates stay FP64.
        var n = len(self.rhs)
        var limit = self.max_refine
        var stride = limit + 1
        var basis = List[List[Float64]]()
        var preconditioned = List[List[Float64]]()
        var h = List[Float64](length=limit * stride, fill=0.0)
        var cosine = List[Float64](length=limit, fill=0.0)
        var sine = List[Float64](length=limit, fill=0.0)
        var g = List[Float64](length=stride, fill=0.0)
        var v = List[Float64](length=n, fill=0.0)
        var scaled_b_sq = 0.0
        for i in range(n):
            v[i] = self.residual[i] * self.weights[i]
            var b = self.rhs[i] * self.weights[i]
            scaled_b_sq = fma(b, b, scaled_b_sq)
        g[0] = sqrt(dot(v, v))
        if g[0] == 0.0 or not isfinite(g[0]):
            return self.form_residual()
        var reciprocal = 1.0 / g[0]
        for i in range(n):
            v[i] *= reciprocal
        basis.append(v^)

        var used = 0
        var tolerance = 1.0e-12 * sqrt(scaled_b_sq)
        for j in range(limit):
            var z = List[Float64](length=n, fill=0.0)
            self.solver.solve(basis[j], z)
            for i in range(n):
                self.scaled_rhs[i] = z[i] * self.weights[i]
                self.residual[i] = 0.0
            subtract_matrix_product(
                self.columns, self.rows, self.values, self.scaled_rhs, self.residual
            )
            for i in range(n):
                self.residual[i] *= -self.weights[i]

            # Modified Gram-Schmidt, repeated once for reliable orthogonality
            # when the preconditioned operator is already near identity.
            var column = j * stride
            for _ in range(2):
                for i in range(j + 1):
                    var projection = dot(basis[i], self.residual)
                    h[column + i] += projection
                    axpy(-projection, basis[i], self.residual)
            var next_norm = sqrt(dot(self.residual, self.residual))
            h[column + j + 1] = next_norm
            for i in range(j):
                var upper = h[column + i]
                var lower = h[column + i + 1]
                h[column + i] = cosine[i] * upper + sine[i] * lower
                h[column + i + 1] = cosine[i] * lower - sine[i] * upper
            var diagonal = h[column + j]
            var rotation = sqrt(fma(diagonal, diagonal, next_norm * next_norm))
            if rotation == 0.0 or not isfinite(rotation):
                break
            cosine[j] = diagonal / rotation
            sine[j] = next_norm / rotation
            h[column + j] = rotation
            g[j + 1] = -sine[j] * g[j]
            g[j] *= cosine[j]
            preconditioned.append(z^)
            used += 1
            if next_norm == 0.0 or abs(g[j + 1]) <= tolerance:
                break
            var next = List[Float64](length=n, fill=0.0)
            reciprocal = 1.0 / next_norm
            for i in range(n):
                next[i] = self.residual[i] * reciprocal
            basis.append(next^)

        for j in range(used - 1, -1, -1):
            g[j] /= h[j * stride + j]
            for i in range(j):
                g[i] -= g[j] * h[j * stride + i]
        for j in range(used):
            for i in range(n):
                self.solution[i] = fma(
                    g[j], preconditioned[j][i] * self.weights[i], self.solution[i]
                )
        self.refinements += used
        return self.form_residual()

    fn solve(mut self, file: UnsafePointer[UInt8]) raises:
        if self.factors == 0:
            abort("missing sparse trace tangent before solve")
        read_vector(file, self.rhs)
        read_vector(file, self.expected)
        self.solves += 1
        if not self.valid:
            self.inaccurate += 1
            return

        var n = len(self.rhs)
        var start = perf_counter_ns()
        if self.equilibrate:
            for i in range(n):
                self.scaled_rhs[i] = self.rhs[i] * self.weights[i]
            self.solver.solve(self.scaled_rhs, self.solution)
            for i in range(n):
                self.solution[i] *= self.weights[i]
        else:
            self.solver.solve(self.rhs, self.solution)
        self.solve_ns += perf_counter_ns() - start
        self.worst_initial_error = max(self.worst_initial_error, self.solution_error())

        start = perf_counter_ns()
        var residual_norm = self.form_residual()
        if self.max_refine > 0:
            if self.gmres and residual_norm > 1.0e-10:
                residual_norm = self.refine_with_gmres()
            elif not self.gmres:
                for _ in range(self.max_refine):
                    if residual_norm <= 1.0e-10 or not isfinite(residual_norm):
                        break
                    for i in range(n):
                        self.scaled_rhs[i] = self.residual[i] * self.weights[i]
                    self.solver.solve(self.scaled_rhs, self.correction)
                    for i in range(n):
                        self.solution[i] = fma(
                            self.weights[i], self.correction[i], self.solution[i]
                        )
                    self.refinements += 1
                    residual_norm = self.form_residual()
            self.refine_ns += perf_counter_ns() - start
        var error = self.solution_error()
        self.worst_error = max(self.worst_error, error)
        self.worst_residual = max(self.worst_residual, residual_norm)
        self.inaccurate += Int(error > 1.0e-5)

    fn report(self):
        print("factors", self.factors, "rejected", self.rejected)
        print("solves", self.solves, "inaccurate", self.inaccurate, "corrections", self.refinements)
        print("symbolic_s", Float64(self.symbolic_ns) * 1e-9)
        print("factor_s", Float64(self.factor_ns) * 1e-9)
        print("solve_s", Float64(self.solve_ns) * 1e-9)
        print("refine_s", Float64(self.refine_ns) * 1e-9)
        print("worst_initial_error", self.worst_initial_error)
        print("worst_error", self.worst_error)
        print("worst_scaled_residual", self.worst_residual)


fn main() raises:
    var args = argv()
    if len(args) < 2 or len(args) > 5:
        raise Error("usage: replay_ldlt TRACE [MAX_REFINEMENTS] [equilibrate [gmres]]")
    var corrections = Int(String(args[2])) if len(args) > 2 else 0
    if (
        corrections < 0
        or (len(args) >= 4 and args[3] != "equilibrate")
        or (len(args) == 5 and args[4] != "gmres")
    ):
        raise Error(
            "expected a nonnegative refinement limit, then equilibrate and gmres as options"
        )
    var trace = TraceFile(String(args[1]))
    var replay = Replay(len(args) >= 4, corrections, len(args) == 5)
    while True:
        var tag = trace.next_tag()
        if not tag:
            break
        var kind = tag.value()
        if kind == 1:
            replay.pattern(trace.handle)
        elif kind == 2:
            replay.factor(trace.handle)
        elif kind == 3:
            replay.solve(trace.handle)
        else:
            raise Error("unrecognized sparse linear trace tag")
    replay.report()
