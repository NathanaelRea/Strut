"""Replay captured Strut tangents and Newton right-hand sides through STRUMPACK."""

from collections import InlineArray, List
from math import inf, isfinite
from os import getenv
from sys import argv
from sys.ffi import DLHandle, dlsym
from time import perf_counter_ns
from trace_io import FilePointer, TraceFile, read_ints, read_vector

alias _Opaque = UnsafePointer[UInt8]
alias _MAX_INDEX = 2_147_483_647
alias _RELATIVE_TOLERANCE = 1e-7
alias _BACKWARD_TOLERANCE = 1e-8

alias _DOUBLE = Int32(1)
alias _MT = Int32(0)
alias _NO_MATCHING = Int32(0)
alias _METIS = Int32(1)
alias _DIRECT = Int32(1)
alias _NO_COMPRESSION = Int32(0)


fn _seconds(start: UInt64) -> Float64:
    return Float64(perf_counter_ns() - start) * 1e-9


fn _check(status: Int32, operation: String) raises:
    if status != 0:
        raise Error("STRUMPACK " + operation + " failed with status " + String(Int(status)))


fn _require_gpu(lib: DLHandle) raises:
    for symbol in ["hipGetDeviceCount", "cudaGetDeviceCount"]:
        var name = String(symbol)
        if dlsym[NoneType](lib.handle, name.unsafe_cstr_ptr()):
            var count = Int32(0)
            var device_count = lib.get_function[fn (UnsafePointer[Int32]) -> Int32](name)
            if device_count(UnsafePointer(to=count)) == 0 and count > 0:
                return
    raise Error("STRUMPACK GPU replay needs a GPU-enabled shared library and a visible device")


struct Matrix(Movable):
    var size: Int
    var row_ptr: List[Int32]
    var columns: List[Int32]
    var source: List[Int32]
    var lower: List[Float64]
    var values: List[Float64]

    fn __init__(out self, file: FilePointer) raises:
        var dimensions = read_ints(file, 2)
        var n = dimensions[0]
        var nnz = dimensions[1]
        if n <= 0 or n >= _MAX_INDEX or nnz < n or nnz > (_MAX_INDEX + n) // 2:
            raise Error("matrix trace exceeds the CSR index limits")
        var ptr = read_ints(file, n + 1)
        var rows = read_ints(file, nnz)
        if ptr[0] != 0 or ptr[n] != nnz:
            raise Error("invalid CSC column pointers")
        var row_ptr = List[Int32](length=n + 1, fill=0)
        for col in range(n):
            if ptr[col] < 0 or ptr[col] >= ptr[col + 1] or ptr[col + 1] > nnz:
                raise Error("invalid CSC column pointers")
            if rows[ptr[col]] != col:
                raise Error("CSC column has no diagonal")
            var previous = col - 1
            for p in range(ptr[col], ptr[col + 1]):
                var row = rows[p]
                if row <= previous or row >= n:
                    raise Error("invalid lower CSC row index")
                previous = row
                row_ptr[row + 1] += 1
                if row != col:
                    row_ptr[col + 1] += 1

        for row in range(n):
            row_ptr[row + 1] += row_ptr[row]
        var full_nnz = Int(row_ptr[n])
        var columns = List[Int32](length=full_nnz, fill=0)
        var source = List[Int32](length=full_nnz, fill=0)
        var next = row_ptr.copy()
        for col in range(n):
            for p in range(ptr[col], ptr[col + 1]):
                var row = rows[p]
                var lower = Int(next[row])
                columns[lower] = Int32(col)
                source[lower] = Int32(p)
                next[row] += 1
                if row != col:
                    var upper = Int(next[col])
                    columns[upper] = Int32(row)
                    source[upper] = Int32(p)
                    next[col] += 1
        self.size = n
        self.row_ptr = row_ptr^
        self.columns = columns^
        self.source = source^
        self.lower = List[Float64](length=nnz, fill=0.0)
        self.values = List[Float64](length=full_nnz, fill=0.0)

    fn read_values(mut self, file: FilePointer) raises:
        read_vector(file, self.lower)
        var lower = self.lower.unsafe_ptr()
        for i in range(len(self.source)):
            self.values[i] = lower[Int(self.source[i])]

    fn max_scaled_residual(self, result: List[Float64], rhs: List[Float64]) -> Float64:
        var maximum = 0.0
        var values = self.values.unsafe_ptr()
        var x = result.unsafe_ptr()
        for row in range(self.size):
            var residual = -rhs[row]
            var scale = abs(rhs[row])
            for p in range(Int(self.row_ptr[row]), Int(self.row_ptr[row + 1])):
                var product = values[p] * x[Int(self.columns[p])]
                residual += product
                scale += abs(product)
            if not isfinite(residual) or not isfinite(scale):
                return inf[DType.float64]()
            if scale > 0.0:
                maximum = max(maximum, abs(residual) / scale)
        return maximum


fn _relative_error(result: List[Float64], reference: List[Float64]) -> Float64:
    var error = 0.0
    var scale = 0.0
    for i in range(len(result)):
        if not isfinite(result[i]) or not isfinite(reference[i]):
            return inf[DType.float64]()
        error = max(error, abs(result[i] - reference[i]))
        scale = max(scale, abs(reference[i]))
    return error / scale if scale > 0.0 else error


# The 64-bit Linux C ABI passes this public 16-byte aggregate in two integer
# registers: the solver pointer, then the precision and interface enum fields.
alias _MatrixFn = fn (
    _Opaque,
    UInt64,
    UnsafePointer[Int32],
    UnsafePointer[Int32],
    UnsafePointer[Int32],
    UnsafePointer[Float64],
    Int32,
) -> NoneType


struct Strumpack(Movable):
    var lib: DLHandle
    var handle: InlineArray[UInt64, 2]

    fn __init__(out self, lib: DLHandle, gpu: Bool):
        self.lib = lib
        self.handle = InlineArray[UInt64, 2](0, 0)
        var init = lib.get_function[
            fn (UnsafePointer[UInt64], Int32, Int32, Int32, _Opaque, Int32) -> NoneType
        ]("STRUMPACK_init_mt")
        init(UnsafePointer(to=self.handle[0]), _DOUBLE, _MT, 0, _Opaque(), 0)
        self._option("STRUMPACK_set_matching", _NO_MATCHING)
        self._option("STRUMPACK_set_reordering_method", _METIS)
        self._option("STRUMPACK_set_Krylov_solver", _DIRECT)
        self._option("STRUMPACK_set_compression", _NO_COMPRESSION)
        var select = lib.get_function[fn (_Opaque, UInt64) -> NoneType](
            "STRUMPACK_enable_gpu" if gpu else "STRUMPACK_disable_gpu"
        )
        select(self._pointer(), self.handle[1])

    fn __del__(deinit self):
        var destroy = self.lib.get_function[fn (UnsafePointer[UInt64]) -> NoneType](
            "STRUMPACK_destroy"
        )
        destroy(UnsafePointer(to=self.handle[0]))

    fn _pointer(self) -> _Opaque:
        return UnsafePointer(to=self.handle[0]).bitcast[_Opaque]()[]

    fn _option(self, name: String, value: Int32):
        var set = self.lib.get_function[fn (_Opaque, UInt64, Int32) -> NoneType](name)
        set(self._pointer(), self.handle[1], value)

    fn matrix(self, matrix: Matrix, update: Bool):
        var install = self.lib.get_function[_MatrixFn](
            "STRUMPACK_update_csr_matrix_values" if update else "STRUMPACK_set_csr_matrix"
        )
        var n = Int32(matrix.size)
        install(
            self._pointer(),
            self.handle[1],
            UnsafePointer(to=n),
            matrix.row_ptr.unsafe_ptr(),
            matrix.columns.unsafe_ptr(),
            matrix.values.unsafe_ptr(),
            1,
        )

    fn reorder(self) raises:
        var run = self.lib.get_function[fn (_Opaque, UInt64) -> Int32]("STRUMPACK_reorder")
        _check(run(self._pointer(), self.handle[1]), "reorder")

    fn factor(self) raises:
        var run = self.lib.get_function[fn (_Opaque, UInt64) -> Int32]("STRUMPACK_factor")
        _check(run(self._pointer(), self.handle[1]), "factor")

    fn solve(self, rhs: List[Float64], mut result: List[Float64]) raises:
        var run = self.lib.get_function[
            fn (_Opaque, UInt64, UnsafePointer[Float64], UnsafePointer[Float64], Int32) -> Int32
        ]("STRUMPACK_solve")
        _check(
            run(self._pointer(), self.handle[1], rhs.unsafe_ptr(), result.unsafe_ptr(), 0), "solve"
        )

    fn factor_nonzeros(self) -> Int64:
        var count = self.lib.get_function[fn (_Opaque, UInt64) -> Int64](
            "STRUMPACK_factor_nonzeros"
        )
        return count(self._pointer(), self.handle[1])


struct Replay(Movable):
    var library: DLHandle
    var gpu: Bool
    var stages: Int
    var factors: Int
    var solves: Int
    var max_error: Float64
    var max_residual: Float64
    var reorder_time: Float64
    var factor_time: Float64
    var solve_time: Float64
    var update_time: Float64

    fn __init__(out self, lib: DLHandle, gpu: Bool):
        self.library = lib
        self.gpu = gpu
        self.stages = 0
        self.factors = 0
        self.solves = 0
        self.max_error = 0.0
        self.max_residual = 0.0
        self.reorder_time = 0.0
        self.factor_time = 0.0
        self.solve_time = 0.0
        self.update_time = 0.0

    fn record(self, kind: String, elapsed: Float64, residual: String = "", error: String = ""):
        print(kind, self.stages, self.factors, self.solves, elapsed, residual, error, sep=",")

    fn next_stage(mut self, trace: TraceFile, limit: Int) raises -> Optional[UInt64]:
        var matrix = Matrix(trace.handle)
        var rhs = List[Float64](length=matrix.size, fill=0.0)
        var expected = rhs.copy()
        var result = rhs.copy()
        var setup = perf_counter_ns()
        var solver = Strumpack(self.library, self.gpu)
        var setup_time = _seconds(setup)
        var has_factor = False
        self.stages += 1
        while True:
            var tag = trace.next_tag()
            if not tag:
                return None
            var kind = tag.value()
            if kind != 3 and limit > 0 and self.factors >= limit:
                return None
            if kind == 1:
                return tag
            if kind == 2:
                matrix.read_values(trace.handle)
                var start = perf_counter_ns()
                solver.matrix(matrix, has_factor)
                if has_factor:
                    self.update_time += _seconds(start)
                else:
                    solver.reorder()
                    var elapsed = setup_time + _seconds(start)
                    self.reorder_time += elapsed
                    self.record("reorder", elapsed)
                start = perf_counter_ns()
                solver.factor()
                var elapsed = _seconds(start)
                self.factor_time += elapsed
                self.factors += 1
                if not has_factor:
                    var info = (
                        "stage "
                        + String(self.stages)
                        + " full CSR entries "
                        + String(len(matrix.values))
                        + ", LU factor entries "
                        + String(solver.factor_nonzeros())
                        + "\n"
                    )
                    print(info, end="", file=FileDescriptor(2))
                has_factor = True
                self.record("factor", elapsed)
            elif kind == 3:
                if not has_factor:
                    raise Error("RHS precedes matrix values")
                read_vector(trace.handle, rhs)
                read_vector(trace.handle, expected)
                var start = perf_counter_ns()
                solver.solve(rhs, result)
                var elapsed = _seconds(start)
                self.solve_time += elapsed
                self.solves += 1
                var residual = matrix.max_scaled_residual(result, rhs)
                var error = _relative_error(result, expected)
                self.max_residual = max(self.max_residual, residual)
                self.max_error = max(self.max_error, error)
                self.record("solve", elapsed, String(residual), String(error))
            else:
                raise Error("unknown matrix trace record")

    fn report(self) raises:
        self.record("total_reorder", self.reorder_time)
        self.record("total_update", self.update_time)
        self.record("total_factor", self.factor_time)
        self.record(
            "total_solve", self.solve_time, String(self.max_residual), String(self.max_error)
        )
        if (
            self.solves == 0
            or self.max_error > _RELATIVE_TOLERANCE
            or self.max_residual > _BACKWARD_TOLERANCE
        ):
            raise Error("replay failed numerical verification")


fn main() raises:
    var args = argv()
    var mode = String(args[1]) if len(args) > 1 else ""
    if len(args) < 3 or len(args) > 4 or (mode != "cpu" and mode != "gpu"):
        raise Error("usage: replay (cpu|gpu) TRACE [MAX_FACTORS]")
    var limit = Int(String(args[3])) if len(args) == 4 else 0
    if len(args) == 4 and limit < 1:
        raise Error("MAX_FACTORS must be a positive integer")
    var from_env = getenv("STRUT_STRUMPACK_LIBRARY")
    var lib = DLHandle(from_env if from_env != "" else "libstrumpack.so")
    var gpu = mode == "gpu"
    if gpu:
        _require_gpu(lib)
    var trace = TraceFile(String(args[2]))
    var replay = Replay(lib, gpu)
    print("record,stage,factors,solves,seconds,max_scaled_residual,relative_inf_error")
    var tag = trace.next_tag()
    while tag:
        if tag.value() != 1:
            raise Error("matrix trace data precedes the CSC pattern")
        tag = replay.next_stage(trace, limit)
    replay.report()
