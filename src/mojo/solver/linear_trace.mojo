"""Opt-in binary replay capture for the sparse solver benchmarks."""

from collections import InlineArray, List
from os import abort, getenv
from sys.ffi import external_call


struct LinearTrace(Movable):
    var path: List[UInt8]

    fn __init__(out self):
        self.path = []
        var filename = getenv("STRUT_LINEAR_TRACE")
        if filename != "":
            for byte in filename.as_bytes():
                self.path.append(byte)
            self.path.append(0)

    fn open_record(self, kind: UInt64) -> UnsafePointer[UInt8]:
        var mode = InlineArray[UInt8, 3](fill=0)
        mode[0] = 97  # a
        mode[1] = 98  # b
        var file = external_call["fopen", UnsafePointer[UInt8]](
            self.path.unsafe_ptr(), UnsafePointer(to=mode[0])
        )
        if not file:
            abort("could not open sparse linear trace")
        var tag = kind
        self.write(file, UnsafePointer(to=tag).bitcast[UInt8](), 1)
        return file

    fn write(self, file: UnsafePointer[UInt8], data: UnsafePointer[UInt8], count: Int):
        if external_call["fwrite", UInt](data, UInt(8), UInt(count), file) != UInt(count):
            abort("could not write sparse linear trace")

    fn close_record(self, file: UnsafePointer[UInt8]):
        if external_call["fclose", Int32](file) != 0:
            abort("could not close sparse linear trace")

    fn pattern(self, n: Int, columns: List[Int], rows: List[Int]):
        if len(self.path) == 0:
            return
        var file = self.open_record(1)
        var dimensions = InlineArray[Int, 2](n, len(rows))
        self.write(file, UnsafePointer(to=dimensions[0]).bitcast[UInt8](), 2)
        self.write(file, columns.unsafe_ptr().bitcast[UInt8](), len(columns))
        self.write(file, rows.unsafe_ptr().bitcast[UInt8](), len(rows))
        self.close_record(file)

    fn tangent(self, values: List[Float64]):
        if len(self.path) == 0:
            return
        var file = self.open_record(2)
        self.write(file, values.unsafe_ptr().bitcast[UInt8](), len(values))
        self.close_record(file)

    fn solution(self, rhs: List[Float64], solution: List[Float64]):
        if len(self.path) == 0:
            return
        var file = self.open_record(3)
        self.write(file, rhs.unsafe_ptr().bitcast[UInt8](), len(rhs))
        self.write(file, solution.unsafe_ptr().bitcast[UInt8](), len(solution))
        self.close_record(file)
