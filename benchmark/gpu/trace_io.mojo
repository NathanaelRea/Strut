"""Read the little-endian sparse linear trace written by Strut."""

from collections import List
from sys.ffi import external_call

alias FilePointer = UnsafePointer[UInt8]


struct TraceFile(Movable):
    var handle: FilePointer

    fn __init__(out self, path: String) raises:
        var name = List[UInt8]()
        for byte in path.as_bytes():
            name.append(byte)
        name.append(0)
        var mode = InlineArray[UInt8, 3](114, 98, 0)  # "rb"
        var file = external_call["fopen", FilePointer](name.unsafe_ptr(), UnsafePointer(to=mode[0]))
        if not file:
            raise Error("could not open sparse linear trace")
        self.handle = file

    fn __del__(deinit self):
        _ = external_call["fclose", Int32](self.handle)

    fn next_tag(self) raises -> Optional[UInt64]:
        var tag = UInt64(0)
        var count = external_call["fread", UInt](
            UnsafePointer(to=tag), UInt(1), UInt(8), self.handle
        )
        if count == 0 and external_call["feof", Int32](self.handle) != 0:
            return None
        if count != 8:
            raise Error("incomplete sparse linear trace tag")
        return tag


fn _read_exact(file: FilePointer, output: FilePointer, count: Int) raises:
    if external_call["fread", UInt](output, UInt(8), UInt(count), file) != UInt(count):
        raise Error("truncated sparse linear trace")


fn read_ints(file: FilePointer, count: Int) raises -> List[Int]:
    var data = List[Int](length=count, fill=0)
    _read_exact(file, data.unsafe_ptr().bitcast[UInt8](), count)
    return data^


fn read_vector(file: FilePointer, mut data: List[Float64]) raises:
    _read_exact(file, data.unsafe_ptr().bitcast[UInt8](), len(data))
