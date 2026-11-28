"""Optional CHOLMOD NESDIS ordering for the sparse factorization benchmark."""

from collections import InlineArray, List
from sys import size_of
from sys.ffi import DLHandle, external_call

alias _Opaque = UnsafePointer[UInt8]

# CHOLMOD 5.3.4 LP64 public struct layout from CHOLMOD/Include/cholmod.h.
# Check the library's version before allocating its common workspace.
alias _COMMON_WORDS = 335
alias _SUPERNODAL = 48
alias _NMETHODS = 168
alias _METHOD_0_ORDERING = 316
alias _POSTORDER = 1864
alias _SPARSE_COLUMNS = 24
alias _SPARSE_ROWS = 32
alias _FACTOR_PERMUTATION = 16


fn nested_dissection_order(
    path: String, n: Int, columns: List[Int], rows: List[Int]
) raises -> List[Int]:
    """Return the NESDIS permutation without CHOLMOD's postordering."""
    var lib = DLHandle(path)
    try:
        var order = _order(lib, n, columns, rows)
        lib.close()
        return order^
    except error:
        lib.close()
        raise error^


fn _order(lib: DLHandle, n: Int, columns: List[Int], rows: List[Int]) raises -> List[Int]:
    var version = lib.get_function[fn (UnsafePointer[Int32]) -> Int32]("cholmod_version")
    var number = InlineArray[Int32, 3](0, 0, 0)
    _ = version(UnsafePointer(to=number[0]))
    if size_of[Int]() != 8 or number[0] != 5 or number[1] != 3 or number[2] != 4:
        raise Error("Strut's optional NESDIS binding requires 64-bit CHOLMOD 5.3.4")

    var start = lib.get_function[fn (_Opaque) -> Int32]("cholmod_l_start")
    var finish = lib.get_function[fn (_Opaque) -> Int32]("cholmod_l_finish")
    var allocate = lib.get_function[
        fn (UInt, UInt, UInt, Int32, Int32, Int32, Int32, _Opaque) -> _Opaque
    ]("cholmod_l_allocate_sparse")
    var analyze = lib.get_function[fn (_Opaque, _Opaque) -> _Opaque]("cholmod_l_analyze")
    var free_sparse = lib.get_function[fn (UnsafePointer[_Opaque], _Opaque) -> Int32](
        "cholmod_l_free_sparse"
    )
    var free_factor = lib.get_function[fn (UnsafePointer[_Opaque], _Opaque) -> Int32](
        "cholmod_l_free_factor"
    )

    var workspace = InlineArray[UInt64, _COMMON_WORDS](fill=0)
    var common = UnsafePointer(to=workspace[0]).bitcast[UInt8]()
    if start(common) == 0:
        raise Error("CHOLMOD initialization failed")
    (common + _SUPERNODAL).bitcast[Int32]()[] = 0  # CHOLMOD_SIMPLICIAL
    (common + _NMETHODS).bitcast[Int32]()[] = 1
    (common + _METHOD_0_ORDERING).bitcast[Int32]()[] = 4  # CHOLMOD_NESDIS
    (common + _POSTORDER).bitcast[Int32]()[] = 0
    # Sorted, packed, lower triangle, pattern only.
    var matrix = allocate(UInt(n), UInt(n), UInt(len(rows)), 1, 1, -1, 0, common)
    if not matrix:
        _ = finish(common)
        raise Error("CHOLMOD sparse pattern allocation failed")
    var p = (matrix + _SPARSE_COLUMNS).bitcast[_Opaque]()[]
    var i = (matrix + _SPARSE_ROWS).bitcast[_Opaque]()[]
    _ = external_call["memcpy", _Opaque](
        p, columns.unsafe_ptr().bitcast[UInt8](), UInt((n + 1) * size_of[Int]())
    )
    _ = external_call["memcpy", _Opaque](
        i, rows.unsafe_ptr().bitcast[UInt8](), UInt(len(rows) * size_of[Int]())
    )

    var factor = analyze(matrix, common)
    var order = List[Int](length=n, fill=0)
    var success = Bool(factor)
    if success:
        var permutation = (factor + _FACTOR_PERMUTATION).bitcast[_Opaque]()[]
        success = Bool(permutation)
        if success:
            _ = external_call["memcpy", _Opaque](
                order.unsafe_ptr().bitcast[UInt8](), permutation, UInt(n * size_of[Int]())
            )
        _ = free_factor(UnsafePointer(to=factor), common)
    _ = free_sparse(UnsafePointer(to=matrix), common)
    _ = finish(common)
    if not success:
        raise Error("CHOLMOD NESDIS ordering failed")
    return order^
