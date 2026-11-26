"""Sparse symmetric LDLᵀ with reusable symbolic analysis.

Input is zero-based CSC of the lower triangle, including every diagonal.
Column pointers have length n+1 and each column's row indices are strictly
increasing. The values passed to factorize use this exact input layout.
Row labels fit in signed 32 bits. CSC pointers and factor offsets can be wider.
Analyze owns its resulting permutation and pattern. Call it again if the
dimension, pattern, ordering, or factorization mode changes. An optional
permutation maps each new column to its original index and is used as given.

The factor stores D on the diagonal of L's packed, column-major panels.
Unit diagonals of L are implicit. Each panel shares one sorted list of row
indices below its diagonal block. All numeric and solve storage is allocated
in analyze. Use solve with pre-sized vectors after a successful factorization.
Both vectors are in the original ordering. Solve uses instance scratch space.
A failed factorization invalidates the factor until the next successful one.
A pivot must be finite and exceed both 1e-300 in magnitude and 1e-14 times
the maximum magnitude in the corresponding original matrix row.

SparseLDLT(0) uses the physical performance-core count. SparseLDLT(1) uses
the caller thread. Large supernodal factors reuse native workers and scratch
space created in analyze. Each instance supports one call at a time.
"""

from algorithm import parallelize
from collections import InlineArray, List
from math import fma, isfinite, sqrt
from memory import memset_zero
from os.atomic import Atomic, Consistency
from solver.simd_contiguous import FLOAT64_SIMD_WIDTH
from sys import CompilationTarget, num_performance_cores
from sys.ffi import external_call


struct SparseOrdering:
    """Auto compares symbolic factor fill and checks a wider AMD cutoff on hub graphs."""

    alias AMD = 0
    alias RCM = 1
    alias Auto = 2


struct SparseFactorMode:
    alias Supernodal = 0
    alias Simplicial = 1


fn _ints(n: Int, value: Int = 0) -> List[Int]:
    var result: List[Int] = []
    result.resize(n, value)
    return result^


fn _floats(n: Int) -> List[Float64]:
    var result: List[Float64] = []
    result.resize(n, 0.0)
    return result^


# Labels are 32-bit. Offsets use the compact representation when their
# address range fits, with wide storage for exceptionally large factors.
struct _Offsets(Movable):
    var narrow: List[UInt32]
    var wide: List[Int]
    var compact: Bool

    fn __init__(out self, count: Int = 0, limit: Int = 0):
        self.narrow = []
        self.wide = []
        self.compact = limit <= 4_294_967_295
        if self.compact:
            self.narrow.resize(count, UInt32(0))
        else:
            self.wide.resize(count, 0)

    @always_inline
    fn __getitem__(self, p: Int) -> Int:
        if self.compact:
            return Int(self.narrow.unsafe_ptr()[p])
        return self.wide.unsafe_ptr()[p]

    @always_inline
    fn set(mut self, p: Int, value: Int):
        if self.compact:
            self.narrow.unsafe_ptr()[p] = UInt32(value)
        else:
            self.wide.unsafe_ptr()[p] = value

    fn copy(self) -> Self:
        var result = Self()
        result.compact = self.compact
        result.narrow = self.narrow.copy()
        result.wide = self.wide.copy()
        return result^


struct _Graph(Movable):
    var ptr: List[Int]
    var rows: List[Int32]

    fn __init__(out self, n: Int, nnz: Int):
        self.ptr = _ints(n + 1)
        self.rows = List[Int32](length=nnz, fill=Int32(0))

    fn allocate_columns(mut self, mut counts: List[Int]):
        for j in range(len(counts)):
            self.ptr[j + 1] = self.ptr[j] + counts[j]
            counts[j] = self.ptr[j]


fn _adjacency(n: Int, col_ptr: List[Int], row_idx: List[Int]) -> _Graph:
    var graph = _Graph(n, 2 * (len(row_idx) - n))
    var cursor = _ints(n)
    for j in range(n):
        cursor[j] += col_ptr[j + 1] - col_ptr[j] - 1
        for p in range(col_ptr[j] + 1, col_ptr[j + 1]):
            cursor[row_idx[p]] += 1
    graph.allocate_columns(cursor)
    for j in range(n):
        for p in range(col_ptr[j] + 1, col_ptr[j + 1]):
            var i = row_idx[p]
            graph.rows[cursor[i]] = Int32(j)
            graph.rows[cursor[j]] = Int32(i)
            cursor[i] += 1
            cursor[j] += 1
    return graph^


struct _DegreeBuckets(Movable):
    var head: List[Int]
    var next: List[Int]
    var prev: List[Int]
    var degree: List[Int]
    var minimum: Int

    fn __init__(out self, n: Int):
        self.head = _ints(n, -1)
        self.next = _ints(n, -1)
        self.prev = _ints(n, -1)
        self.degree = _ints(n)
        self.minimum = n

    @always_inline
    fn insert(mut self, i: Int, degree: Int):
        var head = self.head.unsafe_ptr()
        var next = self.next.unsafe_ptr()
        var prev = self.prev.unsafe_ptr()
        var first = head[degree]
        next[i] = first
        prev[i] = -1
        if first >= 0:
            prev[first] = i
        head[degree] = i
        self.degree.unsafe_ptr()[i] = degree
        self.minimum = min(self.minimum, degree)

    @always_inline
    fn remove(mut self, i: Int):
        var next = self.next.unsafe_ptr()
        var prev = self.prev.unsafe_ptr()
        var before = prev[i]
        var after = next[i]
        if before >= 0:
            next[before] = after
        else:
            self.head.unsafe_ptr()[self.degree.unsafe_ptr()[i]] = after
        if after >= 0:
            prev[after] = before

    fn pop_min(mut self) -> Int:
        while self.head.unsafe_ptr()[self.minimum] < 0:
            self.minimum += 1
        var i = self.head.unsafe_ptr()[self.minimum]
        self.remove(i)
        return i


fn _rcm(graph: _Graph) -> List[Int]:
    var n = len(graph.ptr) - 1
    var buckets = _DegreeBuckets(n)
    var seen = _ints(n)
    var order = _ints(n)
    var ptr = graph.ptr.unsafe_ptr()
    var rows = graph.rows.unsafe_ptr()
    var degrees = buckets.degree.unsafe_ptr()
    var queue = order.unsafe_ptr()

    @parameter
    fn degree_order(a: Int, b: Int) -> Bool:
        return degrees[a] < degrees[b] or (degrees[a] == degrees[b] and a < b)

    for i in range(n - 1, -1, -1):
        buckets.insert(i, ptr[i + 1] - ptr[i])

    var tail = 0
    while tail < n:
        var seed = buckets.pop_min()
        var head = tail
        order[tail] = seed
        tail += 1
        seen[seed] = 1
        while head < tail:
            var i = order[head]
            head += 1
            var first = tail
            for p in range(ptr[i], ptr[i + 1]):
                var j = Int(rows[p])
                if seen[j] == 0:
                    seen[j] = 1
                    buckets.remove(j)
                    queue[tail] = j
                    tail += 1
            # Keep insertion sort for short queues. Large neighborhoods
            # use the library sort to avoid quadratic work at dense hubs.
            if tail - first > 32:
                sort[degree_order](Span(ptr=queue + first, length=tail - first))
            else:
                for p in range(first + 1, tail):
                    var j = queue[p]
                    var k = p
                    while k > first and degrees[queue[k - 1]] > degrees[j]:
                        queue[k] = queue[k - 1]
                        k -= 1
                    queue[k] = j
    for i in range(n // 2):
        var j = n - 1 - i
        var saved = order[i]
        order[i] = order[j]
        order[j] = saved
    return order^


fn _amd(graph: _Graph, dense_multiplier: Int = 8) -> List[Int]:
    # In the quotient graph, a live variable references direct neighbors
    # (>= 0) and incident elements (< 0). An element represents a clique of its
    # remaining variables. A pivot reuses its variable ID as an element.
    var n = len(graph.ptr) - 1
    var ptr = graph.ptr.unsafe_ptr()
    var rows = graph.rows.unsafe_ptr()
    var buckets = _DegreeBuckets(n)
    var links: List[List[Int]] = List[List[Int]](capacity=n)
    var live_list = _ints(n, 1)
    var size_list = _ints(n, -1)
    var stamp_list = _ints(n, -1)
    var external_list = _ints(n)
    var element_stamp_list = _ints(n, -1)
    var dense: List[Int] = []
    var order: List[Int] = List[Int](capacity=n)
    var frontier = _ints(n)
    var live = live_list.unsafe_ptr()
    var size = size_list.unsafe_ptr()
    var stamp = stamp_list.unsafe_ptr()
    var external = external_list.unsafe_ptr()
    var element_stamp = element_stamp_list.unsafe_ptr()
    # Include a degree-relative cutoff so floor masters are detected in
    # large low-degree meshes even when they fall below the sqrt(n) bound.
    var mean_degree = len(graph.rows) // max(1, n)
    var dense_limit = min(
        max(16, Int(10.0 * sqrt(Float64(n)))), max(64, dense_multiplier * mean_degree)
    )

    for i in range(n):
        if ptr[i + 1] - ptr[i] > dense_limit:
            live[i] = 0
            dense.append(i)
    for i in range(n):
        var neighbors: List[Int] = []
        if live[i] != 0:
            for p in range(ptr[i], ptr[i + 1]):
                var j = Int(rows[p])
                if live[j] != 0:
                    neighbors.append(j)
            buckets.insert(i, len(neighbors))
        links.append(neighbors^)

    var remaining = n - len(dense)
    var lists = links.unsafe_ptr()
    while remaining > 0:
        var pivot = buckets.pop_min()
        live[pivot] = 0
        order.append(pivot)
        remaining -= 1

        # Form the new frontier and absorb elements incident on the pivot.
        var count = 0
        for p in range(len(lists[pivot])):
            var entry = lists[pivot].unsafe_ptr()[p]
            if entry >= 0:
                if live[entry] != 0 and stamp[entry] != pivot:
                    stamp[entry] = pivot
                    frontier.unsafe_ptr()[count] = entry
                    count += 1
            else:
                var element = -entry - 1
                if size[element] < 0:
                    continue
                var members = lists[element].unsafe_ptr()
                for k in range(len(lists[element])):
                    var j = members[k]
                    if live[j] != 0 and stamp[j] != pivot:
                        stamp[j] = pivot
                        frontier.unsafe_ptr()[count] = j
                        count += 1
                size[element] = -1
                lists[element] = List[Int]()

        # Count E \ frontier once per old element. Update each degree using
        # the sum of its external element sizes, without forming their union.
        var members = frontier.unsafe_ptr()
        for k in range(count):
            var i = members[k]
            buckets.remove(i)
            var neighbors = lists[i].unsafe_ptr()
            for q in range(len(lists[i])):
                var entry = neighbors[q]
                if entry < 0:
                    var element = -entry - 1
                    if size[element] >= 0:
                        if element_stamp[element] != pivot:
                            element_stamp[element] = pivot
                            external[element] = size[element]
                        external[element] -= 1

        var kept = 0
        for k in range(count):
            var i = members[k]
            var neighbors = lists[i].unsafe_ptr()
            var length = 0
            var degree = 0
            for q in range(len(lists[i])):
                var entry = neighbors[q]
                if entry >= 0:
                    if live[entry] == 0 or stamp[entry] == pivot:
                        continue
                    degree += 1
                else:
                    var element = -entry - 1
                    if size[element] < 0:
                        continue
                    if external[element] == 0:
                        # The new element subsumes this old one.
                        size[element] = -1
                        lists[element] = List[Int]()
                        continue
                    degree += external[element]
                neighbors[length] = entry
                length += 1

            if length == 0:
                # All its neighbors are already in the pivot's clique.
                order.append(i)
                live[i] = 0
                remaining -= 1
                lists[i] = List[Int]()
            else:
                lists[i].resize(length, 0)
                lists[i].append(-pivot - 1)
                buckets.degree.unsafe_ptr()[i] = min(degree, buckets.degree.unsafe_ptr()[i])
                members[kept] = i
                kept += 1

        lists[pivot] = List[Int](capacity=kept)
        for k in range(kept):
            var i = members[k]
            lists[pivot].append(i)
            buckets.insert(i, min(remaining - 1, buckets.degree.unsafe_ptr()[i] + kept - 1))
        size[pivot] = kept
    for i in range(len(dense)):
        order.append(dense[i])
    return order^


struct _PermutedTriangle(Movable):
    var graph: _Graph
    var input: _Offsets

    fn __init__(
        out self,
        inverse: List[Int],
        col_ptr: List[Int],
        row_idx: List[Int],
        upper: Bool,
    ):
        var n = len(inverse)
        var nnz = len(row_idx)
        self.graph = _Graph(n, nnz)
        self.input = _Offsets(nnz, nnz)
        var cursor = _ints(n)
        var perm = inverse.unsafe_ptr()
        var starts = col_ptr.unsafe_ptr()
        var rows = row_idx.unsafe_ptr()
        for j in range(n):
            var k = perm[j]
            for p in range(starts[j], starts[j + 1]):
                var i = perm[rows[p]]
                cursor[max(i, k) if upper else min(i, k)] += 1
        self.graph.allocate_columns(cursor)
        for j in range(n):
            var k = perm[j]
            for p in range(starts[j], starts[j + 1]):
                var i = perm[rows[p]]
                var col = max(i, k) if upper else min(i, k)
                self.graph.rows[cursor[col]] = Int32(min(i, k) if upper else max(i, k))
                self.input.set(cursor[col], p)
                cursor[col] += 1


fn _etree(graph: _Graph, order: List[Int], inverse: List[Int]) -> List[Int]:
    var n = len(order)
    var parent = _ints(n, -1)
    var ancestor = _ints(n, -1)
    var starts = graph.ptr.unsafe_ptr()
    var rows = graph.rows.unsafe_ptr()
    var path = ancestor.unsafe_ptr()
    var perm = inverse.unsafe_ptr()
    for k in range(n):
        var col = order[k]
        for p in range(starts[col], starts[col + 1]):
            var i = perm[Int(rows[p])]
            while i >= 0 and i < k:
                var next = path[i]
                path[i] = k
                if next < 0:
                    parent.unsafe_ptr()[i] = k
                i = next
    return parent^


fn _postorder(parent: List[Int]) -> List[Int]:
    var n = len(parent)
    var child = _ints(n, -1)
    var sibling = _ints(n)
    var stack = _ints(n)
    var post = _ints(n)
    for i in range(n - 1, -1, -1):
        var p = parent[i]
        if p >= 0:
            sibling[i] = child[p]
            child[p] = i
    var written = 0
    for root in range(n):
        if parent[root] >= 0:
            continue
        var top = 0
        stack[top] = root
        while top >= 0:
            var i = stack[top]
            var j = child[i]
            if j < 0:
                post[written] = i
                written += 1
                top -= 1
            else:
                child[i] = sibling[j]
                top += 1
                stack[top] = j
    return post^


fn _column_counts(
    graph: _Graph,
    order: List[Int],
    inverse: List[Int],
    parent: List[Int],
    post: List[Int],
) -> List[Int]:
    # Gilbert–Ng–Peyton column counts use the union of etree paths
    # contributing to each L row. Skeleton leaves and their least common
    # ancestors determine the deltas to accumulate along the tree.
    var n = len(parent)
    var count = _ints(n)
    var first = _ints(n, -1)
    var last_first = _ints(n, -1)
    var last_leaf = _ints(n, -1)
    var ancestor = _ints(n)
    var starts = graph.ptr.unsafe_ptr()
    var rows = graph.rows.unsafe_ptr()
    for k in range(n):
        var j = post[k]
        count[j] = Int(first[j] < 0)
        while j >= 0 and first[j] < 0:
            first[j] = k
            j = parent[j]
        ancestor[k] = k
    for k in range(n):
        var j = post[k]
        var up = parent[j]
        if up >= 0:
            count[up] -= 1
        var col = order[j]
        for p in range(starts[col], starts[col + 1]):
            var i = inverse.unsafe_ptr()[Int(rows[p])]
            if i <= j or first[j] <= last_first[i]:
                continue
            last_first[i] = first[j]
            var previous = last_leaf[i]
            last_leaf[i] = j
            count[j] += 1
            if previous >= 0:
                var root = previous
                while root != ancestor[root]:
                    root = ancestor[root]
                while previous != root:
                    var next = ancestor[previous]
                    ancestor[previous] = root
                    previous = next
                count[root] -= 1
        if up >= 0:
            ancestor[j] = up
    for j in range(n):
        if parent[j] >= 0:
            count[parent[j]] += count[j]
    return count^


struct _SymbolicOrder(Movable):
    var kind: Int
    var order: List[Int]
    var parent: List[Int]
    var post: List[Int]
    var counts: List[Int]
    var nnz: Int
    var flops: Int

    fn __init__(
        out self,
        graph: _Graph,
        kind: Int,
        permutation: List[Int] = [],
        dense_multiplier: Int = 8,
    ):
        self.kind = kind
        if len(permutation) > 0:
            self.order = permutation.copy()
        elif kind == SparseOrdering.RCM:
            self.order = _rcm(graph)
        else:
            self.order = _amd(graph, dense_multiplier)
        var inverse = _ints(len(self.order))
        for j in range(len(self.order)):
            inverse[self.order[j]] = j
        self.parent = _etree(graph, self.order, inverse)
        self.post = _postorder(self.parent)
        self.counts = _column_counts(graph, self.order, inverse, self.parent, self.post)
        self.nnz = 0
        self.flops = 0
        for i in range(len(self.counts)):
            var count = self.counts[i]
            self.nnz += count
            self.flops += count * count - 1

    fn apply_postorder(mut self):
        # Tree postordering improves locality without changing fill.
        # Keep the RCM numbering intact to preserve its bandwidth.
        if self.kind != SparseOrdering.AMD:
            return
        var n = len(self.order)
        var old_to_new = _ints(n)
        var ordered = _ints(n)
        var changed = False
        for j in range(n):
            var old = self.post[j]
            changed |= old != j
            old_to_new[old] = j
            ordered[j] = self.order[old]
        if not changed:
            return
        var parent = _ints(n, -1)
        var counts = _ints(n)
        for j in range(n):
            var old = self.post[j]
            var p = self.parent[old]
            if p >= 0:
                parent[j] = old_to_new[p]
            counts[j] = self.counts[old]
        self.order = ordered^
        self.parent = parent^
        self.counts = counts^


fn _validate_symbolic_input(
    n: Int, col_ptr: List[Int], row_idx: List[Int], ordering: Int, dense_multiplier: Int
) raises:
    if dense_multiplier < 1 or dense_multiplier > 128:
        raise Error("sparse LDLT dense degree multiplier must be from 1 to 128")
    if (
        ordering != SparseOrdering.AMD
        and ordering != SparseOrdering.RCM
        and ordering != SparseOrdering.Auto
    ):
        raise Error("unknown sparse LDLT ordering")
    if n < 0 or n > 2_147_483_647:
        raise Error("sparse LDLT dimension exceeds the supported 32-bit row range")
    if len(col_ptr) != n + 1 or col_ptr[0] != 0:
        raise Error("invalid sparse LDLT column pointers")
    if col_ptr[n] != len(row_idx):
        raise Error("sparse LDLT column pointers do not cover the row indices")
    for j in range(n):
        var start = col_ptr[j]
        var end = col_ptr[j + 1]
        if end <= start or end > len(row_idx) or row_idx[start] != j:
            raise Error("sparse LDLT columns must start with the diagonal")
        var previous = j
        for p in range(start + 1, end):
            var i = row_idx[p]
            if i <= previous or i >= n:
                raise Error("sparse LDLT row indices must be sorted and unique")
            previous = i


fn _choose_symbolic(
    graph: _Graph,
    matrix_nnz: Int,
    ordering: Int,
    permutation: List[Int],
    dense_multiplier: Int,
) -> _SymbolicOrder:
    var custom_order = len(permutation) > 0
    var kind = SparseOrdering.RCM
    if not custom_order and ordering == SparseOrdering.AMD:
        kind = SparseOrdering.AMD
    var symbolic = _SymbolicOrder(graph, kind, permutation, dense_multiplier)
    # RCM can preserve a band without any fill. That is optimal, so
    # skip the more expensive quotient-graph search in that case.
    if not custom_order and ordering == SparseOrdering.Auto and symbolic.nnz > matrix_nnz:
        var amd = _SymbolicOrder(graph, SparseOrdering.AMD, dense_multiplier=dense_multiplier)
        if amd.nnz <= symbolic.nnz:
            symbolic = amd^
    # High-degree hubs can benefit from the looser, sqrt(n)-limited AMD
    # cutoff. Test it only for larger Auto patterns with degree outliers.
    if not custom_order and ordering == SparseOrdering.Auto and dense_multiplier == 8:
        var n = len(graph.ptr) - 1
        if n >= 8192 and symbolic.nnz > matrix_nnz:
            var threshold = 16 * max(1, len(graph.rows) // n)
            for j in range(n):
                if graph.ptr[j + 1] - graph.ptr[j] > threshold:
                    var alternate = _SymbolicOrder(graph, SparseOrdering.AMD, dense_multiplier=64)
                    if alternate.nnz < symbolic.nnz and alternate.flops < symbolic.flops:
                        symbolic = alternate^
                    break
    return symbolic^


@fieldwise_init
struct SparseOrderingEstimate(ImplicitlyCopyable):
    var nnz: Int
    var flops: Int


fn estimate_sparse_ldlt_ordering(
    n: Int,
    col_ptr: List[Int],
    row_idx: List[Int],
    ordering: Int = SparseOrdering.Auto,
    dense_multiplier: Int = 8,
) raises -> SparseOrderingEstimate:
    """Count structural factor entries and operations with no panel or factor allocation."""
    _validate_symbolic_input(n, col_ptr, row_idx, ordering, dense_multiplier)
    var graph = _adjacency(n, col_ptr, row_idx)
    var symbolic = _choose_symbolic(graph, len(row_idx), ordering, [], dense_multiplier)
    return SparseOrderingEstimate(symbolic.nnz, symbolic.flops)


@always_inline
fn _scale_column(column: UnsafePointer[Float64], count: Int, alpha: Float64):
    alias width = FLOAT64_SIMD_WIDTH
    var i = 0
    while i + width <= count:
        column.store[width=width, alignment=1](i, column.load[width=width, alignment=1](i) * alpha)
        i += width
    for k in range(i, count):
        column[k] *= alpha


@always_inline
fn _dot_tail(lhs: UnsafePointer[Float64], rhs: UnsafePointer[Float64], count: Int) -> Float64:
    alias lanes = FLOAT64_SIMD_WIDTH
    var total = SIMD[DType.float64, lanes](0.0)
    var i = 0
    while i + lanes <= count:
        total = fma(
            lhs.load[width=lanes, alignment=1](i),
            rhs.load[width=lanes, alignment=1](i),
            total,
        )
        i += lanes
    var value = total.reduce_add()
    for j in range(i, count):
        value += lhs[j] * rhs[j]
    return value


@always_inline
fn _update_columns[
    rank: Int
](
    dst: UnsafePointer[Float64],
    a: UnsafePointer[Float64],
    b: UnsafePointer[Float64],
    c: UnsafePointer[Float64],
    d: UnsafePointer[Float64],
    count: Int,
    alpha: Float64,
    beta: Float64,
    gamma: Float64,
    delta: Float64,
):
    alias width = FLOAT64_SIMD_WIDTH
    var i = 0
    while i + width <= count:
        var value = fma(
            a.load[width=width, alignment=1](i),
            SIMD[DType.float64, width](alpha),
            dst.load[width=width, alignment=1](i),
        )

        @parameter
        if rank == 4:
            value = fma(
                b.load[width=width, alignment=1](i),
                SIMD[DType.float64, width](beta),
                value,
            )
            value = fma(
                c.load[width=width, alignment=1](i),
                SIMD[DType.float64, width](gamma),
                value,
            )
            value = fma(
                d.load[width=width, alignment=1](i),
                SIMD[DType.float64, width](delta),
                value,
            )
        dst.store[width=width, alignment=1](i, value)
        i += width
    for k in range(i, count):
        var value = fma(a[k], alpha, dst[k])

        @parameter
        if rank == 4:
            value = fma(b[k], beta, value)
            value = fma(c[k], gamma, value)
            value = fma(d[k], delta, value)
        dst[k] = value


@always_inline
fn _update_runs[
    rank: Int
](
    dst: UnsafePointer[Float64],
    a: UnsafePointer[Float64],
    b: UnsafePointer[Float64],
    c: UnsafePointer[Float64],
    d: UnsafePointer[Float64],
    mapped: UnsafePointer[Int],
    runs: UnsafePointer[Int],
    first: Int,
    count: Int,
    alpha: Float64,
    beta: Float64,
    gamma: Float64,
    delta: Float64,
):
    var i = first
    while i < count:
        var end = runs[i]
        if end - i < FLOAT64_SIMD_WIDTH:
            # Scatter short segments without setting up a vector loop.
            for k in range(i, end):
                var j = mapped[k]
                var value = fma(a[k], alpha, dst[j])

                @parameter
                if rank == 4:
                    value = fma(b[k], beta, value)
                    value = fma(c[k], gamma, value)
                    value = fma(d[k], delta, value)
                dst[j] = value
        else:
            _update_columns[rank](
                dst + mapped[i],
                a + i,
                b + i,
                c + i,
                d + i,
                end - i,
                alpha,
                beta,
                gamma,
                delta,
            )
        i = end


# Each microtile keeps all of its output in SIMD registers across the rank
# dimension. Packed left tiles and the packed right panel fit together in L2.
alias _TILE_ROWS = 2 * FLOAT64_SIMD_WIDTH
alias _TILE_COLS = 6
alias _WIDE_TILE_COLS = 12
alias _CACHE_ROWS = 192
alias _MAX_PANEL_COLS = 192
alias _PIVOT_BLOCK = 16
alias _PACK_LEFT = _CACHE_ROWS * _MAX_PANEL_COLS
alias _PACK_RIGHT = (
    (_MAX_PANEL_COLS + _WIDE_TILE_COLS - 1) // _WIDE_TILE_COLS
) * _WIDE_TILE_COLS * _MAX_PANEL_COLS
alias _PACK_WORK = _PACK_LEFT + _PACK_RIGHT + _TILE_ROWS * _WIDE_TILE_COLS


fn _gemm_tile[
    indirect: Bool, tile_cols: Int
](
    lhs: UnsafePointer[Float64],
    rhs: UnsafePointer[Float64],
    dst: UnsafePointer[Float64],
    offsets: UnsafePointer[Int],
    rank: Int,
    row_offsets: UnsafePointer[Int],
):
    alias lanes = FLOAT64_SIMD_WIDTH
    alias Vec = SIMD[DType.float64, lanes]
    var top = InlineArray[Vec, tile_cols](fill=Vec(0))
    var bottom = InlineArray[Vec, tile_cols](fill=Vec(0))
    var gather_lo = SIMD[DType.int64, lanes](0)
    var gather_hi = SIMD[DType.int64, lanes](0)

    @parameter
    if indirect:
        gather_lo = row_offsets.bitcast[Int64]().load[width=lanes, alignment=1](0)
        gather_hi = row_offsets.bitcast[Int64]().load[width=lanes, alignment=1](lanes)

    @parameter
    for j in range(tile_cols):
        var ptr = dst + offsets[j]

        @parameter
        if indirect:
            top[j] = ptr.gather[alignment=8](gather_lo)
            bottom[j] = ptr.gather[alignment=8](gather_hi)
        else:
            top[j] = ptr.load[width=lanes, alignment=1](0)
            bottom[j] = ptr.load[width=lanes, alignment=1](lanes)
    for k in range(rank):
        var a = lhs + k * _TILE_ROWS
        var b = rhs + k * tile_cols
        var lo = a.load[width=lanes, alignment=1](0)
        var hi = a.load[width=lanes, alignment=1](lanes)

        @parameter
        for j in range(tile_cols):
            var weight = Vec(b[j])
            top[j] = fma(lo, weight, top[j])
            bottom[j] = fma(hi, weight, bottom[j])

    @parameter
    for j in range(tile_cols):
        var ptr = dst + offsets[j]

        @parameter
        if indirect:
            ptr.scatter[alignment=8](gather_lo, top[j])
            ptr.scatter[alignment=8](gather_hi, bottom[j])
        else:
            ptr.store[width=lanes, alignment=1](0, top[j])
            ptr.store[width=lanes, alignment=1](lanes, bottom[j])


@always_inline
fn _target_column[sparse: Bool](rows: UnsafePointer[Int32], start: Int, i: Int) -> Int:
    @parameter
    if sparse:
        return Int(rows[i])
    else:
        return start + i


@always_inline
fn _target_row[sparse: Bool](mapped: UnsafePointer[Int], start: Int, i: Int) -> Int:
    @parameter
    if sparse:
        return start + mapped[i]
    else:
        return start + i


fn _blocked_update_layout[
    sparse: Bool, tile_cols: Int
](
    factor: UnsafePointer[Float64],
    col_start: UnsafePointer[Int],
    first_pivot: Int,
    end_pivot: Int,
    source_row: Int,
    target_row: Int,
    target_cols: UnsafePointer[Int32],
    mapped: UnsafePointer[Int],
    first_col: Int,
    end_col: Int,
    first_row: Int,
    end_row: Int,
    scratch: UnsafePointer[Float64],
):
    # Apply C -= A D Bᵀ, including only the lower triangle. For a sparse
    # update, source rows and target columns use the same index sequence.
    # Contiguous portions of the target can be updated in place.
    alias lanes = FLOAT64_SIMD_WIDTH
    var rank = end_pivot - first_pivot
    var b_pack = scratch + _PACK_LEFT
    var stage = b_pack + _PACK_RIGHT
    var slots = InlineArray[Int, _MAX_PANEL_COLS](fill=0)
    var offsets = InlineArray[Int, tile_cols](fill=0)
    var off = UnsafePointer(to=offsets[0])
    for k in range(rank):
        var j = first_pivot + k
        slots[k] = col_start[j] + source_row - j

    for p in range(first_col, end_col, tile_cols):
        var packed = b_pack + (p - first_col) * rank
        var cols = min(tile_cols, end_col - p)
        for k in range(rank):
            var src = factor + slots[k] + p
            var weight = -factor[col_start[first_pivot + k]]
            var dst = packed + k * tile_cols
            for j in range(cols):
                dst[j] = weight * src[j]
            for j in range(cols, tile_cols):
                dst[j] = 0.0

    for base in range(max(first_row, first_col), end_row, _CACHE_ROWS):
        var limit = min(end_row, base + _CACHE_ROWS)
        for r in range(base, limit, _TILE_ROWS):
            var packed = scratch + (r - base) * rank
            var rows = min(_TILE_ROWS, limit - r)
            for k in range(rank):
                var src = factor + slots[k] + r
                var dst = packed + k * _TILE_ROWS
                if rows == _TILE_ROWS:
                    dst.store[width=lanes, alignment=1](0, src.load[width=lanes, alignment=1](0))
                    dst.store[width=lanes, alignment=1](
                        lanes, src.load[width=lanes, alignment=1](lanes)
                    )
                else:
                    for i in range(rows):
                        dst[i] = src[i]
                    for i in range(rows, _TILE_ROWS):
                        dst[i] = 0.0

        for p in range(first_col, min(end_col, limit), tile_cols):
            var packed_b = b_pack + (p - first_col) * rank
            var cols = min(tile_cols, end_col - p)
            var col_base = InlineArray[Int, tile_cols](fill=0)
            for j in range(cols):
                var column = _target_column[sparse](target_cols, target_row, p + j)
                col_base[j] = col_start[column] - column
            for r in range(base + (max(0, p - base) // _TILE_ROWS) * _TILE_ROWS, limit, _TILE_ROWS):
                var packed_a = scratch + (r - base) * rank
                var rows = min(_TILE_ROWS, limit - r)
                var start = _target_row[sparse](mapped, target_row, r)
                var last = _target_row[sparse](mapped, target_row, r + rows - 1)
                if cols == tile_cols and rows == _TILE_ROWS and r >= p + cols - 1:
                    if last - start == rows - 1:
                        for j in range(tile_cols):
                            off[j] = col_base[j] + start
                        _gemm_tile[False, tile_cols](packed_a, packed_b, factor, off, rank, mapped)
                    else:
                        for j in range(tile_cols):
                            off[j] = col_base[j] + target_row
                        _gemm_tile[True, tile_cols](
                            packed_a, packed_b, factor, off, rank, mapped + r
                        )
                else:
                    memset_zero(stage, _TILE_ROWS * tile_cols)
                    for j in range(tile_cols):
                        off[j] = j * _TILE_ROWS
                    for j in range(cols):
                        var real = factor + col_base[j]
                        var temp = stage + off[j]
                        for i in range(max(0, p + j - r), rows):
                            temp[i] = real[_target_row[sparse](mapped, target_row, r + i)]
                    _gemm_tile[False, tile_cols](packed_a, packed_b, stage, off, rank, mapped)
                    for j in range(cols):
                        var real = factor + col_base[j]
                        var temp = stage + off[j]
                        for i in range(max(0, p + j - r), rows):
                            real[_target_row[sparse](mapped, target_row, r + i)] = temp[i]


fn _blocked_update[
    sparse: Bool
](
    factor: UnsafePointer[Float64],
    col_start: UnsafePointer[Int],
    first_pivot: Int,
    end_pivot: Int,
    source_row: Int,
    target_row: Int,
    target_cols: UnsafePointer[Int32],
    mapped: UnsafePointer[Int],
    first_col: Int,
    end_col: Int,
    first_row: Int,
    end_row: Int,
    scratch: UnsafePointer[Float64],
):
    var columns = end_col - first_col
    var wide_cols = (columns + _WIDE_TILE_COLS - 1) // _WIDE_TILE_COLS * _WIDE_TILE_COLS
    var narrow_cols = (columns + _TILE_COLS - 1) // _TILE_COLS * _TILE_COLS
    if columns >= 12 and end_pivot - first_pivot >= 24 and wide_cols <= narrow_cols:
        _blocked_update_layout[sparse, _WIDE_TILE_COLS](
            factor,
            col_start,
            first_pivot,
            end_pivot,
            source_row,
            target_row,
            target_cols,
            mapped,
            first_col,
            end_col,
            first_row,
            end_row,
            scratch,
        )
    else:
        _blocked_update_layout[sparse, _TILE_COLS](
            factor,
            col_start,
            first_pivot,
            end_pivot,
            source_row,
            target_row,
            target_cols,
            mapped,
            first_col,
            end_col,
            first_row,
            end_row,
            scratch,
        )


@always_inline
fn _usable_pivot(pivot: Float64, scale: Float64) -> Bool:
    return isfinite(pivot) and abs(pivot) > max(1.0e-300, 1.0e-14 * scale)


# Mailboxes park workers and the dispatcher when the other side has not
# finished. The state transition publishes the job or its result before
# waking the waiter. Threads and scratch are created only in analyze.
alias _TaskFn = fn (UnsafePointer[UInt8], Int)
alias _Mailbox = Atomic[DType.int32]
alias _IDLE = Int32(0)
alias _READY = Int32(1)
alias _STOP = Int32(2)
alias _JOINING = Int32(3)
alias _PARKED = Int32(4)


@always_inline
fn _mailbox_replace(state: UnsafePointer[Int32], value: Int32) -> Int32:
    var before = _Mailbox.load[ordering = Consistency.ACQUIRE](state)
    while not _Mailbox.compare_exchange(state, before, value):
        before = _Mailbox.load[ordering = Consistency.ACQUIRE](state)
    return before


@always_inline
fn _mailbox_wait(state: UnsafePointer[Int32], expected: Int32):
    @parameter
    if CompilationTarget.is_linux() and CompilationTarget.is_x86():
        # Linux x86-64 SYS_futex with FUTEX_WAIT_PRIVATE.
        _ = external_call["syscall", Int](Int(202), state, Int(128), Int(expected), Int(0))
    else:
        var delay = InlineArray[Int, 2](fill=0)
        delay[1] = 25_000
        _ = external_call["nanosleep", Int32](UnsafePointer(to=delay[0]), Int(0))


@always_inline
fn _mailbox_wake(state: UnsafePointer[Int32]):
    @parameter
    if CompilationTarget.is_linux() and CompilationTarget.is_x86():
        _ = external_call["syscall", Int](Int(202), state, Int(129), Int(1), Int(0))


fn _empty_task(context: UnsafePointer[UInt8], lane: Int):
    _ = context
    _ = lane


struct _WorkerSlot(Movable):
    var signal: Int32
    var fn_ptr: _TaskFn
    var argument: UnsafePointer[UInt8]
    var lane: Int
    var padding: InlineArray[Int, 8]

    fn __init__(out self, lane: Int):
        self.signal = _IDLE
        self.fn_ptr = _empty_task
        self.argument = UnsafePointer[UInt8]()
        self.lane = lane
        self.padding = InlineArray[Int, 8](fill=0)


fn _worker_loop(context: UnsafePointer[UInt8]) -> UnsafePointer[UInt8]:
    var slot = context.bitcast[_WorkerSlot]()
    var state = UnsafePointer(to=slot[].signal)
    while True:
        var signal = _Mailbox.load[ordering = Consistency.ACQUIRE](state)
        if signal == _READY or signal == _JOINING:
            slot[].fn_ptr(slot[].argument, slot[].lane)
            if _mailbox_replace(state, _IDLE) == _JOINING:
                _mailbox_wake(state)
        elif signal == _STOP:
            return context
        elif signal == _PARKED or _Mailbox.compare_exchange(state, signal, _PARKED):
            _mailbox_wait(state, _PARKED)


struct _WorkerPool(Movable):
    var slots: UnsafePointer[_WorkerSlot]
    var handles: List[UInt]

    fn __init__(out self):
        self.slots = UnsafePointer[_WorkerSlot]()
        self.handles = []

    fn __del__(deinit self):
        self._shutdown()

    fn _shutdown(self):
        for i in range(len(self.handles)):
            var state = UnsafePointer(to=(self.slots + i)[].signal)
            if _mailbox_replace(state, _STOP) == _PARKED:
                _mailbox_wake(state)
        for i in range(len(self.handles)):
            var handle = self.handles[i]
            if handle != 0:
                _ = external_call["pthread_join", Int32](handle, Int(0))
            (self.slots + i).destroy_pointee()
        if len(self.handles) > 0:
            self.slots.free()

    fn start(mut self, threads: Int) raises:
        if threads <= 1 or len(self.handles) > 0:
            return
        var count = threads - 1
        self.slots = UnsafePointer[_WorkerSlot].alloc(count)
        self.handles.resize(count, UInt(0))
        var slots = self.slots
        var handles = self.handles.unsafe_ptr()
        for i in range(count):
            (slots + i).init_pointee_move(_WorkerSlot(i + 1))

        @parameter
        fn launch(i: Int):
            var code = external_call["pthread_create", Int32](
                handles + i,
                Int(0),
                _worker_loop,
                (slots + i).bitcast[UInt8](),
            )
            if code != 0:
                handles[i] = 0

        parallelize[launch](count, count)
        for i in range(count):
            if handles[i] == 0:
                self._shutdown()
                self.handles.clear()
                self.slots = UnsafePointer[_WorkerSlot]()
                raise Error("could not start sparse LDLT worker thread")


fn _dispatch(
    slots: UnsafePointer[_WorkerSlot],
    workers: Int,
    work: _TaskFn,
    argument: UnsafePointer[UInt8],
):
    for i in range(workers - 1):
        var slot = slots + i
        slot[].fn_ptr = work
        slot[].argument = argument
        var state = UnsafePointer(to=slot[].signal)
        if _mailbox_replace(state, _READY) == _PARKED:
            _mailbox_wake(state)
    work(argument, 0)
    for i in range(workers - 1):
        var state = UnsafePointer(to=(slots + i)[].signal)
        var expected = _READY
        if _Mailbox.compare_exchange(state, expected, _JOINING):
            while _Mailbox.load[ordering = Consistency.ACQUIRE](state) == _JOINING:
                _mailbox_wait(state, _JOINING)


fn _solve_panel_rows(
    factor: UnsafePointer[Float64],
    col_start: UnsafePointer[Int],
    first: Int,
    stop: Int,
    row_begin: Int,
    row_end: Int,
):
    for row in range(row_begin, row_end, _CACHE_ROWS):
        var height = min(_CACHE_ROWS, row_end - row)
        for j in range(first, stop):
            var diag = col_start[j]
            var pivot = factor[diag]
            var a = factor + diag + row - j
            _scale_column(a, height, 1.0 / pivot)
            for k in range(j + 1, stop):
                var b = factor + col_start[k] + row - k
                _update_columns[1](
                    b,
                    a,
                    a,
                    a,
                    a,
                    height,
                    -pivot * factor[diag + k - j],
                    0.0,
                    0.0,
                    0.0,
                )


@fieldwise_init
struct _PanelBlockWork(ImplicitlyCopyable):
    var factor: UnsafePointer[Float64]
    var col_start: UnsafePointer[Int]
    var scratch: UnsafePointer[Float64]
    var first: Int
    var stop: Int
    var rows: Int
    var trailing: Int
    var workers: Int


fn _parallel_panel_solve(context: UnsafePointer[UInt8], lane: Int):
    var work = context.bitcast[_PanelBlockWork]()[]
    _solve_panel_rows(
        work.factor,
        work.col_start,
        work.first,
        work.stop,
        work.stop + work.rows * lane // work.workers,
        work.stop + work.rows * (lane + 1) // work.workers,
    )


fn _parallel_trailing_update(context: UnsafePointer[UInt8], lane: Int):
    var work = context.bitcast[_PanelBlockWork]()[]
    _blocked_update[False](
        work.factor,
        work.col_start,
        work.first,
        work.stop,
        work.stop,
        work.stop,
        work.col_start.bitcast[Int32](),
        work.col_start,
        work.trailing * lane // work.workers,
        work.trailing * (lane + 1) // work.workers,
        0,
        work.rows,
        work.scratch + lane * _PACK_WORK,
    )


fn _factor_blocked_panel(
    factor: UnsafePointer[Float64],
    col_start: UnsafePointer[Int],
    scale: UnsafePointer[Float64],
    start: Int,
    end: Int,
    row_end: Int,
    scratch: UnsafePointer[Float64],
    threads: Int,
    slots: UnsafePointer[_WorkerSlot],
) -> Int:
    # Factor a small diagonal block, solve its rectangular panel in row
    # tiles, then update the remaining columns with the same GEMM kernel
    # used between supernodes. Encode a failed pivot as -(negatives+1).
    var negative = 0
    for first in range(start, end, _PIVOT_BLOCK):
        var stop = min(end, first + _PIVOT_BLOCK)
        for j in range(first, stop):
            var diag = col_start[j]
            var pivot = factor[diag]
            if not _usable_pivot(pivot, scale[j]):
                return -negative - 1
            negative += Int(pivot < 0.0)
            _scale_column(factor + diag + 1, stop - j - 1, 1.0 / pivot)
            for k in range(j + 1, stop):
                var a = factor + diag + k - j
                _update_columns[1](
                    factor + col_start[k],
                    a,
                    a,
                    a,
                    a,
                    stop - k,
                    -pivot * a[0],
                    0.0,
                    0.0,
                    0.0,
                )
        var rows = row_end - stop
        var width = stop - first
        var trailing = end - stop
        var solve_lanes = min(threads, rows * width * width // _WORK_PER_LANE)
        if solve_lanes > 1:
            var work = _PanelBlockWork(
                factor, col_start, scratch, first, stop, rows, trailing, solve_lanes
            )
            _dispatch(
                slots,
                solve_lanes,
                _parallel_panel_solve,
                UnsafePointer(to=work).bitcast[UInt8](),
            )
        else:
            _solve_panel_rows(factor, col_start, first, stop, stop, row_end)
        if trailing == 0:
            continue
        var workers = min(
            threads,
            min(
                (trailing + _TILE_COLS - 1) // _TILE_COLS,
                2 * rows * width * trailing // _WORK_PER_LANE,
            ),
        )
        if workers > 1:
            var work = _PanelBlockWork(
                factor, col_start, scratch, first, stop, rows, trailing, workers
            )
            _dispatch(
                slots,
                workers,
                _parallel_trailing_update,
                UnsafePointer(to=work).bitcast[UInt8](),
            )
        else:
            _blocked_update[False](
                factor,
                col_start,
                first,
                stop,
                stop,
                stop,
                col_start.bitcast[Int32](),
                col_start,
                0,
                trailing,
                0,
                rows,
                scratch,
            )
    return negative


struct _PanelUpdate(ImplicitlyCopyable, Movable):
    var source: Int
    var first: Int
    var split: Int

    fn __init__(out self, source: Int, first: Int, split: Int):
        self.source = source
        self.first = first
        self.split = split


struct _PanelJob(ImplicitlyCopyable, Movable):
    var first: Int
    var end: Int
    var lane: Int

    fn __init__(out self, first: Int, end: Int, lane: Int):
        self.first = first
        self.end = end
        self.lane = lane


@always_inline
fn _row_bound(
    rows: UnsafePointer[Int32],
    local: UnsafePointer[Int],
    first: Int,
    end: Int,
    row: Int,
) -> Int:
    var lo = first
    var hi = end
    while lo < hi:
        var mid = (lo + hi) // 2
        if local[Int(rows[mid])] < row:
            lo = mid + 1
        else:
            hi = mid
    return lo


alias _MIN_PARALLEL_WORK = 320_000
alias _WORK_PER_LANE = 600_000


@always_inline
fn _incoming_lanes(threads: Int, height: Int, work: Int) -> Int:
    return max(1, min(threads, min(height // 64, work // _WORK_PER_LANE)))


@fieldwise_init
struct _NumericPanels(ImplicitlyCopyable):
    # These borrowed pointers live for the duration of factorize. A task
    # writes only its panel or columns, plus its worker's scratch lane.
    var panels: UnsafePointer[Int]
    var row_ptr: UnsafePointer[Int]
    var rows: UnsafePointer[Int32]
    var col_start: UnsafePointer[Int]
    var matrix_ptr: UnsafePointer[Int]
    var matrix_input: UnsafePointer[UInt32]
    var wide_matrix_input: UnsafePointer[Int]
    var scatter: UnsafePointer[UInt32]
    var wide_scatter: UnsafePointer[Int]
    var compact_input: Bool
    var compact_scatter: Bool
    var updates_ptr: UnsafePointer[Int]
    var updates: UnsafePointer[_PanelUpdate]
    var work: UnsafePointer[Int]
    var cuts: UnsafePointer[Int32]
    var lane_stride: Int
    var factor: UnsafePointer[Float64]
    var scale: UnsafePointer[Float64]
    var values: UnsafePointer[Float64]
    var indices: UnsafePointer[Int]
    var pack: UnsafePointer[Float64]
    var n: Int
    var max_front: Int
    var slots: UnsafePointer[_WorkerSlot]

    fn factor_panel(self, panel: Int, lane: Int, threads: Int) -> Int:
        var start = self.panels[panel]
        var end = self.panels[panel + 1]
        var row_begin = self.row_ptr[panel]
        var row_end = self.row_ptr[panel + 1]
        var height = end - start + row_end - row_begin
        var factor = self.factor
        var col_start = self.col_start
        memset_zero(factor + col_start[start], col_start[end] - col_start[start])
        if self.compact_input and self.compact_scatter:
            for p in range(self.matrix_ptr[start], self.matrix_ptr[end]):
                factor[Int(self.scatter[p])] = self.values[Int(self.matrix_input[p])]
        else:
            for p in range(self.matrix_ptr[start], self.matrix_ptr[end]):
                var dst = Int(self.scatter[p]) if self.compact_scatter else self.wide_scatter[p]
                var src = Int(
                    self.matrix_input[p]
                ) if self.compact_input else self.wide_matrix_input[p]
                factor[dst] = self.values[src]

        var local = self.indices + lane * (self.n + 2 * self.max_front)
        for j in range(start, end):
            local[j] = j - start
        for p in range(row_begin, row_end):
            local[Int(self.rows[p])] = end - start + p - row_begin

        # Incoming panels update disjoint row tiles, including when the
        # destination has only a few pivot columns and a tall rectangle.
        var workers = _incoming_lanes(threads, height, self.work[panel])
        if (
            workers > 1
            and self.work[panel] >= _MIN_PARALLEL_WORK
            and self.updates_ptr[panel] < self.updates_ptr[panel + 1]
        ):
            var work = _IncomingWork(
                self, local, panel, start, self.cuts + panel * self.lane_stride
            )
            _dispatch(
                self.slots,
                workers,
                _parallel_incoming_updates,
                UnsafePointer(to=work).bitcast[UInt8](),
            )
        else:
            self.apply_updates(panel, start, 0, height, local, lane)

        if end - start >= 24 and height >= 48:
            return _factor_blocked_panel(
                factor,
                col_start,
                self.scale,
                start,
                end,
                start + height,
                self.pack + lane * _PACK_WORK,
                threads,
                self.slots,
            )

        var negative = 0
        for j in range(start, end):
            var diag = col_start[j]
            var pivot = factor[diag]
            if not _usable_pivot(pivot, self.scale[j]):
                return -negative - 1
            negative += Int(pivot < 0.0)
            _scale_column(factor + diag + 1, height - (j - start) - 1, 1.0 / pivot)
            for k in range(j + 1, end):
                var a = factor + diag + k - j
                _update_columns[1](
                    factor + col_start[k],
                    a,
                    a,
                    a,
                    a,
                    height - (k - start),
                    -pivot * a[0],
                    0.0,
                    0.0,
                    0.0,
                )
        return negative

    fn apply_updates(
        self,
        panel: Int,
        target_start: Int,
        lo: Int,
        hi: Int,
        local: UnsafePointer[Int],
        lane: Int,
    ):
        var mapped = self.indices + lane * (self.n + 2 * self.max_front) + self.n
        var runs = mapped + self.max_front
        var scratch = self.pack + lane * _PACK_WORK
        for i in range(self.updates_ptr[panel], self.updates_ptr[panel + 1]):
            var update = self.updates[i]
            var first = update.first
            var limit = self.row_ptr[update.source + 1]
            if local[Int(self.rows[first])] >= hi or local[Int(self.rows[limit - 1])] < lo:
                continue
            var row_begin = first
            var row_end = limit
            if local[Int(self.rows[first])] < lo:
                row_begin = _row_bound(self.rows, local, first, limit, lo)
            if local[Int(self.rows[limit - 1])] >= hi:
                row_end = _row_bound(self.rows, local, row_begin, limit, hi)
            self.apply_update(
                update.source,
                target_start,
                first,
                min(update.split, row_end) - first,
                row_begin - first,
                row_end - first,
                local,
                mapped,
                runs,
                scratch,
            )

    fn apply_update(
        self,
        source: Int,
        target_start: Int,
        first: Int,
        overlap: Int,
        row_begin: Int,
        row_end: Int,
        local: UnsafePointer[Int],
        mapped: UnsafePointer[Int],
        runs: UnsafePointer[Int],
        scratch: UnsafePointer[Float64],
    ):
        var factor = self.factor
        var col_start = self.col_start
        var rows = self.rows
        var start = self.panels[source]
        var end = self.panels[source + 1]
        var external = end + first - self.row_ptr[source]
        var rank = end - start
        if rank >= 6 and overlap >= _TILE_COLS and row_end - row_begin >= 48:
            for p in range(row_begin, row_end):
                mapped[p] = local[Int(rows[first + p])]
            _blocked_update[True](
                factor,
                col_start,
                start,
                end,
                external,
                target_start,
                rows + first,
                mapped,
                0,
                overlap,
                row_begin,
                row_end,
                scratch,
            )
            return

        var run_end = row_end
        for p in range(row_end - 1, row_begin - 1, -1):
            var row = local[Int(rows[first + p])]
            if p + 1 < row_end and row + 1 != mapped[p + 1]:
                run_end = p + 1
            mapped[p] = row
            runs[p] = run_end
        for p in range(overlap):
            var j = Int(rows[first + p])
            var dst = factor + col_start[j] + target_start - j
            var k = start
            while k + 4 <= end:
                var a = factor + col_start[k] + external - k
                var b = factor + col_start[k + 1] + external - (k + 1)
                var c = factor + col_start[k + 2] + external - (k + 2)
                var d = factor + col_start[k + 3] + external - (k + 3)
                _update_runs[4](
                    dst,
                    a,
                    b,
                    c,
                    d,
                    mapped,
                    runs,
                    max(p, row_begin),
                    row_end,
                    -factor[col_start[k]] * a[p],
                    -factor[col_start[k + 1]] * b[p],
                    -factor[col_start[k + 2]] * c[p],
                    -factor[col_start[k + 3]] * d[p],
                )
                k += 4
            while k < end:
                var a = factor + col_start[k] + external - k
                _update_runs[1](
                    dst,
                    a,
                    a,
                    a,
                    a,
                    mapped,
                    runs,
                    max(p, row_begin),
                    row_end,
                    -factor[col_start[k]] * a[p],
                    0.0,
                    0.0,
                    0.0,
                )
                k += 1


@fieldwise_init
struct _IncomingWork(ImplicitlyCopyable):
    var view: _NumericPanels
    var local: UnsafePointer[Int]
    var panel: Int
    var start: Int
    var cuts: UnsafePointer[Int32]


fn _parallel_incoming_updates(context: UnsafePointer[UInt8], lane: Int):
    var work = context.bitcast[_IncomingWork]()[]
    work.view.apply_updates(
        work.panel,
        work.start,
        Int(work.cuts[lane]),
        Int(work.cuts[lane + 1]),
        work.local,
        lane,
    )


@fieldwise_init
struct _SubtreeWork(ImplicitlyCopyable):
    var view: _NumericPanels
    var lane_ptr: UnsafePointer[Int]
    var jobs: UnsafePointer[_PanelJob]
    var post: UnsafePointer[Int]
    var status: UnsafePointer[Int]


fn _parallel_subtrees(context: UnsafePointer[UInt8], lane: Int):
    var work = context.bitcast[_SubtreeWork]()[]
    var negative = 0
    for j in range(work.lane_ptr[lane], work.lane_ptr[lane + 1]):
        for k in range(work.jobs[j].first, work.jobs[j].end):
            var result = work.view.factor_panel(work.post[k], lane, 1)
            if result < 0:
                work.status[lane] = result - negative
                return
            negative += result
    work.status[lane] = negative


struct SparseLDLT(Movable):
    var _order: List[Int]
    var _parent: List[Int]
    var _upper: _Graph
    var _upper_input: _Offsets
    var _input_ptr: List[Int]
    var _scale_row: List[Int32]
    var _inverse: List[Int]
    var _matrix_ptr: List[Int]
    var _matrix_input: _Offsets
    var _scatter: _Offsets
    var _pack: List[Float64]
    var _factor: List[Float64]
    var _col_start: List[Int]
    var _panel_of: List[Int]
    var _panel_start: List[Int]
    var _panel_row_ptr: List[Int]
    var _panel_rows: List[Int32]
    var _work: List[Float64]
    var _solve_tail: List[Float64]
    var _scale: List[Float64]
    var _cursor: List[Int]
    var _mark: List[Int]
    var _reach: List[Int]
    var _threads: Int
    var _active_threads: Int
    var _pool: _WorkerPool
    var _indices: List[Int]
    var _updates_ptr: List[Int]
    var _updates: List[_PanelUpdate]
    var _panel_work: List[Int]
    var _row_cuts: List[Int32]
    var _panel_post: List[Int]
    var _jobs: List[_PanelJob]
    var _remaining: List[Int]
    var _lane_ptr: List[Int]
    var _lane_result: List[Int]
    var _max_panel: Int
    var _max_front: Int
    var _flops: Int
    var _nnz: Int
    var _negative: Int
    var _mode: Int
    var _analyzed: Bool
    var _valid: Bool

    fn __init__(out self, threads: Int = 0):
        self._threads = max(1, num_performance_cores() if threads == 0 else threads)
        self._active_threads = 1
        self._pool = _WorkerPool()
        self._order = []
        self._parent = []
        self._upper = _Graph(0, 0)
        self._upper_input = _Offsets()
        self._input_ptr = []
        self._scale_row = []
        self._inverse = []
        self._matrix_ptr = []
        self._matrix_input = _Offsets()
        self._scatter = _Offsets()
        self._pack = []
        self._factor = []
        self._col_start = []
        self._panel_of = []
        self._panel_start = []
        self._panel_row_ptr = []
        self._panel_rows = []
        self._work = []
        self._solve_tail = []
        self._scale = []
        self._cursor = []
        self._mark = []
        self._reach = []
        self._indices = []
        self._updates_ptr = []
        self._updates = []
        self._panel_work = []
        self._row_cuts = []
        self._panel_post = []
        self._jobs = []
        self._remaining = []
        self._lane_ptr = []
        self._lane_result = []
        self._max_panel = 0
        self._max_front = 0
        self._flops = 0
        self._nnz = 0
        self._negative = 0
        self._mode = SparseFactorMode.Supernodal
        self._analyzed = False
        self._valid = False

    fn factor_nnz(self) -> Int:
        """Number of structural entries in L, including its unit diagonal."""
        return self._nnz

    fn supernode_count(self) -> Int:
        """Number of stored panels, including any relaxed amalgamation."""
        return max(0, len(self._panel_start) - 1)

    fn negative_pivots(self) -> Int:
        """Number of negative pivots accepted in the last numeric attempt."""
        return self._negative

    fn factor_flops(self) -> Int:
        """Structural LDLᵀ update and scaling operations from the exact column counts."""
        return self._flops

    fn largest_supernode(self) -> Int:
        """Maximum number of pivot columns in a stored panel."""
        return self._max_panel

    fn largest_front(self) -> Int:
        """Maximum panel height including the external rows."""
        return self._max_front

    fn analyze(
        mut self,
        n: Int,
        col_ptr: List[Int],
        row_idx: List[Int],
        ordering: Int = SparseOrdering.Auto,
        mode: Int = SparseFactorMode.Supernodal,
        permutation: List[Int] = [],
        dense_multiplier: Int = 8,
    ) raises:
        """Analyze and allocate for a lower CSC pattern. Invalid input raises."""
        _validate_symbolic_input(n, col_ptr, row_idx, ordering, dense_multiplier)
        if mode != SparseFactorMode.Supernodal and mode != SparseFactorMode.Simplicial:
            raise Error("unknown sparse LDLT factorization mode")

        var custom_order = len(permutation) > 0
        if custom_order:
            if len(permutation) != n:
                raise Error("sparse LDLT permutation length does not match the dimension")
            var seen = _ints(n)
            for j in range(n):
                var col = permutation[j]
                if col < 0 or col >= n or seen[col] != 0:
                    raise Error("invalid sparse LDLT permutation")
                seen[col] = 1

        self._valid = False
        self._analyzed = False
        self._negative = 0
        self._mode = mode
        var graph = _adjacency(n, col_ptr, row_idx)
        var symbolic = _choose_symbolic(
            graph, len(row_idx), ordering, permutation, dense_multiplier
        )
        symbolic.apply_postorder()
        self._order = symbolic.order.copy()
        self._inverse = _ints(n)
        for j in range(n):
            self._inverse[self._order[j]] = j
        self._input_ptr = col_ptr.copy()
        self._scale_row.resize(len(row_idx), Int32(0))
        for p in range(len(row_idx)):
            self._scale_row[p] = Int32(self._inverse[row_idx[p]])
        self._parent = symbolic.parent.copy()
        self._nnz = symbolic.nnz
        self._flops = symbolic.flops
        self._build_panels(graph, symbolic.counts)
        graph.rows = []
        self._install_pattern(col_ptr, row_idx)

        self._work = _floats(n)
        self._solve_tail.resize(self._max_front, 0.0)
        self._scale = _floats(n)
        self._cursor = _ints(n)
        self._mark = _ints(n)
        self._reach = _ints(n)
        self._active_threads = 1
        if mode == SparseFactorMode.Supernodal:
            if n >= 256 and self._flops >= _MIN_PARALLEL_WORK:
                self._active_threads = self._threads
            self._schedule_panels()
            self._pool.start(self._active_threads)
        self._factor.resize(self._col_start[n], 0.0)
        self._analyzed = True

    fn _install_pattern(mut self, col_ptr: List[Int], row_idx: List[Int]):
        var simple = self._mode == SparseFactorMode.Simplicial
        var pattern = _PermutedTriangle(self._inverse, col_ptr, row_idx, simple)
        if simple:
            self._upper.ptr = pattern.graph.ptr.copy()
            self._upper.rows = pattern.graph.rows.copy()
            self._upper_input = pattern.input.copy()
            self._matrix_ptr = []
            self._matrix_input = _Offsets()
            self._scatter = _Offsets()
            return
        self._upper.ptr = _ints(1)
        self._upper.rows = []
        self._upper_input = _Offsets()
        self._matrix_ptr = pattern.graph.ptr.copy()
        self._matrix_input = pattern.input.copy()
        self._scatter = _Offsets(len(row_idx), self._col_start[len(self._order)])
        var local = _ints(len(self._order))
        for s in range(self.supernode_count()):
            var start = self._panel_start[s]
            var end = self._panel_start[s + 1]
            for j in range(start, end):
                local[j] = j - start
            var first_row = self._panel_row_ptr[s]
            for p in range(first_row, self._panel_row_ptr[s + 1]):
                local[Int(self._panel_rows[p])] = end - start + p - first_row
            for j in range(start, end):
                var base = self._col_start[j] - (j - start)
                for p in range(self._matrix_ptr[j], self._matrix_ptr[j + 1]):
                    self._scatter.set(p, base + local[Int(pattern.graph.rows[p])])

    fn _build_panels(mut self, graph: _Graph, counts: List[Int]):
        var n = len(counts)
        self._panel_start = []
        self._panel_row_ptr = []
        self._panel_of = _ints(n)
        self._col_start = _ints(n + 1)
        var pos = 0
        var row_count = 0
        var start = 0
        self._max_front = 0
        self._max_panel = 0
        while start < n:
            var panel = len(self._panel_start)
            self._panel_start.append(start)
            self._panel_row_ptr.append(row_count)
            var end = start + 1
            var entries = counts[start]
            # Along a consecutive etree chain, each column's external
            # rows are contained in its parent's. The last column supplies
            # the shared row list even when we allow a little padding.
            while end < n and end - start < _MAX_PANEL_COLS and self._parent[end - 1] == end:
                var width = end - start + 1
                var exact = entries + counts[end]
                var packed = width * (counts[end] - 1) + width * (width + 1) // 2
                var padding = packed - exact
                if padding > 0 and (
                    self._mode == SparseFactorMode.Simplicial or padding * 12 > exact
                ):
                    break
                entries = exact
                end += 1
            var external = counts[end - 1] - 1
            self._max_panel = max(self._max_panel, end - start)
            self._max_front = max(self._max_front, end - start + external)
            row_count += external
            for j in range(start, end):
                self._col_start[j] = pos
                self._panel_of[j] = panel
                pos += end - j + external
            start = end
        self._col_start[n] = pos
        self._panel_start.append(n)
        self._panel_row_ptr.append(row_count)
        self._panel_rows = List[Int32](length=row_count, fill=Int32(0))

        # Row-wise symbolic reach on the contracted etree. Visit only the
        # external row list, once per supernode regardless of its width.
        var mark = _ints(self.supernode_count(), -1)
        var next_row = self._panel_row_ptr.copy()
        var panel_parent = _ints(self.supernode_count(), -1)
        for s in range(self.supernode_count()):
            var p = self._parent[self._panel_start[s + 1] - 1]
            if p >= 0:
                panel_parent[s] = self._panel_of[p]
        var panels = self._panel_of.unsafe_ptr()
        var starts = graph.ptr.unsafe_ptr()
        var rows = graph.rows.unsafe_ptr()
        var stamps = mark.unsafe_ptr()
        var cursor = next_row.unsafe_ptr()
        var parents = panel_parent.unsafe_ptr()
        var panel_rows = self._panel_rows.unsafe_ptr()
        var row_ptr = self._panel_row_ptr.unsafe_ptr()
        for i in range(n):
            var target = panels[i]
            var original = self._order[i]
            for p in range(starts[original], starts[original + 1]):
                var s = panels[self._inverse[Int(rows[p])]]
                if s >= target:
                    continue
                while s != target:
                    debug_assert(s >= 0 and s < target, "invalid sparse LDLT panel reach")
                    if stamps[s] == i:
                        break
                    stamps[s] = i
                    debug_assert(cursor[s] < row_ptr[s + 1], "sparse LDLT column count too small")
                    panel_rows[cursor[s]] = Int32(i)
                    cursor[s] += 1
                    s = parents[s]
        for s in range(self.supernode_count()):
            debug_assert(cursor[s] == row_ptr[s + 1], "sparse LDLT column count too large")

    fn factorize(mut self, values: List[Float64]) -> Bool:
        """Replace the numeric factor. Return False if a pivot is unsafe."""
        self._negative = 0
        self._valid = False
        if not self._analyzed or len(values) != len(self._scale_row):
            return False
        var n = len(self._order)
        var scale = self._scale.unsafe_ptr()
        for i in range(n):
            scale[i] = 0.0
        var starts = self._input_ptr.unsafe_ptr()
        var rows = self._scale_row.unsafe_ptr()
        var inverse = self._inverse.unsafe_ptr()
        var input_values = values.unsafe_ptr()
        for column in range(n):
            var i = inverse[column]
            var largest = scale[i]
            for p in range(starts[column], starts[column + 1]):
                var magnitude = abs(input_values[p])
                if not isfinite(magnitude):
                    return False
                var j = Int(rows[p])
                largest = max(largest, magnitude)
                scale[j] = max(scale[j], magnitude)
            scale[i] = largest
        if self._mode == SparseFactorMode.Simplicial:
            self._valid = self._factorize_simplicial(values)
        else:
            self._valid = self._factorize_supernodal(values)
        return self._valid

    @always_inline
    fn _accept_pivot(mut self, k: Int, pivot: Float64) -> Bool:
        var scale = self._scale.unsafe_ptr()[k]
        if not _usable_pivot(pivot, scale):
            return False
        self._negative += Int(pivot < 0.0)
        return True

    fn _factorize_simplicial(mut self, values: List[Float64]) -> Bool:
        # In up-looking LDLᵀ, etree reach gives the topological order in which
        # the previously built column prefixes update the current row.
        var n = len(self._order)
        var starts = self._upper.ptr.unsafe_ptr()
        var rows = self._upper.rows.unsafe_ptr()
        var input_values = values.unsafe_ptr()
        var col_start = self._col_start.unsafe_ptr()
        var parent = self._parent.unsafe_ptr()
        var panel_of = self._panel_of.unsafe_ptr()
        var panel_start = self._panel_start.unsafe_ptr()
        var panel_row_ptr = self._panel_row_ptr.unsafe_ptr()
        var panel_rows = self._panel_rows.unsafe_ptr()
        var work = self._work.unsafe_ptr()
        var factor = self._factor.unsafe_ptr()
        var cursor = self._cursor.unsafe_ptr()
        var mark = self._mark.unsafe_ptr()
        var reach = self._reach.unsafe_ptr()
        for i in range(n):
            work[i] = 0.0
            cursor[i] = col_start[i] + 1
            mark[i] = -1
        for k in range(n):
            var top = n
            mark[k] = k
            for p in range(starts[k], starts[k + 1]):
                var i = Int(rows[p])
                work[i] = input_values[self._upper_input[p]]
                var length = 0
                while mark[i] != k:
                    reach[length] = i
                    length += 1
                    mark[i] = k
                    i = parent[i]
                while length > 0:
                    length -= 1
                    top -= 1
                    reach[top] = reach[length]
            var pivot = work[k]
            work[k] = 0.0
            for p in range(top, n):
                var j = reach[p]
                var y = work[j]
                work[j] = 0.0
                var diag = col_start[j]
                var s = panel_of[j]
                var first = diag + 1
                var external = diag + panel_start[s + 1] - j
                var end = cursor[j]
                for q in range(first, min(end, external)):
                    work[j + q - diag] -= factor[q] * y
                for q in range(external, end):
                    work[Int(panel_rows[panel_row_ptr[s] + q - external])] -= factor[q] * y
                var lij = y / factor[diag]
                factor[end] = lij
                cursor[j] = end + 1
                pivot -= lij * y
            if not self._accept_pivot(k, pivot):
                return False
            factor[col_start[k]] = pivot
        return True

    fn _schedule_panels(mut self):
        # An update contributes to exactly the ancestor panels containing
        # its source's external rows. Store their inputs in CSC-like order.
        var count = self.supernode_count()
        self._updates_ptr = _ints(count + 1)
        self._panel_work = _ints(count)
        var parents = _ints(count, -1)
        var panels = self._panel_start.unsafe_ptr()
        var panel_of = self._panel_of.unsafe_ptr()
        var row_ptr = self._panel_row_ptr.unsafe_ptr()
        var rows = self._panel_rows.unsafe_ptr()
        var work = self._panel_work.unsafe_ptr()
        for s in range(count):
            var width = panels[s + 1] - panels[s]
            var height = width + row_ptr[s + 1] - row_ptr[s]
            for j in range(width):
                var tail = height - j - 1
                var cols = width - j - 1
                work[s] += tail + cols * (2 * tail - cols + 1)
            var first = row_ptr[s]
            var limit = row_ptr[s + 1]
            if first < limit:
                parents[s] = panel_of[Int(rows[first])]
            while first < limit:
                var target = panel_of[Int(rows[first])]
                var split = first + 1
                while split < limit and Int(rows[split]) < panels[target + 1]:
                    split += 1
                self._updates_ptr[target + 1] += 1
                var cols = split - first
                work[target] += width * cols * (2 * (limit - first) - cols + 1)
                first = split
        for s in range(count):
            self._updates_ptr[s + 1] += self._updates_ptr[s]
        self._updates.resize(self._updates_ptr[count], _PanelUpdate(0, 0, 0))
        var cursor = self._updates_ptr.copy()
        for s in range(count - 1, -1, -1):
            var first = row_ptr[s]
            var limit = row_ptr[s + 1]
            while first < limit:
                var target = panel_of[Int(rows[first])]
                var split = first + 1
                while split < limit and Int(rows[split]) < panels[target + 1]:
                    split += 1
                self._updates[cursor[target]] = _PanelUpdate(s, first, split)
                cursor[target] += 1
                first = split
        self._indices.resize(self._active_threads * (len(self._order) + 2 * self._max_front), 0)
        self._pack.resize(self._active_threads * _PACK_WORK, 0.0)
        self._remaining = []
        self._lane_ptr = _ints(self._active_threads + 1)
        self._lane_result = _ints(self._active_threads)
        self._jobs = []
        self._panel_post = []
        if self._active_threads == 1:
            return

        # Cut the etree into independent, bounded-work subtrees. The
        # ancestor panels left above the cut may use all workers per front.
        self._panel_post = _postorder(parents)
        var subtree_work = self._panel_work.copy()
        var subtree_start = _ints(count)
        var total = 0
        for k in range(count):
            subtree_start[self._panel_post[k]] = k
            total += work[k]
        for s in range(count):
            var p = parents[s]
            if p >= 0:
                subtree_start[p] = min(subtree_start[p], subtree_start[s])
                subtree_work[p] += subtree_work[s]
        var budget = max(400_000, min(80_000_000, total // (self._active_threads * 16)))
        var loads = _ints(self._active_threads)
        var jobs: List[_PanelJob] = []
        for k in range(count):
            var s = self._panel_post[k]
            if subtree_work[s] > budget:
                self._remaining.append(s)
            else:
                var p = parents[s]
                if p >= 0 and subtree_work[p] <= budget:
                    continue
                var lane = 0
                for i in range(1, self._active_threads):
                    if loads[i] < loads[lane]:
                        lane = i
                loads[lane] += subtree_work[s] + 128 * (k + 1 - subtree_start[s])
                jobs.append(_PanelJob(subtree_start[s], k + 1, lane))
                self._lane_ptr[lane + 1] += 1
        for i in range(self._active_threads):
            self._lane_ptr[i + 1] += self._lane_ptr[i]
        self._jobs.resize(len(jobs), _PanelJob(0, 0, 0))
        var next_job = self._lane_ptr.copy()
        for j in range(len(jobs)):
            var job = jobs[j]
            self._jobs[next_job[job.lane]] = job
            next_job[job.lane] += 1

        self._build_row_cuts()

    fn _build_row_cuts(mut self):
        # A row split by height can give one lane most of the work when
        # several source panels begin high in the destination front.
        # Use the exact triangular update count below each cut instead.
        var stride = self._active_threads + 1
        self._row_cuts = List[Int32](length=self.supernode_count() * stride, fill=Int32(0))
        var local_array = _ints(len(self._order))
        var local = local_array.unsafe_ptr()
        var rows = self._panel_rows.unsafe_ptr()
        var row_ptr = self._panel_row_ptr.unsafe_ptr()
        var panels = self._panel_start.unsafe_ptr()
        var updates = self._updates.unsafe_ptr()
        for k in range(len(self._remaining)):
            var s = self._remaining[k]
            var start = panels[s]
            var end = panels[s + 1]
            var width = end - start
            var height = width + row_ptr[s + 1] - row_ptr[s]
            var first_update = self._updates_ptr[s]
            var end_update = self._updates_ptr[s + 1]
            var lanes = _incoming_lanes(self._active_threads, height, self._panel_work[s])
            if lanes == 1 or first_update == end_update:
                continue
            for j in range(start, end):
                local[j] = j - start
            for j in range(row_ptr[s], row_ptr[s + 1]):
                local[Int(rows[j])] = width + j - row_ptr[s]
            var total = 0
            for i in range(first_update, end_update):
                var update = updates[i]
                var rank = panels[update.source + 1] - panels[update.source]
                var overlap = update.split - update.first
                total += (
                    rank * overlap * (2 * (row_ptr[update.source + 1] - update.first) - overlap + 1)
                )

            var base = s * stride
            var last = 0
            for lane in range(1, lanes):
                var goal = total * lane // lanes
                var lo = last
                var hi = height
                while lo < hi:
                    var mid = (lo + hi) // 2
                    var subtotal = 0
                    for i in range(first_update, end_update):
                        var update = updates[i]
                        var upto = (
                            _row_bound(
                                rows,
                                local,
                                update.first,
                                row_ptr[update.source + 1],
                                mid,
                            )
                            - update.first
                        )
                        var cols = min(upto, update.split - update.first)
                        var rank = panels[update.source + 1] - panels[update.source]
                        subtotal += rank * cols * (2 * upto - cols + 1)
                    if subtotal < goal:
                        lo = mid + 1
                    else:
                        hi = mid
                self._row_cuts[base + lane] = Int32(lo)
                last = lo
            self._row_cuts[base + lanes] = Int32(height)

    fn _factorize_supernodal(mut self, values: List[Float64]) -> Bool:
        var view = _NumericPanels(
            self._panel_start.unsafe_ptr(),
            self._panel_row_ptr.unsafe_ptr(),
            self._panel_rows.unsafe_ptr(),
            self._col_start.unsafe_ptr(),
            self._matrix_ptr.unsafe_ptr(),
            self._matrix_input.narrow.unsafe_ptr(),
            self._matrix_input.wide.unsafe_ptr(),
            self._scatter.narrow.unsafe_ptr(),
            self._scatter.wide.unsafe_ptr(),
            self._matrix_input.compact,
            self._scatter.compact,
            self._updates_ptr.unsafe_ptr(),
            self._updates.unsafe_ptr(),
            self._panel_work.unsafe_ptr(),
            self._row_cuts.unsafe_ptr(),
            self._active_threads + 1,
            self._factor.unsafe_ptr(),
            self._scale.unsafe_ptr(),
            values.unsafe_ptr(),
            self._indices.unsafe_ptr(),
            self._pack.unsafe_ptr(),
            len(self._order),
            self._max_front,
            self._pool.slots,
        )
        if self._active_threads == 1:
            for s in range(self.supernode_count()):
                var result = view.factor_panel(s, 0, 1)
                if not self._record_pivots(result):
                    return False
            return True

        var valid = True
        if len(self._jobs) > 0:
            var status = self._lane_result.unsafe_ptr()
            var work = _SubtreeWork(
                view,
                self._lane_ptr.unsafe_ptr(),
                self._jobs.unsafe_ptr(),
                self._panel_post.unsafe_ptr(),
                status,
            )
            _dispatch(
                self._pool.slots,
                self._active_threads,
                _parallel_subtrees,
                UnsafePointer(to=work).bitcast[UInt8](),
            )
            for lane in range(self._active_threads):
                valid = self._record_pivots(status[lane]) and valid
        if not valid:
            return False
        for i in range(len(self._remaining)):
            var result = view.factor_panel(self._remaining[i], 0, self._active_threads)
            if not self._record_pivots(result):
                return False
        return True

    @always_inline
    fn _record_pivots(mut self, result: Int) -> Bool:
        if result < 0:
            self._negative += -result - 1
            return False
        self._negative += result
        return True

    fn solve(mut self, rhs: List[Float64], mut x: List[Float64]):
        """Solve with the last successful factor and pre-sized vectors."""
        var n = len(self._order)
        debug_assert(self._valid and len(rhs) == n and len(x) == n, "invalid sparse LDLT solve")
        var order = self._order.unsafe_ptr()
        var col_start = self._col_start.unsafe_ptr()
        var panels = self._panel_start.unsafe_ptr()
        var row_ptr = self._panel_row_ptr.unsafe_ptr()
        var rows = self._panel_rows.unsafe_ptr()
        var factor = self._factor.unsafe_ptr()
        var b = rhs.unsafe_ptr()
        var output = x.unsafe_ptr()
        var y = self._work.unsafe_ptr()
        var scratch = self._solve_tail.unsafe_ptr()
        for i in range(n):
            y[i] = b[order[i]]
        for s in range(self.supernode_count()):
            var start = panels[s]
            var end = panels[s + 1]
            var start_row = row_ptr[s]
            var count = row_ptr[s + 1] - start_row
            var index = rows + start_row
            var direct = count >= 16 and Int(index[count - 1]) - Int(index[0]) == count - 1
            var buffered = not direct and count >= 24 and end - start >= 4
            var tail = y + Int(index[0]) if direct else scratch
            if buffered:
                for i in range(count):
                    tail[i] = y[Int(index[i])]
            for j in range(start, end):
                var pivot_rhs = y[j]
                var base = col_start[j] - j
                for i in range(j + 1, end):
                    y[i] -= factor[base + i] * pivot_rhs
                var external = factor + base + end
                if direct or buffered:
                    _update_columns[1](
                        tail,
                        external,
                        external,
                        external,
                        external,
                        count,
                        -pivot_rhs,
                        0.0,
                        0.0,
                        0.0,
                    )
                else:
                    for i in range(count):
                        y[Int(index[i])] -= external[i] * pivot_rhs
            if buffered:
                for i in range(count):
                    y[Int(index[i])] = tail[i]

        for s in range(self.supernode_count() - 1, -1, -1):
            var start = panels[s]
            var end = panels[s + 1]
            var start_row = row_ptr[s]
            var count = row_ptr[s + 1] - start_row
            var index = rows + start_row
            var direct = count >= 16 and Int(index[count - 1]) - Int(index[0]) == count - 1
            var buffered = not direct and count >= 24 and end - start >= 4
            var tail = y + Int(index[0]) if direct else scratch
            if buffered:
                for i in range(count):
                    tail[i] = y[Int(index[i])]
            for j in range(end - 1, start - 1, -1):
                var pivot_rhs = y[j] / factor[col_start[j]]
                var base = col_start[j] - j
                for i in range(j + 1, end):
                    pivot_rhs -= factor[base + i] * y[i]
                var external = factor + base + end
                if direct or buffered:
                    pivot_rhs -= _dot_tail(external, tail, count)
                else:
                    for i in range(count):
                        pivot_rhs -= external[i] * y[Int(index[i])]
                y[j] = pivot_rhs
        for i in range(n):
            output[order[i]] = y[i]
