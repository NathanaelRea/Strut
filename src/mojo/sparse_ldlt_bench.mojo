"""Standalone sparse LDLT and banded Cholesky benchmark.

Build with uv run mojo build src/mojo/sparse_ldlt_bench.mojo -o build/sparse_ldlt_bench.
Use --case NAME, --ordering amd|rcm|geometric, or --threads 1|8 for synthetic cases.
Use --pattern PATH for a lower CSC dump, --ordering auto|amd|rcm|nd, and --threads 1|2|4|8 for a single thread
count. With a pattern, --analyze-only 1 skips factorization and --counts-only 1
compares orderings without allocating a factor. --dense-multiplier N sets the
AMD degree-relative cutoff multiplier (default 8).
"""

from math import sqrt
from memory import memset_zero
from pathlib import Path
from sys.arg import argv
from time import perf_counter_ns

from solver.simd_contiguous import FLOAT64_SIMD_WIDTH
from solver.sparse_ldlt import estimate_sparse_ldlt_ordering, SparseLDLT, SparseOrdering


struct SparseBenchMatrix(Movable):
    var n: Int
    var bandwidth: Int
    var col_ptr: List[Int]
    var rows: List[Int]
    var values: List[Float64]

    fn __init__(out self, n: Int, bandwidth: Int, capacity: Int):
        self.n = n
        self.bandwidth = bandwidth
        self.col_ptr = List[Int](capacity=n + 1)
        self.rows = List[Int](capacity=capacity)
        self.values = List[Float64](capacity=capacity)

    fn append(mut self, row: Int, value: Float64):
        self.rows.append(row)
        self.values.append(value)


fn _pattern_integer(text: String, mut cursor: Int) raises -> Int:
    var bytes = text.as_bytes()
    while cursor < len(bytes) and bytes[cursor] <= 32:
        cursor += 1
    var start = cursor
    var result = 0
    while cursor < len(bytes):
        var byte = bytes[cursor]
        if byte < 48 or byte > 57:
            break
        var digit = Int(byte) - 48
        if result > (Int.MAX - digit) // 10:
            raise Error("CSC pattern integer overflow")
        result = 10 * result + digit
        cursor += 1
    if cursor == start or (cursor < len(bytes) and bytes[cursor] > 32):
        raise Error("expected a nonnegative integer in CSC pattern")
    return result


fn _set_pattern_values(mut matrix: SparseBenchMatrix, scaled: Bool):
    var scale = List[Float64](length=matrix.n, fill=1.0)
    if scaled:
        var decade: List[Float64] = [
            1.0e3,
            3.162277660168379e3,
            1.0e4,
            3.162277660168379e4,
            1.0e5,
            3.162277660168379e5,
            1.0e6,
            3.162277660168379e6,
            1.0e7,
        ]
        for i in range(matrix.n):
            scale[i] = decade[(23 * i + i // 9) % 9]
    for p in range(len(matrix.values)):
        matrix.values[p] = 1.0
    for j in range(matrix.n):
        var diag = matrix.col_ptr[j]
        for p in range(diag + 1, matrix.col_ptr[j + 1]):
            var i = matrix.rows[p]
            matrix.values[matrix.col_ptr[i]] += 0.01
            matrix.values[diag] += 0.01
            matrix.values[p] = -0.01 * scale[i] * scale[j]
    for j in range(matrix.n):
        matrix.values[matrix.col_ptr[j]] *= scale[j] * scale[j]


fn load_csc_pattern(path: String, scaled: Bool = False) raises -> SparseBenchMatrix:
    """Load a text lower CSC pattern and assign SPD values."""
    var text = Path(path).read_text()
    var cursor = 0
    var n = _pattern_integer(text, cursor)
    var nnz = _pattern_integer(text, cursor)
    if n > nnz or nnz > len(text.as_bytes()) or n == Int.MAX:
        raise Error("invalid CSC pattern dimensions")
    var matrix = SparseBenchMatrix(n, max(0, n - 1), nnz)
    for _ in range(n + 1):
        matrix.col_ptr.append(_pattern_integer(text, cursor))
    if matrix.col_ptr[0] != 0 or matrix.col_ptr[n] != nnz:
        raise Error("invalid CSC pattern column pointers")
    for j in range(n):
        var start = matrix.col_ptr[j]
        var end = matrix.col_ptr[j + 1]
        if end <= start or end > nnz:
            raise Error("invalid CSC pattern column pointers")
        var previous = j - 1
        for p in range(start, end):
            var i = _pattern_integer(text, cursor)
            if i <= previous or i >= n or (p == start and i != j):
                raise Error("invalid CSC pattern row indices")
            matrix.append(i, 0.0)
            previous = i
    var bytes = text.as_bytes()
    for p in range(cursor, len(bytes)):
        if bytes[p] > 32:
            raise Error("unexpected data after CSC pattern")
    _set_pattern_values(matrix, scaled)
    return matrix^


fn _edge_weight(a: Int, b: Int) -> Float64:
    # Nonuniform weights make permutation and indexing errors visible.
    return 0.25 + 0.0625 * Float64((13 * min(a, b) + 7 * max(a, b)) % 11)


fn make_lattice(
    nx: Int, ny: Int, nz: Int, dofs: Int = 1, diaphragms: Bool = False
) -> SparseBenchMatrix:
    # Each element assembles a weighted complete-graph Laplacian on its
    # 2*dofs endpoints. Add a unit shift to every assembled diagonal.
    var nodes = nx * ny * nz
    var edges = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
    var lattice_dofs = nodes * dofs
    var n = lattice_dofs + (3 * nz if diaphragms else 0)
    var master_diag: List[Float64] = []
    if diaphragms:
        master_diag.resize(3 * nz, 1.0)
    var bandwidth = (nx * ny if nz > 1 else (nx if ny > 1 else 1)) * dofs + dofs - 1
    var capacity = nodes * dofs * (dofs + 1) // 2 + edges * dofs * dofs
    if diaphragms:
        bandwidth = n - 1
        capacity += 9 * nodes + 3 * nz
    var result = SparseBenchMatrix(n, bandwidth, capacity)
    var strides: List[Int] = [1, nx, nx * ny]
    var dims: List[Int] = [nx, ny, nz]
    for node in range(nodes):
        var degree = 0
        for axis in range(3):
            var coordinate = (node // strides[axis]) % dims[axis]
            degree += Int(coordinate > 0) + Int(coordinate + 1 < dims[axis])
        for d in range(dofs):
            var j = node * dofs + d
            result.col_ptr.append(len(result.rows))
            var diag = len(result.rows)
            result.append(j, 0.0)
            var sum = 1.0
            for c in range(dofs):
                var i = node * dofs + c
                if c == d:
                    continue
                var weight = Float64(degree) * _edge_weight(i, j)
                sum += weight
                if c > d:
                    result.append(i, -weight)
            for axis in range(3):
                var step = strides[axis]
                var coordinate = (node // step) % dims[axis]
                if coordinate > 0:
                    for c in range(dofs):
                        sum += _edge_weight((node - step) * dofs + c, j)
                if coordinate + 1 < dims[axis]:
                    for c in range(dofs):
                        var i = (node + step) * dofs + c
                        var weight = _edge_weight(i, j)
                        sum += weight
                        result.append(i, -weight)
            if diaphragms and d < 3:
                var first_master = 3 * (node // (nx * ny))
                for c in range(first_master, first_master + 3):
                    var i = lattice_dofs + c
                    var weight = 0.125 * _edge_weight(i, j)
                    sum += weight
                    master_diag[c] += weight
                    result.append(i, -weight)
            result.values[diag] = sum
    for c in range(len(master_diag)):
        result.col_ptr.append(len(result.rows))
        result.append(lattice_dofs + c, master_diag[c])
    result.col_ptr.append(len(result.rows))
    return result^


struct _LatticeBox(ImplicitlyCopyable):
    var x0: Int
    var x1: Int
    var y0: Int
    var y1: Int
    var z0: Int
    var z1: Int

    fn __init__(out self, x0: Int, x1: Int, y0: Int, y1: Int, z0: Int, z1: Int):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1


fn _dissect_box(box: _LatticeBox, nx: Int, ny: Int, mut order: List[Int]):
    var x = box.x1 - box.x0
    var y = box.y1 - box.y0
    var z = box.z1 - box.z0
    if x == 0 or y == 0 or z == 0:
        return
    if x * y * z <= 8:
        for k in range(box.z0, box.z1):
            for j in range(box.y0, box.y1):
                for i in range(box.x0, box.x1):
                    var node = (k * ny + j) * nx + i
                    for d in range(6):
                        order.append(6 * node + d)
    elif z >= max(x, y):
        var mid = box.z0 + z // 2
        _dissect_box(_LatticeBox(box.x0, box.x1, box.y0, box.y1, box.z0, mid), nx, ny, order)
        _dissect_box(_LatticeBox(box.x0, box.x1, box.y0, box.y1, mid + 1, box.z1), nx, ny, order)
        _dissect_box(_LatticeBox(box.x0, box.x1, box.y0, box.y1, mid, mid + 1), nx, ny, order)
    elif y >= x:
        var mid = box.y0 + y // 2
        _dissect_box(_LatticeBox(box.x0, box.x1, box.y0, mid, box.z0, box.z1), nx, ny, order)
        _dissect_box(_LatticeBox(box.x0, box.x1, mid + 1, box.y1, box.z0, box.z1), nx, ny, order)
        _dissect_box(_LatticeBox(box.x0, box.x1, mid, mid + 1, box.z0, box.z1), nx, ny, order)
    else:
        var mid = box.x0 + x // 2
        _dissect_box(_LatticeBox(box.x0, mid, box.y0, box.y1, box.z0, box.z1), nx, ny, order)
        _dissect_box(_LatticeBox(mid + 1, box.x1, box.y0, box.y1, box.z0, box.z1), nx, ny, order)
        _dissect_box(_LatticeBox(mid, mid + 1, box.y0, box.y1, box.z0, box.z1), nx, ny, order)


fn _dissect_floors(nx: Int, ny: Int, z0: Int, z1: Int, master_base: Int, mut order: List[Int]):
    # Whole floors separate the tall axis. In a short slab, subdivide
    # the physical grid and retain each floor's masters to the end.
    if z1 - z0 > min(nx, ny):
        var mid = (z0 + z1) // 2
        _dissect_floors(nx, ny, z0, mid, master_base, order)
        _dissect_floors(nx, ny, mid + 1, z1, master_base, order)
        _dissect_floors(nx, ny, mid, mid + 1, master_base, order)
        return
    _dissect_box(_LatticeBox(0, nx, 0, ny, z0, z1), nx, ny, order)
    for k in range(z0, z1):
        for d in range(3):
            order.append(master_base + 3 * k + d)


fn floor_dissection(nx: Int, ny: Int, nz: Int) -> List[Int]:
    var n = 6 * nx * ny * nz
    var order = List[Int](capacity=n + 3 * nz)
    _dissect_floors(nx, ny, 0, nz, n, order)
    return order^


fn make_banded(n: Int, bandwidth: Int) -> SparseBenchMatrix:
    var result = SparseBenchMatrix(n, bandwidth, n * (bandwidth + 1))
    for j in range(n):
        result.col_ptr.append(len(result.rows))
        var diag = len(result.rows)
        result.append(j, 0.0)
        var sum = 1.0
        for i in range(max(0, j - bandwidth), j):
            sum += _edge_weight(i, j)
        for i in range(j + 1, min(n, j + bandwidth + 1)):
            var weight = _edge_weight(i, j)
            sum += weight
            result.append(i, -weight)
        result.values[diag] = sum
    result.col_ptr.append(len(result.rows))
    return result^


fn make_rhs(n: Int) -> List[Float64]:
    var rhs: List[Float64] = []
    rhs.resize(n, 0.0)
    for i in range(n):
        rhs[i] = 0.25 + Float64((3 * i + 5) % 23) / 23.0
    return rhs^


fn relative_residual(matrix: SparseBenchMatrix, x: List[Float64], rhs: List[Float64]) -> Float64:
    var residual: List[Float64] = []
    residual.resize(matrix.n, 0.0)
    var p = residual.unsafe_ptr()
    var starts = matrix.col_ptr.unsafe_ptr()
    var rows = matrix.rows.unsafe_ptr()
    var values = matrix.values.unsafe_ptr()
    var u = x.unsafe_ptr()
    var b = rhs.unsafe_ptr()
    for j in range(matrix.n):
        var diag = starts[j]
        var xj = u[j]
        var ax = values[diag] * xj
        for k in range(diag + 1, starts[j + 1]):
            var i = rows[k]
            var aij = values[k]
            p[i] += aij * xj
            ax += aij * u[i]
        p[j] += ax - b[j]
    var norm = 0.0
    var rhs_norm = 0.0
    for i in range(matrix.n):
        norm += p[i] * p[i]
        rhs_norm += b[i] * b[i]
    return sqrt(norm / rhs_norm) if matrix.n > 0 else 0.0


fn _banded_cholesky(mut band: List[Float64], n: Int, width: Int):
    alias lanes = FLOAT64_SIMD_WIDTH
    var factor = band.unsafe_ptr()
    for j in range(n):
        var col = factor + j * width
        var pivot = sqrt(col[0])
        col[0] = pivot
        var inv = 1.0 / pivot
        var count = min(width, n - j)
        for k in range(1, count):
            col[k] *= inv
        for k in range(1, count):
            var dst = col + k * width
            var src = col + k
            var alpha = -src[0]
            var i = 0
            while i + lanes <= count - k:
                dst.store[width=lanes, alignment=1](
                    i,
                    dst.load[width=lanes, alignment=1](i)
                    + alpha * src.load[width=lanes, alignment=1](i),
                )
                i += lanes
            for p in range(i, count - k):
                dst[p] += alpha * src[p]


fn _banded_solve(band: List[Float64], width: Int, rhs: List[Float64], mut x: List[Float64]):
    var n = len(rhs)
    var factor = band.unsafe_ptr()
    var y = x.unsafe_ptr()
    for i in range(n):
        y[i] = rhs[i]
    for j in range(n):
        var col = factor + j * width
        var solved = y[j] / col[0]
        y[j] = solved
        for k in range(1, min(width, n - j)):
            y[j + k] -= col[k] * solved
    for j in range(n - 1, -1, -1):
        var col = factor + j * width
        var summed = y[j]
        for k in range(1, min(width, n - j)):
            summed -= col[k] * y[j + k]
        y[j] = summed / col[0]


alias REPEATS = 3


fn _run_banded(
    name: String, matrix: SparseBenchMatrix, rhs: List[Float64], mut x: List[Float64]
) raises:
    var width = matrix.bandwidth + 1
    var n = matrix.n
    # Avoid allocating or timing impractically wide bands.
    if n * width * width > 400_000_000:
        return
    var band: List[Float64] = []
    band.resize(n * width, 0.0)
    var ptr = band.unsafe_ptr()
    var factor_ns = 0
    var solve_ns = 0
    for sample in range(REPEATS):
        memset_zero(ptr, n * width)
        for j in range(n):
            for p in range(matrix.col_ptr[j], matrix.col_ptr[j + 1]):
                ptr[j * width + matrix.rows[p] - j] = matrix.values[p]
        var start = Int(perf_counter_ns())
        _banded_cholesky(band, n, width)
        var factored = Int(perf_counter_ns())
        _banded_solve(band, width, rhs, x)
        var solved = Int(perf_counter_ns())
        factor_ns = factored - start if sample == 0 else min(factor_ns, factored - start)
        solve_ns = solved - factored if sample == 0 else min(solve_ns, solved - factored)
    var residual = relative_residual(matrix, x, rhs)
    if not (residual <= 1.0e-10):
        raise Error("banded Cholesky residual too large for " + name)
    var band_nnz = n * width - width * (width - 1) // 2
    print(
        name,
        "band",
        n,
        2 * len(matrix.rows) - n,
        band_nnz,
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        Float64(factor_ns) * 1.0e-6,
        "-",
        Float64(solve_ns) * 1.0e-6,
        "-",
        residual,
        sep=",",
    )


@fieldwise_init
struct _PanelTreeShape(ImplicitlyCopyable):
    var height: Int
    var leaves: Int
    var roots: Int


fn _panel_tree_shape(solver: SparseLDLT) -> _PanelTreeShape:
    var count = solver.supernode_count()
    var heights = List[Int](length=count, fill=1)
    var leaves = count
    var height = 0
    var roots = 0
    for panel in range(count):
        var begin = solver._panel_row_ptr[panel]
        var end = solver._panel_row_ptr[panel + 1]
        height = max(height, heights[panel])
        if begin == end:
            roots += 1
        else:
            var parent = solver._panel_of[Int(solver._panel_rows[begin])]
            leaves -= Int(heights[parent] == 1)
            heights[parent] = max(heights[parent], heights[panel] + 1)
    return _PanelTreeShape(height, leaves, roots)


@fieldwise_init
struct _SparseTiming(ImplicitlyCopyable):
    var nnz: Int
    var panels: Int
    var largest_panel: Int
    var largest_front: Int
    var flops: Int
    var analyze_ms: Float64
    var factor_ms: Float64
    var solve_ms: Float64
    var residual: Float64
    var tree: _PanelTreeShape


fn _time_sparse(
    name: String,
    matrix: SparseBenchMatrix,
    rhs: List[Float64],
    mut x: List[Float64],
    ordering: Int,
    threads: Int,
    permutation: List[Int],
    dense_multiplier: Int = 8,
) raises -> _SparseTiming:
    var solver = SparseLDLT(threads)
    var analyze_ns = 0
    for sample in range(REPEATS):
        solver = SparseLDLT(threads)
        var start = Int(perf_counter_ns())
        solver.analyze(
            matrix.n,
            matrix.col_ptr,
            matrix.rows,
            ordering,
            permutation=permutation,
            dense_multiplier=dense_multiplier,
        )
        var elapsed = Int(perf_counter_ns()) - start
        analyze_ns = elapsed if sample == 0 else min(analyze_ns, elapsed)
    var factor_ns = 0
    var solve_ns = 0
    for sample in range(REPEATS):
        var start = Int(perf_counter_ns())
        if not solver.factorize(matrix.values):
            raise Error("sparse LDLT pivot failure for " + name)
        var factored = Int(perf_counter_ns())
        solver.solve(rhs, x)
        var solved = Int(perf_counter_ns())
        factor_ns = factored - start if sample == 0 else min(factor_ns, factored - start)
        solve_ns = solved - factored if sample == 0 else min(solve_ns, solved - factored)
    var residual = relative_residual(matrix, x, rhs)
    if not (residual <= 1.0e-10):
        raise Error("sparse LDLT residual too large for " + name + ": " + String(residual))
    return _SparseTiming(
        solver.factor_nnz(),
        solver.supernode_count(),
        solver.largest_supernode(),
        solver.largest_front(),
        solver.factor_flops(),
        Float64(analyze_ns) * 1.0e-6,
        Float64(factor_ns) * 1.0e-6,
        Float64(solve_ns) * 1.0e-6,
        residual,
        _panel_tree_shape(solver),
    )


fn _run_sparse(
    name: String,
    method: String,
    matrix: SparseBenchMatrix,
    rhs: List[Float64],
    mut x: List[Float64],
    ordering: Int,
    threads_filter: Int,
    permutation: List[Int] = [],
) raises:
    var one = _time_sparse(
        name, matrix, rhs, x, ordering, 8 if threads_filter == 8 else 1, permutation
    )
    var eight = one
    if threads_filter == 0:
        eight = _time_sparse(name, matrix, rhs, x, ordering, 8, permutation)
    print(
        name,
        method,
        matrix.n,
        2 * len(matrix.rows) - matrix.n,
        one.nnz,
        one.panels,
        one.largest_panel,
        one.largest_front,
        one.flops,
        "-" if threads_filter == 8 else String(one.analyze_ms),
        "-" if threads_filter == 1 else String(eight.analyze_ms),
        "-" if threads_filter == 8 else String(one.factor_ms),
        "-" if threads_filter == 1 else String(eight.factor_ms),
        "-" if threads_filter == 8 else String(one.solve_ms),
        "-" if threads_filter == 1 else String(eight.solve_ms),
        max(one.residual, eight.residual),
        sep=",",
    )


fn _run_case(
    name: String,
    matrix: SparseBenchMatrix,
    ordering_filter: String,
    threads_filter: Int,
) raises:
    var rhs = make_rhs(matrix.n)
    var x = rhs.copy()
    if ordering_filter == "" or ordering_filter == "amd":
        _run_sparse(name, "amd", matrix, rhs, x, SparseOrdering.AMD, threads_filter)
    if ordering_filter == "" or ordering_filter == "rcm":
        _run_sparse(name, "rcm", matrix, rhs, x, SparseOrdering.RCM, threads_filter)
    if ordering_filter == "" and threads_filter != 8:
        _run_banded(name, matrix, rhs, x)


fn _run_tall(ordering_filter: String, threads_filter: Int) raises:
    var name = String("tall-14x14x125")
    var matrix = make_lattice(14, 14, 125, 6, True)
    var rhs = make_rhs(matrix.n)
    var x = rhs.copy()
    if ordering_filter == "" or ordering_filter == "amd":
        _run_sparse(name, "amd", matrix, rhs, x, SparseOrdering.AMD, threads_filter)
    if ordering_filter == "rcm":
        _run_sparse(name, "rcm", matrix, rhs, x, SparseOrdering.RCM, threads_filter)
    if ordering_filter == "" or ordering_filter == "geometric":
        var permutation = floor_dissection(14, 14, 125)
        _run_sparse(
            name,
            "geometric",
            matrix,
            rhs,
            x,
            SparseOrdering.Auto,
            threads_filter,
            permutation,
        )


fn _run_pattern(
    path: String,
    ordering: String,
    threads_filter: Int,
    analyze_only: Bool,
    counts_only: Bool,
    dense_multiplier: Int,
) raises:
    var matrix = load_csc_pattern(path)
    var rhs = make_rhs(matrix.n)
    var x = rhs.copy()
    var kind = SparseOrdering.Auto
    if ordering == "amd":
        kind = SparseOrdering.AMD
    elif ordering == "rcm":
        kind = SparseOrdering.RCM
    elif ordering == "nd":
        kind = SparseOrdering.NestedDissection
    var method = "auto" if ordering == "" else ordering
    if counts_only:
        var start = Int(perf_counter_ns())
        var stats = estimate_sparse_ldlt_ordering(
            matrix.n, matrix.col_ptr, matrix.rows, kind, dense_multiplier
        )
        var elapsed = Int(perf_counter_ns()) - start
        print("ordering,dense_multiplier,n,nnz(L),flops,estimate_ms")
        print(
            method,
            dense_multiplier,
            matrix.n,
            stats.nnz,
            stats.flops,
            Float64(elapsed) * 1.0e-6,
            sep=",",
        )
        return
    if analyze_only:
        var solver = SparseLDLT(1 if threads_filter == 0 else threads_filter)
        var start = Int(perf_counter_ns())
        solver.analyze(
            matrix.n,
            matrix.col_ptr,
            matrix.rows,
            kind,
            dense_multiplier=dense_multiplier,
        )
        var elapsed = Int(perf_counter_ns()) - start
        var tree = _panel_tree_shape(solver)
        print(
            "ordering,n,nnz(A),nnz(L),supernodes,max_panel,max_front,flops,tree_height,tree_leaves,tree_roots,analyze_ms"
        )
        print(
            method,
            matrix.n,
            2 * len(matrix.rows) - matrix.n,
            solver.factor_nnz(),
            solver.supernode_count(),
            solver.largest_supernode(),
            solver.largest_front(),
            solver.factor_flops(),
            tree.height,
            tree.leaves,
            tree.roots,
            Float64(elapsed) * 1.0e-6,
            sep=",",
        )
        return
    print(
        "case,ordering,threads,n,nnz(A),nnz(L),supernodes,max_panel,max_front,flops,tree_height,tree_leaves,tree_roots,analyze_ms,factor_ms,solve_ms,gflops,residual"
    )
    for scaled in [False, True]:
        if scaled:
            _set_pattern_values(matrix, True)
        var name = "pattern-scaled" if scaled else "pattern"
        for threads in [1, 2, 4, 8]:
            if threads_filter != 0 and threads_filter != threads:
                continue
            var timing = _time_sparse(name, matrix, rhs, x, kind, threads, [], dense_multiplier)
            print(
                name,
                method,
                threads,
                matrix.n,
                2 * len(matrix.rows) - matrix.n,
                timing.nnz,
                timing.panels,
                timing.largest_panel,
                timing.largest_front,
                timing.flops,
                timing.tree.height,
                timing.tree.leaves,
                timing.tree.roots,
                timing.analyze_ms,
                timing.factor_ms,
                timing.solve_ms,
                Float64(timing.flops) * 1.0e-6 / timing.factor_ms,
                timing.residual,
                sep=",",
            )


fn main() raises:
    var args = argv()
    var case_filter = String("")
    var ordering_filter = String("")
    var threads_filter = 0
    var pattern_path = String("")
    var analyze_only = False
    var counts_only = False
    var dense_multiplier = 8
    if (len(args) - 1) % 2 != 0:
        raise Error("options require a value. Use --case NAME or --pattern PATH")
    for i in range(1, len(args), 2):
        if args[i] == "--case":
            case_filter = String(args[i + 1])
        elif args[i] == "--ordering":
            ordering_filter = String(args[i + 1])
            if (
                ordering_filter != "amd"
                and ordering_filter != "rcm"
                and ordering_filter != "geometric"
                and ordering_filter != "auto"
                and ordering_filter != "nd"
            ):
                raise Error("--ordering expects auto, amd, rcm, nd, or geometric")
        elif args[i] == "--threads":
            threads_filter = Int(args[i + 1])
            if (
                threads_filter != 1
                and threads_filter != 2
                and threads_filter != 4
                and threads_filter != 8
            ):
                raise Error("--threads expects 1, 2, 4, or 8")
        elif args[i] == "--pattern":
            pattern_path = String(args[i + 1])
        elif args[i] == "--analyze-only":
            if args[i + 1] != "1":
                raise Error("--analyze-only expects 1")
            analyze_only = True
        elif args[i] == "--counts-only":
            if args[i + 1] != "1":
                raise Error("--counts-only expects 1")
            counts_only = True
        elif args[i] == "--dense-multiplier":
            dense_multiplier = Int(args[i + 1])
        else:
            raise Error("unknown option " + String(args[i]))

    if pattern_path != "":
        if analyze_only and counts_only:
            raise Error("choose either --analyze-only or --counts-only")
        if case_filter != "" or ordering_filter == "geometric":
            raise Error("--pattern accepts auto, amd, rcm, or nd and supplies the case")
        _run_pattern(
            pattern_path,
            ordering_filter,
            threads_filter,
            analyze_only,
            counts_only,
            dense_multiplier,
        )
        return
    if (
        analyze_only
        or counts_only
        or dense_multiplier != 8
        or threads_filter == 2
        or threads_filter == 4
        or ordering_filter == "auto"
        or ordering_filter == "nd"
    ):
        raise Error("this option requires --pattern")

    if ordering_filter == "geometric":
        if case_filter != "" and case_filter != "tall-14x14x125":
            raise Error("geometric ordering is defined for the tall case")
        case_filter = "tall-14x14x125"

    print(
        "case,ordering,n,nnz(A),nnz(L),supernodes,max_panel,max_front,flops,"
        "analyze_1_ms,analyze_8_ms,factor_1_ms,factor_8_ms,solve_1_ms,solve_8_ms,residual"
    )
    var matched = False
    for size in [100, 300, 600]:
        var name = "grid2d-" + String(size)
        if case_filter == "" or case_filter == name:
            _run_case(name, make_lattice(size, size, 1), ordering_filter, threads_filter)
            matched = True
    for size in [20, 30, 40]:
        var name = "grid3d-" + String(size)
        if case_filter == "" or case_filter == name:
            _run_case(name, make_lattice(size, size, size), ordering_filter, threads_filter)
            matched = True
    for size in [10, 20]:
        var name = "frame-" + String(size) + "x" + String(size) + "x" + String(3 * size)
        if case_filter == "" or case_filter == name:
            _run_case(
                name,
                make_lattice(size, size, size * 3, 6),
                ordering_filter,
                threads_filter,
            )
            matched = True
    if case_filter == "" or case_filter == "band-30000":
        _run_case("band-30000", make_banded(30000, 100), ordering_filter, threads_filter)
        matched = True
    if case_filter == "" or case_filter == "tall-14x14x125":
        _run_tall(ordering_filter, threads_filter)
        matched = True
    if (case_filter == "" and ordering_filter == "") or case_filter == "dense-3000":
        var matrix = make_banded(3000, 2999)
        var rhs = make_rhs(matrix.n)
        var x = rhs.copy()
        var identity = List[Int](capacity=matrix.n)
        for j in range(matrix.n):
            identity.append(j)
        _run_sparse(
            "dense-3000",
            "natural",
            matrix,
            rhs,
            x,
            SparseOrdering.Auto,
            threads_filter,
            identity,
        )
        matched = True
    if not matched:
        raise Error("unknown benchmark case " + case_filter)
