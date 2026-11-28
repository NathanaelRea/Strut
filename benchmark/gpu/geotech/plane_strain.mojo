"""Matrix-free FP64 Q1 plane-strain elasticity with a geometric V-cycle."""

from algorithm import parallelize
from collections import List
from gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import AddressSpace
from math import cos, isfinite, sin, sqrt
from memory import stack_allocation
from sys import argv
from sys.info import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator
from time import perf_counter_ns

alias Vec = SIMD[DType.float64, 2]
alias VecPtr = UnsafePointer[Vec]
alias TOLERANCE = 1e-9
alias MAX_ITERATIONS = 100
alias BLOCK = 256
alias CHECK_INTERVAL = 4

alias APPLY = 0
alias RESIDUAL = 1
alias SMOOTH_ZERO = 2
alias SMOOTH = 3
alias RESTRICT = 4
alias PROLONG = 5

alias RHO = 0
alias BETA = 1
alias ALPHA = 2
alias TARGET = 3
alias ITERATIONS = 4
alias STATUS = 5
alias CONTROL_SIZE = 6


@fieldwise_init
struct Stencil(ImplicitlyCopyable):
    var center: Float64
    var horizontal: Float64
    var vertical: Float64
    var corner: Float64
    var cross: Float64

    @staticmethod
    fn from_poisson(nu: Float64) -> Self:
        var mu = 1.0 / (2.0 * (1.0 + nu))
        var lame = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return Self(
            (4.0 / 3.0) * (lame + 3.0 * mu),
            -(2.0 * lame + 3.0 * mu) / 3.0,
            lame / 3.0,
            -(lame + 3.0 * mu) / 6.0,
            -(lame + mu) / 4.0,
        )

    @always_inline
    fn apply(self, p: Int, i: Int, u: VecPtr) -> Vec:
        var h = u[i - 1] + u[i + 1]
        var v = u[i - p] + u[i + p]
        var same = u[i - p - 1] + u[i + p + 1]
        var opposite = u[i - p + 1] + u[i + p - 1]
        var mix = same - opposite
        return (
            self.center * u[i]
            + Vec(self.horizontal, self.vertical) * h
            + Vec(self.vertical, self.horizontal) * v
            + self.corner * (same + opposite)
            + self.cross * Vec(mix[1], mix[0])
        )


@fieldwise_init
struct GridStep(ImplicitlyCopyable, Movable):
    var kind: Int
    var cells: Int
    var other_cells: Int
    var a: Int
    var b: Int
    var dest: Int
    var weight: Float64

    @always_inline
    fn cell[operation: Int = -1](self, x: Int, y: Int, stencil: Stencil, data: VecPtr):
        var p = self.cells + 1
        var i = y * p + x
        var a = data + self.a
        var b = data + self.b
        var out = data + self.dest
        var kind = self.kind if operation < 0 else operation
        if kind == APPLY:
            out[i] = stencil.apply(p, i, a)
        elif kind == RESIDUAL:
            out[i] = a[i] - stencil.apply(p, i, b)
        elif kind == SMOOTH_ZERO:
            out[i] = a[i] * self.weight
        elif kind == SMOOTH:
            out[i] = b[i] + (a[i] - stencil.apply(p, i, b)) * self.weight
        elif kind == RESTRICT:
            # R = Pᵀ for the integrated stiffness. Boundaries hold zero.
            var fine = self.other_cells + 1
            var j = 2 * y * fine + 2 * x
            var edges = (a[j - 1] + a[j + 1]) + (a[j - fine] + a[j + fine])
            var corners = (a[j - fine - 1] + a[j + fine + 1]) + (a[j - fine + 1] + a[j + fine - 1])
            out[i] = a[j] + edges * 0.5 + corners * 0.25
        else:
            var coarse = self.other_cells + 1
            var j = (y // 2) * coarse + x // 2
            var right = x & 1
            var down = (y & 1) * coarse
            var top = (a[j] + a[j + right]) * 0.5
            var bottom = (a[j + down] + a[j + down + right]) * 0.5
            out[i] += (top + bottom) * 0.5


@fieldwise_init
struct Level(ImplicitlyCopyable, Movable):
    var cells: Int
    var rhs: Int
    var u: Int
    var work: Int


fn append_cycle(levels: List[Level], index: Int, diagonal: Float64, mut steps: List[GridStep]):
    var level = levels[index]
    var n = level.cells
    if n == 2:
        # The single free node has a diagonal two-component stiffness.
        steps.append(GridStep(SMOOTH_ZERO, n, 0, level.rhs, 0, level.u, 1.0 / diagonal))
        return
    var scale = 0.65 / diagonal
    var coarse = levels[index + 1]
    steps.append(GridStep(SMOOTH_ZERO, n, 0, level.rhs, 0, level.work, scale))
    steps.append(GridStep(SMOOTH, n, 0, level.rhs, level.work, level.u, scale))
    steps.append(GridStep(RESIDUAL, n, 0, level.rhs, level.u, level.work, 0))
    steps.append(GridStep(RESTRICT, coarse.cells, n, level.work, 0, coarse.rhs, 0))
    append_cycle(levels, index + 1, diagonal, steps)
    steps.append(GridStep(PROLONG, n, coarse.cells, coarse.u, 0, level.u, 0))
    steps.append(GridStep(SMOOTH, n, 0, level.rhs, level.u, level.work, scale))
    steps.append(GridStep(SMOOTH, n, 0, level.rhs, level.work, level.u, scale))


struct Plan(Movable):
    var size: Int
    var entries: Int
    var z: Int
    var apply: GridStep
    var cycle: List[GridStep]

    fn __init__(out self, cells: Int, stencil: Stencil):
        var size = (cells + 1) ** 2
        # The first four vectors are x, residual, direction and A*direction.
        var cursor = 4 * size
        var levels = List[Level]()
        var n = cells
        while n >= 2:
            var count = (n + 1) ** 2
            var rhs = size
            if n != cells:
                rhs = cursor
                cursor += count
            var u = cursor
            cursor += count
            var scratch = cursor
            if n != 2:
                cursor += count
            levels.append(Level(n, rhs, u, scratch))
            n //= 2
        self.size = size
        self.entries = cursor
        self.z = levels[0].u
        self.apply = GridStep(APPLY, cells, 0, 2 * size, 0, 3 * size, 0)
        self.cycle = []
        append_cycle(levels, 0, stencil.center, self.cycle)


@always_inline
fn dot(a: Vec, b: Vec) -> Float64:
    return (a * b).reduce_add()


fn _cpu_grid[operation: Int](step: GridStep, stencil: Stencil, data: VecPtr, threads: Int):
    var rows = step.cells - 1
    var workers = min(threads, rows) if step.cells >= 128 else 1

    @parameter
    fn run(worker: Int):
        for y in range(1 + rows * worker // workers, 1 + rows * (worker + 1) // workers):
            for x in range(1, step.cells):
                step.cell[operation](x, y, stencil, data)

    if workers == 1:
        run(0)
    else:
        parallelize[run](workers, workers)


fn cpu_grid(step: GridStep, stencil: Stencil, data: VecPtr, threads: Int):
    # Specialize the inner loop for each stencil or transfer.
    if step.kind == APPLY:
        _cpu_grid[APPLY](step, stencil, data, threads)
    elif step.kind == RESIDUAL:
        _cpu_grid[RESIDUAL](step, stencil, data, threads)
    elif step.kind == SMOOTH_ZERO:
        _cpu_grid[SMOOTH_ZERO](step, stencil, data, threads)
    elif step.kind == SMOOTH:
        _cpu_grid[SMOOTH](step, stencil, data, threads)
    elif step.kind == RESTRICT:
        _cpu_grid[RESTRICT](step, stencil, data, threads)
    else:
        _cpu_grid[PROLONG](step, stencil, data, threads)


fn cpu_reduce[
    update: Bool = False
](
    n: Int,
    a: VecPtr,
    b: VecPtr,
    threads: Int,
    x: VecPtr = VecPtr(),
    r: VecPtr = VecPtr(),
    alpha: Float64 = 0.0,
) -> Float64:
    var workers = min(n, threads) if n >= 16384 else 1
    var sums = List[Float64](length=workers, fill=0)
    var out = sums.unsafe_ptr()

    @parameter
    fn run(worker: Int):
        var sum = 0.0
        for i in range(n * worker // workers, n * (worker + 1) // workers):

            @parameter
            if update:
                x[i] += a[i] * alpha
                r[i] -= b[i] * alpha
                sum += dot(r[i], r[i])
            else:
                sum += dot(a[i], b[i])
        out[worker] = sum

    if workers == 1:
        run(0)
    else:
        parallelize[run](workers, workers)
    var result = 0.0
    for sum in sums:
        result += sum
    return result


fn cpu_solve(
    plan: Plan, stencil: Stencil, data: VecPtr, b: VecPtr, norm: Float64, threads: Int
) raises -> Int:
    var n = plan.size
    var x = data
    var r = data + n
    var p = data + 2 * n
    var ap = data + 3 * n
    var z = data + plan.z
    for i in range(n):
        x[i] = Vec(0.0)
        p[i] = Vec(0.0)
        r[i] = b[i]
    var old_rho = 1.0
    var target = TOLERANCE * TOLERANCE * norm
    var workers = min(n, threads) if n >= 16384 else 1
    for iteration in range(1, MAX_ITERATIONS + 1):
        for step in plan.cycle:
            cpu_grid(step, stencil, data, threads)
        var rho = cpu_reduce(n, r, z, threads)
        if rho <= 0.0 or not isfinite(rho):
            raise Error("CPU PCG preconditioner breakdown")
        var beta = rho / old_rho
        old_rho = rho

        @parameter
        fn grow(worker: Int):
            for i in range(n * worker // workers, n * (worker + 1) // workers):
                p[i] = z[i] + p[i] * beta

        if workers == 1:
            grow(0)
        else:
            parallelize[grow](workers, workers)
        cpu_grid(plan.apply, stencil, data, threads)
        var denom = cpu_reduce(n, p, ap, threads)
        if denom <= 0.0 or not isfinite(denom):
            raise Error("CPU PCG operator breakdown")
        if cpu_reduce[True](n, p, ap, threads, x, r, rho / denom) <= target:
            return iteration
    raise Error("CPU PCG reached the iteration limit")


fn grid_kernel(
    data: UnsafePointer[Float64],
    control: UnsafePointer[Float64],
    kind: Int,
    cells: Int,
    other: Int,
    a: Int,
    b: Int,
    dest: Int,
    weight: Float64,
    center: Float64,
    horizontal: Float64,
    vertical: Float64,
    corner: Float64,
    cross: Float64,
):
    var x = block_idx.x * block_dim.x + thread_idx.x + 1
    var y = block_idx.y * block_dim.y + thread_idx.y + 1
    if x < cells and y < cells and control[STATUS] == 0.0:
        GridStep(kind, cells, other, a, b, dest, weight).cell(
            x, y, Stencil(center, horizontal, vertical, corner, cross), data.bitcast[Vec]()
        )


@always_inline
fn block_sum(value: Float64) -> Float64:
    var shared = stack_allocation[BLOCK, Float64, address_space = AddressSpace.SHARED]()
    var lane = thread_idx.x
    shared[lane] = value
    barrier()
    var stride = BLOCK // 2
    while stride > 0:
        if lane < stride:
            shared[lane] += shared[lane + stride]
        barrier()
        stride //= 2
    return shared[0]


fn reduce_kernel(
    data: UnsafePointer[Float64],
    control: UnsafePointer[Float64],
    partial: UnsafePointer[Float64],
    stage: Int,
    n: Int,
    z: Int,
):
    if control[STATUS] != 0:
        return
    var x = data.bitcast[Vec]()
    var r = x + n
    var p = x + 2 * n
    var ap = x + 3 * n
    var sum = 0.0
    var first = block_idx.x * BLOCK + thread_idx.x
    var stride = grid_dim.x * BLOCK
    if stage == RHO:
        for i in range(first, n, stride):
            sum += dot(r[i], x[z + i])
    elif stage == ALPHA:
        for i in range(first, n, stride):
            sum += dot(p[i], ap[i])
    else:
        var alpha = control[ALPHA]
        for i in range(first, n, stride):
            x[i] += p[i] * alpha
            r[i] -= ap[i] * alpha
            sum += dot(r[i], r[i])
    var total = block_sum(sum)
    if thread_idx.x == 0:
        partial[block_idx.x] = total


fn finish_kernel(
    control: UnsafePointer[Float64], partial: UnsafePointer[Float64], stage: Int, blocks: Int
):
    if control[STATUS] != 0:
        return
    var lane = thread_idx.x
    var total = block_sum(partial[lane] if lane < blocks else 0.0)
    if lane != 0:
        return
    if not isfinite(total) or (stage != ITERATIONS and total <= 0.0):
        control[STATUS] = -1.0
    elif stage == RHO:
        control[BETA] = total / control[RHO]
        control[RHO] = total
    elif stage == ALPHA:
        control[ALPHA] = control[RHO] / total
    else:
        control[ITERATIONS] += 1.0
        if total <= control[TARGET]:
            control[STATUS] = 1.0


fn grow_kernel(data: UnsafePointer[Float64], control: UnsafePointer[Float64], n: Int, z: Int):
    var i = block_idx.x * BLOCK + thread_idx.x
    if i < n and control[STATUS] == 0.0:
        var x = data.bitcast[Vec]()
        var direction = 2 * n + i
        x[direction] = x[z + i] + x[direction] * control[BETA]


struct DeviceSolver(Copyable, Movable):
    var ctx: DeviceContext
    var data: DeviceBuffer[DType.float64]
    var x: DeviceBuffer[DType.float64]
    var r: DeviceBuffer[DType.float64]
    var p: DeviceBuffer[DType.float64]
    var control: DeviceBuffer[DType.float64]
    var partial: DeviceBuffer[DType.float64]

    fn __init__(out self, plan: Plan) raises:
        self.ctx = DeviceContext()
        self.data = self.ctx.enqueue_create_buffer[DType.float64](2 * plan.entries)
        _ = self.data.enqueue_fill(0.0)
        self.x = self.data.create_sub_buffer[DType.float64](0, 2 * plan.size)
        self.r = self.data.create_sub_buffer[DType.float64](2 * plan.size, 2 * plan.size)
        self.p = self.data.create_sub_buffer[DType.float64](4 * plan.size, 2 * plan.size)
        self.control = self.ctx.enqueue_create_buffer[DType.float64](CONTROL_SIZE)
        self.partial = self.ctx.enqueue_create_buffer[DType.float64](BLOCK)
        self.ctx.synchronize()

    fn solve(
        self, plan: Plan, stencil: Stencil, rhs: VecPtr, solution: VecPtr, norm: Float64
    ) raises -> Int:
        var ctx = self.ctx
        var grid = ctx.compile_function[grid_kernel]()
        var reduce = ctx.compile_function[reduce_kernel]()
        var finish = ctx.compile_function[finish_kernel]()
        var grow = ctx.compile_function[grow_kernel]()
        var n = plan.size
        var vector_blocks = (n + BLOCK - 1) // BLOCK
        var blocks = min(BLOCK, vector_blocks)
        _ = self.x.enqueue_fill(0.0)
        _ = self.p.enqueue_fill(0.0)
        self.r.enqueue_copy_from(rhs.bitcast[Float64]())
        var state = InlineArray[Float64, CONTROL_SIZE](
            1.0, 0.0, 0.0, TOLERANCE * TOLERANCE * norm, 0.0, 0.0
        )
        var host_state = UnsafePointer(to=state[0])
        self.control.enqueue_copy_from(host_state)

        @parameter
        fn launch(step: GridStep) raises:
            ctx.enqueue_function(
                grid,
                self.data,
                self.control,
                step.kind,
                step.cells,
                step.other_cells,
                step.a,
                step.b,
                step.dest,
                step.weight,
                stencil.center,
                stencil.horizontal,
                stencil.vertical,
                stencil.corner,
                stencil.cross,
                grid_dim=((step.cells + 30) // 32, (step.cells + 6) // 8),
                block_dim=(32, 8),
            )

        @parameter
        fn reduction(stage: Int) raises:
            ctx.enqueue_function(
                reduce,
                self.data,
                self.control,
                self.partial,
                stage,
                n,
                plan.z,
                grid_dim=blocks,
                block_dim=BLOCK,
            )
            ctx.enqueue_function(
                finish,
                self.control,
                self.partial,
                stage,
                blocks,
                grid_dim=1,
                block_dim=BLOCK,
            )

        for _ in range(0, MAX_ITERATIONS, CHECK_INTERVAL):
            for _ in range(CHECK_INTERVAL):
                for step in plan.cycle:
                    launch(step)
                reduction(RHO)
                ctx.enqueue_function(
                    grow,
                    self.data,
                    self.control,
                    n,
                    plan.z,
                    grid_dim=vector_blocks,
                    block_dim=BLOCK,
                )
                launch(plan.apply)
                reduction(ALPHA)
                reduction(ITERATIONS)
            self.control.enqueue_copy_to(host_state)
            ctx.synchronize()
            if state[STATUS] == 1.0:
                self.x.enqueue_copy_to(solution.bitcast[Float64]())
                ctx.synchronize()
                return Int(state[ITERATIONS])
            if state[STATUS] < 0.0:
                raise Error("GPU PCG breakdown")
        raise Error("GPU PCG reached the iteration limit")


fn verify_stencil(nu: Float64, stencil: Stencil) raises:
    # Independently integrate Q1 cell forces with 2×2 Gauss quadrature.
    alias CELLS = 4
    alias PITCH = CELLS + 1
    var dx = InlineArray[Int, 4](0, 1, 1, 0)
    var dy = InlineArray[Int, 4](0, 0, 1, 1)
    var mu = 1.0 / (2.0 * (1.0 + nu))
    var lame = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    var q = 0.5 / sqrt(3.0)
    var gauss = InlineArray[Float64, 2](0.5 - q, 0.5 + q)
    var u = List[Vec](length=PITCH * PITCH, fill=Vec(0.0))
    var expected = u.copy()
    for y in range(1, CELLS):
        for x in range(1, CELLS):
            u[y * PITCH + x] = Vec(sin(Float64(x + 2 * y)), cos(Float64(2 * x + y)))
    for y in range(CELLS):
        for x in range(CELLS):
            for iy in range(2):
                for ix in range(2):
                    var py = gauss[iy]
                    var px = gauss[ix]
                    var gx = InlineArray[Float64, 4](py - 1, 1 - py, py, -py)
                    var gy = InlineArray[Float64, 4](px - 1, -px, px, 1 - px)
                    var ex = 0.0
                    var ey = 0.0
                    var shear = 0.0
                    for a in range(4):
                        var value = u[(y + dy[a]) * PITCH + x + dx[a]]
                        ex += gx[a] * value[0]
                        ey += gy[a] * value[1]
                        shear += gy[a] * value[0] + gx[a] * value[1]
                    var sx = (lame + 2 * mu) * ex + lame * ey
                    var sy = (lame + 2 * mu) * ey + lame * ex
                    var tau = mu * shear
                    for a in range(4):
                        expected[(y + dy[a]) * PITCH + x + dx[a]] += 0.25 * Vec(
                            gx[a] * sx + gy[a] * tau, gy[a] * sy + gx[a] * tau
                        )
    var input = u.unsafe_ptr()
    for y in range(1, CELLS):
        for x in range(1, CELLS):
            var i = y * PITCH + x
            var error = expected[i] - stencil.apply(PITCH, i, input)
            if sqrt(dot(error, error)) > 1e-12 * stencil.center:
                raise Error("matrix-free stencil failed independent Gauss assembly")


fn manufacture(
    cells: Int,
    load: Int,
    stencil: Stencil,
    exact: VecPtr,
    rhs: VecPtr,
):
    var pitch = cells + 1
    var kx = Float64(1 + load % 5)
    var ky = Float64(1 + (load // 5 + load) % 7)
    var xshape = List[Vec](length=pitch, fill=Vec(0.0))
    var yshape = xshape.copy()
    for i in range(1, cells):
        var t = Float64(i) * 3.141592653589793 / Float64(cells)
        xshape[i] = Vec(sin(kx * t), sin((ky + 1.0) * t))
        yshape[i] = Vec(sin(ky * t), sin((kx + 1.0) * t))
    for y in range(1, cells):
        for x in range(1, cells):
            exact[y * pitch + x] = xshape[x] * yshape[y] * Vec(1.0, -0.7)
    for y in range(1, cells):
        for x in range(1, cells):
            var i = y * pitch + x
            rhs[i] = stencil.apply(pitch, i, exact)


fn run[gpu: Bool](cells: Int, stencil: Stencil, loads: Int, threads: Int) raises:
    var label = "gpu" if gpu else "cpu"
    var start = perf_counter_ns()
    var plan = Plan(cells, stencil)
    var n = plan.size
    var solver: Optional[DeviceSolver] = None
    var data = List[Vec](length=n if gpu else plan.entries, fill=Vec(0.0))
    var output = data.unsafe_ptr()

    @parameter
    if gpu:
        solver = DeviceSolver(plan)
    var setup = Float64(perf_counter_ns() - start) * 1e-6
    var exact = List[Vec](length=n, fill=Vec(0.0))
    var rhs = exact.copy()
    var exact_ptr = exact.unsafe_ptr()
    var rhs_ptr = rhs.unsafe_ptr()
    var first = 0.0
    var all = setup
    var pitch = cells + 1
    print(label, "setup", cells, -1, 0, setup, 0, 0, sep=",")
    for load in range(loads):
        manufacture(cells, load, stencil, exact_ptr, rhs_ptr)
        var norm = cpu_reduce(n, rhs_ptr, rhs_ptr, threads)
        var started = perf_counter_ns()
        var iterations: Int

        @parameter
        if gpu:
            iterations = solver.value().solve(plan, stencil, rhs_ptr, output, norm)
        else:
            iterations = cpu_solve(plan, stencil, output, rhs_ptr, norm, threads)
        var time = Float64(perf_counter_ns() - started) * 1e-6
        if load == 0:
            first = time
        all += time
        var error = 0.0
        var residual = 0.0
        for y in range(1, cells):
            for x in range(1, cells):
                var i = y * pitch + x
                var e = exact_ptr[i] - output[i]
                var r = rhs_ptr[i] - stencil.apply(pitch, i, output)
                error += dot(e, e)
                residual += dot(r, r)
        var relative_error = sqrt(error / cpu_reduce(n, exact_ptr, exact_ptr, threads))
        var relative_residual = sqrt(residual / norm)
        if not (relative_residual <= 5e-8 and relative_error <= 5e-8):
            raise Error(label + " failed manufactured solution validation")
        print(
            label, "load", cells, load, iterations, time, relative_residual, relative_error, sep=","
        )
    print(label, "setup_plus_first", cells, 0, 0, setup + first, 0, 0, sep=",")
    print(label, "setup_plus_all", cells, -1, 0, all, 0, 0, sep=",")


fn main() raises:
    var args = argv()
    var cells = Int(String(args[1])) if len(args) > 1 else 512
    var loads = Int(String(args[2])) if len(args) > 2 else 8
    var threads = Int(String(args[3])) if len(args) > 3 else 8
    var nu = atof(String(args[4])) if len(args) > 4 else 0.30
    var engine = String(args[5]) if len(args) > 5 else "both"
    if (
        len(args) > 6
        or cells < 8
        or cells > 4096
        or (cells & (cells - 1)) != 0
        or loads < 1
        or threads < 1
        or not (0.0 <= nu < 0.5)
        or (engine != "both" and engine != "cpu" and engine != "gpu")
    ):
        raise Error(
            "usage: plane_strain [power-of-two cells 8..4096] [loads] [CPU threads] [nu in [0,"
            " 0.5)] [both|cpu|gpu]"
        )
    var stencil = Stencil.from_poisson(nu)
    verify_stencil(nu, stencil)
    print("engine,kind,cells,load,iterations,ms,relative_residual,relative_solution_error")
    if engine != "gpu":
        run[False](cells, stencil, loads, threads)
    if engine != "cpu":

        @parameter
        if has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator():
            run[True](cells, stencil, loads, threads)
        else:
            raise Error("This plane-strain build has no supported GPU target")
