"""Standalone FP64 plane-strain material-point benchmark. Tensile stress is positive."""

from algorithm import parallelize
from collections import List
from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from math import isfinite, sqrt
from sys import argv
from time import perf_counter_ns

alias BULK = 10.0
alias SHEAR = 6.0
alias YIELD_SLOPE = 0.7
alias FLOW_SLOPE = 0.27
alias COHESION = 0.09
alias HARDENING = 0.4
alias DEVIATOR_SLOPE = 3.0 * SHEAR
alias PRESSURE_SLOPE = YIELD_SLOPE * BULK * FLOW_SLOPE + HARDENING
alias SCALE_PERIOD = 1021
alias BLOCK = 256
alias ELASTIC = 0
alias CONE = 1
alias APEX = 2


@fieldwise_init
struct Strain(ImplicitlyCopyable, Movable):
    var xx: Float64
    var yy: Float64
    var xy: Float64


@fieldwise_init
struct State(ImplicitlyCopyable, Movable):
    var p: Float64
    var sxx: Float64
    var syy: Float64
    var sxy: Float64
    var kappa: Float64


@always_inline
fn load_state(data: UnsafePointer[Float64], n: Int, i: Int) -> State:
    return State(data[i], data[n + i], data[2 * n + i], data[3 * n + i], data[4 * n + i])


@always_inline
fn store_state(data: UnsafePointer[Float64], n: Int, i: Int, state: State):
    data[i] = state.p
    data[n + i] = state.sxx
    data[2 * n + i] = state.syy
    data[3 * n + i] = state.sxy
    data[4 * n + i] = state.kappa


@always_inline
fn predictor(state: State, increment: Strain, scale: Float64) -> State:
    var xx = scale * increment.xx
    var yy = scale * increment.yy
    var mean = (xx + yy) / 3.0
    return State(
        state.p + BULK * (xx + yy),
        state.sxx + 2.0 * SHEAR * (xx - mean),
        state.syy + 2.0 * SHEAR * (yy - mean),
        state.sxy + 2.0 * SHEAR * (scale * increment.xy),
        state.kappa,
    )


@always_inline
fn equivalent_stress(state: State) -> Float64:
    # The deviator's zz component is -sxx - syy.
    var szz = -state.sxx - state.syy
    return sqrt(
        1.5
        * (state.sxx * state.sxx + state.syy * state.syy + szz * szz + 2.0 * state.sxy * state.sxy)
    )


@always_inline
fn correct(mut state: State) -> Int:
    # Backward Euler for f = q + a*p - (c + H*kappa) and g = q + b*p.
    var q = equivalent_stress(state)
    var margin = YIELD_SLOPE * state.p - COHESION - HARDENING * state.kappa
    if q + margin <= 0.0:
        return ELASTIC
    var multiplier = (q + margin) / (DEVIATOR_SLOPE + PRESSURE_SLOPE)
    var deviator = 0.0
    var apex = multiplier * DEVIATOR_SLOPE >= q
    if apex:
        multiplier = margin / PRESSURE_SLOPE
    else:
        deviator = 1.0 - multiplier * DEVIATOR_SLOPE / q
    state.p -= multiplier * BULK * FLOW_SLOPE
    state.sxx *= deviator
    state.syy *= deviator
    state.sxy *= deviator
    state.kappa += multiplier
    return APEX if apex else CONE


@always_inline
fn update[
    plastic: Bool
](data: UnsafePointer[Float64], n: Int, i: Int, increment: Strain, scale: Float64):
    var next = predictor(load_state(data, n, i), increment, scale)

    @parameter
    if plastic:
        _ = correct(next)
    store_state(data, n, i, next)


fn step[
    plastic: Bool
](
    n: Int,
    data: UnsafePointer[Float64],
    scale: UnsafePointer[Float64],
    xx: Float64,
    yy: Float64,
    xy: Float64,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        update[plastic](data, n, i, Strain(xx, yy, xy), scale[i])


fn near(a: Float64, b: Float64) -> Bool:
    return isfinite(a) and isfinite(b) and abs(a - b) <= 5e-13 * max(1.0, abs(a))


fn same(a: State, b: State) -> Bool:
    return (
        near(a.p, b.p)
        and near(a.sxx, b.sxx)
        and near(a.syy, b.syy)
        and near(a.sxy, b.sxy)
        and near(a.kappa, b.kappa)
    )


fn bisect(trial: State) -> State:
    # Check the closed-form correction against the scalar yield equation.
    var q = equivalent_stress(trial)

    @parameter
    fn yield_at(multiplier: Float64) -> Float64:
        return (
            max(q - DEVIATOR_SLOPE * multiplier, 0.0)
            + YIELD_SLOPE * (trial.p - BULK * FLOW_SLOPE * multiplier)
            - COHESION
            - HARDENING * (trial.kappa + multiplier)
        )

    if yield_at(0.0) <= 0.0:
        return trial
    var low = 0.0
    var high = 1.0
    while yield_at(high) > 0.0:
        high *= 2.0
    for _ in range(80):
        var mid = (low + high) * 0.5
        if yield_at(mid) > 0.0:
            low = mid
        else:
            high = mid
    var multiplier = (low + high) * 0.5
    var deviator = max(q - DEVIATOR_SLOPE * multiplier, 0.0) / q if q > 0.0 else 0.0
    return State(
        trial.p - BULK * FLOW_SLOPE * multiplier,
        deviator * trial.sxx,
        deviator * trial.syy,
        deviator * trial.sxy,
        trial.kappa + multiplier,
    )


fn verify_returns() raises:
    var samples = InlineArray[State, 4](
        State(-0.2, 0.01, 0.01, 0.0, 0.0),
        State(-0.2, 0.07, 0.03, 0.30, 0.10),
        State(1.0, 0.0, 0.0, 0.0, 0.03),
        State(1.0, 0.01, -0.01, 0.0, 0.03),
    )
    var expected = InlineArray[Int, 4](ELASTIC, CONE, APEX, APEX)
    for i in range(4):
        var actual = samples[i]
        if correct(actual) != expected[i] or not same(actual, bisect(samples[i])):
            raise Error("Drucker–Prager return mapping check failed")


fn verify_history(
    history: List[Strain], scales: List[Float64], data: UnsafePointer[Float64]
) raises -> Tuple[Int, Int]:
    var n = len(scales)
    var cone = 0
    var apex = 0
    for i in range(min(n, SCALE_PERIOD)):
        var copies = 1 + (n - 1 - i) // SCALE_PERIOD
        var state = State(0, 0, 0, 0, 0)
        for increment in history:
            state = predictor(state, increment, scales[i])
            var expected = bisect(state)
            var branch = correct(state)
            if not same(expected, state):
                raise Error("Drucker–Prager cyclic return check failed")
            cone += copies * Int(branch == CONE)
            apex += copies * Int(branch == APEX)
        if not same(load_state(data, n, i), state):
            raise Error("Drucker–Prager reference history mismatch")
    return (cone, apex)


fn cpu_history[
    plastic: Bool
](
    history: List[Strain], scales: List[Float64], data: UnsafePointer[Float64], threads: Int
) -> Float64:
    var n = len(scales)
    var workers = min(threads, n) if n >= 16384 else 1
    var scale = scales.unsafe_ptr()
    var start = perf_counter_ns()
    for carry in history:
        var increment = carry

        @parameter
        fn work(worker: Int):
            for i in range(n * worker // workers, n * (worker + 1) // workers):
                update[plastic](data, n, i, increment, scale[i])

        if workers == 1:
            work(0)
        else:
            parallelize[work](workers, workers)
    return Float64(perf_counter_ns() - start) * 1e-6


fn gpu_history[
    plastic: Bool
](
    ctx: DeviceContext, history: List[Strain], scales: List[Float64], data: UnsafePointer[Float64]
) raises -> Float64:
    var n = len(scales)
    var state = ctx.enqueue_create_buffer[DType.float64](5 * n)
    var scale = ctx.enqueue_create_buffer[DType.float64](n)
    scale.enqueue_copy_from(scales.unsafe_ptr())
    _ = state.enqueue_fill(0.0)
    var kernel = ctx.compile_function[step[plastic]]()
    ctx.synchronize()
    var start = perf_counter_ns()
    for increment in history:
        ctx.enqueue_function(
            kernel,
            n,
            state,
            scale,
            increment.xx,
            increment.yy,
            increment.xy,
            grid_dim=(n + BLOCK - 1) // BLOCK,
            block_dim=BLOCK,
        )
    ctx.synchronize()
    var elapsed = Float64(perf_counter_ns() - start) * 1e-6
    state.enqueue_copy_to(data)
    ctx.synchronize()
    return elapsed


fn run[
    plastic: Bool
](ctx: DeviceContext, history: List[Strain], scales: List[Float64], threads: Int) raises:
    var n = len(scales)
    var cpu = List[Float64](length=5 * n, fill=0.0)
    var gpu = List[Float64](length=5 * n, fill=0.0)
    var cpu_ms = cpu_history[plastic](history, scales, cpu.unsafe_ptr(), threads)
    var gpu_ms = gpu_history[plastic](ctx, history, scales, gpu.unsafe_ptr())
    var worst = 0.0
    for i in range(len(cpu)):
        if not near(cpu[i], gpu[i]):
            raise Error("Drucker–Prager CPU/GPU state mismatch")
        worst = max(worst, abs(cpu[i] - gpu[i]))
    var cone = 0
    var apex = 0

    @parameter
    if plastic:
        (cone, apex) = verify_history(history, scales, cpu.unsafe_ptr())
    print(
        "drucker_prager" if plastic else "elastic",
        n,
        len(history),
        cpu_ms,
        gpu_ms,
        cone,
        apex,
        worst,
        sep=",",
    )


fn main() raises:
    var args = argv()
    var points = Int(String(args[1])) if len(args) > 1 else 1_048_576
    var increments = Int(String(args[2])) if len(args) > 2 else 40
    var threads = Int(String(args[3])) if len(args) > 3 else 8
    if len(args) > 4 or points < 1 or points > 16_777_216 or increments < 1 or threads < 1:
        raise Error("usage: drucker_prager [points <= 16777216] [increments] [CPU threads]")
    verify_returns()

    # Synthetic mixed loading, unloading, shear, and tension.
    var cycle = InlineArray[Strain, 8](
        Strain(-0.007, -0.005, 0.001),
        Strain(0.000, -0.003, 0.020),
        Strain(0.000, 0.000, 0.015),
        Strain(0.004, 0.005, -0.015),
        Strain(0.010, 0.009, 0.000),
        Strain(0.009, 0.009, 0.000),
        Strain(-0.008, -0.012, 0.000),
        Strain(0.001, -0.002, -0.018),
    )
    var history = List[Strain](capacity=increments)
    for i in range(increments):
        history.append(cycle[i % 8])
    var scales = List[Float64](capacity=points)
    for i in range(points):
        scales.append(
            0.05 + 1.95 * Float64((i % SCALE_PERIOD * 463) % SCALE_PERIOD) / (SCALE_PERIOD - 1)
        )
    var ctx = DeviceContext()
    print("model,points,increments,cpu_ms,gpu_ms,cone_updates,apex_updates,max_difference")
    run[False](ctx, history, scales, threads)
    run[True](ctx, history, scales, threads)
