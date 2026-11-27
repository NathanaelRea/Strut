"""Standalone CPU/GPU replay of Strut's Steel02 and Concrete01 fiber updates."""

from algorithm import parallelize, vectorize
from collections import List
from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from materials import UniMaterialDef, UniMaterialState, uniaxial_commit, uniaxial_set_trial_strain
from memory import UnsafePointer
from sections.fiber_kernels import _concrete01_eval_simd_step, _steel02_eval_simd_step
from solver.simd_contiguous import FLOAT64_SIMD_WIDTH
from sys import size_of
from sys.arg import argv
from sys.ffi import external_call
from tag_types import UniMaterialTypeTag
from time import perf_counter_ns


alias STATE_FIELDS = 11
alias PARAM_FIELDS = 9
alias THREADS = 256
alias STEPS = 40
alias AUDIT_MAX_FIBERS = 16384


@always_inline
fn _update[
    steel: Bool, width: Int
](
    state: UnsafePointer[Float64],
    strain_params: UnsafePointer[Float64],
    defs: UnsafePointer[Float64],
    n: Int,
    i: Int,
    group: Int,
    movement: Float64,
    gravity: Float64,
):
    var mat = defs + group * PARAM_FIELDS
    var strain = (
        strain_params.load[width=width](i) * movement
        + strain_params.load[width=width](n + i) * gravity
    )

    @parameter
    if steel:
        var response = _steel02_eval_simd_step[width](
            strain,
            state.load[width=width](i),
            state.load[width=width](n + i),
            state.load[width=width](2 * n + i),
            state.load[width=width](3 * n + i),
            state.load[width=width](4 * n + i),
            state.load[width=width](5 * n + i),
            state.load[width=width](6 * n + i),
            state.load[width=width](7 * n + i),
            state.load[width=width](8 * n + i),
            state.load[width=width](9 * n + i),
            state.load[width=width](10 * n + i).cast[DType.int32](),
            mat[0],
            mat[1],
            mat[2],
            mat[3],
            mat[4],
            mat[5],
            0.0,
            0.0,
            0.0,
            mat[6],
            mat[7],
            0.0,
            mat[8],
            mat[8],
        )
        state.store(i, response[0])
        state.store(n + i, response[1])
        state.store(2 * n + i, response[2])
        state.store(3 * n + i, response[3])
        state.store(4 * n + i, response[4])
        state.store(5 * n + i, response[5])
        state.store(6 * n + i, response[6])
        state.store(7 * n + i, response[7])
        state.store(8 * n + i, response[8])
        state.store(9 * n + i, response[9])
        state.store(10 * n + i, response[10].cast[DType.float64]())
    else:
        var concrete = _concrete01_eval_simd_step[width](
            strain,
            state.load[width=width](i),
            state.load[width=width](n + i),
            state.load[width=width](2 * n + i),
            state.load[width=width](3 * n + i),
            state.load[width=width](4 * n + i),
            state.load[width=width](5 * n + i),
            mat[0],
            mat[1],
            mat[2],
            mat[3],
            mat[4],
            mat[5],
        )
        state.store(i, strain)
        state.store(n + i, concrete[0])
        state.store(2 * n + i, concrete[1])
        state.store(3 * n + i, concrete[2])
        state.store(4 * n + i, concrete[3])
        state.store(5 * n + i, concrete[4])


fn _gpu_tagged(
    state: UnsafePointer[Float64],
    strains: UnsafePointer[Float64],
    defs: UnsafePointer[Float64],
    ids: UnsafePointer[Int32],
    n: Int,
    steel_groups: Int,
    movement: Float64,
    gravity: Float64,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < n:
        var group = Int(ids[i])
        if group < steel_groups:
            _update[True, 1](state, strains, defs, n, i, group, movement, gravity)
        else:
            _update[False, 1](state, strains, defs, n, i, group, movement, gravity)


fn _gpu_grouped[
    steel: Bool
](
    state: UnsafePointer[Float64],
    strains: UnsafePointer[Float64],
    defs: UnsafePointer[Float64],
    n: Int,
    first: Int,
    end: Int,
    group: Int,
    movement: Float64,
    gravity: Float64,
):
    var i = first + Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < end:
        _update[steel, 1](state, strains, defs, n, i, group, movement, gravity)


fn _cpu_grouped[
    steel: Bool
](
    state: UnsafePointer[Float64],
    strains: UnsafePointer[Float64],
    defs: UnsafePointer[Float64],
    n: Int,
    first: Int,
    end: Int,
    group: Int,
    movement: Float64,
    gravity: Float64,
    workers: Int,
):
    @parameter
    fn work(worker: Int):
        var start = first + worker * (end - first) // workers
        var count = first + (worker + 1) * (end - first) // workers - start

        @parameter
        fn chunk[width: Int](i: Int):
            _update[steel, width](state, strains, defs, n, start + i, group, movement, gravity)

        vectorize[chunk, FLOAT64_SIMD_WIDTH](count)

    parallelize[work](workers, workers)


fn _reset(
    state: UnsafePointer[Float64],
    defs: UnsafePointer[Float64],
    ids: UnsafePointer[Int32],
    n: Int,
    steel_groups: Int,
):
    for i in range(n):
        var group = Int(ids[i])
        var mat = defs + group * PARAM_FIELDS
        for field in range(STATE_FIELDS):
            state[field * n + i] = 0
        if group < steel_groups:
            state[2 * n + i] = mat[1]
            state[3 * n + i] = -mat[7]
            state[4 * n + i] = mat[7]
        else:
            state[2 * n + i] = mat[4]
            state[5 * n + i] = mat[4]


fn _open(path: String, writing: Bool) raises -> UnsafePointer[UInt8]:
    var name = List[UInt8]()
    for byte in path.as_bytes():
        name.append(byte)
    name.append(0)
    var mode = InlineArray[UInt8, 3](119 if writing else 114, 98, 0)
    var file = external_call["fopen", UnsafePointer[UInt8]](
        name.unsafe_ptr(), UnsafePointer(to=mode[0])
    )
    if not file:
        raise Error("could not open " + path)
    return file


fn _read[T: AnyType](file: UnsafePointer[UInt8], destination: UnsafePointer[T], count: Int) raises:
    if external_call["fread", UInt](destination, UInt(size_of[T]()), UInt(count), file) != UInt(
        count
    ):
        raise Error("truncated fiber benchmark input")


fn _write_results(path: String, data: UnsafePointer[Float64], count: Int) raises:
    var file = _open(path, True)
    var written = external_call["fwrite", UInt](data, UInt(8), UInt(count), file)
    if external_call["fclose", Int32](file) != 0 or written != UInt(count):
        raise Error("could not write fiber benchmark results")


fn _load_step(step: Int) -> (Float64, Float64):
    if step < 10:
        return (0.0, Float64(step + 1) / 10.0)
    # Six-step pulses with tension, compression and reversals. Strain history is
    # synthetic for this benchmark. The model supplies material definitions and counts.
    var phase = (step - 10) % 6
    var swing = InlineArray[Float64, 6](0.25, 1.0, 0.35, -0.4, -1.0, -0.1)[phase] * (
        0.5 + 0.1 * Float64((step - 10) // 6)
    )
    return (swing, 1.0)


fn _cpu_oracle(
    defs: UnsafePointer[Float64],
    strains: UnsafePointer[Float64],
    ids: UnsafePointer[Int32],
    steel_groups: Int,
    n: Int,
    result: UnsafePointer[Float64],
):
    for i in range(n):
        var group = Int(ids[i])
        var mat = defs + group * PARAM_FIELDS
        var material: UniMaterialDef
        if group < steel_groups:
            material = UniMaterialDef(
                UniMaterialTypeTag.Steel02,
                mat[0],
                mat[1],
                mat[2],
                mat[3],
                mat[4],
                mat[5],
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
            )
        else:
            material = UniMaterialDef(UniMaterialTypeTag.Concrete01, mat[0], mat[1], mat[2], mat[3])
        var state = UniMaterialState(material)
        for step in range(STEPS):
            var movement, gravity = _load_step(step)
            var strain = strains[i] * movement + strains[n + i] * gravity
            uniaxial_set_trial_strain(material, state, strain)
            result[2 * step * n + i] = state.sig_t
            result[(2 * step + 1) * n + i] = state.tangent_t
            uniaxial_commit(state)


fn _cpu_run(
    state_ptr: UnsafePointer[Float64],
    def_ptr: UnsafePointer[Float64],
    strain_ptr: UnsafePointer[Float64],
    ids: UnsafePointer[Int32],
    offsets: List[Int],
    steel_groups: Int,
    n: Int,
    workers: Int,
    samples: Int,
    output: String,
) raises:
    var audit = UnsafePointer[Float64]()
    if n <= AUDIT_MAX_FIBERS:
        audit = UnsafePointer[Float64].alloc(2 * n * STEPS)
    for sample in range(samples + 1):
        _reset(state_ptr, def_ptr, ids, n, steel_groups)
        var start_ns = perf_counter_ns()
        for step in range(STEPS):
            var movement, gravity = _load_step(step)
            for group in range(len(offsets) - 1):
                var start = offsets[group]
                var end = offsets[group + 1]
                if start == end:
                    continue
                if group < steel_groups:
                    _cpu_grouped[True](
                        state_ptr,
                        strain_ptr,
                        def_ptr,
                        n,
                        start,
                        end,
                        group,
                        movement,
                        gravity,
                        workers,
                    )
                else:
                    _cpu_grouped[False](
                        state_ptr,
                        strain_ptr,
                        def_ptr,
                        n,
                        start,
                        end,
                        group,
                        movement,
                        gravity,
                        workers,
                    )
            if audit and sample == 0:
                for i in range(2 * n):
                    audit[step * 2 * n + i] = state_ptr[n + i]
        var elapsed_ns = perf_counter_ns() - start_ns
        if sample > 0:
            print("cpu_grouped", Float64(elapsed_ns) * 1e-9)
    _write_results(output, state_ptr + n, 2 * n)
    if audit:
        _write_results(output + ".history", audit, 2 * n * STEPS)
        audit.free()


fn _gpu_run[
    grouped: Bool
](
    ctx: DeviceContext,
    host_ptr: UnsafePointer[Float64],
    defs: UnsafePointer[Float64],
    strains: UnsafePointer[Float64],
    ids: UnsafePointer[Int32],
    offsets: List[Int],
    steel_groups: Int,
    n: Int,
    samples: Int,
    output: String,
) raises:
    var state = ctx.enqueue_create_buffer[DType.float64](STATE_FIELDS * n)
    var parameters = ctx.enqueue_create_buffer[DType.float64](PARAM_FIELDS * (len(offsets) - 1))
    var strains_gpu = ctx.enqueue_create_buffer[DType.float64](2 * n)
    var groups_gpu = ctx.enqueue_create_buffer[DType.int32](n)
    with parameters.map_to_host() as host:
        for i in range(PARAM_FIELDS * (len(offsets) - 1)):
            host[i] = defs[i]
    with strains_gpu.map_to_host() as host:
        for i in range(2 * n):
            host[i] = strains[i]
    with groups_gpu.map_to_host() as host:
        for i in range(n):
            host[i] = ids[i]
    _reset(host_ptr, defs, ids, n, steel_groups)
    var audit = UnsafePointer[Float64]()
    if n <= AUDIT_MAX_FIBERS:
        audit = UnsafePointer[Float64].alloc(2 * n * STEPS)
    for sample in range(samples + 1):
        with state.map_to_host() as host:
            for i in range(STATE_FIELDS * n):
                host[i] = host_ptr[i]
        ctx.synchronize()
        var start_ns = perf_counter_ns()
        for step in range(STEPS):
            var movement, gravity = _load_step(step)

            @parameter
            if grouped:
                for group in range(len(offsets) - 1):
                    var start = offsets[group]
                    var end = offsets[group + 1]
                    if start == end:
                        continue
                    var blocks = (end - start + THREADS - 1) // THREADS
                    if group < steel_groups:
                        ctx.enqueue_function[_gpu_grouped[True]](
                            state,
                            strains_gpu,
                            parameters,
                            n,
                            start,
                            end,
                            group,
                            movement,
                            gravity,
                            grid_dim=blocks,
                            block_dim=THREADS,
                        )
                    else:
                        ctx.enqueue_function[_gpu_grouped[False]](
                            state,
                            strains_gpu,
                            parameters,
                            n,
                            start,
                            end,
                            group,
                            movement,
                            gravity,
                            grid_dim=blocks,
                            block_dim=THREADS,
                        )
            else:
                ctx.enqueue_function[_gpu_tagged](
                    state,
                    strains_gpu,
                    parameters,
                    groups_gpu,
                    n,
                    steel_groups,
                    movement,
                    gravity,
                    grid_dim=(n + THREADS - 1) // THREADS,
                    block_dim=THREADS,
                )
            if audit and sample == 0:
                with state.map_to_host() as host:
                    for i in range(2 * n):
                        audit[step * 2 * n + i] = host[n + i]
        ctx.synchronize()
        var elapsed_ns = perf_counter_ns() - start_ns
        var label = "gpu_grouped" if grouped else "gpu_tagged"
        if sample > 0:
            print(label, Float64(elapsed_ns) * 1e-9)
    with state.map_to_host() as host:
        for i in range(2 * n):
            host_ptr[i] = host[n + i]
    _write_results(output, host_ptr, 2 * n)
    if audit:
        _write_results(output + ".history", audit, 2 * n * STEPS)
        audit.free()


fn main() raises:
    var args = argv()
    if len(args) != 6:
        raise Error("usage: sweep input cpu|gpu_grouped|gpu_tagged|oracle samples workers output")
    var mode = String(args[2])
    var workers = Int(args[4])
    var samples = Int(args[3])
    var output = String(args[5])
    var file = _open(String(args[1]), False)
    var header = InlineArray[Int, 3](fill=0)
    _read(file, UnsafePointer(to=header[0]), 3)
    var n = header[0]
    var groups = header[1]
    var steel_groups = header[2]
    if (
        n <= 0
        or groups <= 0
        or steel_groups < 0
        or steel_groups > groups
        or workers <= 0
        or samples <= 0
    ):
        raise Error("invalid fiber benchmark dimensions")
    var offsets = List[Int](length=groups + 1, fill=0)
    _read(file, offsets.unsafe_ptr(), len(offsets))
    var defs = UnsafePointer[Float64].alloc(PARAM_FIELDS * groups)
    _read(file, defs, PARAM_FIELDS * groups)
    var ids = UnsafePointer[Int32].alloc(n)
    _read(file, ids, n)
    var strains = UnsafePointer[Float64].alloc(2 * n)
    _read(file, strains, 2 * n)
    _ = external_call["fclose", Int32](file)

    var state = UnsafePointer[Float64].alloc(STATE_FIELDS * n)
    if mode == "oracle":
        if n > AUDIT_MAX_FIBERS:
            raise Error("scalar correctness oracle is limited to 16384 fibers")
        var result = UnsafePointer[Float64].alloc(2 * n * STEPS)
        _cpu_oracle(defs, strains, ids, steel_groups, n, result)
        _write_results(output, result, 2 * n * STEPS)
        result.free()
    elif mode == "cpu":
        _cpu_run(state, defs, strains, ids, offsets, steel_groups, n, workers, samples, output)
    else:
        with DeviceContext() as ctx:
            if mode == "gpu_grouped":
                _gpu_run[True](
                    ctx, state, defs, strains, ids, offsets, steel_groups, n, samples, output
                )
            elif mode == "gpu_tagged":
                _gpu_run[False](
                    ctx, state, defs, strains, ids, offsets, steel_groups, n, samples, output
                )
            else:
                raise Error("unknown sweep mode " + mode)
    state.free()
    strains.free()
    ids.free()
    defs.free()
