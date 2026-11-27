"""Numeric LDLᵀ on a Mojo device, using the CPU solver's packed supernodes."""

from collections import InlineArray, List
from gpu import WARP_SIZE, barrier, block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import AddressSpace
from math import fma, isfinite
from memory import UnsafePointer, stack_allocation
from os.atomic import Atomic, Consistency

alias _THREADS = 128
alias _UPDATE_COLUMNS = 4
alias _PIVOTS = 16
alias _TILE_ROWS = 32
alias _TILE_COLS = 16

alias _ADD_UPDATES = 0
alias _FACTOR = 1
alias _SOLVE_ROWS = 2
alias _UPDATE_TAIL = 3
alias _DENSE_UPDATES = 4


@fieldwise_init
struct PanelUpdate(ImplicitlyCopyable, Movable):
    var source: Int
    var first: Int
    var split: Int


@fieldwise_init
struct _Batch(ImplicitlyCopyable, Movable):
    var kind: Int
    var offset: Int
    var blocks: Int


struct _Round(Copyable, Movable):
    var diagonal: List[Int32]
    var solve_rows: List[Int32]
    var tail: List[Int32]

    fn __init__(out self):
        self.diagonal = []
        self.solve_rows = []
        self.tail = []


struct _Level(Copyable, Movable):
    var updates: List[Int32]
    var dense_updates: List[Int32]
    var small_panels: List[Int32]
    var rounds: List[_Round]

    fn __init__(out self):
        self.updates = []
        self.dense_updates = []
        self.small_panels = []
        self.rounds = []


fn _add_batch(
    mut batches: List[_Batch],
    mut jobs: List[Int32],
    kind: Int,
    width: Int,
    data: List[Int32],
):
    if len(data) > 0:
        batches.append(_Batch(kind, len(jobs), len(data) // width))
        for value in data:
            jobs.append(value)


fn _upload(context: DeviceContext, data: List[Int32]) raises -> DeviceBuffer[DType.int32]:
    # Empty lists may have a null host pointer even when no copy is needed.
    var buf = context.enqueue_create_buffer[DType.int32](max(1, len(data)))
    if len(data) > 0:
        buf.enqueue_copy_from(data.unsafe_ptr())
        context.synchronize()
    return buf^


fn _int32_list(values: List[Int]) -> List[Int32]:
    var result = List[Int32](capacity=len(values))
    for i in range(len(values)):
        result.append(Int32(values[i]))
    return result^


fn _scatter_values(
    count: Int,
    input: UnsafePointer[UInt32],
    scatter: UnsafePointer[UInt32],
    values: UnsafePointer[Float64],
    factor: UnsafePointer[Float64],
):
    var i = block_idx.x * _THREADS + thread_idx.x
    if i < count:
        factor[Int(scatter[i])] = values[Int(input[i])]


@always_inline
fn _find_row(rows: UnsafePointer[Int32], first: Int, end: Int, row: Int) -> Int:
    var lo = first
    var hi = end
    var first_row = Int(rows[first])
    var last_row = Int(rows[end - 1])
    if row < first_row or row > last_row:
        return -1
    if last_row - first_row == end - first - 1:
        return first + row - first_row
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if Int(rows[mid]) < row:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < end and Int(rows[lo]) == row else -1


fn _add_updates(
    jobs: UnsafePointer[Int32],
    offset: Int,
    panels: UnsafePointer[Int32],
    rows: UnsafePointer[Int32],
    columns: UnsafePointer[Int32],
    updates: UnsafePointer[Int32],
    factor: UnsafePointer[Float64],
):
    var job = jobs + offset + 5 * block_idx.x
    var panel = panels + 4 * Int(job[0])
    var end = Int(panel[1])
    var row_start = Int(panel[2])
    var row_offset = Int(job[2]) + thread_idx.x
    if row_offset >= end + Int(panel[3]) - row_start:
        return
    var row = row_offset if row_offset < end else Int(rows[row_start + row_offset - end])
    var col_base = Int(job[1])
    var value = InlineArray[Float64, _UPDATE_COLUMNS](fill=0.0)
    var destination = InlineArray[Int, _UPDATE_COLUMNS](fill=-1)

    @parameter
    for i in range(_UPDATE_COLUMNS):
        var col = col_base + i
        if col < end and col <= row_offset:
            destination[i] = Int(columns[col]) + row_offset - col
            value[i] = factor[destination[i]]

    # A workgroup owns these destinations. It can accumulate all incoming
    # contributions in registers without global floating-point atomics.
    for u in range(Int(job[3]), Int(job[4]), 2 + _UPDATE_COLUMNS):
        var source = panels + 4 * Int(updates[u])
        var slot = _find_row(rows, Int(updates[u + 1]), Int(source[3]), row)
        if slot < 0:
            continue
        var external = Int(source[1]) - Int(source[2])
        for k in range(Int(source[0]), Int(source[1])):
            var diag = Int(columns[k])
            var base = diag + external - k
            var d = -factor[diag]
            var left = factor[base + slot]

            @parameter
            for i in range(_UPDATE_COLUMNS):
                var col_slot = Int(updates[u + 2 + i])
                if destination[i] >= 0 and col_slot >= 0:
                    value[i] = fma(d * factor[base + col_slot], left, value[i])

    @parameter
    for i in range(_UPDATE_COLUMNS):
        if destination[i] >= 0:
            factor[destination[i]] = value[i]


fn _add_dense_updates(
    jobs: UnsafePointer[Int32],
    offset: Int,
    panels: UnsafePointer[Int32],
    rows: UnsafePointer[Int32],
    columns: UnsafePointer[Int32],
    destinations: UnsafePointer[Int32],
    factor: UnsafePointer[Float64],
):
    var job = jobs + offset + 6 * block_idx.x
    var source = panels + 4 * Int(job[0])
    var split = Int(job[1])
    var map_base = Int(job[2])
    var row_base = Int(job[3])
    var col_base = Int(job[4])
    var first_slot = Int(job[5])
    var source_end = Int(source[1])
    var rows_end = Int(source[3])
    var base = source_end - Int(source[2])
    var lane = thread_idx.x
    var local_row = lane % _TILE_ROWS
    var row = row_base + local_row
    var col_group = lane // _TILE_ROWS
    alias _COLUMNS_PER_THREAD = _TILE_COLS // (_THREADS // _TILE_ROWS)
    var left = stack_allocation[
        _PIVOTS * _TILE_ROWS, Float64, address_space = AddressSpace.SHARED
    ]()
    var right = stack_allocation[
        _PIVOTS * _TILE_COLS, Float64, address_space = AddressSpace.SHARED
    ]()
    var update = InlineArray[Float64, _COLUMNS_PER_THREAD](fill=0.0)

    # Each workgroup computes one lower-triangle tile of the source's
    # Schur complement. Different source panels may share a destination.
    for rank in range(Int(source[0]), source_end, _PIVOTS):
        for i in range(lane, _PIVOTS * _TILE_ROWS, _THREADS):
            var k = rank + i // _TILE_ROWS
            var r = row_base + i % _TILE_ROWS
            left[i] = (
                factor[Int(columns[k]) + base + r - k] if k < source_end and r < rows_end else 0.0
            )
        for i in range(lane, _PIVOTS * _TILE_COLS, _THREADS):
            var k = rank + i // _TILE_COLS
            var c = col_base + i % _TILE_COLS
            var weighted = 0.0
            if k < source_end and c < split:
                var diag = Int(columns[k])
                weighted = -factor[diag] * factor[diag + base + c - k]
            right[i] = weighted
        barrier()

        @parameter
        for k in range(_PIVOTS):

            @parameter
            for i in range(_COLUMNS_PER_THREAD):
                var col = col_group + i * (_THREADS // _TILE_ROWS)
                update[i] = fma(
                    left[k * _TILE_ROWS + local_row], right[k * _TILE_COLS + col], update[i]
                )
        barrier()

    if row >= rows_end:
        return
    var dest_row = Int(destinations[map_base + row - first_slot])

    @parameter
    for i in range(_COLUMNS_PER_THREAD):
        var col = col_base + col_group + i * (_THREADS // _TILE_ROWS)
        if col < split and col <= row:
            var dest_col = Int(rows[col])
            var entry = Int(columns[dest_col]) + dest_row - dest_col
            _ = Atomic[DType.float64].fetch_add[ordering = Consistency.MONOTONIC](
                factor + entry, update[i]
            )


fn _factor_pivots(
    jobs: UnsafePointer[Int32],
    offset: Int,
    columns: UnsafePointer[Int32],
    scale: UnsafePointer[Float64],
    factor: UnsafePointer[Float64],
    status: UnsafePointer[Int32],
):
    var job = jobs + offset + 3 * block_idx.x
    var first = Int(job[0])
    var end = Int(job[1])
    var row_end = Int(job[2])
    var lane = thread_idx.x
    var shared_pivot = stack_allocation[1, Float64, address_space = AddressSpace.SHARED]()
    for col in range(first, end):
        var diag = Int(columns[col])
        if lane == 0:
            var pivot = factor[diag]
            if not isfinite(pivot) or abs(pivot) <= max(1e-300, 1e-14 * scale[col]):
                _ = Atomic[DType.int32].fetch_add[ordering = Consistency.MONOTONIC](status, 1)
                pivot = 0.0
            shared_pivot[0] = pivot
        barrier()
        var pivot = shared_pivot[0]
        if pivot == 0.0:
            return
        var reciprocal = 1.0 / pivot
        for row in range(col + 1 + lane, row_end, _THREADS):
            factor[diag + row - col] *= reciprocal
        barrier()
        for target in range(col + 1 + lane // WARP_SIZE, end, _THREADS // WARP_SIZE):
            var dest = Int(columns[target])
            var weight = -pivot * factor[diag + target - col]
            for row in range(target + lane % WARP_SIZE, row_end, WARP_SIZE):
                var entry = dest + row - target
                factor[entry] = fma(weight, factor[diag + row - col], factor[entry])
        barrier()


fn _solve_rows(
    jobs: UnsafePointer[Int32],
    offset: Int,
    columns: UnsafePointer[Int32],
    factor: UnsafePointer[Float64],
):
    var job = jobs + offset + 4 * block_idx.x
    var first = Int(job[0])
    var width = Int(job[1]) - first
    var lane = thread_idx.x
    var row = Int(job[2]) + lane
    var weights = stack_allocation[
        _PIVOTS * _PIVOTS, Float64, address_space = AddressSpace.SHARED
    ]()
    var reciprocals = stack_allocation[_PIVOTS, Float64, address_space = AddressSpace.SHARED]()
    if lane < width:
        reciprocals[lane] = 1.0 / factor[Int(columns[first + lane])]
    for i in range(lane, _PIVOTS * _PIVOTS, _THREADS):
        var k = i // _PIVOTS
        var j = i % _PIVOTS
        if j < width and k < j:
            var diag = Int(columns[first + k])
            weights[i] = -factor[diag] * factor[diag + j - k]
    barrier()
    if row >= Int(job[3]):
        return
    var solved = InlineArray[Float64, _PIVOTS](fill=0.0)

    @parameter
    for j in range(_PIVOTS):
        if j < width:
            var col = first + j
            var entry = Int(columns[col]) + row - col
            var value = factor[entry]

            @parameter
            for k in range(j):
                value = fma(weights[k * _PIVOTS + j], solved[k], value)
            solved[j] = value * reciprocals[j]
            factor[entry] = solved[j]


fn _update_tail(
    jobs: UnsafePointer[Int32],
    offset: Int,
    columns: UnsafePointer[Int32],
    factor: UnsafePointer[Float64],
):
    var job = jobs + offset + 6 * block_idx.x
    var first = Int(job[0])
    var end = Int(job[1])
    var row_base = Int(job[2])
    var col_base = Int(job[3])
    var row_end = Int(job[4])
    var col_end = Int(job[5])
    var lane = thread_idx.x
    var row = row_base + lane % _TILE_ROWS
    var column_group = lane // _TILE_ROWS
    var left = stack_allocation[
        _PIVOTS * _TILE_ROWS, Float64, address_space = AddressSpace.SHARED
    ]()
    var right = stack_allocation[
        _PIVOTS * _TILE_COLS, Float64, address_space = AddressSpace.SHARED
    ]()
    for i in range(lane, _PIVOTS * _TILE_ROWS, _THREADS):
        var k = first + i // _TILE_ROWS
        var r = row_base + i % _TILE_ROWS
        left[i] = factor[Int(columns[k]) + r - k] if k < end and r < row_end else 0.0
    for i in range(lane, _PIVOTS * _TILE_COLS, _THREADS):
        var k = first + i // _TILE_COLS
        var c = col_base + i % _TILE_COLS
        var weighted = 0.0
        if k < end and c < col_end:
            var diag = Int(columns[k])
            weighted = -factor[diag] * factor[diag + c - k]
        right[i] = weighted
    barrier()
    if row >= row_end:
        return

    @parameter
    for i in range(_TILE_COLS // (_THREADS // _TILE_ROWS)):
        var local_col = column_group + i * (_THREADS // _TILE_ROWS)
        var col = col_base + local_col
        if col < col_end and col <= row:
            var entry = Int(columns[col]) + row - col
            var value = factor[entry]

            @parameter
            for k in range(_PIVOTS):
                value = fma(
                    left[k * _TILE_ROWS + lane % _TILE_ROWS],
                    right[k * _TILE_COLS + local_col],
                    value,
                )
            factor[entry] = value


struct GpuLdlt(Copyable, Movable):
    var context: DeviceContext
    var batches: List[_Batch]
    var jobs: DeviceBuffer[DType.int32]
    var panels: DeviceBuffer[DType.int32]
    var rows: DeviceBuffer[DType.int32]
    var columns: DeviceBuffer[DType.int32]
    var updates: DeviceBuffer[DType.int32]
    var dense_destinations: DeviceBuffer[DType.int32]
    var input: DeviceBuffer[DType.uint32]
    var scatter: DeviceBuffer[DType.uint32]
    var values: DeviceBuffer[DType.float64]
    var scale: DeviceBuffer[DType.float64]
    var factor: DeviceBuffer[DType.float64]
    var status: DeviceBuffer[DType.int32]

    fn __init__(
        out self,
        context: DeviceContext,
        panels: List[Int],
        row_ptr: List[Int],
        rows: List[Int32],
        columns: List[Int],
        incoming_ptr: List[Int],
        incoming: List[PanelUpdate],
        input: List[UInt32],
        scatter: List[UInt32],
    ) raises:
        var count = len(panels) - 1
        var depths = List[Int](length=count, fill=0)
        var levels = List[_Level]()
        var descriptors = List[Int32](capacity=4 * count)
        var updates = List[Int32]()
        var dense_destinations = List[Int32]()
        for panel in range(count):
            var first = panels[panel]
            var end = panels[panel + 1]
            var row_end = end + row_ptr[panel + 1] - row_ptr[panel]
            descriptors.extend(
                [Int32(first), Int32(end), Int32(row_ptr[panel]), Int32(row_ptr[panel + 1])]
            )
            var depth = 0
            var in_first = incoming_ptr[panel]
            var in_end = incoming_ptr[panel + 1]
            for i in range(in_first, in_end):
                depth = max(depth, depths[incoming[i].source] + 1)
            depths[panel] = depth
            while len(levels) <= depth:
                levels.append(_Level())
            ref level = levels[depth]

            if end - first < 32 or row_end - first < 96:
                level.small_panels.extend([Int32(first), Int32(end), Int32(row_end)])
            else:
                for p in range(first, end, _PIVOTS):
                    var round_index = (p - first) // _PIVOTS
                    while len(level.rounds) <= round_index:
                        level.rounds.append(_Round())
                    ref round = level.rounds[round_index]
                    var stop = min(p + _PIVOTS, end)
                    var p32 = Int32(p)
                    var stop32 = Int32(stop)
                    var row_end32 = Int32(row_end)
                    round.diagonal.extend([p32, stop32, stop32])
                    for row in range(stop, row_end, _THREADS):
                        round.solve_rows.extend([p32, stop32, Int32(row), row_end32])
                    for col in range(stop, end, _TILE_COLS):
                        for row in range(col, row_end, _TILE_ROWS):
                            round.tail.extend(
                                [p32, stop32, Int32(row), Int32(col), row_end32, Int32(end)]
                            )

            var sparse_incoming = List[PanelUpdate]()
            for i in range(in_first, in_end):
                var entry = incoming[i]
                var source_end = panels[entry.source + 1]
                var source_row_end = row_ptr[entry.source + 1]
                if (
                    source_end - panels[entry.source] < 16
                    or entry.split - entry.first < 12
                    or source_row_end - entry.first < 64
                ):
                    sparse_incoming.append(entry)
                    continue
                var map_start = Int32(len(dense_destinations))
                var cursor = row_ptr[panel]
                var end_slot = row_ptr[panel + 1]
                for slot in range(entry.first, source_row_end):
                    var label = Int(rows[slot])
                    if label < end:
                        dense_destinations.append(Int32(label))
                    else:
                        while cursor < end_slot and Int(rows[cursor]) < label:
                            cursor += 1
                        if cursor == end_slot or Int(rows[cursor]) != label:
                            raise Error("Invalid row in GPU LDLT dense update")
                        dense_destinations.append(Int32(end + cursor - row_ptr[panel]))
                for col in range(entry.first, entry.split, _TILE_COLS):
                    for row in range(col, source_row_end, _TILE_ROWS):
                        level.dense_updates.extend(
                            [
                                Int32(entry.source),
                                Int32(entry.split),
                                map_start,
                                Int32(row),
                                Int32(col),
                                Int32(entry.first),
                            ]
                        )

            for col in range(first, end, _UPDATE_COLUMNS):
                var update_start = len(updates)
                for entry in sparse_incoming:
                    if (
                        Int(rows[entry.first]) >= col + _UPDATE_COLUMNS
                        or Int(rows[entry.split - 1]) < col
                    ):
                        continue
                    var slots = InlineArray[Int32, _UPDATE_COLUMNS](fill=-1)
                    var overlapping = False
                    for k in range(entry.first, entry.split):
                        var label = Int(rows[k])
                        if label >= col + _UPDATE_COLUMNS:
                            break
                        if label >= col:
                            slots[label - col] = Int32(k)
                            overlapping = True
                    if overlapping:
                        updates.append(Int32(entry.source))
                        updates.append(Int32(entry.first))
                        for k in range(_UPDATE_COLUMNS):
                            updates.append(slots[k])
                if update_start < len(updates):
                    for row in range(col, row_end, _THREADS):
                        level.updates.extend(
                            [
                                Int32(panel),
                                Int32(col),
                                Int32(row),
                                Int32(update_start),
                                Int32(len(updates)),
                            ]
                        )

        var batches = List[_Batch]()
        var jobs = List[Int32]()
        for depth in range(len(levels)):
            ref level = levels[depth]
            _add_batch(batches, jobs, _DENSE_UPDATES, 6, level.dense_updates)
            _add_batch(batches, jobs, _ADD_UPDATES, 5, level.updates)
            _add_batch(batches, jobs, _FACTOR, 3, level.small_panels)
            for i in range(len(level.rounds)):
                ref round = level.rounds[i]
                _add_batch(batches, jobs, _FACTOR, 3, round.diagonal)
                _add_batch(batches, jobs, _SOLVE_ROWS, 4, round.solve_rows)
                _add_batch(batches, jobs, _UPDATE_TAIL, 6, round.tail)

        self.context = context
        self.batches = batches^
        self.jobs = _upload(context, jobs)
        self.panels = _upload(context, descriptors)
        self.rows = _upload(context, rows)
        self.columns = _upload(context, _int32_list(columns))
        self.updates = _upload(context, updates)
        self.dense_destinations = _upload(context, dense_destinations)
        self.input = context.enqueue_create_buffer[DType.uint32](len(input))
        self.scatter = context.enqueue_create_buffer[DType.uint32](len(scatter))
        self.values = context.enqueue_create_buffer[DType.float64](len(input))
        self.scale = context.enqueue_create_buffer[DType.float64](len(columns) - 1)
        self.factor = context.enqueue_create_buffer[DType.float64](columns[len(columns) - 1])
        self.status = context.enqueue_create_buffer[DType.int32](1)
        self.input.enqueue_copy_from(input.unsafe_ptr())
        self.scatter.enqueue_copy_from(scatter.unsafe_ptr())
        context.synchronize()

    fn factorize(
        self, values: List[Float64], scale: List[Float64], mut result: List[Float64]
    ) raises -> Bool:
        self.values.enqueue_copy_from(values.unsafe_ptr())
        self.scale.enqueue_copy_from(scale.unsafe_ptr())
        _ = self.factor.enqueue_fill(0.0)
        _ = self.status.enqueue_fill(0)
        var ctx = self.context
        var scatter_kernel = ctx.compile_function[_scatter_values]()
        var dense_updates_kernel = ctx.compile_function[_add_dense_updates]()
        var updates_kernel = ctx.compile_function[_add_updates]()
        var factor_kernel = ctx.compile_function[_factor_pivots]()
        var solve_kernel = ctx.compile_function[_solve_rows]()
        var tail_kernel = ctx.compile_function[_update_tail]()
        ctx.enqueue_function(
            scatter_kernel,
            len(values),
            self.input,
            self.scatter,
            self.values,
            self.factor,
            grid_dim=(len(values) + _THREADS - 1) // _THREADS,
            block_dim=_THREADS,
        )
        for i in range(len(self.batches)):
            var batch = self.batches[i]
            if batch.kind == _DENSE_UPDATES:
                ctx.enqueue_function(
                    dense_updates_kernel,
                    self.jobs,
                    batch.offset,
                    self.panels,
                    self.rows,
                    self.columns,
                    self.dense_destinations,
                    self.factor,
                    grid_dim=batch.blocks,
                    block_dim=_THREADS,
                )
            elif batch.kind == _ADD_UPDATES:
                ctx.enqueue_function(
                    updates_kernel,
                    self.jobs,
                    batch.offset,
                    self.panels,
                    self.rows,
                    self.columns,
                    self.updates,
                    self.factor,
                    grid_dim=batch.blocks,
                    block_dim=_THREADS,
                )
            elif batch.kind == _FACTOR:
                ctx.enqueue_function(
                    factor_kernel,
                    self.jobs,
                    batch.offset,
                    self.columns,
                    self.scale,
                    self.factor,
                    self.status,
                    grid_dim=batch.blocks,
                    block_dim=_THREADS,
                )
            elif batch.kind == _SOLVE_ROWS:
                ctx.enqueue_function(
                    solve_kernel,
                    self.jobs,
                    batch.offset,
                    self.columns,
                    self.factor,
                    grid_dim=batch.blocks,
                    block_dim=_THREADS,
                )
            else:
                ctx.enqueue_function(
                    tail_kernel,
                    self.jobs,
                    batch.offset,
                    self.columns,
                    self.factor,
                    grid_dim=batch.blocks,
                    block_dim=_THREADS,
                )
        var invalid = Int32(0)
        self.status.enqueue_copy_to(UnsafePointer(to=invalid))
        self.factor.enqueue_copy_to(result.unsafe_ptr())
        ctx.synchronize()
        return invalid == 0
