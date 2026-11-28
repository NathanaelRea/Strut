"""Bounded compute-then-scatter storage for element assembly.

Workers compute contiguous ranges. Each has one exclusive scratch slice and
writes one exclusive matrix and force slice per element. The caller gives each
element exclusive ownership of any mutable constitutive state it accesses and
keeps all captured buffers alive and at fixed addresses until compute returns.
Scatter runs synchronously in original element order on the caller's thread.
"""

from algorithm import parallelize
from collections import List
from os import abort

from solver.threads import element_worker_count


struct ElementWorkPlan(Movable):
    """Cumulative estimates in source order, reusable for each nonlinear iteration."""

    var cumulative: List[Int]
    var minimum_work: Int
    var threads: Int

    fn __init__(out self, minimum_work: Int):
        self.cumulative = [0]
        self.minimum_work = minimum_work
        self.threads = 0

    fn append(mut self, work: Int):
        self.cumulative.append(self.cumulative[len(self.cumulative) - 1] + work)

    fn worker_count(mut self, start: Int, count: Int) -> Int:
        var work = self.cumulative[start + count] - self.cumulative[start]
        return element_worker_count(count, work, self.minimum_work, self.threads)


fn element_work_boundary(
    cumulative: UnsafePointer[Int], start: Int, count: Int, worker: Int, workers: Int
) -> Int:
    if worker == 0:
        return start
    if worker == workers:
        return start + count
    var work = cumulative[start + count] - cumulative[start]
    var target = cumulative[start] + work * worker // workers
    var low = start
    var high = start + count
    while high - low > 1:
        var middle = (low + high) // 2
        if cumulative[middle] < target:
            low = middle
        else:
            high = middle
    if target - cumulative[low] < cumulative[high] - target:
        return low
    return high


struct ParallelElementWork(Movable):
    """Flat matrix, force, and worker scratch storage reusable across sweeps."""

    var matrices: List[Float64]
    var forces: List[Float64]
    var _scratch: List[Float64]
    var capacity: Int
    var matrix_size: Int
    var force_size: Int
    var _scratch_size: Int
    var plan: ElementWorkPlan

    fn __init__(
        out self,
        capacity: Int = 0,
        matrix_size: Int = 0,
        force_size: Int = 0,
        scratch_size: Int = 0,
        minimum_work: Int = 1,
    ):
        self.capacity = capacity
        self.matrix_size = matrix_size
        self.force_size = force_size
        self._scratch_size = scratch_size
        self.plan = ElementWorkPlan(minimum_work)
        self.matrices = []
        self.forces = []
        self._scratch = []
        self.matrices.resize(capacity * matrix_size, 0.0)
        self.forces.resize(capacity * force_size, 0.0)

    fn prepare_scratch(mut self):
        var count = len(self.plan.cumulative) - 1
        var workers = self.plan.worker_count(0, count)
        self._scratch.resize(workers * self._scratch_size, 0.0)

    fn compute[
        Context: AnyType,
        kernel: fn (
            Context,
            Int,
            Int,
            UnsafePointer[Float64],
            UnsafePointer[Float64],
            UnsafePointer[Float64],
        ),
    ](mut self, context: Context, start: Int, count: Int):
        """Run kernel(context, index, matrix, force, scratch) for one batch."""
        if start < 0 or count < 0 or count > self.capacity:
            abort("parallel element compute range exceeds workspace")
        var workers = self.plan.worker_count(start, count)
        var cumulative = self.plan.cumulative.unsafe_ptr()
        var matrices = self.matrices.unsafe_ptr()
        var forces = self.forces.unsafe_ptr()
        var scratch = self._scratch.unsafe_ptr()
        var matrix_size = self.matrix_size
        var force_size = self.force_size
        var scratch_size = self._scratch_size

        @parameter
        fn run(worker: Int):
            var local_scratch = scratch + worker * scratch_size
            var first = element_work_boundary(cumulative, start, count, worker, workers)
            var end = element_work_boundary(cumulative, start, count, worker + 1, workers)
            for index in range(first, end):
                var local_idx = index - start
                kernel(
                    context,
                    index,
                    worker,
                    matrices + local_idx * matrix_size,
                    forces + local_idx * force_size,
                    local_scratch,
                )

        if workers == 1:
            run(0)
        else:
            parallelize[run](workers, workers)

    fn scatter[
        Context: AnyType,
        add: fn (Context, Int, Int, Int),
    ](self, context: Context, start: Int, count: Int):
        """Run add(context, index, matrix_offset, force_offset) in source order."""
        if start < 0 or count < 0 or count > self.capacity:
            abort("parallel element scatter range exceeds workspace")
        for local_idx in range(count):
            add(
                context,
                start + local_idx,
                local_idx * self.matrix_size,
                local_idx * self.force_size,
            )
