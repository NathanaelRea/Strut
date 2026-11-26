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


struct ParallelElementWork(Movable):
    """Flat matrix, force, and worker scratch storage reusable across sweeps."""

    var matrices: List[Float64]
    var forces: List[Float64]
    var _scratch: List[Float64]
    var capacity: Int
    var _matrix_size: Int
    var _force_size: Int
    var _scratch_size: Int
    var _threads: Int
    var _min_per_worker: Int

    fn __init__(
        out self,
        capacity: Int = 0,
        matrix_size: Int = 0,
        force_size: Int = 0,
        scratch_size: Int = 0,
        threads: Int = 1,
        min_per_worker: Int = 1,
    ):
        self.capacity = capacity
        self._matrix_size = matrix_size
        self._force_size = force_size
        self._scratch_size = scratch_size
        self._min_per_worker = max(1, min_per_worker)
        self._threads = max(1, min(threads, capacity // self._min_per_worker))
        self.matrices = []
        self.forces = []
        self._scratch = []
        self.matrices.resize(capacity * matrix_size, 0.0)
        self.forces.resize(capacity * force_size, 0.0)
        self._scratch.resize(self._threads * scratch_size, 0.0)

    fn compute[
        Context: AnyType,
        kernel: fn(Context, Int, UnsafePointer[Float64], UnsafePointer[Float64], UnsafePointer[Float64]),
    ](mut self, context: Context, start: Int, count: Int):
        """Run kernel(context, index, matrix, force, scratch) for one batch."""
        if start < 0 or count < 0 or count > self.capacity:
            abort("parallel element compute range exceeds workspace")
        var workers = max(1, min(self._threads, count // self._min_per_worker))
        var matrices = self.matrices.unsafe_ptr()
        var forces = self.forces.unsafe_ptr()
        var scratch = self._scratch.unsafe_ptr()
        var matrix_size = self._matrix_size
        var force_size = self._force_size
        var scratch_size = self._scratch_size

        @parameter
        fn run(worker: Int):
            var local_scratch = scratch + worker * scratch_size
            var end = (worker + 1) * count // workers
            for local_idx in range(worker * count // workers, end):
                kernel(
                    context,
                    start + local_idx,
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
        add: fn(Context, Int, Int, Int),
    ](self, context: Context, start: Int, count: Int):
        """Run add(context, index, matrix_offset, force_offset) in source order."""
        if start < 0 or count < 0 or count > self.capacity:
            abort("parallel element scatter range exceeds workspace")
        for local_idx in range(count):
            add(
                context,
                start + local_idx,
                local_idx * self._matrix_size,
                local_idx * self._force_size,
            )
