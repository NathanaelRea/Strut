"""Driver-owned scratch for 3D beam-column assembly.

`main` is the kernel scratch the serial paths use. The threaded path gives every
worker its own kernel scratch and element buffers, so no two threads share
mutable kernel state, and collects element results in `matrices` and `forces`
so they can be scattered afterwards in element order.

Workers keep their own geometry and load caches. `reset_frame3d_assembly_scratch`
and `invalidate_frame3d_assembly_load_cache` clear them together with `main`, so
drivers treat this as one object.
"""

from collections import List

from elements import (
    ForceBeamColumn3dScratch,
    invalidate_force_beam_column3d_load_cache,
    reset_force_beam_column3d_scratch,
)
from solver.threads import solver_thread_count


alias FRAME3D_DOFS: Int = 12
alias FRAME3D_MATRIX_SIZE: Int = 144
# Elements computed per parallel sweep; bounds the result buffers to about 2.5 MB.
alias FRAME3D_BATCH: Int = 2048
# Below this many elements per worker the dispatch costs more than it saves.
alias FRAME3D_MIN_PER_WORKER: Int = 32


struct Frame3dAssemblyScratch(Movable):
    var main: ForceBeamColumn3dScratch
    var workers: List[ForceBeamColumn3dScratch]
    var worker_u: List[List[Float64]]
    var worker_k: List[List[List[Float64]]]
    var worker_f: List[List[Float64]]
    var matrices: List[Float64]
    var forces: List[Float64]
    # 0 until the first threaded sweep resolves it from the environment.
    var threads: Int

    fn __init__(out self):
        self.main = ForceBeamColumn3dScratch()
        self.workers = []
        self.worker_u = []
        self.worker_k = []
        self.worker_f = []
        self.matrices = []
        self.forces = []
        self.threads = 0

    fn worker_count(mut self, element_count: Int) -> Int:
        """Workers to use for `element_count` elements; 1 means stay serial."""
        if element_count < 2 * FRAME3D_MIN_PER_WORKER:
            return 1
        if self.threads == 0:
            self.threads = solver_thread_count()
        return max(1, min(self.threads, element_count // FRAME3D_MIN_PER_WORKER))

    fn ensure_workers(mut self, count: Int):
        while len(self.workers) < count:
            self.workers.append(ForceBeamColumn3dScratch())
            var u: List[Float64] = []
            u.resize(FRAME3D_DOFS, 0.0)
            self.worker_u.append(u^)
            var k: List[List[Float64]] = []
            for _ in range(FRAME3D_DOFS):
                var row: List[Float64] = []
                row.resize(FRAME3D_DOFS, 0.0)
                k.append(row^)
            self.worker_k.append(k^)
            var f: List[Float64] = []
            f.resize(FRAME3D_DOFS, 0.0)
            self.worker_f.append(f^)
        if len(self.forces) < FRAME3D_BATCH * FRAME3D_DOFS:
            self.matrices.resize(FRAME3D_BATCH * FRAME3D_MATRIX_SIZE, 0.0)
            self.forces.resize(FRAME3D_BATCH * FRAME3D_DOFS, 0.0)


fn reset_frame3d_assembly_scratch(mut scratch: Frame3dAssemblyScratch):
    reset_force_beam_column3d_scratch(scratch.main)
    for worker in range(len(scratch.workers)):
        reset_force_beam_column3d_scratch(scratch.workers[worker])


fn invalidate_frame3d_assembly_load_cache(mut scratch: Frame3dAssemblyScratch):
    invalidate_force_beam_column3d_load_cache(scratch.main)
    for worker in range(len(scratch.workers)):
        invalidate_force_beam_column3d_load_cache(scratch.workers[worker])
