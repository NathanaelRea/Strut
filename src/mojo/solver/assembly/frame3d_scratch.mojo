"""Driver-owned scratch for parallel beam-column assembly.

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
from solver.assembly.parallel_elements import ElementWorkPlan
from solver.assembly.frame2d_scratch import Frame2dAssemblyScratch


alias FRAME3D_DOFS: Int = 12
alias FRAME3D_MATRIX_SIZE: Int = 144
# Elements computed per parallel sweep; bounds the result buffers to about 2.5 MB.
alias FRAME3D_BATCH: Int = 2048
# A fiber trial is the unit of work. Elastic elements contribute a small fixed
# equivalent for their geometry and global response.
alias FRAME3D_MINIMUM_WORK: Int = 512


struct Frame3dAssemblyScratch(Movable):
    var frame2d: Frame2dAssemblyScratch
    var main: ForceBeamColumn3dScratch
    var workers: List[ForceBeamColumn3dScratch]
    var worker_u: List[List[Float64]]
    var worker_k: List[List[List[Float64]]]
    var worker_f: List[List[Float64]]
    var matrices: List[Float64]
    var forces: List[Float64]
    var plan: ElementWorkPlan
    var has_corotational: Bool

    fn __init__(out self):
        self.frame2d = Frame2dAssemblyScratch()
        self.main = ForceBeamColumn3dScratch()
        self.workers = []
        self.worker_u = []
        self.worker_k = []
        self.worker_f = []
        self.matrices = []
        self.forces = []
        self.plan = ElementWorkPlan(FRAME3D_MINIMUM_WORK)
        self.has_corotational = False

    fn ensure_workers(mut self, count: Int, batch_capacity: Int):
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
        if len(self.forces) < batch_capacity * FRAME3D_DOFS:
            self.matrices.resize(batch_capacity * FRAME3D_MATRIX_SIZE, 0.0)
            self.forces.resize(batch_capacity * FRAME3D_DOFS, 0.0)

    fn commit_corotational(mut self):
        for i in range(len(self.main.corotational_states)):
            self.main.corotational_states[i].commit()

    fn revert_corotational(mut self):
        for i in range(len(self.main.corotational_states)):
            self.main.corotational_states[i].revert()


fn reset_frame3d_assembly_scratch(mut scratch: Frame3dAssemblyScratch):
    scratch.frame2d.reset()
    reset_force_beam_column3d_scratch(scratch.main)
    scratch.plan.cumulative.resize(1, 0)
    scratch.has_corotational = False
    for worker in range(len(scratch.workers)):
        reset_force_beam_column3d_scratch(scratch.workers[worker])


fn invalidate_frame3d_assembly_load_cache(mut scratch: Frame3dAssemblyScratch):
    scratch.frame2d.invalidate_load_cache()
    invalidate_force_beam_column3d_load_cache(scratch.main)
    for worker in range(len(scratch.workers)):
        invalidate_force_beam_column3d_load_cache(scratch.workers[worker])
