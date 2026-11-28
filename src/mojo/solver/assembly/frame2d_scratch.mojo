"""Worker-local kernel buffers and staged results for 2D fiber frames."""

from elements import (
    ForceBeamColumn2dScratch,
    reset_force_beam_column2d_scratch,
    invalidate_force_beam_column2d_load_cache,
)
from solver.assembly.parallel_elements import ParallelElementWork


alias FRAME2D_DOFS = 6
alias FRAME2D_BATCH = 2048
alias FRAME2D_MINIMUM_WORK = 384


struct Frame2dWorker(Copyable, Movable):
    var kernel: ForceBeamColumn2dScratch
    var u: List[Float64]
    var matrix: List[List[Float64]]
    var force: List[Float64]

    fn __init__(out self):
        self.kernel = ForceBeamColumn2dScratch()
        self.u = []
        self.u.resize(FRAME2D_DOFS, 0.0)
        self.matrix = []
        self.force = []


struct Frame2dAssemblyScratch(Movable):
    var work: ParallelElementWork
    var workers: List[Frame2dWorker]

    fn __init__(out self):
        self.work = ParallelElementWork(minimum_work=FRAME2D_MINIMUM_WORK)
        self.workers = []

    fn ensure_workers(mut self, count: Int):
        while len(self.workers) < count:
            self.workers.append(Frame2dWorker())

    fn reset(mut self):
        self.work.plan.cumulative.resize(1, 0)
        for i in range(len(self.workers)):
            reset_force_beam_column2d_scratch(self.workers[i].kernel)

    fn invalidate_load_cache(mut self):
        for i in range(len(self.workers)):
            invalidate_force_beam_column2d_load_cache(self.workers[i].kernel)
