from collections import List

from solver.sparse_pattern import SparsePattern
from solver.dof import node_dof_index, require_dof_in_range
from solver.run_case.input_types import NodeInput, SupportMotionInput
from solver.time_series import TimeSeriesInput, eval_time_series_input, find_time_series_input
from os import abort


fn append_penalty_pairs(mut offsets: List[Int], mut pool: List[Int], mp_dofs: List[Int]):
    for i in range(0, len(mp_dofs), 2):
        pool.append(mp_dofs[i])
        pool.append(mp_dofs[i + 1])
        offsets.append(len(pool))


struct PenaltyConstraints(Movable):
    var sp_equations: List[Int]
    var mp_equations: List[Int]
    var stiffness: List[Float64]
    var sp_values: List[Float64]
    var sp_target: List[Float64]
    var motion_equations: List[Int]
    var motion_series: List[Int]
    var alpha_sp: Float64
    var alpha_mp: Float64

    fn __init__(out self):
        self.sp_equations = []
        self.mp_equations = []
        self.stiffness = []
        self.sp_values = []
        self.sp_target = []
        self.motion_equations = []
        self.motion_series = []
        self.alpha_sp = 0.0
        self.alpha_mp = 0.0

    fn prepare(
        mut self,
        pattern: SparsePattern,
        free_index: List[Int],
        sp_dofs: List[Int],
        mp_dofs: List[Int],
        alpha_sp: Float64,
        alpha_mp: Float64,
        sp_values: List[Float64] = [],
    ):
        self.alpha_sp = alpha_sp
        self.alpha_mp = alpha_mp
        self.sp_values.resize(len(sp_dofs), 0.0)
        var offset = len(sp_dofs) - len(sp_values)
        for i in range(len(sp_values)):
            self.sp_values[offset + i] = sp_values[i]
        self.stiffness.resize(pattern.nnz(), 0.0)
        for dof in sp_dofs:
            var eq = free_index[dof]
            self.sp_equations.append(eq)
            self.stiffness[pattern.diag_slot[eq]] += alpha_sp
        for i in range(0, len(mp_dofs), 2):
            var retained = free_index[mp_dofs[i]]
            var constrained = free_index[mp_dofs[i + 1]]
            self.mp_equations.append(retained)
            self.mp_equations.append(constrained)
            self.stiffness[pattern.diag_slot[retained]] += alpha_mp
            self.stiffness[pattern.diag_slot[constrained]] += alpha_mp
            self.stiffness[pattern.require_slot(retained, constrained)] -= alpha_mp
            self.stiffness[pattern.require_slot(constrained, retained)] -= alpha_mp

    fn add_motions(
        mut self,
        motions: List[SupportMotionInput],
        nodes: List[NodeInput],
        id_to_index: List[Int],
        ndf: Int,
        free_index: List[Int],
        pattern: SparsePattern,
        time_series: List[TimeSeriesInput],
    ):
        for motion in motions:
            if motion.node < 0 or motion.node >= len(id_to_index) or id_to_index[motion.node] < 0:
                abort("imposed-motion node not found")
            var node = id_to_index[motion.node]
            require_dof_in_range(motion.dof, nodes[node].ndf, "imposed motion")
            var eq = free_index[node_dof_index(node, motion.dof, ndf)]
            var series = find_time_series_input(time_series, motion.time_series)
            if series < 0:
                abort("imposed-motion time series not found")
            self.motion_equations.append(eq)
            self.motion_series.append(series)
            self.stiffness[pattern.diag_slot[eq]] += self.alpha_sp
        self.sp_target.resize(len(self.sp_equations) + len(self.motion_equations), 0.0)

    fn update_targets(
        mut self,
        time: Float64,
        scale: Float64,
        series: List[TimeSeriesInput],
        values: List[Float64],
        times: List[Float64],
    ) raises:
        for i in range(len(self.sp_values)):
            self.sp_target[i] = scale * self.sp_values[i]
        for i in range(len(self.motion_equations)):
            self.sp_target[len(self.sp_values) + i] = eval_time_series_input(
                series[self.motion_series[i]], time, values, times
            )

    fn subtract_force(
        self,
        u: List[Float64],
        mut residual: List[Float64],
        free: List[Int] = [],
    ):
        for i in range(len(self.sp_equations)):
            var eq = self.sp_equations[i]
            var dof = free[eq] if len(free) > 0 else eq
            residual[eq] += self.alpha_sp * (self.sp_target[i] - u[dof])
        for i in range(len(self.motion_equations)):
            var eq = self.motion_equations[i]
            var dof = free[eq] if len(free) > 0 else eq
            residual[eq] += self.alpha_sp * (self.sp_target[len(self.sp_equations) + i] - u[dof])
        for i in range(0, len(self.mp_equations), 2):
            var retained = self.mp_equations[i]
            var constrained = self.mp_equations[i + 1]
            var constrained_dof = free[constrained] if len(free) > 0 else constrained
            var retained_dof = free[retained] if len(free) > 0 else retained
            var force = self.alpha_mp * (u[constrained_dof] - u[retained_dof])
            residual[constrained] -= force
            residual[retained] += force
