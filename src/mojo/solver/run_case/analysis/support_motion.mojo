from collections import List
from os import abort

from solver.dof import node_dof_index, require_dof_in_range
from solver.run_case.input_types import NodeInput, SupportMotionInput
from solver.time_series import TimeSeriesInput, eval_time_series_input, find_time_series_input


struct SupportKinematics(Movable):
    var dofs: List[Int]
    var velocities: List[Int]
    var accelerations: List[Int]
    var factors: List[Float64]
    var velocity: List[Float64]
    var acceleration: List[Float64]

    fn __init__(
        out self,
        motions: List[SupportMotionInput],
        nodes: List[NodeInput],
        id_to_index: List[Int],
        ndf: Int,
        series: List[TimeSeriesInput],
    ):
        self.dofs = []
        self.velocities = []
        self.accelerations = []
        self.factors = []
        self.velocity = []
        self.acceleration = []
        for motion in motions:
            if motion.node < 0 or motion.node >= len(id_to_index) or id_to_index[motion.node] < 0:
                abort("imposed-motion node not found")
            var node = id_to_index[motion.node]
            require_dof_in_range(motion.dof, nodes[node].ndf, "imposed motion")
            self.dofs.append(node_dof_index(node, motion.dof, ndf))
            self.velocities.append(_support_series_index(series, motion.velocity_series))
            self.accelerations.append(_support_series_index(series, motion.acceleration_series))
            self.factors.append(motion.factor)
            self.velocity.append(0.0)
            self.acceleration.append(0.0)

    fn update(
        mut self,
        time: Float64,
        series: List[TimeSeriesInput],
        values: List[Float64],
        times: List[Float64],
    ) raises:
        for i in range(len(self.dofs)):
            self.velocity[i] = self.factors[i] * _support_series_value(
                time, self.velocities[i], series, values, times
            )
            self.acceleration[i] = self.factors[i] * _support_series_value(
                time, self.accelerations[i], series, values, times
            )

    fn apply(self, mut v: List[Float64], mut a: List[Float64]):
        for i in range(len(self.dofs)):
            v[self.dofs[i]] = self.velocity[i]
            a[self.dofs[i]] = self.acceleration[i]

    fn subtract_force(
        self,
        columns: List[Float64],
        response: List[Float64],
        scale: Float64,
        mut residual: List[Float64],
    ):
        if len(columns) == 0 or scale == 0.0:
            return
        for row in range(len(residual)):
            var force = 0.0
            var offset = row * len(self.dofs)
            for i in range(len(self.dofs)):
                force += columns[offset + i] * response[i]
            residual[row] -= scale * force


fn _support_series_index(series: List[TimeSeriesInput], tag: Int) -> Int:
    if tag < 0:
        return -1
    var index = find_time_series_input(series, tag)
    if index < 0:
        abort("imposed-motion time series not found")
    return index


fn _support_series_value(
    time: Float64,
    index: Int,
    series: List[TimeSeriesInput],
    values: List[Float64],
    times: List[Float64],
) raises -> Float64:
    if time < 0.0 or index < 0:
        return 0.0
    return eval_time_series_input(series[index], time, values, times)
