from collections import List

from solver.run_case.input_types import RecorderInput


struct TransientState(Movable):
    var velocity: List[Float64]
    var acceleration: List[Float64]
    var next_sample_times: List[Float64]
    var stage: Int

    fn __init__(out self, dofs: Int, recorders: Int):
        self.velocity = []
        self.velocity.resize(dofs, 0.0)
        self.acceleration = []
        self.acceleration.resize(dofs, 0.0)
        self.next_sample_times = []
        self.next_sample_times.resize(recorders, 0.0)
        self.stage = 0

    fn should_sample(mut self, recorder: RecorderInput, index: Int, time: Float64) -> Bool:
        if self.stage < recorder.start_stage or self.stage >= recorder.end_stage:
            return False
        if recorder.delta_t <= 0.0:
            return True
        if time - self.next_sample_times[index] < -recorder.delta_t * 1.0e-5:
            return False
        self.next_sample_times[index] = time + recorder.delta_t
        return True
