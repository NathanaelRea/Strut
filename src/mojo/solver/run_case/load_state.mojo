from collections import List
from os import abort

from solver.run_case.helpers import _collapse_load_keeping_restrained
from solver.run_case.input_types import ElementInput, ElementLoadInput, UniformExcitationInput
from solver.run_case.loader import _build_element_load_index
from solver.time_series import TimeSeriesInput, eval_time_series_input


struct ActiveElementLoadState(Movable):
    var element_loads: List[ElementLoadInput]
    var elem_load_offsets: List[Int]
    var elem_load_pool: List[Int]

    fn __init__(out self):
        self.element_loads = []
        self.elem_load_offsets = []
        self.elem_load_pool = []


@always_inline
fn _scaled_element_load(load: ElementLoadInput, scale: Float64) -> ElementLoadInput:
    var scaled = ElementLoadInput(
        load.element,
        load.type,
        load.type_tag,
        load.wy * scale,
        load.wz * scale,
        load.wx * scale,
        load.py * scale,
        load.pz * scale,
        load.px * scale,
        load.x,
    )
    scaled.thermal = load.thermal
    scaled.thermal.scale(scale)
    return scaled^


fn build_active_nodal_load(
    const_load: List[Float64], pattern_load: List[Float64], pattern_scale: Float64
) -> List[Float64]:
    if len(const_load) != len(pattern_load):
        abort("nodal load vectors must have the same size")
    var active_load: List[Float64] = []
    active_load.resize(len(const_load), 0.0)
    for i in range(len(const_load)):
        active_load[i] = const_load[i] + pattern_scale * pattern_load[i]
    return active_load^


fn add_uniform_excitation_load(
    excitations: List[UniformExcitationInput],
    t: Float64,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    mass: List[Float64],
    ndf: Int,
    has_transformation_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    mut load: List[Float64],
) raises:
    var inertia: List[Float64] = []
    inertia.resize(len(mass), 0.0)
    for excitation in excitations:
        var accel = eval_time_series_input(
            time_series[excitation.accel], t, time_series_values, time_series_times
        )
        for idx in range(excitation.direction - 1, len(mass), ndf):
            inertia[idx] -= mass[idx] * accel
    if has_transformation_mpc:
        inertia = _collapse_load_keeping_restrained(
            inertia, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
        )
    for i in range(len(mass)):
        load[i] += inertia[i]


fn set_uniform_excitation_initial_velocity(
    excitations: List[UniformExcitationInput],
    free: List[Int],
    ndf: Int,
    mut velocity: List[Float64],
):
    for excitation in excitations:
        if excitation.initial_velocity != 0.0:
            for i in range(len(free)):
                if free[i] % ndf + 1 == excitation.direction:
                    velocity[i] = excitation.initial_velocity


fn append_scaled_element_loads(
    mut dst: List[ElementLoadInput],
    src: List[ElementLoadInput],
    scale: Float64,
):
    if scale == 0.0:
        return
    for i in range(len(src)):
        dst.append(_scaled_element_load(src[i], scale))


fn build_active_element_load_state(
    const_element_loads: List[ElementLoadInput],
    pattern_element_loads: List[ElementLoadInput],
    pattern_scale: Float64,
    typed_elements: List[ElementInput],
    elem_id_to_index: List[Int],
    ndm: Int,
    ndf: Int,
) -> ActiveElementLoadState:
    var active_element_loads = const_element_loads.copy()
    if pattern_scale != 0.0:
        for i in range(len(pattern_element_loads)):
            active_element_loads.append(
                _scaled_element_load(pattern_element_loads[i], pattern_scale)
            )
    var active_elem_load_offsets: List[Int] = []
    var active_elem_load_pool: List[Int] = []
    _build_element_load_index(
        active_element_loads,
        typed_elements,
        elem_id_to_index,
        ndm,
        ndf,
        active_elem_load_offsets,
        active_elem_load_pool,
    )
    var active_state = ActiveElementLoadState()
    active_state.element_loads = active_element_loads^
    active_state.elem_load_offsets = active_elem_load_offsets^
    active_state.elem_load_pool = active_elem_load_pool^
    return active_state^
