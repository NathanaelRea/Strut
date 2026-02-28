from collections import List
from os import abort

from solver.run_case.input_types import ElementInput, ElementLoadInput
from solver.run_case.loader import _build_element_load_index


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
    return ElementLoadInput(
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
