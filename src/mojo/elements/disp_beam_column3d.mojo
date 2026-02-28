from collections import List

from solver.run_case.input_types import ElementLoadInput

from elements.force_beam_column3d import force_beam_column3d_global_tangent_and_internal


fn disp_beam_column3d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    disp_beam_column3d_global_tangent_and_internal(
        0,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
        k_global_out,
        f_global_out,
    )


fn disp_beam_column3d_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    force_beam_column3d_global_tangent_and_internal(
        elem_index,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
        k_global_out,
        f_global_out,
    )
