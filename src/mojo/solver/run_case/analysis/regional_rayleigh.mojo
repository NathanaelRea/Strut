from collections import Dict
from elements import beam_global_stiffness
from os import abort
from solver.dof import node_dof_index
from solver.run_case.analysis.transient_sparse import TransientSparseMatrices
from solver.run_case.input_types import ElementInput, MassInput, RayleighInput, SectionInput
from solver.sparse_pattern import build_reduced_mass_values
from tag_types import ElementTypeTag


fn assemble_regional_rayleigh(
    rayleigh: RayleighInput,
    masses: List[MassInput],
    elements: List[ElementInput],
    sections: List[SectionInput],
    node_x: List[Float64],
    node_y: List[Float64],
    ndf: Int,
    id_to_index: List[Int],
    elem_id_to_index: List[Int],
    elem_dof_offsets: List[Int],
    elem_free_pool: List[Int],
    free_index: List[Int],
    has_transformation_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    mut mats: TransientSparseMatrices,
) raises:
    # The global coefficients are already assembled. Add only the difference
    # on objects whose regional coefficient overrides the global assignment.
    var node_alpha = Dict[Int, Float64]()
    for entry in rayleigh.node_alpha_m:
        node_alpha[entry.tag] = entry.coefficient - rayleigh.alpha_m
    var weighted_mass: List[Float64] = []
    weighted_mass.resize(len(free_index), 0.0)
    for mass in masses:
        if mass.node in node_alpha:
            var index = node_dof_index(id_to_index[mass.node], mass.dof, ndf)
            weighted_mass[index] += node_alpha[mass.node] * mass.value
    mats.regional_damping = build_reduced_mass_values(
        mats.pattern,
        weighted_mass,
        free_index,
        has_transformation_mpc,
        mpc_row_offsets,
        mpc_dof_pool,
        mpc_coeff_pool,
    )
    swap(mats.assembler.values, mats.regional_damping)
    for entry in rayleigh.element_beta_k_init:
        if entry.tag < 0 or entry.tag >= len(elem_id_to_index) or elem_id_to_index[entry.tag] < 0:
            abort("regional Rayleigh element not found")
        var e = elem_id_to_index[entry.tag]
        var element = elements[e]
        if element.type_tag != ElementTypeTag.ElasticBeamColumn2d:
            abort("regional Rayleigh betaKInit requires elasticBeamColumn2d")
        var section = sections[element.section]
        var i = element.node_index_1
        var j = element.node_index_2
        var stiffness = beam_global_stiffness(
            section.E * (entry.coefficient - rayleigh.beta_k_init),
            section.A,
            section.I,
            node_x[i],
            node_y[i],
            node_x[j],
            node_y[j],
        )
        mats.assembler.add_element_matrix(e, elem_free_pool, elem_dof_offsets[e], 6, stiffness)
    swap(mats.assembler.values, mats.regional_damping)
