from collections import List
from os import abort

from materials import UniMaterialDef, uni_mat_initial_tangent, uni_mat_is_elastic
from solver.run_case.input_types import SectionInput


fn _uniaxial_component(material: Int, definitions_by_id: List[Int], response: String) -> Int:
    if material < 0 or material >= len(definitions_by_id) or definitions_by_id[material] < 0:
        abort("section aggregator requires a uniaxial material for " + response)
    return definitions_by_id[material]


fn axial_bending_components(
    section: SectionInput,
    sections: List[SectionInput],
    definitions_by_id: List[Int],
) -> (Int, Int):
    var sec = section
    if sec.base_section >= 0:
        ref base = sections[sec.base_section]
        if (
            base.type != "AggregatorSection2d"
            or base.base_section >= 0
            or base.shear_y_material >= 0
            or (base.axial_material >= 0 and sec.axial_material >= 0)
            or (base.flexural_material >= 0 and sec.flexural_material >= 0)
        ):
            abort("unsupported 2D beam section aggregator base")
        if sec.axial_material < 0:
            sec.axial_material = base.axial_material
        if sec.flexural_material < 0:
            sec.flexural_material = base.flexural_material
    return (
        _uniaxial_component(sec.axial_material, definitions_by_id, "P"),
        _uniaxial_component(sec.flexural_material, definitions_by_id, "Mz"),
    )


fn elastic_aggregator_3d(
    mut sec: SectionInput,
    definitions: List[UniMaterialDef],
    definitions_by_id: List[Int],
) -> SectionInput:
    if sec.base_section >= 0 or sec.shear_y_material >= 0 or sec.shear_z_material >= 0:
        abort("unsupported 3D beam section aggregator")
    var rigidities: List[Float64] = []
    for material in [
        sec.axial_material,
        sec.flexural_material,
        sec.moment_y_material,
        sec.torsion_material,
    ]:
        var index = _uniaxial_component(material, definitions_by_id, "P, Mz, My or T")
        ref definition = definitions[index]
        var rigidity = uni_mat_initial_tangent(definition)
        if not uni_mat_is_elastic(definition) or rigidity <= 0.0:
            abort("3D beam section aggregator requires positive elastic P, Mz, My and T")
        rigidities.append(rigidity)

    sec.type = "ElasticSection3d"
    sec.E = 1.0
    sec.A = rigidities[0]
    sec.Iz = rigidities[1]
    sec.Iy = rigidities[2]
    sec.G = rigidities[3]
    sec.J = 1.0
    return sec


fn fiber_aggregated_shear_flexibility(
    material: Int, definitions: List[UniMaterialDef], definitions_by_id: List[Int]
) -> Float64:
    if material < 0:
        return 0.0
    var index = _uniaxial_component(material, definitions_by_id, "Vy or Vz")
    ref definition = definitions[index]
    var rigidity = uni_mat_initial_tangent(definition)
    if not uni_mat_is_elastic(definition) or rigidity <= 0.0:
        abort("fiber beam section requires positive elastic aggregated shear stiffness")
    return 1.0 / rigidity
