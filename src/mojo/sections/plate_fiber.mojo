from collections import InlineArray, List
from os import abort

from materials.plate_fiber import elastic_plate_fiber_tangent
from solver.run_case.input_types import MaterialInput


fn elastic_plate_fiber_base(material_id: Int, materials: List[MaterialInput]) -> MaterialInput:
    if material_id < 0 or material_id >= len(materials):
        abort("PlateFiber material not found")
    var material = materials[material_id]
    if material.type == "PlateFiber":
        if material.base_material < 0 or material.base_material >= len(materials):
            abort("PlateFiber base material not found")
        material = materials[material.base_material]
    if material.type != "ElasticIsotropic":
        abort("PlateFiber currently requires an ElasticIsotropic base material")
    return material


fn elastic_plate_fiber_section_tangent(
    E: Float64, nu: Float64, h: Float64
) -> InlineArray[Float64, 64]:
    """Integrate the elastic fiber response using the exact thickness moments.

    Both OpenSees five-point rules integrate the constant elastic tangent
    through thickness exactly. The shear factors multiply strain and force.
    """
    if h <= 0.0:
        abort("PlateFiber section requires positive thickness")
    var fiber = elastic_plate_fiber_tangent(E, nu)
    var tangent = InlineArray[Float64, 64](fill=0.0)
    var second_moment = h * h * h / 12.0
    for i in range(3):
        for j in range(3):
            tangent[8 * i + j] = h * fiber[5 * i + j]
            tangent[8 * (i + 3) + j + 3] = -second_moment * fiber[5 * i + j]
    tangent[54] = (5.0 / 6.0) * h * fiber[18]
    tangent[63] = (5.0 / 6.0) * h * fiber[24]
    return tangent^
