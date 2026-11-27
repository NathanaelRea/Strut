# Shear–flexure interaction wall from OpenSees 3.7.1 SFI_MVLEM.cpp.
from elements.mvlem import MVLEMGeometry
from materials.fsam import PanelVector
from materials.uniaxial.core import UniMaterialState
from materials.uniaxial.ops import fsam_set_trial_strain


struct SFIMVLEMResponse(Movable):
    var force: List[Float64]
    var tangent: List[List[Float64]]

    fn __init__(out self, size: Int):
        self.force = List[Float64]()
        self.force.resize(size, 0.0)
        self.tangent = List[List[Float64]](capacity=size)
        for _ in range(size):
            var row: List[Float64] = []
            row.resize(size, 0.0)
            self.tangent.append(row^)


fn sfi_mvlem_shear_deformation(wall: MVLEMGeometry, u: List[Float64]) -> Float64:
    return u[3] - u[0] + wall.c * wall.height * u[2] + (1.0 - wall.c) * wall.height * u[5]


fn sfi_mvlem_response(
    wall: MVLEMGeometry,
    u: List[Float64],
    offset: Int,
    state_ids: List[Int],
    mut states: List[UniMaterialState],
    initial_tangent: Bool = False,
    committed: Bool = False,
) -> SFIMVLEMResponse:
    var result = SFIMVLEMResponse(6 + len(wall.fibers))
    var h = wall.height
    var shear_row = [1.0, 0.0, -wall.c * h, -1.0, 0.0, -(1.0 - wall.c) * h]
    var gamma = sfi_mvlem_shear_deformation(wall, u) / h
    var shear_force = 0.0
    var shear_stiffness = 0.0
    for i in range(len(wall.fibers)):
        ref fiber = wall.fibers[i]
        var x = fiber.center
        var area = fiber.concrete_area
        var horizontal_area = h * area / fiber.width
        var axial_row = [0.0, -1.0, -x, 0.0, 1.0, x]
        var eps_y = (-u[1] - x * u[2] + u[4] + x * u[5]) / h
        ref state = states[state_ids[offset + i]]
        if not committed:
            fsam_set_trial_strain(state, PanelVector(u[6 + i] / fiber.width, eps_y, gamma, 0.0))
        ref panel = state.fsam[0]
        var response = panel.committed if committed else panel.trial
        # The OpenSees element force recorder reads the last assembled force.
        # Committing a panel can initiate a crack and recompute its stresses.
        var stress = panel.converged_stress if committed else response.stress
        var tangent = panel.params.initial_tangent() if initial_tangent else response.tangent

        # OpenSees uses the three diagonal material moduli for the element
        # tangent. The force still depends on the coupled panel stresses.
        result.force[6 + i] = stress[0] * horizontal_area
        result.tangent[6 + i][6 + i] = tangent[0][0] * horizontal_area / fiber.width
        shear_force -= stress[2] * area
        shear_stiffness += tangent[2][2] * area / h
        var axial_force = stress[1] * area
        var axial_stiffness = tangent[1][1] * area / h
        for j in range(6):
            result.force[j] += axial_row[j] * axial_force
            for k in range(6):
                result.tangent[j][k] += axial_row[j] * axial_stiffness * axial_row[k]

    for j in range(6):
        result.force[j] += shear_row[j] * shear_force
        for k in range(6):
            result.tangent[j][k] += shear_row[j] * shear_stiffness * shear_row[k]
    return result^
