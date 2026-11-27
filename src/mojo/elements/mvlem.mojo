# Multiple-vertical-line element from OpenSees 3.7.1 MVLEM.cpp.
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
)


struct MVLEMFiber(ImplicitlyCopyable, Movable):
    var width: Float64
    var center: Float64
    var concrete_area: Float64
    var steel_area: Float64
    var concrete: Int
    var steel: Int

    fn __init__(
        out self,
        width: Float64,
        thickness: Float64,
        rho: Float64,
        concrete: Int,
        steel: Int,
    ):
        self.width = width
        self.center = 0.0
        var area = width * thickness
        self.steel_area = area * rho
        self.concrete_area = area - self.steel_area
        self.concrete = concrete
        self.steel = steel


struct MVLEMGeometry(ImplicitlyCopyable, Movable):
    var fibers: List[MVLEMFiber]
    var c: Float64
    var height: Float64

    fn __init__(out self):
        self.fibers = []
        self.c = 0.0
        self.height = 0.0

    fn __copyinit__(out self, existing: Self):
        self.fibers = existing.fibers.copy()
        self.c = existing.c
        self.height = existing.height

    fn set_height(mut self, height: Float64) -> Float64:
        self.height = height
        var wall_width = 0.0
        for fiber in self.fibers:
            wall_width += fiber.width
        var panel_right = 0.0
        var wall_area = 0.0
        for i in range(len(self.fibers)):
            ref fiber = self.fibers[i]
            panel_right += fiber.width
            fiber.center = panel_right - fiber.width / 2.0 - wall_width / 2.0
            wall_area += fiber.concrete_area + fiber.steel_area
        return wall_area


struct MVLEMResponse(Movable):
    var force: List[Float64]
    var tangent: List[List[Float64]]

    fn __init__(out self):
        self.force = []
        self.force.resize(6, 0.0)
        self.tangent = []
        for _ in range(6):
            var row: List[Float64] = []
            row.resize(6, 0.0)
            self.tangent.append(row^)


fn mvlem_response(
    geometry: MVLEMGeometry,
    u: List[Float64],
    offset: Int,
    state_ids: List[Int],
    state_defs: List[Int],
    defs: List[UniMaterialDef],
    mut states: List[UniMaterialState],
    initial_tangent: Bool = False,
) -> MVLEMResponse:
    var result = MVLEMResponse()
    var height = geometry.height
    var c = geometry.c
    var count = len(geometry.fibers)
    var axial = 0.0
    var moment = 0.0
    var kv = 0.0
    var km = 0.0
    var eccentric = 0.0
    for i in range(count):
        ref fiber = geometry.fibers[i]
        var x = fiber.center
        var strain = (-u[1] - x * u[2] + u[4] + x * u[5]) / height
        var concrete = state_ids[offset + i]
        var steel = state_ids[offset + count + i]
        uniaxial_set_trial_strain(defs[state_defs[concrete]], states[concrete], strain)
        uniaxial_set_trial_strain(defs[state_defs[steel]], states[steel], strain)
        var force_concrete = states[concrete].sig_t * fiber.concrete_area
        var force_steel = states[steel].sig_t * fiber.steel_area
        axial += force_concrete + force_steel
        moment += force_concrete * x + force_steel * x
        var concrete_tangent = states[concrete].tangent_t
        var steel_tangent = states[steel].tangent_t
        if initial_tangent:
            concrete_tangent = uni_mat_initial_tangent(defs[state_defs[concrete]])
            steel_tangent = uni_mat_initial_tangent(defs[state_defs[steel]])
        var stiffness = (
            concrete_tangent * fiber.concrete_area / height
            + steel_tangent * fiber.steel_area / height
        )
        kv += stiffness
        km += stiffness * x * x
        eccentric += stiffness * x
    var shear_i = c * height
    var shear_j = (1.0 - c) * height
    var shear_deformation = u[0] - u[3] - shear_i * u[2] - shear_j * u[5]
    var shear_state = state_ids[offset + 2 * count]
    uniaxial_set_trial_strain(defs[state_defs[shear_state]], states[shear_state], shear_deformation)
    var shear_force = states[shear_state].sig_t
    var kh = states[shear_state].tangent_t
    if initial_tangent:
        kh = uni_mat_initial_tangent(defs[state_defs[shear_state]])
    result.force = [
        shear_force,
        -axial,
        -moment - shear_force * shear_i,
        -shear_force,
        axial,
        moment - shear_force * shear_j,
    ]

    # The translational and rotational rows come from the axial fiber
    # compatibility and the shear spring at height c*h.
    var shear_row = [1.0, 0.0, -shear_i, -1.0, 0.0, -shear_j]
    for i in range(6):
        for j in range(6):
            result.tangent[i][j] = shear_row[i] * kh * shear_row[j]
    result.tangent[1][1] += kv
    result.tangent[1][2] += eccentric
    result.tangent[1][4] -= kv
    result.tangent[1][5] -= eccentric
    result.tangent[2][1] += eccentric
    result.tangent[2][2] += km
    result.tangent[2][4] -= eccentric
    result.tangent[2][5] -= km
    result.tangent[4][1] -= kv
    result.tangent[4][2] -= eccentric
    result.tangent[4][4] += kv
    result.tangent[4][5] += eccentric
    result.tangent[5][1] -= eccentric
    result.tangent[5][2] -= km
    result.tangent[5][4] += eccentric
    result.tangent[5][5] += km
    return result^
