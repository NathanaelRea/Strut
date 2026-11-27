from collections import List
from os import abort

from elements.ssp_brick_up import (
    BrickCoordinates,
    BrickUPVector,
    SpatialVector,
    SSPBrickUPGeometry,
)
from materials.multi_yield import SoilTangent, SoilVector
from materials.pressure_depend_multi_yield import (
    PressureDependMultiYieldDef,
    PressureDependMultiYieldState,
)
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput


fn _elastic_solid_tangent(E: Float64, nu: Float64) -> SoilTangent:
    var result = SoilTangent(fill=0.0)
    var shear = E / (2.0 * (1.0 + nu))
    var lame = 2.0 * shear * nu / (1.0 - 2.0 * nu)
    for i in range(3):
        for j in range(3):
            result[6 * i + j] = lame + (2.0 * shear if i == j else 0.0)
        result[6 * (i + 3) + i + 3] = shear
    return result


fn _solid_stress(tangent: SoilTangent, strain: SoilVector) -> SoilVector:
    var result = SoilVector(fill=0.0)
    for i in range(6):
        for j in range(6):
            result[i] += tangent[6 * i + j] * strain[j]
    return result


struct SSPBrickUPAssemblyEntry(Copyable, Movable):
    var element: Int
    var material: Int
    var stage: Int
    var pdmy_definition: Int
    var pdmy: List[PressureDependMultiYieldState]
    var geometry: SSPBrickUPGeometry
    var elastic_tangent: SoilTangent
    var density: Float64
    var permeability: SpatialVector
    var body: SpatialVector
    var external_force: BrickUPVector
    var mass: List[Float64]
    var solid_mass: List[Float64]
    var damping: List[Float64]

    fn __init__(
        out self,
        element: Int,
        elem: ElementInput,
        nodes: List[NodeInput],
        node_pool: List[Int],
        node_offset: Int,
        material: MaterialInput,
        definitions: List[PressureDependMultiYieldDef],
    ):
        self.element = element
        self.material = elem.material
        self.stage = 0
        self.pdmy_definition = material.pdmy_index
        self.pdmy = []
        self.elastic_tangent = SoilTangent(fill=0.0)
        if self.pdmy_definition >= 0:
            self.pdmy.append(PressureDependMultiYieldState(definitions[self.pdmy_definition]))
            self.elastic_tangent = self.pdmy[0].tangent(
                definitions[self.pdmy_definition], 0, 3, True
            )
        else:
            self.elastic_tangent = _elastic_solid_tangent(material.E, material.nu)

        var coordinates = BrickCoordinates(fill=SpatialVector(fill=0.0))
        for i in range(8):
            ref node = nodes[node_pool[node_offset + i]]
            coordinates[i] = SpatialVector(node.x, node.y, node.z)
        self.geometry = SSPBrickUPGeometry(coordinates, self.elastic_tangent)
        self.density = elem.quad_density
        self.permeability = SpatialVector(elem.up_perm_x, elem.up_perm_y, elem.up_perm_z)
        self.body = SpatialVector(elem.quad_body_x, elem.quad_body_y, elem.up_body_z)
        self.external_force = BrickUPVector(fill=0.0)
        self.solid_mass = []
        self.solid_mass.resize(1024, 0.0)
        self.damping = []
        self.damping.resize(1024, 0.0)
        var volume = self.geometry.volume
        for i in range(8):
            for axis in range(3):
                var row = 4 * i + axis
                var mass = material.rho * self.geometry.mass_weights[i]
                self.solid_mass[32 * row + row] = mass
                self.external_force[row] = (
                    material.rho * self.geometry.body_weights[i] * self.body[axis]
                )
                var coupling = -0.125 * volume * self.geometry.gradient[i][axis]
                for j in range(8):
                    var pressure = 4 * j + 3
                    self.damping[32 * row + pressure] = coupling
                    self.damping[32 * pressure + row] = coupling
        self.mass = self.solid_mass.copy()
        var porosity = elem.up_void_ratio / (1.0 + elem.up_void_ratio)
        var compressibility = -0.015625 * volume * porosity / elem.up_bulk
        for i in range(8):
            for j in range(8):
                var stabilization = 0.0
                for point in self.geometry.permeability_gradients:
                    for axis in range(3):
                        stabilization += (
                            point.det * point.gradient[i][axis] * point.gradient[j][axis]
                        )
                self.mass[32 * (4 * i + 3) + 4 * j + 3] = (
                    compressibility - elem.up_stabilization * stabilization
                )
        self._form_fluid_terms()

    fn _form_fluid_terms(mut self):
        for i in range(8):
            var row = 4 * i + 3
            self.external_force[row] = 0.0
            for axis in range(3):
                self.external_force[row] -= (
                    self.geometry.volume
                    * self.density
                    * self.geometry.gradient[i][axis]
                    * self.permeability[axis]
                    * self.body[axis]
                )
            for j in range(8):
                var coefficient = 0.0
                for point in self.geometry.permeability_gradients:
                    for axis in range(3):
                        coefficient -= (
                            point.det
                            * point.gradient[i][axis]
                            * self.permeability[axis]
                            * point.gradient[j][axis]
                        )
                self.damping[32 * row + 4 * j + 3] = coefficient

    fn update_permeability(mut self, axis: String, value: Float64):
        if value < 0.0:
            abort("permeability must be nonnegative")
        if axis == "x":
            self.permeability[0] = value
        elif axis == "y":
            self.permeability[1] = value
        elif axis == "z":
            self.permeability[2] = value
        else:
            abort("permeability axis must be x, y, or z")
        self._form_fluid_terms()

    fn evaluate(
        mut self,
        definitions: List[PressureDependMultiYieldDef],
        u: BrickUPVector,
        form_tangent: Bool,
        mut force: BrickUPVector,
        mut matrix: List[Float64],
    ):
        var strain = self.geometry.strain(u)
        var tangent = self.elastic_tangent
        var stress: SoilVector
        if self.pdmy_definition >= 0:
            ref definition = definitions[self.pdmy_definition]
            self.pdmy[0].set_trial_strain(definition, self.stage, strain)
            stress = self.pdmy[0].stress_t.vector()
            if form_tangent:
                tangent = self.pdmy[0].tangent(definition, self.stage, 3)
        else:
            stress = _solid_stress(tangent, strain)
        for i in range(32):
            force[i] = -self.external_force[i]
        self.geometry.add_solid_response(stress, tangent, u, form_tangent, force, matrix)

    fn record(
        self,
        want_stress: Bool,
        components: Int,
        u: BrickUPVector,
        definitions: List[PressureDependMultiYieldDef],
    ) -> List[Float64]:
        var result: List[Float64] = []
        var response: SoilVector
        if self.pdmy_definition >= 0:
            ref state = self.pdmy[0]
            response = state.stress_c.vector() if want_stress else state.strain_c.vector()
            if not want_stress:
                for i in range(3, 6):
                    response[i] *= 2.0
        else:
            var strain = self.geometry.strain(u)
            response = _solid_stress(self.elastic_tangent, strain) if want_stress else strain
        for i in range(6):
            result.append(response[i])
        if self.pdmy_definition >= 0 and want_stress and components != 6:
            result.append(
                self.pdmy[0].committed_stress_ratio(definitions[self.pdmy_definition], self.stage)
            )
        return result^

    fn commit(mut self):
        if self.pdmy_definition >= 0:
            self.pdmy[0].commit(self.stage)
