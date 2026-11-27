from collections import List
from os import abort

from elements.quad4 import PlaneMatrix, quad4_elastic_tangent, quad4_plane_tangent
from elements.quad_up import (
    QuadUPPoint,
    QuadUPShape,
    QuadUPVector,
    quad_up_displacement_dof,
    quad_up_points,
)
from materials.multi_yield import SoilVector
from materials.pressure_depend_multi_yield import (
    PressureDependMultiYieldDef,
    PressureDependMultiYieldState,
)
from materials.pressure_independ_multi_yield import (
    PressureIndependMultiYieldDef,
    PressureIndependMultiYieldState,
)
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput


struct QuadUPAssemblyEntry(Copyable, Movable):
    var element: Int
    var material: Int
    var stage: Int
    var nodes: Int
    var dofs: Int
    var solid_points: List[QuadUPPoint]
    var pressure_points: List[QuadUPPoint]
    var fluid_density: Float64
    var body_x: Float64
    var body_y: Float64
    var perm_x: Float64
    var perm_y: Float64
    var pimy_definition: Int
    var pdmy_definition: Int
    var pimy: List[PressureIndependMultiYieldState]
    var pdmy: List[PressureDependMultiYieldState]
    var elastic_tangent: PlaneMatrix
    var external_force: QuadUPVector
    var mass: List[Float64]
    var solid_mass: List[Float64]
    var damping: List[Float64]
    var solid_row_mass: QuadUPVector
    var initial_stiffness: List[Float64]

    fn __init__(
        out self,
        element: Int,
        elem: ElementInput,
        nodes: List[NodeInput],
        node_pool: List[Int],
        node_offset: Int,
        material: MaterialInput,
        pimy_definitions: List[PressureIndependMultiYieldDef],
        pdmy_definitions: List[PressureDependMultiYieldDef],
    ):
        self.element = element
        self.material = elem.material
        self.stage = 0
        self.nodes = elem.node_count
        self.dofs = elem.dof_count
        var x = QuadUPShape(fill=0.0)
        var y = QuadUPShape(fill=0.0)
        for i in range(self.nodes):
            ref node = nodes[node_pool[node_offset + i]]
            x[i] = node.x
            y[i] = node.y
        self.solid_points = quad_up_points(self.nodes, self.nodes, elem.thickness, x, y)
        self.pressure_points = quad_up_points(4, 4, elem.thickness, x, y)
        self.fluid_density = elem.quad_density
        self.body_x = elem.quad_body_x
        self.body_y = elem.quad_body_y
        self.perm_x = elem.up_perm_x
        self.perm_y = elem.up_perm_y
        self.pimy_definition = material.pimy_index
        self.pdmy_definition = material.pdmy_index
        self.pimy = []
        self.pdmy = []
        self.elastic_tangent = PlaneMatrix(fill=0.0)
        self.initial_stiffness = []
        if self.pimy_definition >= 0:
            for _ in range(self.nodes):
                self.pimy.append(
                    PressureIndependMultiYieldState(pimy_definitions[self.pimy_definition])
                )
        elif self.pdmy_definition >= 0:
            for _ in range(self.nodes):
                self.pdmy.append(
                    PressureDependMultiYieldState(pdmy_definitions[self.pdmy_definition])
                )
        else:
            self.elastic_tangent = quad4_elastic_tangent(material.E, material.nu, True)

        self.external_force = QuadUPVector(fill=0.0)
        self.solid_mass = []
        self.solid_mass.resize(self.dofs * self.dofs, 0.0)
        self.solid_row_mass = QuadUPVector(fill=0.0)
        self.damping = []
        self.damping.resize(self.dofs * self.dofs, 0.0)
        for point in self.solid_points:
            for i in range(self.nodes):
                var row = quad_up_displacement_dof(i)
                var weighted_shape = point.weight * material.rho * point.shape[i]
                self.external_force[row] += weighted_shape * elem.quad_body_x
                self.external_force[row + 1] += weighted_shape * elem.quad_body_y
                for j in range(self.nodes):
                    var col = quad_up_displacement_dof(j)
                    var mass = weighted_shape * point.shape[j]
                    self.solid_mass[row * self.dofs + col] += mass
                    self.solid_mass[(row + 1) * self.dofs + col + 1] += mass
                    self.solid_row_mass[row] += mass
                    self.solid_row_mass[row + 1] += mass
        self.mass = self.solid_mass.copy()
        var coupling_points = quad_up_points(self.nodes, 4, elem.thickness, x, y)
        for q in range(4):
            var p = self.pressure_points[q]
            var coupling = coupling_points[q]
            for i in range(4):
                var row = 3 * i + 2
                for j in range(4):
                    var col = 3 * j + 2
                    self.mass[row * self.dofs + col] -= (
                        p.weight * p.shape[i] * p.shape[j] / elem.up_bulk
                    )
                for j in range(self.nodes):
                    var col = quad_up_displacement_dof(j)
                    var dx = -coupling.weight * coupling.dx[j] * p.shape[i]
                    var dy = -coupling.weight * coupling.dy[j] * p.shape[i]
                    self.damping[row * self.dofs + col] += dx
                    self.damping[col * self.dofs + row] += dx
                    self.damping[row * self.dofs + col + 1] += dy
                    self.damping[(col + 1) * self.dofs + row] += dy

        self._form_fluid_terms()
        if elem.quad_pressure != 0.0:
            # OpenSees integrates the constant edge traction using the edge length.
            for i in range(4):
                var j = (i + 1) % 4
                var fx = 0.5 * elem.thickness * elem.quad_pressure * (y[j] - y[i])
                var fy = -0.5 * elem.thickness * elem.quad_pressure * (x[j] - x[i])
                self.external_force[3 * i] += fx
                self.external_force[3 * j] += fx
                self.external_force[3 * i + 1] += fy
                self.external_force[3 * j + 1] += fy

    fn _form_fluid_terms(mut self):
        for i in range(4):
            var row = 3 * i + 2
            self.external_force[row] = 0.0
            for point in self.pressure_points:
                self.external_force[row] -= (
                    point.weight
                    * self.fluid_density
                    * (
                        self.perm_x * self.body_x * point.dx[i]
                        + self.perm_y * self.body_y * point.dy[i]
                    )
                )
            for j in range(4):
                var col = 3 * j + 2
                var damping = 0.0
                for point in self.pressure_points:
                    damping -= point.weight * (
                        self.perm_x * point.dx[i] * point.dx[j]
                        + self.perm_y * point.dy[i] * point.dy[j]
                    )
                self.damping[row * self.dofs + col] = damping

    fn update_permeability(mut self, axis: String, value: Float64):
        if value < 0.0:
            abort("permeability must be nonnegative")
        if axis == "x":
            self.perm_x = value
        elif axis == "y":
            self.perm_y = value
        else:
            abort("permeability axis must be x or y")
        self._form_fluid_terms()

    fn evaluate(
        mut self,
        pimy_definitions: List[PressureIndependMultiYieldDef],
        pdmy_definitions: List[PressureDependMultiYieldDef],
        u: QuadUPVector,
        form_tangent: Bool,
        mut force: QuadUPVector,
        mut matrix: List[Float64],
    ):
        for i in range(self.dofs):
            force[i] = -self.external_force[i]
        if form_tangent:
            matrix.resize(self.dofs * self.dofs, 0.0)
            for i in range(len(matrix)):
                matrix[i] = 0.0
        var tangent = self.elastic_tangent
        for i in range(len(self.solid_points)):
            var point = self.solid_points[i]
            var strain = point.strain(self.nodes, u)
            var stress = SoilVector(fill=0.0)
            if self.pimy_definition >= 0:
                ref definition = pimy_definitions[self.pimy_definition]
                ref state = self.pimy[i]
                state.set_trial_strain(definition, self.stage, strain)
                stress = state.stress_t.vector()
                if form_tangent:
                    tangent = quad4_plane_tangent(state.tangent(definition, self.stage))
            elif self.pdmy_definition >= 0:
                ref definition = pdmy_definitions[self.pdmy_definition]
                ref state = self.pdmy[i]
                state.set_trial_strain(definition, self.stage, strain)
                stress = state.stress_t.vector()
                if form_tangent:
                    tangent = quad4_plane_tangent(state.tangent(definition, self.stage, 2))
            else:
                stress[0] = tangent[0] * strain[0] + tangent[1] * strain[1]
                stress[1] = tangent[3] * strain[0] + tangent[4] * strain[1]
                stress[3] = tangent[8] * strain[3]
            for j in range(self.nodes):
                var row = quad_up_displacement_dof(j)
                force[row] += point.weight * (point.dx[j] * stress[0] + point.dy[j] * stress[3])
                force[row + 1] += point.weight * (point.dy[j] * stress[1] + point.dx[j] * stress[3])
            if form_tangent:
                point.add_stiffness(self.nodes, self.dofs, tangent, matrix)

    fn commit(mut self):
        for i in range(len(self.pimy)):
            self.pimy[i].commit(self.stage)
        for i in range(len(self.pdmy)):
            self.pdmy[i].commit(self.stage)

    fn initial_stiffness_difference(
        mut self,
        pimy_definitions: List[PressureIndependMultiYieldDef],
        pdmy_definitions: List[PressureDependMultiYieldDef],
        mut difference: List[Float64],
    ):
        # OpenSees caches getInitialStiff() on its first use. Subtract the
        # already-assembled current stiffness while retaining the cached one.
        var initialize = len(self.initial_stiffness) == 0
        var entries = self.dofs * self.dofs
        self.initial_stiffness.resize(entries, 0.0)
        difference.resize(entries, 0.0)
        for i in range(entries):
            difference[i] = 0.0
        for i in range(len(self.solid_points)):
            var point = self.solid_points[i]
            var current = self.elastic_tangent
            var initial = self.elastic_tangent
            if self.pimy_definition >= 0:
                ref material = pimy_definitions[self.pimy_definition]
                ref state = self.pimy[i]
                current = quad4_plane_tangent(state.tangent(material, self.stage))
                if initialize:
                    initial = quad4_plane_tangent(state.initial_tangent())
            elif self.pdmy_definition >= 0:
                ref material = pdmy_definitions[self.pdmy_definition]
                ref state = self.pdmy[i]
                current = quad4_plane_tangent(state.tangent(material, self.stage, 2))
                if initialize:
                    initial = quad4_plane_tangent(state.tangent(material, self.stage, 2, True))
            point.add_stiffness(self.nodes, self.dofs, current, difference)
            if initialize:
                point.add_stiffness(self.nodes, self.dofs, initial, self.initial_stiffness)
        for i in range(entries):
            difference[i] = self.initial_stiffness[i] - difference[i]

    fn record(
        self,
        point: Int,
        want_stress: Bool,
        components: Int,
        u: QuadUPVector,
        pdmy_definitions: List[PressureDependMultiYieldDef],
    ) -> List[Float64]:
        if point < 0 or point >= len(self.solid_points):
            abort("u-p material recorder integration point out of range")
        if self.pimy_definition >= 0:
            ref state = self.pimy[point]
            if want_stress:
                var stress = state.stress_c.vector()
                return [
                    stress[0],
                    stress[1],
                    stress[2],
                    stress[3],
                    state.committed_stress_ratio(self.stage),
                ]
            var strain = state.strain_c
            return [strain[0], strain[1], strain[3]]
        if self.pdmy_definition >= 0:
            ref state = self.pdmy[point]
            if want_stress:
                var stress = state.stress_c.vector()
                if components == 3:
                    return [stress[0], stress[1], stress[3]]
                if components == 4:
                    return [stress[0], stress[1], stress[2], stress[3]]
                ref defn = pdmy_definitions[self.pdmy_definition]
                return [
                    stress[0],
                    stress[1],
                    stress[2],
                    stress[3],
                    state.committed_stress_ratio(defn, self.stage),
                ]
            var strain = state.strain_c.vector()
            return [strain[0], strain[1], 2.0 * strain[3]]
        var strain = self.solid_points[point].strain(self.nodes, u)
        if not want_stress:
            return [strain[0], strain[1], strain[3]]
        var tangent = self.elastic_tangent
        return [
            tangent[0] * strain[0] + tangent[1] * strain[1],
            tangent[3] * strain[0] + tangent[4] * strain[1],
            tangent[8] * strain[3],
        ]
