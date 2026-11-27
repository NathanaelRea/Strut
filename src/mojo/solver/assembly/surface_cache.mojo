from elements.beam_contact2d import BeamContact2D, ContactDofs, ContactVector, Point2d
from tag_types import ElementTypeTag
from solver.assembly.continuum_cache import ContinuumAssemblyCache
from collections import List
from os import abort
from materials.initial_state_analysis import InitialStateAnalysisPoint

from elements.quad4 import (
    PlaneMatrix,
    QuadMatrix,
    QuadPoint,
    QuadVector,
    quad4_elastic_tangent,
    quad4_plane_tangent,
    quad4_points,
)
from materials.pressure_depend_multi_yield import (
    PressureDependMultiYieldDef,
    PressureDependMultiYieldState,
)
from materials.pressure_independ_multi_yield import (
    PressureIndependMultiYieldDef,
    PressureIndependMultiYieldState,
)
from elements.quad_up import QuadUPVector
from elements.ssp_brick_up import BrickUPVector
from solver.assembly.ssp_brick_up_cache import SSPBrickUPAssemblyEntry
from solver.assembly.quad_up_cache import QuadUPAssemblyEntry
from solver.assembly.shell_cache import ShellAssemblyCache
from solver.sparse_pattern import SparseAssembler, zero_values
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput
from solver.assembly.stiffness_internal_shared import _elem_dof


struct QuadAssemblyEntry(Copyable, Movable):
    var points: List[QuadPoint]
    var elastic_tangent: PlaneMatrix
    var soil: List[PressureIndependMultiYieldState]
    var pressure_dependent_soil: List[PressureDependMultiYieldState]
    var initial_state_points: List[InitialStateAnalysisPoint]
    var material: Int
    var soil_definition: Int
    var pressure_dependent_definition: Int
    var stage: Int
    var body_x: Float64
    var body_y: Float64
    var pressure_force: QuadVector
    var mass: QuadVector

    fn __init__(
        out self,
        elem: ElementInput,
        nodes: List[NodeInput],
        material: MaterialInput,
        soil_definitions: List[PressureIndependMultiYieldDef],
        pressure_dependent_definitions: List[PressureDependMultiYieldDef],
    ):
        var x = [
            nodes[elem.node_index_1].x,
            nodes[elem.node_index_2].x,
            nodes[elem.node_index_3].x,
            nodes[elem.node_index_4].x,
        ]
        var y = [
            nodes[elem.node_index_1].y,
            nodes[elem.node_index_2].y,
            nodes[elem.node_index_3].y,
            nodes[elem.node_index_4].y,
        ]
        self.points = quad4_points(x, y, elem.thickness)
        var wraps_initial_state = material.type == "InitialStateAnalysisWrapper"
        self.material = material.base_material if wraps_initial_state else elem.material
        self.soil_definition = material.pimy_index
        self.pressure_dependent_definition = material.pdmy_index
        self.stage = 0
        self.body_x = elem.quad_body_x
        self.body_y = elem.quad_body_y
        self.pressure_force = QuadVector(fill=0.0)
        self.mass = QuadVector(fill=0.0)
        self.soil = []
        self.pressure_dependent_soil = []
        self.initial_state_points = []
        if wraps_initial_state:
            self.initial_state_points.resize(4, InitialStateAnalysisPoint())
        self.elastic_tangent = PlaneMatrix(fill=0.0)
        if self.soil_definition >= 0:
            ref definition = soil_definitions[self.soil_definition]
            for _ in range(4):
                self.soil.append(PressureIndependMultiYieldState(definition))
        elif self.pressure_dependent_definition >= 0:
            ref definition = pressure_dependent_definitions[self.pressure_dependent_definition]
            for _ in range(4):
                self.pressure_dependent_soil.append(PressureDependMultiYieldState(definition))
        else:
            self.elastic_tangent = quad4_elastic_tangent(
                material.E, material.nu, elem.formulation == "PlaneStrain"
            )

        var density = material.rho
        if elem.quad_density != 0.0:
            density = elem.quad_density
        for i in range(4):
            var j = (i + 1) % 4
            var fx = 0.5 * elem.quad_pressure * (y[j] - y[i])
            var fy = -0.5 * elem.quad_pressure * (x[j] - x[i])
            self.pressure_force[2 * i] += fx
            self.pressure_force[2 * j] += fx
            self.pressure_force[2 * i + 1] += fy
            self.pressure_force[2 * j + 1] += fy
            for point in self.points:
                var mass = point.shape[i] * (density * point.weight)
                self.mass[2 * i] += mass
                self.mass[2 * i + 1] += mass

    fn evaluate(
        mut self,
        soil_definitions: List[PressureIndependMultiYieldDef],
        pressure_dependent_definitions: List[PressureDependMultiYieldDef],
        u: QuadVector,
        form_tangent: Bool,
        mut force: QuadVector,
        mut matrix: QuadMatrix,
    ):
        var tangent = self.elastic_tangent
        for i in range(8):
            force[i] = -self.pressure_force[i]
        for i in range(4):
            var point = self.points[i]
            var strain = point.strain(u)
            if len(self.initial_state_points) != 0:
                strain = self.initial_state_points[i].material_strain(strain)
            var stress = strain
            if self.soil_definition >= 0:
                ref definition = soil_definitions[self.soil_definition]
                ref state = self.soil[i]
                state.set_trial_strain(definition, self.stage, strain)
                stress = state.stress_t.vector()
                if form_tangent:
                    tangent = quad4_plane_tangent(state.tangent(definition, self.stage))
            elif self.pressure_dependent_definition >= 0:
                ref definition = pressure_dependent_definitions[self.pressure_dependent_definition]
                ref state = self.pressure_dependent_soil[i]
                state.set_trial_strain(definition, self.stage, strain)
                stress = state.stress_t.vector()
                if form_tangent:
                    tangent = quad4_plane_tangent(state.tangent(definition, self.stage, 2))
            else:
                stress[0] = tangent[0] * strain[0] + tangent[1] * strain[1]
                stress[1] = tangent[3] * strain[0] + tangent[4] * strain[1]
                stress[3] = tangent[8] * strain[3]
            point.add_force(stress, self.body_x, self.body_y, force)
            if form_tangent:
                point.add_stiffness(tangent, matrix)

    fn record(
        self,
        point: Int,
        want_stress: Bool,
        stress_components: Int,
        u: QuadVector,
        pressure_dependent_definitions: List[PressureDependMultiYieldDef],
    ) -> List[Float64]:
        if self.soil_definition >= 0:
            ref state = self.soil[point]
            if want_stress:
                var s = state.stress_c.vector()
                if stress_components == 3:
                    return [s[0], s[1], s[3]]
                if stress_components == 4:
                    return [s[0], s[1], s[2], s[3]]
                return [s[0], s[1], s[2], s[3], state.committed_stress_ratio(self.stage)]
            var strain = state.strain_c
            return [strain[0], strain[1], strain[3]]
        if self.pressure_dependent_definition >= 0:
            ref state = self.pressure_dependent_soil[point]
            if want_stress:
                var s = state.stress_c.vector()
                if stress_components == 3:
                    return [s[0], s[1], s[3]]
                if stress_components == 4:
                    return [s[0], s[1], s[2], s[3]]
                ref definition = pressure_dependent_definitions[self.pressure_dependent_definition]
                return [
                    s[0],
                    s[1],
                    s[2],
                    s[3],
                    state.committed_stress_ratio(definition, self.stage),
                ]
            var strain = state.strain_c.vector()
            return [strain[0], strain[1], 2.0 * strain[3]]
        var strain = self.points[point].strain(u)
        if not want_stress:
            return [strain[0], strain[1], strain[3]]
        var D = self.elastic_tangent
        return [
            D[0] * strain[0] + D[1] * strain[1],
            D[3] * strain[0] + D[4] * strain[1],
            D[8] * strain[3],
        ]

    fn commit(mut self):
        for i in range(len(self.soil)):
            self.soil[i].commit(self.stage)
        for i in range(len(self.pressure_dependent_soil)):
            self.pressure_dependent_soil[i].commit(self.stage)


struct SurfaceAssemblyCache(Movable):
    var shell: ShellAssemblyCache
    var continuum: ContinuumAssemblyCache
    var quads: List[QuadAssemblyEntry]
    var contacts: List[BeamContact2D]
    var contact_index: List[Int]
    var contact_friction: Bool
    var quad_index: List[Int]
    var up_quads: List[QuadUPAssemblyEntry]
    var up_quad_index: List[Int]
    var up_bricks: List[SSPBrickUPAssemblyEntry]
    var up_brick_index: List[Int]
    var pimy_definitions: List[PressureIndependMultiYieldDef]
    var pdmy_definitions: List[PressureDependMultiYieldDef]

    fn __init__(out self):
        self.shell = ShellAssemblyCache()
        self.continuum = ContinuumAssemblyCache()
        self.quads = []
        self.contacts = []
        self.contact_index = []
        self.contact_friction = True
        self.quad_index = []
        self.up_quads = []
        self.up_quad_index = []
        self.up_bricks = []
        self.up_brick_index = []
        self.pimy_definitions = []
        self.pdmy_definitions = []

    fn prepare_quads(
        mut self,
        indices: List[Int],
        elements: List[ElementInput],
        nodes: List[NodeInput],
        materials: List[MaterialInput],
        node_offsets: List[Int],
        node_pool: List[Int],
    ):
        if len(indices) == 0:
            return
        self.quad_index.resize(len(elements), -1)
        self.contact_index.resize(len(elements), -1)
        self.up_quad_index.resize(len(elements), -1)
        self.up_brick_index.resize(len(elements), -1)
        for e in indices:
            ref elem = elements[e]
            if elem.type_tag == ElementTypeTag.SSPBrickUP:
                self.up_brick_index[e] = len(self.up_bricks)
                self.up_bricks.append(
                    SSPBrickUPAssemblyEntry(
                        e,
                        elem,
                        nodes,
                        node_pool,
                        node_offsets[e],
                        materials[elem.material],
                        self.pdmy_definitions,
                    )
                )
                continue
            if elem.type_tag == ElementTypeTag.QuadUP:
                self.up_quad_index[e] = len(self.up_quads)
                self.up_quads.append(
                    QuadUPAssemblyEntry(
                        e,
                        elem,
                        nodes,
                        node_pool,
                        node_offsets[e],
                        materials[elem.material],
                        self.pimy_definitions,
                        self.pdmy_definitions,
                    )
                )
                continue
            if elem.type_tag == ElementTypeTag.BeamContact2D:
                self.contact_index[e] = len(self.contacts)
                var dofs = ContactDofs(fill=0)
                for i in range(10):
                    dofs[i] = _elem_dof(elem, i)
                var a = nodes[elem.node_index_1]
                var b = nodes[elem.node_index_2]
                var secondary = nodes[elem.node_index_3]
                ref parameters = materials[elem.material].contact_parameters
                self.contacts.append(
                    BeamContact2D(
                        e,
                        dofs,
                        Point2d(a.x, a.y),
                        Point2d(b.x, b.y),
                        Point2d(secondary.x, secondary.y),
                        elem.thickness,
                        elem.contact_gap_tol,
                        elem.contact_force_tol,
                        elem.contact_initial,
                        parameters[0],
                        parameters[1],
                        parameters[2],
                        parameters[3],
                    )
                )
                continue
            if elem.type_tag != ElementTypeTag.FourNodeQuad:
                continue
            self.quad_index[e] = len(self.quads)
            self.quads.append(
                QuadAssemblyEntry(
                    elem,
                    nodes,
                    materials[elem.material],
                    self.pimy_definitions,
                    self.pdmy_definitions,
                )
            )

    fn evaluate_contact(
        mut self, element: Int, u: List[Float64], form_tangent: Bool, mut tangent: List[Float64]
    ) -> ContactVector:
        return self.contacts[self.contact_index[element]].evaluate(
            u, self.contact_friction, form_tangent, tangent
        )

    fn evaluate_up(
        mut self,
        element: Int,
        dof_offset: Int,
        dof_pool: List[Int],
        u: List[Float64],
        tangent: Bool,
        mut force: QuadUPVector,
        mut matrix: List[Float64],
    ):
        ref entry = self.up_quads[self.up_quad_index[element]]
        var local_u = QuadUPVector(fill=0.0)
        for i in range(entry.dofs):
            local_u[i] = u[dof_pool[dof_offset + i]]
        entry.evaluate(
            self.pimy_definitions, self.pdmy_definitions, local_u, tangent, force, matrix
        )

    fn evaluate_brick_up(
        mut self,
        element: Int,
        dof_offset: Int,
        dof_pool: List[Int],
        u: List[Float64],
        tangent: Bool,
        mut force: BrickUPVector,
        mut matrix: List[Float64],
    ):
        ref entry = self.up_bricks[self.up_brick_index[element]]
        var local_u = BrickUPVector(fill=0.0)
        for i in range(32):
            local_u[i] = u[dof_pool[dof_offset + i]]
        entry.evaluate(self.pdmy_definitions, local_u, tangent, force, matrix)

    fn add_up_transient_matrices(
        self,
        mut assembler: SparseAssembler,
        dof_offsets: List[Int],
        free_dof_pool: List[Int],
        mut mass: List[Float64],
        mut mass_rayleigh: List[Float64],
        mut damping: List[Float64],
    ):
        zero_values(assembler.values, len(mass))
        for entry in self.up_quads:
            assembler.add_element_matrix_flat(
                entry.element, free_dof_pool, dof_offsets[entry.element], entry.dofs, entry.mass
            )
        for brick in self.up_bricks:
            assembler.add_element_matrix_flat(
                brick.element, free_dof_pool, dof_offsets[brick.element], 32, brick.mass
            )
        for i in range(len(mass)):
            mass[i] += assembler.values[i]

        if len(mass_rayleigh) > 0:
            zero_values(assembler.values, len(mass_rayleigh))
            for entry in self.up_quads:
                assembler.add_element_matrix_flat(
                    entry.element,
                    free_dof_pool,
                    dof_offsets[entry.element],
                    entry.dofs,
                    entry.solid_mass,
                )
            for brick in self.up_bricks:
                assembler.add_element_matrix_flat(
                    brick.element, free_dof_pool, dof_offsets[brick.element], 32, brick.solid_mass
                )
            for i in range(len(mass_rayleigh)):
                mass_rayleigh[i] += assembler.values[i]

        zero_values(assembler.values, len(mass))
        for entry in self.up_quads:
            assembler.add_element_matrix_flat(
                entry.element, free_dof_pool, dof_offsets[entry.element], entry.dofs, entry.damping
            )
        for brick in self.up_bricks:
            assembler.add_element_matrix_flat(
                brick.element, free_dof_pool, dof_offsets[brick.element], 32, brick.damping
            )
        damping = assembler.values.copy()

    fn add_up_excitation_mass(
        self, dof_offsets: List[Int], dof_pool: List[Int], mut mass: List[Float64]
    ):
        for entry in self.up_quads:
            for i in range(entry.dofs):
                mass[dof_pool[dof_offsets[entry.element] + i]] += entry.solid_row_mass[i]

        for brick in self.up_bricks:
            for i in range(32):
                mass[dof_pool[dof_offsets[brick.element] + i]] += brick.solid_mass[32 * i + i]

    fn replace_quad_up_stiffness_with_initial(
        mut self,
        mut assembler: SparseAssembler,
        dof_offsets: List[Int],
        free_dof_pool: List[Int],
    ):
        var difference: List[Float64] = []
        for i in range(len(self.up_quads)):
            ref entry = self.up_quads[i]
            entry.initial_stiffness_difference(
                self.pimy_definitions, self.pdmy_definitions, difference
            )
            assembler.add_element_matrix_flat(
                entry.element, free_dof_pool, dof_offsets[entry.element], entry.dofs, difference
            )

    fn add_up_prescribed_pressure_load(
        self,
        dof_offsets: List[Int],
        dof_pool: List[Int],
        pressure: List[Float64],
        mut force: List[Float64],
    ):
        for entry in self.up_quads:
            var offset = dof_offsets[entry.element]
            for j in range(entry.dofs):
                var value = pressure[dof_pool[offset + j]]
                if value != 0.0:
                    for i in range(entry.dofs):
                        force[dof_pool[offset + i]] -= entry.damping[entry.dofs * i + j] * value
        for brick in self.up_bricks:
            var offset = dof_offsets[brick.element]
            for j in range(32):
                var value = pressure[dof_pool[offset + j]]
                if value != 0.0:
                    for i in range(32):
                        force[dof_pool[offset + i]] -= brick.damping[32 * i + j] * value

    fn has_coupled_elements(self) -> Bool:
        return len(self.up_quads) > 0 or len(self.up_bricks) > 0

    fn has_soil_materials(self) -> Bool:
        return len(self.pimy_definitions) > 0 or len(self.pdmy_definitions) > 0

    fn capture_initial_state(mut self):
        for i in range(len(self.quads)):
            ref quad = self.quads[i]
            for gp in range(len(quad.initial_state_points)):
                var strain = quad.soil[gp].strain_c if quad.soil_definition >= 0 else (
                    quad.pressure_dependent_soil[gp].strain_c.vector()
                )
                if quad.pressure_dependent_definition >= 0:
                    for j in range(3, 6):
                        strain[j] *= 2.0
                quad.initial_state_points[gp].capture(strain)

    fn commit(mut self):
        for i in range(len(self.contacts)):
            if not self.contacts[i].removed:
                self.contacts[i].commit()
        self.shell.commit_layered_states()
        self.continuum.commit()
        if self.has_soil_materials():
            for i in range(len(self.quads)):
                self.quads[i].commit()
            for i in range(len(self.up_quads)):
                self.up_quads[i].commit()
            for i in range(len(self.up_bricks)):
                self.up_bricks[i].commit()

    fn record_quad(
        self,
        element: Int,
        point: Int,
        want_stress: Bool,
        stress_components: Int,
        dof_offset: Int,
        elem_dof_pool: List[Int],
        u: List[Float64],
        want_state: Bool = False,
    ) -> List[Float64]:
        if element < len(self.up_brick_index) and self.up_brick_index[element] >= 0:
            var local_u = BrickUPVector(fill=0.0)
            for i in range(32):
                local_u[i] = u[elem_dof_pool[dof_offset + i]]
            return self.up_bricks[self.up_brick_index[element]].record(
                want_stress, stress_components, local_u, self.pdmy_definitions
            )
        if element < len(self.up_quad_index) and self.up_quad_index[element] >= 0:
            ref entry = self.up_quads[self.up_quad_index[element]]
            var local_u = QuadUPVector(fill=0.0)
            for i in range(entry.dofs):
                local_u[i] = u[elem_dof_pool[dof_offset + i]]
            return entry.record(
                point, want_stress, stress_components, local_u, self.pdmy_definitions
            )
        if self.continuum.contains(element):
            return self.continuum.response(
                element, point, want_stress, stress_components, u, want_state
            )
        if want_state:
            abort("material state recorder requires DruckerPrager")
        if element >= len(self.quad_index) or self.quad_index[element] < 0:
            abort("material recorder requires a supported solid element")
        if point < 0 or point >= 4:
            abort("fourNodeQuad material recorder integration point out of range")
        var local_u = QuadVector(fill=0.0)
        for i in range(8):
            local_u[i] = u[elem_dof_pool[dof_offset + i]]
        return self.quads[self.quad_index[element]].record(
            point, want_stress, stress_components, local_u, self.pdmy_definitions
        )

    fn update_material_stage(mut self, material: Int, stage: Int):
        for i in range(len(self.quads)):
            if self.quads[i].material == material:
                self.quads[i].stage = stage
        for i in range(len(self.up_quads)):
            if self.up_quads[i].material == material:
                self.up_quads[i].stage = stage
        for i in range(len(self.up_bricks)):
            if self.up_bricks[i].material == material:
                self.up_bricks[i].stage = stage
