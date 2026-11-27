from collections import List
from elements.continuum import (
    ContinuumElement,
    ContinuumVector,
    isotropic_continuum_tangent,
    prepare_continuum,
)
from materials.elastic_cross_anisotropic import cross_anisotropic_tangent
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput
from solver.sparse_pattern import ElementMatrixSink, SparseAssembler
from os import abort


@fieldwise_init
struct ContinuumAssemblyEntry(Copyable, Movable):
    var index: Int
    var dofs: List[Int]
    var element: ContinuumElement

    fn gather(self, u: List[Float64]) -> List[Float64]:
        var local: List[Float64] = []
        for d in self.dofs:
            local.append(u[d])
        return local^

    fn add_force(mut self, u: List[Float64], tangent: Bool, mut force: List[Float64]):
        var local = self.element.evaluate(self.gather(u), tangent)
        for a in range(len(self.dofs)):
            force[self.dofs[a]] += local[a]


struct ContinuumAssemblyCache(Movable):
    var nonlinear: List[Int]
    var index_by_element: List[Int]
    var entries: List[ContinuumAssemblyEntry]

    fn __init__(out self):
        self.nonlinear = []
        self.index_by_element = []
        self.entries = []

    fn prepare(
        mut self,
        indices: List[Int],
        elements: List[ElementInput],
        nodes: List[NodeInput],
        materials: List[MaterialInput],
        node_offsets: List[Int],
        node_pool: List[Int],
        dof_offsets: List[Int],
        dof_pool: List[Int],
    ):
        if len(indices) == 0:
            return
        self.index_by_element.resize(len(elements), -1)
        for i in indices:
            ref elem = elements[i]
            ref mat = materials[elem.material]
            var coords: List[ContinuumVector] = []
            for a in range(node_offsets[i], node_offsets[i + 1]):
                ref node = nodes[node_pool[a]]
                coords.append(ContinuumVector(node.x, node.y, node.z, 0.0))
            var dofs: List[Int] = []
            for a in range(dof_offsets[i], dof_offsets[i + 1]):
                dofs.append(dof_pool[a])
            var components = (
                3 if elem.dof_count == 2 * elem.node_count or elem.type == "quad3d" else 6
            )
            if elem.type == "bbarQuad":
                components = 6
            var tangent = isotropic_continuum_tangent(
                mat.E, mat.nu, components, elem.formulation == "PlaneStrain"
            )
            if mat.type == "ElasticCrossAnisotropic":
                var p = mat.cross_anisotropic
                if p[0] <= 0.0 or p[1] <= 0.0 or p[4] <= 0.0:
                    abort("ElasticCrossAnisotropic requires positive moduli")
                tangent = cross_anisotropic_tangent(p[0], p[1], p[2], p[3], p[4])
            if mat.drucker_prager:
                tangent = mat.drucker_prager.value().elastic_tangent()
            var density = elem.quad_density if elem.quad_density != 0.0 else mat.rho
            var body = ContinuumVector(
                elem.quad_body_x, elem.quad_body_y, elem.continuum_body_z, 0.0
            )
            if elem.type == "Brick8N" or elem.type == "Brick27N":
                body *= density
            self.index_by_element[i] = len(self.entries)
            var solid = prepare_continuum(
                elem.type, coords, elem.thickness, density, body, elem.quad_pressure, tangent
            )
            if mat.drucker_prager:
                solid.set_drucker_prager(mat.drucker_prager.value())
                self.nonlinear.append(len(self.entries))
            self.entries.append(ContinuumAssemblyEntry(i, dofs^, solid^))

    fn commit(mut self):
        for i in self.nonlinear:
            self.entries[i].element.commit()

    fn contains(self, index: Int) -> Bool:
        return (
            index >= 0 and index < len(self.index_by_element) and self.index_by_element[index] >= 0
        )

    fn assemble_native[
        S: ElementMatrixSink
    ](
        mut self,
        free_offsets: List[Int],
        free_pool: List[Int],
        u: List[Float64],
        mut sink: S,
        mut force: List[Float64],
    ):
        var tangent = sink.accepts_element_matrices()
        for i in range(len(self.entries)):
            ref entry = self.entries[i]
            entry.add_force(u, tangent, force)
            if tangent:
                sink.add_element_matrix_flat(
                    entry.index,
                    free_pool,
                    free_offsets[entry.index],
                    entry.element.dofs,
                    entry.element.stiffness,
                )

    fn assemble_dense_element(
        mut self,
        index: Int,
        u: List[Float64],
        mut K: List[List[Float64]],
        mut force: List[Float64],
    ):
        ref entry = self.entries[self.index_by_element[index]]
        entry.add_force(u, True, force)
        var count = len(entry.dofs)
        for a in range(count):
            for b in range(count):
                K[entry.dofs[a]][entry.dofs[b]] += entry.element.stiffness[a * count + b]

    fn assemble_dense(
        mut self, u: List[Float64], mut K: List[List[Float64]], mut force: List[Float64]
    ):
        for i in range(len(self.entries)):
            self.assemble_dense_element(self.entries[i].index, u, K, force)

    fn accumulate_lumped_mass(self, mut mass: List[Float64], mut rayleigh: List[Float64]):
        for i in range(len(self.entries)):
            ref entry = self.entries[i]
            for a in range(len(entry.dofs)):
                var d = entry.dofs[a]
                var value = entry.element.lumped_mass[a]
                mass[d] += value
                rayleigh[d] += value

    fn add_consistent_mass(
        self,
        mut assembler: SparseAssembler,
        mut mass: List[Float64],
        free_offsets: List[Int],
        free_pool: List[Int],
    ):
        if len(self.entries) == 0:
            return
        swap(assembler.values, mass)
        for i in range(len(self.entries)):
            ref entry = self.entries[i]
            if len(entry.element.mass) > 0:
                assembler.add_element_matrix_flat(
                    entry.index,
                    free_pool,
                    free_offsets[entry.index],
                    entry.element.dofs,
                    entry.element.mass,
                )
        swap(assembler.values, mass)

    fn resisting_force(self, index: Int, u: List[Float64]) -> List[Float64]:
        ref entry = self.entries[self.index_by_element[index]]
        if entry.element.drucker_prager:
            return entry.element.trial_force.copy()
        return entry.element.resisting_force(entry.gather(u))

    fn add_modal_mass(
        self,
        free_index: List[Int],
        row_offsets: List[Int],
        dof_pool: List[Int],
        coeff_pool: List[Float64],
        mut mass: List[List[Float64]],
    ):
        for i in range(len(self.entries)):
            ref entry = self.entries[i]
            if len(entry.element.mass) == 0:
                continue
            var count = len(entry.dofs)
            for a in range(count):
                var row = entry.dofs[a]
                for b in range(count):
                    var value = entry.element.mass[a * count + b]
                    if value == 0.0:
                        continue
                    var col = entry.dofs[b]
                    for ia in range(row_offsets[row], row_offsets[row + 1]):
                        var r = free_index[dof_pool[ia]]
                        if r < 0:
                            continue
                        for ib in range(row_offsets[col], row_offsets[col + 1]):
                            var c = free_index[dof_pool[ib]]
                            if c >= 0:
                                mass[r][c] += value * coeff_pool[ia] * coeff_pool[ib]

    fn response(
        self,
        index: Int,
        point: Int,
        stress: Bool,
        components: Int,
        u: List[Float64],
        state: Bool = False,
    ) -> List[Float64]:
        ref entry = self.entries[self.index_by_element[index]]
        var values = entry.element.response(entry.gather(u), point, stress, state)
        if components > 0 and components < len(values):
            values.resize(components, 0.0)
        return values^
