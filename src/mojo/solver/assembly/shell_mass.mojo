from collections import List

from elements.shell_mass import ShellMass
from solver.sparse_pattern import SparseAssembler


fn add_shell_mass_sparse(
    masses: List[ShellMass], mut assembler: SparseAssembler, mut values: List[Float64]
):
    if len(masses) == 0:
        return
    var matrix: List[Float64] = []
    var unused_free_pool: List[Int] = []
    swap(assembler.values, values)
    for mass in masses:
        var ndof = 6 * mass.node_count
        matrix.resize(ndof * ndof, 0.0)
        for i in range(mass.node_count):
            for j in range(mass.node_count):
                for d in range(3):
                    matrix[(6 * i + d) * ndof + 6 * j + d] = mass.weights[4 * i + j]
        assembler.add_element_matrix_flat(mass.element, unused_free_pool, 0, ndof, matrix)
        matrix.clear()
    swap(assembler.values, values)


fn shell_excitation_mass(masses: List[ShellMass], diagonal: List[Float64]) -> List[Float64]:
    # Every node has the same translational influence vector under
    # UniformExcitation, so its force uses the full element mass row sum.
    var out = diagonal.copy()
    for mass in masses:
        for i in range(mass.node_count):
            var row_sum = 0.0
            for j in range(mass.node_count):
                row_sum += mass.weights[4 * i + j]
            for d in range(3):
                out[mass.dofs[i] + d] += row_sum
    return out^


fn add_shell_mass_reduced(
    masses: List[ShellMass],
    free_index: List[Int],
    offsets: List[Int],
    dofs: List[Int],
    coeffs: List[Float64],
    mut matrix: List[List[Float64]],
):
    for mass in masses:
        for i in range(mass.node_count):
            for j in range(mass.node_count):
                var weight = mass.weights[4 * i + j]
                for d in range(3):
                    var row = mass.dofs[i] + d
                    var col = mass.dofs[j] + d
                    for a in range(offsets[row], offsets[row + 1]):
                        var free_a = free_index[dofs[a]]
                        if free_a < 0:
                            continue
                        for b in range(offsets[col], offsets[col + 1]):
                            var free_b = free_index[dofs[b]]
                            if free_b >= 0:
                                matrix[free_a][free_b] += weight * coeffs[a] * coeffs[b]
