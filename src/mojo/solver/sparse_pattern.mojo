"""Shared sparse storage for reduced (free-DOF) global matrices.

Every global matrix of a transient analysis (current tangent, initial and
committed tangents, mass, link-filtered Rayleigh variants, zero-length damping)
shares one `SparsePattern`. A matrix is then just a flat `List[Float64]` of
`pattern.nnz()` values, indexed by pattern slot.

Reduced space
-------------
The pattern lives in the reduced equation space used by the linear solver:
single-point constraints are removed and multi-point constraints
(Transformation handler) are already applied. A full-space DOF `d` expands to
the reduced terms `(free_index[mpc_dof_pool[t]], mpc_coeff_pool[t])` for
`t in [mpc_row_offsets[d], mpc_row_offsets[d + 1])` (terms whose retained DOF
is constrained are dropped). Without MPCs a free DOF expands to itself with
coefficient 1.0 and a constrained DOF to nothing.

`SparsePattern` layout (CSR, structurally symmetric)
----------------------------------------------------
- `n`: number of reduced equations.
- `row_start[n + 1]`: row `i` owns slots `[row_start[i], row_start[i + 1])`.
- `col_index[nnz]`: column of each slot, strictly increasing inside a row.
- `diag_slot[n]`: slot of `(i, i)`; every row stores its diagonal.
The pattern is the union of element couplings, reduced mass couplings and the
diagonal, all symmetric, so `(i, j)` is stored iff `(j, i)` is stored.

Lower-triangular CSC view (seam for a sparse LDL^T solver)
---------------------------------------------------------
Because the pattern is structurally symmetric, the lower triangle in CSC form
is the upper triangle of the CSR rows: column `j` of the lower CSC holds rows
`col_index[s]` for `s in [diag_slot[j], row_start[j + 1])` (diagonal first,
rows ascending). For a symmetric matrix the value of lower entry `(i, j)` is
`values[s]` for that same slot `s` (it equals the CSR entry `(j, i)`).
`build_lower_csc` materializes `col_start`, `row_index` and the matching
pattern slots, so a factorization can read `values[slot[p]]` directly.
"""

from collections import List
from os import abort


trait ElementMatrixSink:
    """Target that element routines scatter their global matrices into.

    `elem_index` identifies the element; `elem_free_pool[free_offset + a]` is
    the reduced equation of local DOF `a` (-1 when constrained) for sinks that
    address storage by equation number. Sinks with precomputed slot maps use
    only `elem_index`.
    """

    fn accepts_element_matrices(self) -> Bool:
        ...

    fn zero_matrix(mut self):
        ...

    fn add_element_matrix(
        mut self,
        elem_index: Int,
        elem_free_pool: List[Int],
        free_offset: Int,
        dof_count: Int,
        k_elem: List[List[Float64]],
    ):
        ...

    fn add_element_matrix_flat(
        mut self,
        elem_index: Int,
        elem_free_pool: List[Int],
        free_offset: Int,
        dof_count: Int,
        k_flat: List[Float64],
        k_offset: Int = 0,
    ):
        """`k_flat[k_offset + a * dof_count + b]` is local entry (a, b)."""
        ...


struct SparsePattern(Copyable, Movable):
    var n: Int
    var row_start: List[Int]
    var col_index: List[Int]
    var diag_slot: List[Int]

    fn __init__(out self):
        self.n = 0
        self.row_start = [0]
        self.col_index = []
        self.diag_slot = []

    @always_inline
    fn nnz(self) -> Int:
        return len(self.col_index)

    fn find_slot(self, row: Int, col: Int) -> Int:
        """Binary search for `(row, col)`; returns -1 when not stored."""
        var lo = self.row_start[row]
        var hi = self.row_start[row + 1] - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            var c = self.col_index[mid]
            if c == col:
                return mid
            if c < col:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    fn require_slot(self, row: Int, col: Int) -> Int:
        var slot = self.find_slot(row, col)
        if slot < 0:
            abort("sparse pattern missing structural entry")
        return slot


fn build_transpose_slots(pattern: SparsePattern) -> List[Int]:
    """`out[s]` is the slot of `(j, i)` for slot `s` at `(i, j)`. O(nnz)."""
    var n = pattern.n
    var cursor: List[Int] = []
    cursor.resize(n, 0)
    for i in range(n):
        cursor[i] = pattern.row_start[i]
    var out: List[Int] = []
    out.resize(pattern.nnz(), -1)
    for i in range(n):
        for s in range(pattern.row_start[i], pattern.row_start[i + 1]):
            var j = pattern.col_index[s]
            var t = cursor[j]
            if t >= pattern.row_start[j + 1] or pattern.col_index[t] != i:
                abort("sparse pattern is not structurally symmetric")
            out[s] = t
            cursor[j] = t + 1
    return out^


fn build_lower_csc(
    pattern: SparsePattern,
    mut col_start: List[Int],
    mut row_index: List[Int],
    mut slot: List[Int],
):
    """Materialize the lower-triangular CSC view described in the module doc."""
    var n = pattern.n
    col_start.resize(n + 1, 0)
    col_start[0] = 0
    for j in range(n):
        col_start[j + 1] = col_start[j] + (pattern.row_start[j + 1] - pattern.diag_slot[j])
    var total = col_start[n]
    row_index.resize(total, 0)
    slot.resize(total, 0)
    var p = 0
    for j in range(n):
        for s in range(pattern.diag_slot[j], pattern.row_start[j + 1]):
            row_index[p] = pattern.col_index[s]
            slot[p] = s
            p += 1


struct ElementScatterMap(Movable):
    """Precomputed element-to-pattern scatter.

    For element `e` with local DOFs at `elem_dof_pool[elem_dof_start[e] + a]`:
    - local DOF `a` expands to reduced terms `t` in
      `[dof_term_start[elem_dof_start[e] + a], dof_term_start[... + a + 1])`
      with equation `term_free[t]` and coefficient `term_coeff[t]`;
    - the element's terms are contiguous, `T_e` of them starting at
      `t_base = dof_term_start[elem_dof_start[e]]`;
    - term pair `(p, q)` (element-relative term indices) lands in pattern slot
      `slots[elem_slot_start[e] + p * T_e + q]`.
    `unit_terms` is True when every DOF has at most one term with coefficient
    1.0 (no MPC expansion), which enables a multiply-free scatter.
    `elem_unit_terms[e]` says the same for one element, so a model with a few
    constrained nodes still scatters its unconstrained elements that way.
    """

    var elem_count: Int
    var elem_dof_start: List[Int]
    var dof_term_start: List[Int]
    var term_free: List[Int]
    var term_coeff: List[Float64]
    var elem_slot_start: List[Int]
    var slots: List[Int32]
    var unit_terms: Bool
    var elem_unit_terms: List[Bool]

    fn __init__(out self):
        self.elem_count = 0
        self.elem_dof_start = [0]
        self.dof_term_start = [0]
        self.term_free = []
        self.term_coeff = []
        self.elem_slot_start = [0]
        self.slots = []
        self.unit_terms = True
        self.elem_unit_terms = []


struct NullMatrixSink(ElementMatrixSink, Movable):
    """Sink that discards element matrices.

    Residual-only sweeps (line search, Krylov residuals, reactions) need the
    internal forces and the material state update but no tangent. Assembling
    into this sink keeps them on the same element code path as the tangent
    assembly without allocating or filling a global matrix.
    """

    fn __init__(out self):
        pass

    fn accepts_element_matrices(self) -> Bool:
        return False

    fn zero_matrix(mut self):
        pass

    fn add_element_matrix(
        mut self,
        elem_index: Int,
        elem_free_pool: List[Int],
        free_offset: Int,
        dof_count: Int,
        k_elem: List[List[Float64]],
    ):
        pass

    fn add_element_matrix_flat(
        mut self,
        elem_index: Int,
        elem_free_pool: List[Int],
        free_offset: Int,
        dof_count: Int,
        k_flat: List[Float64],
        k_offset: Int = 0,
    ):
        pass


struct SparseAssembler(ElementMatrixSink, Movable):
    """Scatters element matrices into `values` (one value array over the pattern).

    To assemble into another value array over the same pattern, `swap` it with
    `values`, assemble, and swap back.
    """

    var scatter: ElementScatterMap
    var values: List[Float64]

    fn __init__(out self, var scatter: ElementScatterMap, nnz: Int):
        self.scatter = scatter^
        self.values = []
        self.values.resize(nnz, 0.0)

    fn accepts_element_matrices(self) -> Bool:
        return True

    fn zero_matrix(mut self):
        for i in range(len(self.values)):
            self.values[i] = 0.0

    @always_inline
    fn _check(self, elem_index: Int, dof_count: Int):
        if elem_index < 0 or elem_index >= self.scatter.elem_count:
            abort("sparse assembler element index out of range")
        if (
            self.scatter.elem_dof_start[elem_index + 1] - self.scatter.elem_dof_start[elem_index]
            != dof_count
        ):
            abort("sparse assembler element dof count mismatch")

    fn add_element_matrix(
        mut self,
        elem_index: Int,
        elem_free_pool: List[Int],
        free_offset: Int,
        dof_count: Int,
        k_elem: List[List[Float64]],
    ):
        self._check(elem_index, dof_count)
        var dof0 = self.scatter.elem_dof_start[elem_index]
        var t_base = self.scatter.dof_term_start[dof0]
        var t_count = self.scatter.dof_term_start[dof0 + dof_count] - t_base
        var slot_base = self.scatter.elem_slot_start[elem_index] - t_base * t_count - t_base
        if self.scatter.elem_unit_terms[elem_index]:
            for a in range(dof_count):
                var ta = self.scatter.dof_term_start[dof0 + a]
                if ta == self.scatter.dof_term_start[dof0 + a + 1]:
                    continue
                var row_base = slot_base + ta * t_count
                for b in range(dof_count):
                    var tb = self.scatter.dof_term_start[dof0 + b]
                    if tb == self.scatter.dof_term_start[dof0 + b + 1]:
                        continue
                    self.values[Int(self.scatter.slots[row_base + tb])] += k_elem[a][b]
            return
        for a in range(dof_count):
            var ta0 = self.scatter.dof_term_start[dof0 + a]
            var ta1 = self.scatter.dof_term_start[dof0 + a + 1]
            if ta0 == ta1:
                continue
            for b in range(dof_count):
                var tb0 = self.scatter.dof_term_start[dof0 + b]
                var tb1 = self.scatter.dof_term_start[dof0 + b + 1]
                if tb0 == tb1:
                    continue
                var value = k_elem[a][b]
                for ta in range(ta0, ta1):
                    var ca_value = self.scatter.term_coeff[ta] * value
                    var row_base = slot_base + ta * t_count
                    for tb in range(tb0, tb1):
                        self.values[Int(self.scatter.slots[row_base + tb])] += (
                            ca_value * self.scatter.term_coeff[tb]
                        )

    fn add_element_matrix_flat(
        mut self,
        elem_index: Int,
        elem_free_pool: List[Int],
        free_offset: Int,
        dof_count: Int,
        k_flat: List[Float64],
        k_offset: Int = 0,
    ):
        self._check(elem_index, dof_count)
        if k_offset < 0 or len(k_flat) - k_offset < dof_count * dof_count:
            abort("sparse assembler flat element matrix size mismatch")
        var dof0 = self.scatter.elem_dof_start[elem_index]
        var t_base = self.scatter.dof_term_start[dof0]
        var t_count = self.scatter.dof_term_start[dof0 + dof_count] - t_base
        var matrix = k_flat.unsafe_ptr() + k_offset
        var elem_slot = self.scatter.elem_slot_start[elem_index]
        var slots = self.scatter.slots.unsafe_ptr() + elem_slot
        var values = self.values.unsafe_ptr()
        var terms = self.scatter.dof_term_start.unsafe_ptr() + dof0
        if self.scatter.elem_unit_terms[elem_index]:
            if t_count == dof_count:
                # Every local DOF is active. The row-major slots and matrix
                # entries have the same layout, including repeated equations.
                for i in range(dof_count * dof_count):
                    values[Int(slots[i])] += matrix[i]
                return
            for a in range(dof_count):
                var ta = terms[a]
                if ta == terms[a + 1]:
                    continue
                var row_slots = slots + (ta - t_base) * t_count
                var row_values = matrix + a * dof_count
                for b in range(dof_count):
                    var tb = terms[b]
                    if tb != terms[b + 1]:
                        values[Int(row_slots[tb - t_base])] += row_values[b]
            return

        # Constrained element: expand each local DOF into its reduced terms.
        var coeff = self.scatter.term_coeff.unsafe_ptr()
        for a in range(dof_count):
            var ta0 = terms[a]
            var ta1 = terms[a + 1]
            if ta0 == ta1:
                continue
            var row_values = matrix + a * dof_count
            for b in range(dof_count):
                var tb0 = terms[b]
                var tb1 = terms[b + 1]
                if tb0 == tb1:
                    continue
                var value = row_values[b]
                for ta in range(ta0, ta1):
                    var ca_value = coeff[ta] * value
                    var row_slots = slots + (ta - t_base) * t_count - t_base
                    for tb in range(tb0, tb1):
                        values[Int(row_slots[tb])] += ca_value * coeff[tb]


@always_inline
fn _append_dof_terms(
    dof: Int,
    use_mpc: Bool,
    free_index: List[Int],
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    mut term_free: List[Int],
    mut term_coeff: List[Float64],
):
    if use_mpc:
        for t in range(mpc_row_offsets[dof], mpc_row_offsets[dof + 1]):
            var reduced = free_index[mpc_dof_pool[t]]
            if reduced < 0:
                continue
            term_free.append(reduced)
            term_coeff.append(mpc_coeff_pool[t])
        return
    var reduced = free_index[dof]
    if reduced >= 0:
        term_free.append(reduced)
        term_coeff.append(1.0)


fn build_sparse_assembler(
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    free_index: List[Int],
    free_count: Int,
    use_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    mass_diag: List[Float64],
    mass_diag_secondary: List[Float64],
    mut pattern: SparsePattern,
) -> SparseAssembler:
    """Build the shared reduced pattern and the element scatter map.

    The pattern is the union of the diagonal, every element's reduced term
    couplings, and the reduced couplings of every full DOF with nonzero mass in
    `mass_diag` or `mass_diag_secondary` (MPC expansion makes these non-diagonal).
    """
    var elem_count = len(elem_dof_offsets) - 1
    if elem_count < 0 or len(elem_dof_pool) != elem_dof_offsets[elem_count]:
        abort("sparse pattern invalid element dof map")
    var total_dofs = len(free_index)
    if use_mpc and len(mpc_row_offsets) != total_dofs + 1:
        abort("sparse pattern invalid mpc row offsets")

    var scatter = ElementScatterMap()
    scatter.elem_count = elem_count
    scatter.elem_dof_start = elem_dof_offsets.copy()
    scatter.dof_term_start.resize(len(elem_dof_pool) + 1, 0)
    for p in range(len(elem_dof_pool)):
        scatter.dof_term_start[p] = len(scatter.term_free)
        _append_dof_terms(
            elem_dof_pool[p],
            use_mpc,
            free_index,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            scatter.term_free,
            scatter.term_coeff,
        )
        if scatter.dof_term_start[p] + 1 < len(scatter.term_free):
            scatter.unit_terms = False
    scatter.dof_term_start[len(elem_dof_pool)] = len(scatter.term_free)
    for t in range(len(scatter.term_coeff)):
        if scatter.term_coeff[t] != 1.0:
            scatter.unit_terms = False
            break
    scatter.elem_unit_terms.resize(elem_count, True)
    if not scatter.unit_terms:
        for e in range(elem_count):
            var unit = True
            for p in range(elem_dof_offsets[e], elem_dof_offsets[e + 1]):
                var t0 = scatter.dof_term_start[p]
                var t1 = scatter.dof_term_start[p + 1]
                if t1 - t0 > 1 or (t1 - t0 == 1 and scatter.term_coeff[t0] != 1.0):
                    unit = False
                    break
            scatter.elem_unit_terms[e] = unit

    # Coupling groups: elements first, then MPC-expanded mass rows.
    var group_start: List[Int] = []
    var group_member: List[Int] = []
    for e in range(elem_count):
        group_start.append(len(group_member))
        var t0 = scatter.dof_term_start[elem_dof_offsets[e]]
        var t1 = scatter.dof_term_start[elem_dof_offsets[e + 1]]
        for t in range(t0, t1):
            group_member.append(scatter.term_free[t])
    var mass_free: List[Int] = []
    var mass_coeff: List[Float64] = []
    for d in range(total_dofs):
        var has_mass = False
        if d < len(mass_diag) and mass_diag[d] != 0.0:
            has_mass = True
        if d < len(mass_diag_secondary) and mass_diag_secondary[d] != 0.0:
            has_mass = True
        if not has_mass:
            continue
        mass_free.clear()
        mass_coeff.clear()
        _append_dof_terms(
            d,
            use_mpc,
            free_index,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            mass_free,
            mass_coeff,
        )
        if len(mass_free) < 2:
            continue
        group_start.append(len(group_member))
        for i in range(len(mass_free)):
            group_member.append(mass_free[i])
    var group_count = len(group_start)
    group_start.append(len(group_member))

    # Equation -> group incidence (CSR).
    var inc_start: List[Int] = []
    inc_start.resize(free_count + 1, 0)
    for m in range(len(group_member)):
        inc_start[group_member[m] + 1] += 1
    for i in range(free_count):
        inc_start[i + 1] += inc_start[i]
    var inc_fill = inc_start.copy()
    var inc_group: List[Int] = []
    inc_group.resize(len(group_member), 0)
    for g in range(group_count):
        for m in range(group_start[g], group_start[g + 1]):
            var r = group_member[m]
            inc_group[inc_fill[r]] = g
            inc_fill[r] += 1

    pattern.n = free_count
    pattern.row_start.resize(free_count + 1, 0)
    pattern.row_start[0] = 0
    pattern.col_index.clear()
    pattern.diag_slot.resize(free_count, 0)
    var marker: List[Int] = []
    marker.resize(free_count, -1)
    var row_cols: List[Int] = []
    for r in range(free_count):
        row_cols.clear()
        row_cols.append(r)
        marker[r] = r
        for k in range(inc_start[r], inc_start[r + 1]):
            var g = inc_group[k]
            for m in range(group_start[g], group_start[g + 1]):
                var c = group_member[m]
                if marker[c] != r:
                    marker[c] = r
                    row_cols.append(c)
        sort(row_cols)
        for k in range(len(row_cols)):
            if row_cols[k] == r:
                pattern.diag_slot[r] = len(pattern.col_index)
            pattern.col_index.append(row_cols[k])
        pattern.row_start[r + 1] = len(pattern.col_index)

    # Element slot blocks.
    scatter.elem_slot_start.resize(elem_count + 1, 0)
    var slot_total = 0
    for e in range(elem_count):
        scatter.elem_slot_start[e] = slot_total
        var count = (
            scatter.dof_term_start[elem_dof_offsets[e + 1]]
            - scatter.dof_term_start[elem_dof_offsets[e]]
        )
        slot_total += count * count
    scatter.elem_slot_start[elem_count] = slot_total
    scatter.slots.resize(slot_total, Int32(-1))
    for e in range(elem_count):
        var t0 = scatter.dof_term_start[elem_dof_offsets[e]]
        var t1 = scatter.dof_term_start[elem_dof_offsets[e + 1]]
        var count = t1 - t0
        var base = scatter.elem_slot_start[e]
        for p in range(count):
            var row = scatter.term_free[t0 + p]
            for q in range(count):
                var col = scatter.term_free[t0 + q]
                scatter.slots[base + p * count + q] = Int32(pattern.require_slot(row, col))

    return SparseAssembler(scatter^, pattern.nnz())


fn build_reduced_mass_values(
    pattern: SparsePattern,
    mass_diag: List[Float64],
    free_index: List[Int],
    use_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
) -> List[Float64]:
    """Reduced mass `T^T diag(m) T` as values over `pattern`."""
    var values: List[Float64] = []
    values.resize(pattern.nnz(), 0.0)
    var term_free: List[Int] = []
    var term_coeff: List[Float64] = []
    for d in range(len(mass_diag)):
        var mass = mass_diag[d]
        if mass == 0.0:
            continue
        term_free.clear()
        term_coeff.clear()
        _append_dof_terms(
            d,
            use_mpc,
            free_index,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            term_free,
            term_coeff,
        )
        for i in range(len(term_free)):
            var mass_ci = mass * term_coeff[i]
            for j in range(len(term_free)):
                var slot = pattern.require_slot(term_free[i], term_free[j])
                values[slot] += mass_ci * term_coeff[j]
    return values^


fn sparse_values_any_nonzero(values: List[Float64]) -> Bool:
    for i in range(len(values)):
        if values[i] != 0.0:
            return True
    return False


fn sparse_matvec(
    pattern: SparsePattern, values: List[Float64], x: List[Float64], mut y: List[Float64]
):
    """y = A x."""
    for i in range(pattern.n):
        var sum = 0.0
        for s in range(pattern.row_start[i], pattern.row_start[i + 1]):
            sum += values[s] * x[pattern.col_index[s]]
        y[i] = sum


fn copy_values(mut dst: List[Float64], src: List[Float64]):
    if len(dst) != len(src):
        dst.resize(len(src), 0.0)
    for i in range(len(src)):
        dst[i] = src[i]


fn zero_values(mut values: List[Float64], nnz: Int):
    if len(values) != nnz:
        values.resize(nnz, 0.0)
    for i in range(nnz):
        values[i] = 0.0


fn link_filtered_values(
    mut dst: List[Float64],
    k_values: List[Float64],
    link_all: List[Float64],
    link_rayleigh: List[Float64],
):
    """dst = K - K_link_all + K_link_rayleigh (slot-wise)."""
    if len(dst) != len(k_values):
        dst.resize(len(k_values), 0.0)
    for s in range(len(k_values)):
        dst[s] = k_values[s] - link_all[s] + link_rayleigh[s]


fn build_elem_free_pool(elem_dof_pool: List[Int], free_index: List[Int]) -> List[Int]:
    """Element DOF pool mapped to reduced equations (-1 when not free)."""
    var out: List[Int] = []
    out.resize(len(elem_dof_pool), -1)
    for i in range(len(elem_dof_pool)):
        var dof = elem_dof_pool[i]
        if dof >= 0 and dof < len(free_index):
            out[i] = free_index[dof]
    return out^
